import unittest
from nitwit import webanswer
from nitwit.router import Endpoint


class _Resp:
    def __init__(self, content): self.content = content


class _Client:
    """Fake non-streaming chat client. `script` maps call-index -> content string."""
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def chat(self, messages, *, temperature, max_tokens, response_format=None):
        self.calls.append(messages)
        return _Resp(self._replies.pop(0) if self._replies else "")


class TestSynthesize(unittest.TestCase):
    def test_uses_context_and_query(self):
        c = _Client(["Chapter 1140 is the latest (https://viz.com)."])
        out = webanswer.synthesize("latest chapter?", "CONTEXT text 1140", client=c)
        self.assertEqual(out, "Chapter 1140 is the latest (https://viz.com).")
        joined = "\n".join(m["content"] for m in c.calls[0])
        self.assertIn("CONTEXT text 1140", joined)
        self.assertIn("ONLY using the CONTEXT", joined)

    def test_never_raises(self):
        class Boom:
            def chat(self, *a, **k): raise RuntimeError("down")
        self.assertEqual(webanswer.synthesize("q", "ctx", client=Boom()), "")


class TestVerifyGrounding(unittest.TestCase):
    def test_returns_unsupported_list(self):
        c = _Client(['{"unsupported": ["chapter 1187", "July 5 2026"]}'])
        self.assertEqual(webanswer.verify_grounding("ans", "ctx", client=c),
                         ["chapter 1187", "July 5 2026"])

    def test_fails_open_on_garbage(self):
        c = _Client(["not json at all"])
        self.assertEqual(webanswer.verify_grounding("ans", "ctx", client=c), [])

    def test_fails_open_on_exception(self):
        class Boom:
            def chat(self, *a, **k): raise RuntimeError("x")
        self.assertEqual(webanswer.verify_grounding("ans", "ctx", client=Boom()), [])

    def test_empty_answer_short_circuits(self):
        c = _Client(['{"unsupported": ["x"]}'])
        self.assertEqual(webanswer.verify_grounding("   ", "ctx", client=c), [])
        self.assertEqual(c.calls, [])  # no model call for an empty answer


class TestHedgeAndClean(unittest.TestCase):
    def test_hedge_drops_sentence_with_unsupported_number(self):
        ans = "Chapter 1187 releases July 5, 2026. You can read it free on VIZ."
        out = webanswer._hedge(ans, ["chapter 1187", "July 5, 2026"])
        self.assertNotIn("1187", out)
        self.assertIn("VIZ", out)

    def test_clean_strips_lead_disclaimer(self):
        out = webanswer.clean("I can't perform real-time web searches, but chapter 1140 is out.\n\n\n")
        self.assertNotIn("real-time", out)
        self.assertIn("1140", out)


def _route(stage, *, gpu_up=True, cpu_up=True, health=None):
    # returns synth->8080 when gpu_up, else chat->8086; verify->8086
    if stage in ("synth", "code"):
        return Endpoint("http://127.0.0.1:8080", "coder", {}) if gpu_up else Endpoint("http://127.0.0.1:8086", "4b", {})
    return Endpoint("http://127.0.0.1:8086", "4b", {})


class TestAnswerWeb(unittest.TestCase):
    def _search(self, q, limit=6):
        return "WEB RESULTS:\n- A: blurb (https://a.example/x)"

    def _fetch(self, u):
        return "chapter 1140 is the current latest chapter"

    def test_clean_answer_passes_through_no_correction(self):
        # synth -> good answer; verify -> no unsupported. Exactly one synth call, no cooldown.
        clients = {}
        def factory(url, model, extra_body=None):
            # synth client (8080) then verify client (8086)
            c = _Client(["Chapter 1140 is the latest (https://a.example/x)."]) if "8080" in url \
                else _Client(['{"unsupported": []}'])
            clients.setdefault(url, c)
            return clients[url]
        slept = []
        out = []
        ans = webanswer.answer_web("latest chapter?", out=out.append,
                                   route=lambda s: _route(s, gpu_up=True),
                                   factory=factory, search=self._search, fetch=self._fetch,
                                   sleep=slept.append)
        self.assertIn("1140", ans)
        self.assertEqual(slept, [])                       # no correction → no cooldown
        self.assertEqual(len(clients["http://127.0.0.1:8080"].calls), 1)  # one GPU prefill

    def test_unsupported_triggers_exactly_one_gpu_correction_with_cooldown(self):
        gpu = _Client(["Chapter 1187 releases July 5 (https://a.example/x).",  # synth (wrong)
                       "The sources don't give an exact chapter number (https://a.example/x)."])  # correction
        ver = _Client(['{"unsupported": ["chapter 1187"]}'])
        def factory(url, model, extra_body=None):
            return gpu if "8080" in url else ver
        slept = []
        out = []
        ans = webanswer.answer_web("latest chapter?", out=out.append,
                                   route=lambda s: _route(s, gpu_up=True),
                                   factory=factory, search=self._search, fetch=self._fetch,
                                   sleep=slept.append)
        self.assertEqual(len(gpu.calls), 2)               # synth + exactly one correction, no loop
        self.assertEqual(len(slept), 1)                   # cooldown between the two prefills
        self.assertNotIn("1187", ans)

    def test_cpu_path_hedges_without_gpu_prefill(self):
        # gpu_up=False → synth resolves to CPU (8086). Unsupported → hedge, never a GPU call.
        cpu = _Client(["Chapter 1187 releases July 5. You can read it on VIZ (https://a.example/x)."])
        ver = _Client(['{"unsupported": ["chapter 1187", "July 5"]}'])
        calls = {"cpu": 0}
        def factory(url, model, extra_body=None):
            # both synth-fallback and verify land on 8086; hand out cpu first, then ver
            if calls["cpu"] == 0:
                calls["cpu"] += 1
                return cpu
            return ver
        slept = []
        ans = webanswer.answer_web("latest chapter?", out=lambda s: None,
                                   route=lambda s: _route(s, gpu_up=False),
                                   factory=factory, search=self._search, fetch=self._fetch,
                                   sleep=slept.append)
        self.assertEqual(slept, [])                        # no cooldown, no GPU correction
        self.assertNotIn("1187", ans)
        self.assertIn("VIZ", ans)

    def test_never_raises_when_everything_fails(self):
        class Boom:
            def chat(self, *a, **k): raise RuntimeError("down")
        def bad_search(q, limit=6): raise RuntimeError("x")
        out = []
        ans = webanswer.answer_web("q", out=out.append,
                                   route=lambda s: _route(s, gpu_up=True),
                                   factory=lambda *a, **k: Boom(),
                                   search=bad_search, fetch=lambda u: (_ for _ in ()).throw(RuntimeError()))
        self.assertIsInstance(ans, str)                    # produced something, did not raise


if __name__ == "__main__":
    unittest.main()
