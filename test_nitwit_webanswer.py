import json
import unittest
from nitwit import webanswer
from nitwit.router import Endpoint


class _Resp:
    def __init__(self, content): self.content = content


class _Client:
    """Fake non-streaming chat client. Pops `replies` in order; records calls."""
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def chat(self, messages, *, temperature, max_tokens, response_format=None):
        self.calls.append(messages)
        return _Resp(self._replies.pop(0) if self._replies else "")


def _claims(*pairs):
    """Build a verifier JSON payload from (claim, supported) pairs."""
    return json.dumps({"claims": [{"claim": c, "supported": s, "quote": ""} for c, s in pairs]})


class TestSynthesize(unittest.TestCase):
    def test_uses_context_and_query(self):
        c = _Client(["Chapter 1140 is the latest (https://viz.com)."])
        out = webanswer.synthesize("latest chapter?", "CONTEXT text 1140", client=c)
        self.assertEqual(out, "Chapter 1140 is the latest (https://viz.com).")
        joined = "\n".join(m["content"] for m in c.calls[0])
        self.assertIn("CONTEXT text 1140", joined)
        self.assertIn("using ONLY the CONTEXT", joined)

    def test_never_raises(self):
        class Boom:
            def chat(self, *a, **k): raise RuntimeError("down")
        self.assertEqual(webanswer.synthesize("q", "ctx", client=Boom()), "")


class TestVerifyGrounding(unittest.TestCase):
    def test_returns_unsupported_claims_only(self):
        c = _Client([_claims(("chapter 1187", True), ("July 5 2026", False))])
        self.assertEqual(webanswer.verify_grounding("ans", "ctx", client=c), ["July 5 2026"])

    def test_all_supported_returns_empty(self):
        c = _Client([_claims(("1140", True))])
        self.assertEqual(webanswer.verify_grounding("ans", "ctx", client=c), [])

    def test_fails_open_on_garbage(self):
        self.assertEqual(webanswer.verify_grounding("ans", "ctx", client=_Client(["not json"])), [])

    def test_fails_open_on_exception(self):
        class Boom:
            def chat(self, *a, **k): raise RuntimeError("x")
        self.assertEqual(webanswer.verify_grounding("ans", "ctx", client=Boom()), [])

    def test_empty_answer_short_circuits(self):
        c = _Client([_claims(("x", False))])
        self.assertEqual(webanswer.verify_grounding("   ", "ctx", client=c), [])
        self.assertEqual(c.calls, [])


class TestCleanAndWarmup(unittest.TestCase):
    def test_clean_strips_lead_disclaimer(self):
        out = webanswer.clean("I can't perform real-time web searches, but chapter 1140 is out.\n\n\n")
        self.assertNotIn("real-time", out)
        self.assertIn("1140", out)

    def test_warmup_true_on_success_false_on_error(self):
        self.assertTrue(webanswer._warmup(_Client(["ok"])))
        class Boom:
            def chat(self, *a, **k): raise RuntimeError("cold")
        self.assertFalse(webanswer._warmup(Boom()))


def _route(stage, *, gpu=True):
    if stage == "synth":
        return Endpoint("http://127.0.0.1:8080" if gpu else "http://127.0.0.1:8086", "synth", {})
    return Endpoint("http://127.0.0.1:8086", "4b", {})   # verify


class TestAnswerWebLoop(unittest.TestCase):
    """One model does both synth and verify; its `replies` interleave answers and verifier JSON in
    call order: [answer, verify, (corrected, verify)...]."""

    def _factory(self, client):
        return lambda url, model, extra_body=None: client

    def test_grounded_first_pass_no_correction(self):
        client = _Client(["Chapter 1140 is the latest (https://a/x).", _claims(("1140", True))])
        slept, warmed = [], []
        ans = webanswer.answer_web("latest chapter?", out=lambda s: None,
                                   route=lambda s: _route(s, gpu=True), factory=self._factory(client),
                                   search=lambda q, limit=6: "WEB RESULTS:\n- a (https://a/x)",
                                   fetch=lambda u: "1140 is the latest chapter",
                                   sleep=slept.append, warmup=lambda c: warmed.append(1) or True,
                                   gpu_ok=lambda: True)
        self.assertIn("1140", ans)
        self.assertEqual(len(client.calls), 2)              # synth + verify, no correction
        self.assertEqual(len(slept), 1)                     # one cooldown (before the verify prefill)
        self.assertEqual(warmed, [1])                       # GPU warmed once

    def test_self_corrects_until_grounded(self):
        client = _Client(["Chapter 1187 releases July 5 2026 (https://a/x).",   # synth (fabricated)
                          _claims(("July 5 2026", False)),                       # verify: flag date
                          "Chapter 1187 is out; sources give no date (https://a/x).",  # correction
                          _claims(("1187", True))])                              # verify: grounded
        slept = []
        ans = webanswer.answer_web("latest chapter?", out=lambda s: None,
                                   route=lambda s: _route(s, gpu=True), factory=self._factory(client),
                                   search=lambda q, limit=6: "WEB RESULTS:\n- a (https://a/x)",
                                   fetch=lambda u: "chapter 1187", sleep=slept.append,
                                   warmup=lambda c: True, gpu_ok=lambda: True)
        self.assertEqual(len(client.calls), 4)              # synth, verify, correct, verify
        self.assertEqual(len(slept), 3)                     # cooldown before each GPU prefill after #1
        self.assertNotIn("July 5", ans)

    def test_budget_exhaustion_keeps_best_effort(self):
        client = _Client(["a1 (https://a/x)", _claims(("x", False)),
                          "a2 (https://a/x)", _claims(("x", False))])
        slept = []
        ans = webanswer.answer_web("q", out=lambda s: None,
                                   route=lambda s: _route(s, gpu=True), factory=self._factory(client),
                                   search=lambda q, limit=6: "WEB RESULTS:\n- a (https://a/x)",
                                   fetch=lambda u: "text", max_iters=2, sleep=slept.append,
                                   warmup=lambda c: True, gpu_ok=lambda: True)
        self.assertEqual(len(client.calls), 4)              # synth, verify, correct, verify → stop
        self.assertTrue(ans)                                # returns best-effort, not blank

    def test_cpu_path_skips_warmup_and_cooldown(self):
        client = _Client(["a1", _claims(("x", False)), "a2", _claims(("x", True))])
        slept, warmed = [], []
        webanswer.answer_web("q", out=lambda s: None,
                             route=lambda s: _route(s, gpu=False),          # synth on CPU (8086)
                             factory=self._factory(client),
                             search=lambda q, limit=6: "WEB RESULTS:\n- a (https://a/x)",
                             fetch=lambda u: "text", sleep=slept.append,
                             warmup=lambda c: warmed.append(1) or True)
        self.assertEqual(warmed, [])                         # CPU → no warm-up
        self.assertEqual(slept, [])                          # CPU → no cooldown

    def test_never_raises_when_everything_fails(self):
        class Boom:
            def chat(self, *a, **k): raise RuntimeError("down")
        ans = webanswer.answer_web("q", out=lambda s: None,
                                   route=lambda s: _route(s, gpu=True),
                                   factory=lambda *a, **k: Boom(),
                                   search=lambda q, limit=6: (_ for _ in ()).throw(RuntimeError()),
                                   fetch=lambda u: (_ for _ in ()).throw(RuntimeError()),
                                   warmup=lambda c: False, gpu_ok=lambda: True)
        self.assertIsInstance(ans, str)

    def test_gpu_gated_to_cpu_when_undervolt_absent(self):
        # synth routes to GPU (8080), but CoreCtrl undervolt is NOT applied → must fall back to the
        # CPU model (8086): no warm-up, no cooldown, and the client is built for the CPU endpoint.
        client = _Client(["answer (https://a/x)", _claims(("x", True))])
        seen_urls, slept, warmed = [], [], []
        def factory(url, model, extra_body=None):
            seen_urls.append(url)
            return client
        webanswer.answer_web("q", out=lambda s: None,
                             route=lambda s: _route(s, gpu=True), factory=factory,
                             search=lambda q, limit=6: "WEB RESULTS:\n- a (https://a/x)",
                             fetch=lambda u: "text", sleep=slept.append,
                             warmup=lambda c: warmed.append(1) or True,
                             gpu_ok=lambda: False)                 # no undervolt
        self.assertTrue(all("8086" in u for u in seen_urls))      # never built a GPU client
        self.assertEqual(warmed, [])                              # CPU path → no warm-up
        self.assertEqual(slept, [])                               # CPU path → no cooldown


class TestGpuUndervoltActive(unittest.TestCase):
    def test_true_when_corectrl_running(self):
        self.assertTrue(webanswer._gpu_undervolt_active(_probe=lambda: True))

    def test_false_when_corectrl_absent(self):
        self.assertFalse(webanswer._gpu_undervolt_active(_probe=lambda: False))

    def test_env_override_forces_true(self):
        import os
        os.environ["NITWIT_GPU_UNPROTECTED_OK"] = "1"
        try:
            self.assertTrue(webanswer._gpu_undervolt_active(_probe=lambda: False))
        finally:
            del os.environ["NITWIT_GPU_UNPROTECTED_OK"]

    def test_never_raises(self):
        def boom(): raise RuntimeError("x")
        self.assertFalse(webanswer._gpu_undervolt_active(_probe=boom))


if __name__ == "__main__":
    unittest.main()
