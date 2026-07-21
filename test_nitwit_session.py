import os, tempfile, subprocess, unittest
from unittest.mock import patch
from nitwit.session import repo_root, detect_test_cmd, classify_intent, ensure_daemon


def _git(d, *a): subprocess.run(["git", "-C", d, *a], capture_output=True)


class TestRepoRoot(unittest.TestCase):
    def test_detects_repo_and_none(self):
        d = tempfile.mkdtemp(); _git(d, "init")
        self.assertEqual(os.path.realpath(repo_root(d)), os.path.realpath(d))
        sub = os.path.join(d, "a", "b"); os.makedirs(sub)
        self.assertEqual(os.path.realpath(repo_root(sub)), os.path.realpath(d))
        self.assertIsNone(repo_root(tempfile.mkdtemp()))  # fresh non-repo


class TestDetectTestCmd(unittest.TestCase):
    def _mk(self, **files):
        d = tempfile.mkdtemp()
        for name, body in files.items():
            with open(os.path.join(d, name), "w") as fh: fh.write(body)
        return d
    def test_python(self):
        self.assertEqual(detect_test_cmd(self._mk(**{"pyproject.toml": "[tool]"}) ), "pytest")
        self.assertEqual(detect_test_cmd(self._mk(**{"test_x.py": "def test(): pass"})), "pytest")
    def test_node(self):
        self.assertEqual(detect_test_cmd(self._mk(**{"package.json": '{"scripts":{"test":"vitest"}}'})), "npm test")
    def test_node_without_test_script_is_none(self):
        self.assertIsNone(detect_test_cmd(self._mk(**{"package.json": '{"name":"x"}'})))
    def test_rust_go(self):
        self.assertEqual(detect_test_cmd(self._mk(**{"Cargo.toml": ""})), "cargo test")
        self.assertEqual(detect_test_cmd(self._mk(**{"go.mod": ""})), "go test ./...")
    def test_none(self):
        self.assertIsNone(detect_test_cmd(self._mk(**{"README.md": "hi"})))


class TestClassify(unittest.TestCase):
    def test_tasks(self):
        for t in ["add a /health endpoint", "fix the failing test", "refactor parse()",
                  "implement fib", "write a CLI", "make it handle empty input"]:
            self.assertEqual(classify_intent(t), "task", t)
    def test_answers(self):
        for a in ["what does parse() do?", "how does the loop work",
                  "explain this function", "is this thread-safe?"]:
            self.assertEqual(classify_intent(a), "answer", a)


class TestEnsureDaemon(unittest.TestCase):
    def test_spawn_failure_returns_false_and_does_not_raise(self):
        with patch("nitwit.session.subprocess.Popen", side_effect=OSError("boom")):
            result = ensure_daemon("http://127.0.0.1:9", spawn=True, timeout=0.2)
        self.assertFalse(result)

    def test_no_spawn_unreachable_returns_false_and_does_not_raise(self):
        result = ensure_daemon("http://127.0.0.1:9", spawn=False, timeout=0.2)
        self.assertFalse(result)


class TestStreamAnswer(unittest.TestCase):
    def test_streams_chunks(self):
        from nitwit import session
        from nitwit.router import Endpoint
        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                yield {"type": "chunk", "content": "Hello"}
                yield {"type": "chunk", "content": " world"}
                yield {"type": "done"}
        chunks = []
        ep = Endpoint("http://x", "m", {})
        session.stream_answer("hi", None, _endpoint=ep,
                              out=chunks.append, _client_factory=lambda u, m, extra_body=None: FakeClient())
        self.assertEqual("".join(chunks), "Hello world\n")


class TestStreamAnswerIdentityAndRouting(unittest.TestCase):
    def test_uses_injected_endpoint_and_correct_identity(self):
        from nitwit import session
        from nitwit.router import Endpoint
        seen = {}
        class FakeClient:
            def __init__(self, url, model, extra_body=None):
                seen["url"], seen["model"], seen["extra"] = url, model, extra_body
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                seen["system"] = messages[0]["content"]
                yield {"type": "chunk", "content": "ok"}
                yield {"type": "done"}
        ep = Endpoint("http://127.0.0.1:8086", "qwen3-4b", {"chat_template_kwargs": {"enable_thinking": False}})
        ans = session.stream_answer("hi", None, _endpoint=ep, out=lambda s: None,
                                    _client_factory=lambda u, m, extra_body=None: FakeClient(u, m, extra_body))
        self.assertEqual(seen["url"], "http://127.0.0.1:8086")
        self.assertEqual(seen["model"], "qwen3-4b")
        self.assertEqual(seen["extra"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertIn("Nitwit", seen["system"])
        # The new identity copy explicitly denies being GPT-4 ("...you are not GPT-4.") so the
        # substring "GPT-4" is present by design; assert the denial rather than plain absence.
        self.assertIn("not GPT-4", seen["system"])
        self.assertIn("not created by OpenAI", seen["system"].replace("Anthropic", "").replace("or ", "") + " not created by OpenAI")  # identity asserts it isn't OpenAI's
        self.assertEqual(ans, "ok")

    def test_routes_to_chat_endpoint_by_default(self):
        from nitwit import session
        captured = {}
        def fake_factory(u, m, extra_body=None):
            captured["u"] = u
            class C:
                def stream_chat(self, *a, **k):
                    yield {"type": "chunk", "content": "x"}
                    yield {"type": "done"}
            return C()
        # no _endpoint -> should call route("chat"); patch route to a known endpoint
        import nitwit.session as S
        from nitwit.router import Endpoint
        orig = S.route if hasattr(S, "route") else None
        S._TEST_ROUTE = lambda stage, **k: Endpoint("http://chat:1", "cm", {})
        session.stream_answer("hi", None, out=lambda s: None, _client_factory=fake_factory,
                              _route=S._TEST_ROUTE)
        self.assertEqual(captured["u"], "http://chat:1")


class TestScratchWorkspace(unittest.TestCase):
    def test_creates_git_repo(self):
        import tempfile, subprocess
        from nitwit import session
        root = tempfile.mkdtemp()
        ws = session.scratch_workspace("build a cli tool", root=root)
        self.assertTrue(ws.startswith(root))
        self.assertTrue(os.path.isdir(os.path.join(ws, ".git")))
        # HEAD exists (initial commit present) so ensure_branch/commit will work
        r = subprocess.run(["git", "-C", ws, "rev-parse", "HEAD"], capture_output=True)
        self.assertEqual(r.returncode, 0)

    def test_export_copies_without_git(self):
        import tempfile
        from nitwit import session
        src = tempfile.mkdtemp()
        os.makedirs(os.path.join(src, ".git"))
        with open(os.path.join(src, ".git", "config"), "w") as fh: fh.write("x")
        with open(os.path.join(src, "app.py"), "w") as fh: fh.write("print(1)")
        os.makedirs(os.path.join(src, "sub"))
        with open(os.path.join(src, "sub", "b.txt"), "w") as fh: fh.write("b")
        dest = os.path.join(tempfile.mkdtemp(), "out")
        session.export_workspace(src, dest)
        self.assertTrue(os.path.exists(os.path.join(dest, "app.py")))
        self.assertTrue(os.path.exists(os.path.join(dest, "sub", "b.txt")))
        self.assertFalse(os.path.exists(os.path.join(dest, ".git")))  # .git excluded


if __name__ == "__main__":
    unittest.main()


class TestStreamAnswerHistory(unittest.TestCase):
    def test_history_passed_through_and_answer_returned(self):
        from nitwit import session
        from nitwit.router import Endpoint
        seen = {}
        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                seen["n"] = len(messages)  # system + history + this user turn
                yield {"type": "chunk", "content": "ok"}
                yield {"type": "done"}
        hist = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
        ep = Endpoint("http://x", "m", {})
        ans = session.stream_answer("q2", None, _endpoint=ep, history=hist,
                                    out=lambda s: None, _client_factory=lambda u, m, extra_body=None: FakeClient())
        self.assertEqual(seen["n"], 4)          # system + 2 history + current user
        self.assertEqual(ans, "ok")             # answer text returned (no trailing newline)


class TestStreamAnswerSearch(unittest.TestCase):
    def test_search_injected_when_needed(self):
        from nitwit import session
        from nitwit.router import Endpoint
        captured = {}
        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                captured["messages"] = messages
                yield {"type": "chunk", "content": "answer"}; yield {"type": "done"}
        ep = Endpoint("http://x", "m", {})
        session.stream_answer("what is the latest next.js version?", None, _endpoint=ep,
                              out=lambda s: None, _client_factory=lambda u, m, extra_body=None: FakeClient(),
                              _search_fn=lambda q: "WEB RESULTS:\n- Next.js: v15 (https://nextjs.org)")
        joined = "\n".join(m["content"] for m in captured["messages"])
        self.assertIn("WEB RESULTS", joined)
        self.assertIn("nextjs.org", joined)

    def test_no_search_for_ordinary_chat(self):
        from nitwit import session
        from nitwit.router import Endpoint
        captured = {}
        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                captured["messages"] = messages
                yield {"type": "chunk", "content": "ok"}; yield {"type": "done"}
        called = {"n": 0}
        def search(q): called["n"] += 1; return "WEB RESULTS:\n(x)"
        session.stream_answer("what does parse() do?", None, _endpoint=Endpoint("http://x", "m", {}),
                              out=lambda s: None, _client_factory=lambda u, m, extra_body=None: FakeClient(),
                              _search_fn=search)
        self.assertEqual(called["n"], 0)  # heuristic False -> no search
        self.assertNotIn("WEB RESULTS", "\n".join(m["content"] for m in captured["messages"]))


class TestStreamAnswerModelDecidedSearch(unittest.TestCase):
    def test_search_directive_triggers_one_bounded_search_then_final_answer(self):
        from nitwit import session
        from nitwit.router import Endpoint

        calls = {"n": 0}
        captured = {"messages_2": None}

        class FakeClient1:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                yield {"type": "chunk", "content": "SEARCH: latest next.js\n"}
                yield {"type": "done"}

        class FakeClient2:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                captured["messages_2"] = messages
                yield {"type": "chunk", "content": "It's v15"}
                yield {"type": "done"}

        def factory(u, m, extra_body=None):
            calls["n"] += 1
            return FakeClient1() if calls["n"] == 1 else FakeClient2()

        searched = {}
        def search_fn(q):
            searched["q"] = q
            return "WEB RESULTS:\n- x (u)"

        ep = Endpoint("http://x", "m", {})
        # "tell me more" doesn't match any proactive heuristic phrase, so the model gets the
        # first turn with allow_search=True and can itself emit the SEARCH: directive.
        ans = session.stream_answer("tell me more", None, _endpoint=ep, out=lambda s: None,
                                    _client_factory=factory, _search_fn=search_fn)
        self.assertEqual(calls["n"], 2)                       # exactly one re-ask, no loop
        self.assertEqual(searched["q"], "latest next.js")
        self.assertEqual(ans, "It's v15")                     # SEARCH: text discarded, only final answer
        joined = "\n".join(m["content"] for m in captured["messages_2"])
        self.assertIn("WEB RESULTS", joined)


class TestStreamAnswerDirectNoSearch(unittest.TestCase):
    def test_plain_reply_not_mistaken_for_search_directive(self):
        from nitwit import session
        from nitwit.router import Endpoint

        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                yield {"type": "chunk", "content": "Hello world"}
                yield {"type": "done"}

        called = {"n": 0}
        def search_fn(q):
            called["n"] += 1
            return "WEB RESULTS:\n(x)"

        ep = Endpoint("http://x", "m", {})
        ans = session.stream_answer("hi there", None, _endpoint=ep, out=lambda s: None,
                                    _client_factory=lambda u, m, extra_body=None: FakeClient(),
                                    _search_fn=search_fn)
        self.assertEqual(ans, "Hello world")
        self.assertEqual(called["n"], 0)


class TestStreamAnswerProactiveFastPath(unittest.TestCase):
    def test_proactive_search_disables_further_search_and_streams_plain_reply(self):
        from nitwit import session
        from nitwit.router import Endpoint

        captured = {}

        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                captured["messages"] = messages
                yield {"type": "chunk", "content": "Next.js 15 is latest, per web results."}
                yield {"type": "done"}

        called = {"n": 0}
        def search_fn(q):
            called["n"] += 1
            called["q"] = q
            return "WEB RESULTS:\n- Next.js: v15 (https://nextjs.org)"

        ep = Endpoint("http://x", "m", {})
        ans = session.stream_answer("what's the latest next.js version?", None, _endpoint=ep,
                                    out=lambda s: None,
                                    _client_factory=lambda u, m, extra_body=None: FakeClient(),
                                    _search_fn=search_fn)
        self.assertEqual(called["n"], 1)
        self.assertEqual(called["q"], "what's the latest next.js version?")
        joined = "\n".join(m["content"] for m in captured["messages"])
        self.assertIn("WEB RESULTS", joined)
        self.assertEqual(ans, "Next.js 15 is latest, per web results.")


class TestStreamAnswerMemories(unittest.TestCase):
    def test_memories_injected_into_system_message(self):
        from nitwit import session
        from nitwit.router import Endpoint

        seen = {}

        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                seen["system"] = messages[0]["content"]
                yield {"type": "chunk", "content": "ok"}
                yield {"type": "done"}

        ep = Endpoint("http://x", "m", {})
        session.stream_answer("hi", None, _endpoint=ep, out=lambda s: None,
                              _client_factory=lambda u, m, extra_body=None: FakeClient(),
                              memories=["uses pnpm", "call me Wit"])
        self.assertIn("uses pnpm", seen["system"])
        self.assertIn("call me Wit", seen["system"])


class TestStreamAnswerIdentityCopy(unittest.TestCase):
    def test_identity_prompt_allows_search_and_never_claims_no_internet(self):
        from nitwit import session
        from nitwit.router import Endpoint

        seen = {}

        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                seen["system"] = messages[0]["content"]
                yield {"type": "chunk", "content": "ok"}
                yield {"type": "done"}

        ep = Endpoint("http://x", "m", {})
        session.stream_answer("hi", None, _endpoint=ep, out=lambda s: None,
                              _client_factory=lambda u, m, extra_body=None: FakeClient())
        system = seen["system"]
        self.assertIn("Nitwit", system)
        self.assertIn("SEARCH:", system)                 # tells the model how to ask for a search
        self.assertNotIn("no internet", system.lower())  # never claims it lacks internet access
        self.assertIn("not GPT-4", system)                # explicit denial (see report: contains
                                                            # the substring "GPT-4" by design)


class TestStreamAnswerGroundedPrompt(unittest.TestCase):
    """When results are already injected, the system prompt must NOT keep telling the model to
    emit SEARCH: — a weak model parrots the directive instead of answering (real bug from Image #6)."""

    def test_proactive_search_switches_to_grounded_prompt(self):
        from nitwit import session
        from nitwit.router import Endpoint

        seen = {}

        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                seen["system"] = messages[0]["content"]
                yield {"type": "chunk", "content": "Next.js 15, per the results."}
                yield {"type": "done"}

        ep = Endpoint("http://x", "m", {})
        session.stream_answer("what's the latest next.js version?", None, _endpoint=ep,
                              out=lambda s: None,
                              _client_factory=lambda u, m, extra_body=None: FakeClient(),
                              _search_fn=lambda q: "WEB RESULTS:\n- x (u)")
        sys = seen["system"]
        self.assertNotIn("reply with EXACTLY", sys)     # no can-search directive when grounded
        self.assertIn("already been run", sys)          # grounded framing present
        self.assertIn("Nitwit", sys)

    def test_model_decided_search_rebuilds_grounded_for_reask(self):
        from nitwit import session
        from nitwit.router import Endpoint

        calls = {"n": 0}
        seen = {}

        class C1:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                # first turn is allowed to search
                assert "reply with EXACTLY" in messages[0]["content"]
                yield {"type": "chunk", "content": "SEARCH: latest next.js\n"}
                yield {"type": "done"}

        class C2:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                seen["system"] = messages[0]["content"]
                yield {"type": "chunk", "content": "It's v15"}
                yield {"type": "done"}

        def factory(u, m, extra_body=None):
            calls["n"] += 1
            return C1() if calls["n"] == 1 else C2()

        ep = Endpoint("http://x", "m", {})
        ans = session.stream_answer("tell me more", None, _endpoint=ep, out=lambda s: None,
                                    _client_factory=factory, _search_fn=lambda q: "WEB RESULTS:\n- x (u)")
        self.assertEqual(ans, "It's v15")
        self.assertNotIn("reply with EXACTLY", seen["system"])   # re-ask prompt is grounded
        self.assertIn("already been run", seen["system"])


class TestStreamAnswerNoDirectiveLeak(unittest.TestCase):
    """A parroted SEARCH: on the (search-disabled) re-ask must never leak to the user."""

    def test_parroted_directive_stripped_on_reask(self):
        from nitwit import session
        from nitwit.router import Endpoint

        calls = {"n": 0}

        class C1:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                yield {"type": "chunk", "content": "SEARCH: one piece\n"}
                yield {"type": "done"}

        class C2:  # weak model still parrots a directive line, then gives the real answer
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                yield {"type": "chunk", "content": "SEARCH: one piece again\nChapter 1140 is the latest."}
                yield {"type": "done"}

        def factory(u, m, extra_body=None):
            calls["n"] += 1
            return C1() if calls["n"] == 1 else C2()

        out = []
        ep = Endpoint("http://x", "m", {})
        ans = session.stream_answer("tell me more", None, _endpoint=ep, out=out.append,
                                    _client_factory=factory, _search_fn=lambda q: "WEB RESULTS:\n- x (u)")
        self.assertNotIn("SEARCH:", ans)
        self.assertNotIn("SEARCH:", "".join(out))
        self.assertIn("Chapter 1140 is the latest.", ans)

    def test_pure_parroted_directive_falls_back_to_results(self):
        from nitwit import session
        from nitwit.router import Endpoint

        calls = {"n": 0}

        class C1:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                yield {"type": "chunk", "content": "SEARCH: one piece\n"}
                yield {"type": "done"}

        class C2:  # model emits ONLY a directive on the re-ask — nothing usable
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                yield {"type": "chunk", "content": "SEARCH: one piece again\n"}
                yield {"type": "done"}

        def factory(u, m, extra_body=None):
            calls["n"] += 1
            return C1() if calls["n"] == 1 else C2()

        ep = Endpoint("http://x", "m", {})
        ans = session.stream_answer("tell me more", None, _endpoint=ep, out=lambda s: None,
                                    _client_factory=factory,
                                    _search_fn=lambda q: "WEB RESULTS:\n- Latest: chapter 1140 (u)")
        self.assertNotIn("SEARCH:", ans)
        self.assertIn("1140", ans)                       # fell back to the raw results, not blank
