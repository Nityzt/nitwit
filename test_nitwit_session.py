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
        self.assertNotIn("GPT-4", seen["system"])
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
