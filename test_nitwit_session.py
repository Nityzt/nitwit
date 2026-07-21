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
        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                yield {"type": "chunk", "content": "Hello"}
                yield {"type": "chunk", "content": " world"}
                yield {"type": "done"}
        chunks = []
        session.stream_answer("hi", None, coder_url="http://x", coder_model="m",
                              out=chunks.append, _client_factory=lambda u, m: FakeClient())
        self.assertEqual("".join(chunks), "Hello world\n")


if __name__ == "__main__":
    unittest.main()


class TestStreamAnswerHistory(unittest.TestCase):
    def test_history_passed_through_and_answer_returned(self):
        from nitwit import session
        seen = {}
        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                seen["n"] = len(messages)  # system + history + this user turn
                yield {"type": "chunk", "content": "ok"}
                yield {"type": "done"}
        hist = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
        ans = session.stream_answer("q2", None, coder_url="x", coder_model="m", history=hist,
                                    out=lambda s: None, _client_factory=lambda u, m: FakeClient())
        self.assertEqual(seen["n"], 4)          # system + 2 history + current user
        self.assertEqual(ans, "ok")             # answer text returned (no trailing newline)
