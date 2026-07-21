import os, tempfile, subprocess, unittest
from nitwit.session import repo_root, detect_test_cmd, classify_intent


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


if __name__ == "__main__":
    unittest.main()
