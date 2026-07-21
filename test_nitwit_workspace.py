import os
import subprocess
import tempfile
import unittest
from nitwit.workspace import Workspace, FileEdit, DirtyRepo, UnsafeEditPath, git, TestResult


def make_repo() -> str:
    d = tempfile.mkdtemp()
    git(d, "init", "-q")
    git(d, "config", "user.email", "t@t")
    git(d, "config", "user.name", "t")
    with open(os.path.join(d, "README.md"), "w") as fh:
        fh.write("seed\n")
    git(d, "add", "-A")
    git(d, "commit", "-q", "-m", "seed")
    return d


class TestWorkspaceGit(unittest.TestCase):
    def setUp(self):
        self.repo = make_repo()
        self.ws = Workspace(self.repo)

    def test_is_clean(self):
        self.assertTrue(self.ws.is_clean())
        with open(os.path.join(self.repo, "x.txt"), "w") as fh:
            fh.write("dirty")
        self.assertFalse(self.ws.is_clean())

    def test_ensure_branch_creates_and_is_reentrant(self):
        self.ws.ensure_branch("agent/test")
        self.assertEqual(git(self.repo, "branch", "--show-current"), "agent/test")
        self.ws.ensure_branch("agent/test")  # second call must not fail
        self.assertEqual(git(self.repo, "branch", "--show-current"), "agent/test")

    def test_ensure_branch_refuses_dirty(self):
        with open(os.path.join(self.repo, "x.txt"), "w") as fh:
            fh.write("dirty")
        with self.assertRaises(DirtyRepo):
            self.ws.ensure_branch("agent/test")

    def test_apply_edits_and_commit(self):
        self.ws.ensure_branch("agent/test")
        self.ws.apply_edits([FileEdit("src/app.py", "print('hi')\n"),
                             FileEdit("README.md", "changed\n")])
        self.assertTrue(os.path.exists(os.path.join(self.repo, "src/app.py")))
        sha = self.ws.commit("add app")
        self.assertTrue(sha)
        # committing again with no change returns ""
        self.assertEqual(self.ws.commit("noop"), "")

    def test_apply_edits_rejects_relative_escape(self):
        self.ws.ensure_branch("agent/test")
        with self.assertRaises(UnsafeEditPath):
            self.ws.apply_edits([FileEdit("../evil.txt", "x")])

    def test_apply_edits_rejects_absolute_escape(self):
        self.ws.ensure_branch("agent/test")
        with self.assertRaises(UnsafeEditPath):
            self.ws.apply_edits([FileEdit("/tmp/evil_abs.txt", "x")])

    def test_apply_edits_allows_legitimate_nested_path(self):
        self.ws.ensure_branch("agent/test")
        self.ws.apply_edits([FileEdit("src/deep/app.py", "print('nested')\n")])
        nested = os.path.join(self.repo, "src/deep/app.py")
        self.assertTrue(os.path.exists(nested))
        with open(nested) as fh:
            self.assertEqual(fh.read(), "print('nested')\n")

    def test_apply_edits_rejects_symlink_escape(self):
        # A symlink INSIDE the repo whose target resolves OUTSIDE it must be caught by
        # the realpath check, even though the edit path itself never contains "..".
        self.ws.ensure_branch("agent/test")
        os.symlink(tempfile.gettempdir(), os.path.join(self.repo, "outlink"))
        with self.assertRaises(UnsafeEditPath):
            self.ws.apply_edits([FileEdit("outlink/evil.txt", "x")])


class TestWorkspaceResetHard(unittest.TestCase):
    def setUp(self):
        self.repo = make_repo()
        self.ws = Workspace(self.repo)

    def test_reset_hard_discards_dirty_and_untracked(self):
        self.ws.ensure_branch("agent/reset")
        self.ws.apply_edits([FileEdit("README.md", "checkpoint\n")])
        sha = self.ws.commit("checkpoint")
        self.assertTrue(sha)
        # simulate a crashed half-iteration: dirty tracked file + untracked file
        with open(os.path.join(self.repo, "README.md"), "w") as fh:
            fh.write("half-applied garbage\n")
        with open(os.path.join(self.repo, "untracked.txt"), "w") as fh:
            fh.write("crash debris\n")
        self.assertFalse(self.ws.is_clean())

        self.ws.reset_hard()

        self.assertTrue(self.ws.is_clean())
        self.assertFalse(os.path.exists(os.path.join(self.repo, "untracked.txt")))
        with open(os.path.join(self.repo, "README.md")) as fh:
            self.assertEqual(fh.read(), "checkpoint\n")


class TestWorkspaceRunTests(unittest.TestCase):
    def setUp(self):
        self.repo = make_repo()
        self.ws = Workspace(self.repo)

    def test_passing_command(self):
        r = self.ws.run_tests("true")
        self.assertTrue(r.passed)

    def test_failing_command_captures_output(self):
        r = self.ws.run_tests("echo boom && false")
        self.assertFalse(r.passed)
        self.assertIn("boom", r.output)

    def test_python_test_file(self):
        with open(os.path.join(self.repo, "check.py"), "w") as fh:
            fh.write("assert 1 + 1 == 2\nprint('ok')\n")
        r = self.ws.run_tests("python3 check.py")
        self.assertTrue(r.passed)
        self.assertIn("ok", r.output)

    def test_timeout(self):
        r = self.ws.run_tests("sleep 5", timeout=1)
        self.assertFalse(r.passed)
        self.assertIn("TIMEOUT", r.output)


if __name__ == "__main__":
    unittest.main()
