import os
import subprocess
import tempfile
import unittest
from nitwit.workspace import Workspace, FileEdit, DirtyRepo, UnsafeEditPath, git


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


if __name__ == "__main__":
    unittest.main()
