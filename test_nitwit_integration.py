"""End-to-end: a mission with a real failing test in a real git repo, driven to green by a
deterministic coder, must branch, iterate, commit each round, and stop `done`. Also proves
resume: reconcile after an interrupted run, then finish."""
import os
import tempfile
import unittest
from nitwit.missions import MissionStore
from nitwit.workspace import FileEdit, git
from nitwit.coder import CoderResponse, FakeCoder, FakeVerifier
from nitwit.engine import MissionEngine
from test_nitwit_workspace import make_repo


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "m.db"))
        self.repo = make_repo()
        # a real failing test: pytest-free, just a python assert on a module the coder must write
        with open(os.path.join(self.repo, "test_feature.py"), "w") as fh:
            fh.write("from feature import add\nassert add(2, 3) == 5\nprint('PASS')\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "add failing test")

    def test_mission_reaches_green_and_commits(self):
        m = self.store.create(
            "implement feature.add so the test passes",
            repos=[{"path": self.repo, "branch": "agent/feature", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=[{"type": "tests", "repo": self.repo, "cmd": "python3 test_feature.py"},
                              {"type": "verifier", "description": "is the implementation meaningful?"}],
        )
        # iteration 1: a wrong impl (returns 0); iteration 2: the correct impl
        coder = FakeCoder([
            CoderResponse(edits=[FileEdit("feature.py", "def add(a, b):\n    return 0\n")], note="stub"),
            CoderResponse(edits=[FileEdit("feature.py", "def add(a, b):\n    return a + b\n")], note="fix"),
        ])
        engine = MissionEngine(self.store, coder, FakeVerifier(True), max_iterations=5)
        result = engine.run_mission(m.id)

        self.assertEqual(result.state, "done")
        # on the agent branch, with a commit per iteration (2) on top of the seed+test commits
        self.assertEqual(git(self.repo, "branch", "--show-current"), "agent/feature")
        subjects = git(self.repo, "log", "--format=%s").splitlines()
        self.assertTrue(any("iteration 2" in s for s in subjects))
        self.assertTrue(any("iteration 1" in s for s in subjects))
        # the deliverable actually exists and is correct
        with open(os.path.join(self.repo, "feature.py")) as fh:
            self.assertIn("a + b", fh.read())

    def test_resume_after_interruption(self):
        m = self.store.create(
            "resume me",
            repos=[{"path": self.repo, "branch": "agent/resume", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=[{"type": "tests", "repo": self.repo, "cmd": "python3 test_feature.py"}],
        )
        # simulate an engine that died mid-run: mission stuck in 'running'
        self.store.set_state(m.id, "running")

        # a fresh engine reconciles (rewinds running -> queued), then runs to done
        coder = FakeCoder([CoderResponse(edits=[FileEdit("feature.py", "def add(a, b):\n    return a + b\n")], note="fix")])
        engine = MissionEngine(self.store, coder, FakeVerifier(True))
        self.assertEqual(engine.reconcile(), 1)
        self.assertEqual(self.store.get(m.id).state, "queued")
        result = engine.run_mission(m.id)
        self.assertEqual(result.state, "done")


if __name__ == "__main__":
    unittest.main()
