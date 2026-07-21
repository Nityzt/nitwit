import os
import tempfile
import unittest
from nitwit.missions import MissionStore
from nitwit.workspace import Workspace, FileEdit
from nitwit.coder import CoderResponse, FakeCoder, FakeVerifier
from nitwit.engine import MissionEngine
from test_nitwit_workspace import make_repo


class TestEngineIteration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "m.db"))
        self.repo = make_repo()

    def _mission(self, criteria):
        return self.store.create(
            "make target.txt say ok",
            repos=[{"path": self.repo, "branch": "agent/t", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=criteria,
        )

    def test_evaluate_tests_criterion(self):
        # target.txt must contain 'ok' for the test cmd to pass
        m = self._mission([{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        ws = Workspace(self.repo); ws.ensure_branch("agent/t")
        engine = MissionEngine(self.store, FakeCoder([]), FakeVerifier(True))
        passed, _ = engine.evaluate_criteria(m, {self.repo: ws})
        self.assertFalse(passed)  # target.txt doesn't exist yet
        ws.apply_edits([FileEdit("target.txt", "ok\n")]); ws.commit("add")
        passed, _ = engine.evaluate_criteria(m, {self.repo: ws})
        self.assertTrue(passed)

    def test_run_iteration_applies_edits_and_commits(self):
        m = self._mission([{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        coder = FakeCoder([CoderResponse(edits=[FileEdit("target.txt", "ok\n")], note="wrote target")])
        engine = MissionEngine(self.store, coder, FakeVerifier(True))
        ws = Workspace(self.repo); ws.ensure_branch("agent/t")
        m, done = engine.run_iteration(m, {self.repo: ws})
        self.assertTrue(done)                      # criterion now satisfied
        self.assertEqual(m.iteration, 1)
        self.assertIn("target", m.notes)
        self.assertTrue(os.path.exists(os.path.join(self.repo, "target.txt")))

    def test_verifier_criterion_uses_injected_verifier(self):
        m = self._mission([{"type": "verifier", "description": "is it meaningful?"}])
        engine = MissionEngine(self.store, FakeCoder([]), FakeVerifier(verdict=False))
        ws = Workspace(self.repo); ws.ensure_branch("agent/t")
        passed, _ = engine.evaluate_criteria(m, {self.repo: ws})
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
