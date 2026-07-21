import os
import tempfile
import unittest
from nitwit.missions import MissionStore
from nitwit.workspace import Workspace, FileEdit, DirtyRepo, git
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


class TestEngineLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "m.db"))
        self.repo = make_repo()

    def _mission(self, criteria):
        return self.store.create(
            "loop to green",
            repos=[{"path": self.repo, "branch": "agent/loop", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=criteria,
        )

    def test_loop_reaches_done_in_two_iterations(self):
        # first response writes the wrong content, second writes the right one
        coder = FakeCoder([
            CoderResponse(edits=[FileEdit("target.txt", "nope\n")], note="attempt 1"),
            CoderResponse(edits=[FileEdit("target.txt", "ok\n")], note="attempt 2"),
        ])
        m = self._mission([{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        engine = MissionEngine(self.store, coder, FakeVerifier(True))
        result = engine.run_mission(m.id)
        self.assertEqual(result.state, "done")
        self.assertEqual(result.iteration, 2)

    def test_loop_hits_cap_and_needs_input(self):
        coder = FakeCoder([])  # never produces a fix
        m = self._mission([{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        engine = MissionEngine(self.store, coder, FakeVerifier(True), max_iterations=3)
        result = engine.run_mission(m.id)
        self.assertEqual(result.state, "needs_input")
        self.assertEqual(result.iteration, 3)

    def test_pause_stops_the_loop(self):
        coder = FakeCoder([CoderResponse(edits=[FileEdit("a.txt", "1\n")], note="x")] * 10)
        m = self._mission([{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        engine = MissionEngine(self.store, coder, FakeVerifier(True), max_iterations=10)
        engine.pause()  # paused before it starts
        result = engine.run_mission(m.id)
        self.assertEqual(result.state, "paused")
        self.assertEqual(result.iteration, 0)

    def test_reconcile_rewinds_running(self):
        m = self._mission([{"type": "verifier", "description": "x"}])
        self.store.set_state(m.id, "running")
        engine = MissionEngine(self.store, FakeCoder([]), FakeVerifier(True))
        n = engine.reconcile()
        self.assertEqual(n, 1)
        self.assertEqual(self.store.get(m.id).state, "queued")

    def test_resume_after_crash_resets_dirty_tree(self):
        # Build a mission + repo and run one full iteration so a checkpoint commit exists
        # on the agent branch (the "intended" last-good state).
        m = self._mission([{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        branch = "agent/loop"
        ws = Workspace(self.repo)
        ws.ensure_branch(branch)
        first_coder = FakeCoder([CoderResponse(edits=[FileEdit("target.txt", "nope\n")], note="attempt 1")])
        engine1 = MissionEngine(self.store, first_coder, FakeVerifier(True))
        m, done = engine1.run_iteration(m, {self.repo: ws})
        self.assertFalse(done)
        self.assertEqual(m.iteration, 1)
        self.assertTrue(ws.is_clean())  # checkpoint commit landed; tree is clean

        # Simulate a crash mid-iteration-2: a tracked file is half-edited (uncommitted)
        # and an untracked file was created by the partial edit application.
        with open(os.path.join(self.repo, "target.txt"), "w") as fh:
            fh.write("half-written garbage\n")
        with open(os.path.join(self.repo, "crash_debris.txt"), "w") as fh:
            fh.write("leftover\n")
        self.assertFalse(ws.is_clean())
        self.store.set_state(m.id, "running")  # mission never made it back out of "running"

        # A fresh engine, as if the process restarted after the crash.
        second_coder = FakeCoder([CoderResponse(edits=[FileEdit("target.txt", "ok\n")], note="attempt 2")])
        engine2 = MissionEngine(self.store, second_coder, FakeVerifier(True))
        rewound = engine2.reconcile()
        self.assertEqual(rewound, 1)
        self.assertEqual(self.store.get(m.id).state, "queued")

        result = engine2.run_mission(m.id)

        self.assertEqual(result.state, "done")
        with open(os.path.join(self.repo, "target.txt")) as fh:
            self.assertEqual(fh.read(), "ok\n")
        self.assertFalse(os.path.exists(os.path.join(self.repo, "crash_debris.txt")))

    def test_resume_never_clobbers_user_work_on_a_different_branch(self):
        # Build a mission + repo, run one full iteration so the agent branch + a
        # checkpoint commit exist (mirrors test_resume_after_crash_resets_dirty_tree).
        m = self._mission([{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        branch = "agent/loop"
        original_branch = git(self.repo, "rev-parse", "--abbrev-ref", "HEAD")
        ws = Workspace(self.repo)
        ws.ensure_branch(branch)
        first_coder = FakeCoder([CoderResponse(edits=[FileEdit("target.txt", "nope\n")], note="attempt 1")])
        engine1 = MissionEngine(self.store, first_coder, FakeVerifier(True))
        m, done = engine1.run_iteration(m, {self.repo: ws})
        self.assertFalse(done)
        self.assertEqual(m.iteration, 1)
        self.assertTrue(ws.is_clean())  # checkpoint commit landed; tree is clean

        # Mission "crashed" (never made it back out of running), but in the meantime the
        # USER checked the repo back out to their own branch and did normal work there:
        # an untracked scratch file, plus an uncommitted edit to a tracked file the
        # mission never touched.
        git(self.repo, "checkout", "-q", original_branch)
        with open(os.path.join(self.repo, "user_scratch.txt"), "w") as fh:
            fh.write("my own untracked work\n")
        readme_path = os.path.join(self.repo, "README.md")
        with open(readme_path, "a") as fh:
            fh.write("user edit - do not delete\n")
        with open(readme_path) as fh:
            readme_before = fh.read()
        self.store.set_state(m.id, "running")  # mission never made it back out of "running"

        # A fresh engine, as if the process restarted after the crash.
        second_coder = FakeCoder([CoderResponse(edits=[FileEdit("target.txt", "ok\n")], note="attempt 2")])
        engine2 = MissionEngine(self.store, second_coder, FakeVerifier(True))
        rewound = engine2.reconcile()
        self.assertEqual(rewound, 1)

        with self.assertRaises(DirtyRepo):
            engine2.run_mission(m.id)

        # The user's work must survive the refusal untouched.
        self.assertTrue(os.path.exists(os.path.join(self.repo, "user_scratch.txt")))
        with open(readme_path) as fh:
            self.assertEqual(fh.read(), readme_before)

    def test_empty_success_criteria_never_vacuously_done(self):
        # Zero success_criteria must not be treated as trivially satisfied; the mission
        # should keep iterating (and hit the cap) rather than complete after one no-op pass.
        coder = FakeCoder([])  # never proposes edits; nothing to check anyway
        m = self._mission([])  # no success criteria at all
        engine = MissionEngine(self.store, coder, FakeVerifier(True), max_iterations=3)
        result = engine.run_mission(m.id)
        self.assertEqual(result.state, "needs_input")
        self.assertNotEqual(result.state, "done")


if __name__ == "__main__":
    unittest.main()
