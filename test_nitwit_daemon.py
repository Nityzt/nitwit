import os
import tempfile
import time
import unittest
from nitwit.missions import MissionStore
from nitwit.engine import MissionEngine
from nitwit.coder import CoderResponse, FakeCoder, FakeVerifier
from nitwit.workspace import FileEdit
from nitwit.daemon import MissionDaemon, EventBus
from test_nitwit_workspace import make_repo


class TestEventBus(unittest.TestCase):
    def test_pub_sub(self):
        bus = EventBus()
        q = bus.subscribe()
        bus.publish({"event": "x"})
        self.assertEqual(q.get_nowait()["event"], "x")
        bus.unsubscribe(q)
        bus.publish({"event": "y"})  # no subscribers, must not raise
        self.assertTrue(q.empty())


def _wait_until(pred, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


class TestMissionDaemon(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "m.db"))
        self.repo = make_repo()

    def _engine(self):
        coder = FakeCoder([CoderResponse(edits=[FileEdit("target.txt", "ok\n")], note="x")])
        return MissionEngine(self.store, coder, FakeVerifier(True), max_iterations=3)

    def test_worker_runs_queued_mission_to_done(self):
        daemon = MissionDaemon(self.store, self._engine())
        m = self.store.create(
            "d", repos=[{"path": self.repo, "branch": "agent/d", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=[{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        daemon.start()
        daemon.turn_on()
        try:
            self.assertTrue(_wait_until(lambda: self.store.get(m.id).state == "done"))
        finally:
            daemon.stop()

    def test_off_does_not_dispatch(self):
        daemon = MissionDaemon(self.store, self._engine())
        m = self.store.create(
            "d", repos=[{"path": self.repo, "branch": "agent/d2", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=[{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        daemon.start()  # default OFF
        try:
            time.sleep(0.6)
            self.assertEqual(self.store.get(m.id).state, "queued")  # never dispatched
            self.assertFalse(daemon.is_on())
        finally:
            daemon.stop()


if __name__ == "__main__":
    unittest.main()


class TestDaemonErrorRouting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "m.db"))
        self.repo = make_repo()
        # make the repo dirty (untracked, non-ignored file) so ensure_branch raises DirtyRepo
        with open(os.path.join(self.repo, "scratch.txt"), "w") as fh:
            fh.write("uncommitted\n")

    def test_failing_mission_routed_to_failed_not_stuck_running(self):
        engine = MissionEngine(self.store, FakeCoder([]), FakeVerifier(True), max_iterations=2)
        daemon = MissionDaemon(self.store, engine)
        m = self.store.create(
            "x", repos=[{"path": self.repo, "branch": "agent/x", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=[{"type": "tests", "repo": self.repo, "cmd": "true"}])
        daemon.start(); daemon.turn_on()
        try:
            # it must reach a TERMINAL failed state, never sit stuck in running
            self.assertTrue(_wait_until(lambda: self.store.get(m.id).state == "failed"))
            self.assertIn("ERROR", self.store.get(m.id).notes)
        finally:
            daemon.stop()
