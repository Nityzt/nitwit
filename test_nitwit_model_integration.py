"""Gated end-to-end: runs a REAL small mission against the live coder (:8080) + verifier
(:8086). Skipped automatically when either endpoint is down, so the suite stays green offline.
Bounded (max_iterations small) — GPU-safe per the crash envelope."""
import os
import tempfile
import unittest
from nitwit.missions import MissionStore
from nitwit.workspace import git
from nitwit.factory import build_model_engine, endpoint_healthy
from test_nitwit_workspace import make_repo

CODER_URL = os.environ.get("NITWIT_CODER_URL", "http://127.0.0.1:8080")
VERIFIER_URL = os.environ.get("NITWIT_VERIFIER_URL", "http://127.0.0.1:8086")
LIVE = endpoint_healthy(CODER_URL) and endpoint_healthy(VERIFIER_URL)


@unittest.skipUnless(LIVE, "live coder/verifier endpoints not available")
class TestModelMissionLive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "m.db"))
        self.repo = make_repo()
        with open(os.path.join(self.repo, "test_feature.py"), "w") as fh:
            fh.write("from feature import add\nassert add(2, 3) == 5\nprint('PASS')\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "failing test")

    def test_real_mission_reaches_green(self):
        m = self.store.create(
            "implement feature.add(a, b) so test_feature.py passes",
            repos=[{"path": self.repo, "branch": "agent/feat", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=[{"type": "tests", "repo": self.repo, "cmd": "python3 test_feature.py"}],
        )
        engine = build_model_engine(self.store, max_iterations=6)
        result = engine.run_mission(m.id)
        self.assertEqual(result.state, "done", f"mission ended {result.state}; notes:\n{result.notes}")
        with open(os.path.join(self.repo, "feature.py")) as fh:
            self.assertIn("a+b", fh.read().replace(" ", "") or "")


if __name__ == "__main__":
    unittest.main()
