"""End-to-end offline: daemon + HTTP API + worker drive a FakeCoder mission to done."""
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from nitwit.missions import MissionStore
from nitwit.engine import MissionEngine
from nitwit.coder import CoderResponse, FakeCoder, FakeVerifier
from nitwit.workspace import FileEdit, git
from nitwit.daemon import MissionDaemon
from nitwit.api import make_server
from test_nitwit_workspace import make_repo


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(url=req, timeout=5) as r:
        return json.loads(r.read().decode())


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode())


class TestDaemonEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "m.db"))
        self.repo = make_repo()
        coder = FakeCoder([CoderResponse(edits=[FileEdit("target.txt", "ok\n")], note="write target")])
        engine = MissionEngine(self.store, coder, FakeVerifier(True), max_iterations=3)
        self.daemon = MissionDaemon(self.store, engine)
        self.daemon.start()
        self.server = make_server(self.daemon, port=0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()
        self.daemon.stop()

    def test_mission_via_api_reaches_done(self):
        m = _post(self.base + "/missions", {
            "goal": "make target.txt say ok",
            "repos": [{"path": self.repo, "branch": "agent/e2e", "test_cmd": "", "checkpoint_commit": ""}],
            "success_criteria": [{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}],
        })
        _post(self.base + "/control/on", {})
        mid = m["id"]
        end = time.time() + 10
        state = None
        while time.time() < end:
            state = _get(self.base + f"/missions/{mid}")["state"]
            if state in ("done", "failed", "needs_input"):
                break
            time.sleep(0.1)
        self.assertEqual(state, "done")


if __name__ == "__main__":
    unittest.main()
