import json
import os
import tempfile
import threading
import unittest
import urllib.request
from nitwit.missions import MissionStore
from nitwit.engine import MissionEngine
from nitwit.coder import FakeCoder, FakeVerifier
from nitwit.daemon import MissionDaemon
from nitwit.api import make_server


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, json.loads(r.read().decode())


def _post(url, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read().decode())


class TestApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "m.db"))
        engine = MissionEngine(self.store, FakeCoder([]), FakeVerifier(True))
        self.daemon = MissionDaemon(self.store, engine)
        self.server = make_server(self.daemon, port=0)  # 0 => ephemeral port
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()

    def test_status(self):
        st, body = _get(self.base + "/status")
        self.assertEqual(st, 200)
        self.assertIn("on", body)

    def test_control_toggle(self):
        _post(self.base + "/control/on")
        _, body = _get(self.base + "/status")
        self.assertTrue(body["on"])
        _post(self.base + "/control/off")
        _, body = _get(self.base + "/status")
        self.assertFalse(body["on"])

    def test_create_list_get_mission(self):
        st, m = _post(self.base + "/missions", {"goal": "do a thing", "repos": [], "success_criteria": []})
        self.assertEqual(st, 200)
        mid = m["id"]
        st, lst = _get(self.base + "/missions")
        self.assertTrue(any(x["id"] == mid for x in lst))
        st, got = _get(self.base + f"/missions/{mid}")
        self.assertEqual(got["goal"], "do a thing")

    def test_missing_mission_404(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            _get(self.base + "/missions/nope")
        self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
