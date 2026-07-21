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

    def test_pause_resume_roundtrip(self):
        st, m = _post(self.base + "/missions", {"goal": "pause me", "repos": [], "success_criteria": []})
        self.assertEqual(st, 200)
        mid = m["id"]
        st, body = _post(self.base + f"/missions/{mid}/pause")
        self.assertEqual(st, 200)
        self.assertEqual(body["state"], "paused")
        st, body = _post(self.base + f"/missions/{mid}/resume")
        self.assertEqual(st, 200)
        self.assertEqual(body["state"], "queued")

    def test_needs_input_answer_roundtrip(self):
        st, m = _post(self.base + "/missions", {"goal": "answer me", "repos": [], "success_criteria": []})
        self.assertEqual(st, 200)
        mid = m["id"]
        self.daemon.store.set_state(mid, "running")
        self.daemon.store.set_state(mid, "needs_input")
        st, body = _post(self.base + f"/missions/{mid}/answer", {"answer": "do X"})
        self.assertEqual(st, 200)
        self.assertEqual(body["state"], "queued")
        self.assertIn("do X", body["notes"])

    def test_cancel(self):
        st, m = _post(self.base + "/missions", {"goal": "cancel me", "repos": [], "success_criteria": []})
        self.assertEqual(st, 200)
        mid = m["id"]
        st, body = _post(self.base + f"/missions/{mid}/cancel")
        self.assertEqual(st, 200)
        self.assertEqual(body["state"], "cancelled")

    def test_wrong_shape_body_returns_400(self):
        data = json.dumps([1, 2, 3]).encode()
        req = urllib.request.Request(
            self.base + "/missions", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(cm.exception.code, 400)

    def test_spoofed_host_header_rejected(self):
        # Simulates a DNS-rebinding / CSRF attack: a browser page that resolved
        # "evil.com" to 127.0.0.1 and issues a request with that Host header.
        # Loopback binding alone does not stop this -- the Host header must be checked.
        req = urllib.request.Request(self.base + "/status", headers={"Host": "evil.com"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(cm.exception.code, 403)

        data = json.dumps({"goal": "pwn"}).encode()
        req = urllib.request.Request(
            self.base + "/missions", data=data,
            headers={"Content-Type": "application/json", "Host": "evil.com"}, method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(cm.exception.code, 403)

    def test_normal_host_header_still_works(self):
        # Default urllib/browser requests to 127.0.0.1:<port> send a matching Host
        # header and must keep working.
        st, body = _get(self.base + "/status")
        self.assertEqual(st, 200)
        self.assertIn("on", body)


if __name__ == "__main__":
    unittest.main()
