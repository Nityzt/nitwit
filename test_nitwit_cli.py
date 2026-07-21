import io
import json
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import redirect_stdout
from nitwit import cli


class _Stub(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/status": return self._send({"on": True, "active_mission": None, "counts": {"done": 2}})
        if self.path == "/missions": return self._send([{"id": "m1", "goal": "g", "state": "done", "iteration": 1}])
        self._send({"error": "nf"})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); self.rfile.read(n)
        self._send({"id": "m2", "goal": "new goal", "state": "queued", "iteration": 0})


class TestCli(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()

    def test_humanize_event(self):
        line = cli.humanize_event({"event": "iteration_started", "mission_id": "m1", "iteration": 3})
        self.assertIn("iteration", line.lower())
        self.assertIn("3", line)

    def test_cmd_status(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["status", "--url", self.base])
        self.assertIn("on", buf.getvalue().lower())

    def test_cmd_ls(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["ls", "--url", self.base])
        self.assertIn("m1", buf.getvalue())

    def test_cmd_new(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["new", "new goal", "--url", self.base])
        self.assertIn("m2", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
