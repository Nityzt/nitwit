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

    def test_url_before_subcommand_is_honored(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["--url", self.base, "status"])
        out = buf.getvalue()
        self.assertIn("on", out.lower())
        # active_mission only appears in the real stub /status payload, never in
        # the "cannot reach daemon" fallback message -- this is what actually
        # proves --url before the subcommand reached the stub server.
        self.assertIn("active_mission", out)

    def test_new_multiword_goal_without_quotes(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["new", "fix", "the", "bug", "--url", self.base])
        self.assertIn("m2", buf.getvalue())

    def test_new_test_without_repo_is_friendly_error(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["new", "goal", "--test", "pytest", "--url", self.base])
        out = buf.getvalue()
        self.assertIn("--repo", out)

    def test_bare_interactive_routes_to_session(self):
        # main() with no subcommand and stdin closed should attempt the interactive session,
        # which reads EOF immediately and exits cleanly (no traceback).
        import sys
        old = sys.stdin
        sys.stdin = io.StringIO("")  # immediate EOF
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main(["--url", self.base])  # ensure_daemon sees the stub (status 200)
            self.assertIn("nitwit", buf.getvalue().lower())  # printed a banner
        finally:
            sys.stdin = old


if __name__ == "__main__":
    unittest.main()
