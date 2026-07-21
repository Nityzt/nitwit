import io
import json
import threading
import unittest
import unittest.mock
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
        if self.path.startswith("/missions/"): return self._send({"id": "m1", "repos": []})
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

    def test_start_mission_without_repo_creates_scratch_workspace(self):
        # _start_mission(base, repo=None, ...) should create a scratch workspace
        # and proceed with the mission (not refuse).
        calls = []
        orig_api_call = cli.api_call
        def spy(base, method, path, body=None):
            calls.append((method, path))
            if method == "GET" and "/missions/" in path:
                return (200, {"id": "m1", "state": "done"})
            return orig_api_call(base, method, path, body)
        buf = io.StringIO()
        with redirect_stdout(buf):
            with unittest.mock.patch.object(cli, "api_call", side_effect=spy):
                cli._start_mission(self.base, None, None, "add x")
        out = buf.getvalue().lower()
        self.assertIn("scratch workspace", out)
        self.assertIn("export", out)
        # Should have called the API (POST /missions and GET /missions/{id})
        self.assertGreater(len(calls), 0)
        self.assertIn(("POST", "/missions"), calls)

    def test_start_mission_with_repo_uses_branch_not_scratch(self):
        # _start_mission(base, repo="/repo", ...) should use the repo and NOT create a scratch workspace.
        import tempfile
        repo = tempfile.mkdtemp()
        calls = []
        orig_api_call = cli.api_call
        def spy(base, method, path, body=None):
            calls.append((method, path, body))
            if method == "GET" and "/missions/" in path:
                return (200, {"id": "m1", "state": "done"})
            return orig_api_call(base, method, path, body)
        buf = io.StringIO()
        with redirect_stdout(buf):
            with unittest.mock.patch.object(cli, "api_call", side_effect=spy):
                cli._start_mission(self.base, repo, None, "add x")
        out = buf.getvalue().lower()
        # Should NOT mention scratch workspace
        self.assertNotIn("scratch", out)
        # Should mention the branch
        self.assertIn("agent/add-x", out)
        # Should have used the provided repo in the API call
        post_missions = [call for call in calls if call[0] == "POST" and "/missions" in call[1]]
        self.assertGreater(len(post_missions), 0)
        body = post_missions[0][2]
        self.assertIn(repo, body["repos"][0]["path"])

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

    def test_cmd_export(self):
        import tempfile, os
        # the stub returns a mission dict; ensure it has repos[0].path pointing at a real dir
        src = tempfile.mkdtemp()
        with open(os.path.join(src, "f.txt"), "w") as fh: fh.write("hi")
        # monkeypatch api_call to return a mission with that path
        orig = cli.api_call
        cli.api_call = lambda base, method, path, body=None: (200, {"id": "m1", "repos": [{"path": src}]})
        try:
            dest = os.path.join(tempfile.mkdtemp(), "exported")
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main(["export", "m1", dest, "--url", self.base])
            self.assertTrue(os.path.exists(os.path.join(dest, "f.txt")))
        finally:
            cli.api_call = orig


if __name__ == "__main__":
    unittest.main()
