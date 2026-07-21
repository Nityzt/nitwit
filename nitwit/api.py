"""Loopback HTTP + SSE API over a MissionDaemon. Binds 127.0.0.1 only."""
from __future__ import annotations

import dataclasses
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _mission_dict(m) -> dict:
    return dataclasses.asdict(m)


def make_server(daemon, port: int = 8807, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    store = daemon.store

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode() or "{}")
            except ValueError:
                return {}

        def _host_ok(self) -> bool:
            # Loopback binding stops remote hosts, not a browser: any page the user has
            # open can still have JS POST to http://127.0.0.1:<port>, and a DNS-rebinding
            # attacker can point a hostname at 127.0.0.1. The Host header is the only
            # signal available to a same-origin-unaware handler like this one, so reject
            # anything that isn't explicitly loopback/localhost.
            host = self.headers.get("Host", "")
            if host.startswith("["):
                end = host.find("]")
                hostname = host[:end + 1] if end != -1 else host
            elif ":" in host:
                hostname = host.rsplit(":", 1)[0]
            else:
                hostname = host
            return hostname in ("127.0.0.1", "localhost", "::1", "[::1]")

        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "forbidden host"})
            if self.path == "/status":
                return self._json(200, daemon.status())
            if self.path == "/missions":
                return self._json(200, [_mission_dict(m) for m in store.list()])
            m = re.fullmatch(r"/missions/([\w-]+)", self.path)
            if m:
                mission = store.get(m.group(1))
                return self._json(200, _mission_dict(mission)) if mission else self._json(404, {"error": "not found"})
            if self.path == "/events":
                return self._sse()
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._host_ok():
                return self._json(403, {"error": "forbidden host"})
            if self.path == "/control/on":
                daemon.turn_on(); return self._json(200, daemon.status())
            if self.path == "/control/off":
                daemon.turn_off(); return self._json(200, daemon.status())
            if self.path == "/missions":
                d = self._read_json()
                if not isinstance(d, dict):
                    return self._json(400, {"error": "body must be a JSON object"})
                mission = store.create(d.get("goal", ""), title=d.get("title", ""),
                                       constraints=d.get("constraints"),
                                       success_criteria=d.get("success_criteria"),
                                       repos=d.get("repos"))
                return self._json(200, _mission_dict(mission))
            m = re.fullmatch(r"/missions/([\w-]+)/(pause|resume|cancel|answer)", self.path)
            if m:
                mid, action = m.group(1), m.group(2)
                if store.get(mid) is None:
                    return self._json(404, {"error": "not found"})
                try:
                    if action == "pause":
                        store.set_state(mid, "paused")
                    elif action == "resume":
                        store.set_state(mid, "queued")
                    elif action == "cancel":
                        store.set_state(mid, "cancelled")
                    elif action == "answer":
                        ans_body = self._read_json()
                        if not isinstance(ans_body, dict):
                            return self._json(400, {"error": "body must be a JSON object"})
                        store.append_note(mid, "USER ANSWER: " + str(ans_body.get("answer", "")))
                        store.set_state(mid, "queued")
                except Exception as exc:
                    return self._json(400, {"error": str(exc)})
                return self._json(200, _mission_dict(store.get(mid)))
            return self._json(404, {"error": "not found"})

        def _sse(self):
            q = daemon.bus.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    event = q.get()
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                daemon.bus.unsubscribe(q)

    server = ThreadingHTTPServer((host, port), Handler)
    return server
