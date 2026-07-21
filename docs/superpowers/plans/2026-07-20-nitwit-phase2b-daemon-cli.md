# Nitwit Phase 2b: Daemon + HTTP/SSE API + `wit` CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the working engine into the always-on product: a headless daemon that runs missions 24/7, exposes them over a loopback HTTP + SSE API, and is driven by a `wit` CLI (interactive REPL + one-shot + control subcommands). Close the REPL and missions keep running in the daemon; reopen and see them.

**Architecture:** The daemon owns a `MissionEngine` (built via `build_model_engine`) and a single background worker thread that runs queued missions one at a time (the single GPU slot). Progress flows as structured events: the engine gets an `on_event` callback (Task 1) → the daemon's in-process event bus (Task 2) → SSE clients (Task 3). The `wit` CLI (Task 4) is a thin client over the loopback API. A systemd user unit (Task 5) runs the daemon; `reconcile()` on start rewinds any mission left `running` by a crash. Everything binds `127.0.0.1` only.

**Tech Stack:** Python 3.14 (uv), stdlib only — `http.server` (ThreadingHTTPServer), `urllib`, `threading`, `queue`, `json`, `argparse`, `unittest`. No new deps. Reuses `nitwit.factory.build_model_engine`, `nitwit.missions.MissionStore`, `nitwit.engine.MissionEngine`.

## Global Constraints

- **Stdlib only** — `http.server`, `urllib`, `threading`, `queue`, `json`, `argparse`. No pip deps.
- **Loopback only** — the API server binds `127.0.0.1`, never `0.0.0.0` (CLAUDE.md house rule; reach it from other machines via `ssh -L`).
- **One GPU slot** — the daemon worker runs **one mission at a time** (global FIFO over `state="queued"`, oldest first).
- **Durable + resumable** — all mission state lives in SQLite; the daemon holds no authoritative state in RAM. On start it calls `engine.reconcile()`.
- **The daemon must survive a closed client** — missions run in the worker thread, independent of any HTTP/SSE connection.
- **Toggle frees the GPU** — `off` pauses the worker (finishes the current model call, parks); it does not need to stop the coder container in this phase (that is a later ops concern), but it must stop dispatching new missions.
- **Test convention:** root-level `test_nitwit_*.py`, `unittest`. Tests use a **FakeCoder/FakeVerifier** daemon (no GPU) and an ephemeral port; a gated live test is optional. The offline suite must stay green.

## File Structure

- `nitwit/engine.py` — MODIFY: add optional `on_event` callback + emit structured events.
- `nitwit/daemon.py` — `MissionDaemon`: worker thread, control flag, in-process `EventBus`.
- `nitwit/api.py` — `make_server(daemon, port)`: ThreadingHTTPServer with REST + SSE handlers.
- `nitwit/cli.py` — `wit` client: subcommands + one-shot + REPL, over the API.
- `nitwit/__main__.py` — daemon entrypoint (`python3 -m nitwit`): build engine, start daemon + server.
- `wit` — thin executable shim calling `nitwit.cli:main` (or document `python3 -m nitwit.cli`).
- `~/.config/systemd/user/nitwit.service` — daemon unit (delivered as a file in-repo: `deploy/nitwit.service`).
- Tests: `test_nitwit_daemon.py`, `test_nitwit_api.py`, `test_nitwit_cli.py`, `test_nitwit_daemon_integration.py`.

---

## Task 1: Engine event emission

**Files:**
- Modify: `nitwit/engine.py`
- Test: `test_nitwit_engine.py`

**Interfaces:**
- Consumes: existing `MissionEngine`.
- Produces: `MissionEngine.__init__(..., on_event=None)` stored as `self.on_event`; a private `self._emit(event_type:str, mission_id:str, **data)` that calls `on_event({"event": event_type, "mission_id": mission_id, "time": <float>, **data})` when set. Emits at least: `mission_started`, `iteration_started` (with `iteration`), `edits_applied` (with `paths`), `criteria_evaluated` (with `passed`, `summary`), `mission_finished` (with `state`). No behavior change when `on_event` is None.

- [ ] **Step 1: Write the failing test**

Append to `test_nitwit_engine.py` (new class):

```python
class TestEngineEvents(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "m.db"))
        self.repo = make_repo()

    def test_emits_lifecycle_events(self):
        events = []
        coder = FakeCoder([CoderResponse(edits=[FileEdit("target.txt", "ok\n")], note="x")])
        m = self.store.create(
            "e", repos=[{"path": self.repo, "branch": "agent/e", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=[{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        engine = MissionEngine(self.store, coder, FakeVerifier(True), on_event=events.append)
        engine.run_mission(m.id)
        kinds = [e["event"] for e in events]
        self.assertIn("mission_started", kinds)
        self.assertIn("iteration_started", kinds)
        self.assertIn("criteria_evaluated", kinds)
        self.assertIn("mission_finished", kinds)
        finished = [e for e in events if e["event"] == "mission_finished"][0]
        self.assertEqual(finished["state"], "done")
        self.assertEqual(finished["mission_id"], m.id)

    def test_no_callback_is_safe(self):
        coder = FakeCoder([CoderResponse(edits=[FileEdit("target.txt", "ok\n")], note="x")])
        m = self.store.create(
            "e", repos=[{"path": self.repo, "branch": "agent/e2", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=[{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        engine = MissionEngine(self.store, coder, FakeVerifier(True))  # no on_event
        self.assertEqual(engine.run_mission(m.id).state, "done")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_nitwit_engine -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'on_event'`

- [ ] **Step 3: Implement**

In `nitwit/engine.py`, add `import time` (already present) and modify `__init__` to accept `on_event=None`:

```python
    def __init__(self, store, coder, verifier, workspace_factory=Workspace,
                 max_iterations=20, cooldown_s=0.0, on_event=None):
        # ... existing assignments ...
        self.on_event = on_event
```

Add the emit helper:

```python
    def _emit(self, event_type: str, mission_id: str, **data) -> None:
        if self.on_event:
            self.on_event({"event": event_type, "mission_id": mission_id,
                           "time": round(time.time(), 3), **data})
```

Wire emits: in `run_mission` after setting `running`, `self._emit("mission_started", mission.id)`; before the final return in each terminal branch, `self._emit("mission_finished", mission.id, state=<the state>)`. In `run_iteration`, at the top `self._emit("iteration_started", mission.id, iteration=mission.iteration + 1)`; after applying edits `self._emit("edits_applied", mission.id, paths=[e.path for e in response.edits])`; after evaluating `self._emit("criteria_evaluated", mission.id, passed=done, summary=summary)`. (Keep the existing return values unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_nitwit_engine -v`
Expected: PASS (all engine tests incl. 2 new)

- [ ] **Step 5: Commit**

```bash
git add nitwit/engine.py test_nitwit_engine.py
git commit -m "feat(nitwit): engine emits structured lifecycle events"
```

---

## Task 2: MissionDaemon (worker + event bus)

**Files:**
- Create: `nitwit/daemon.py`
- Test: `test_nitwit_daemon.py`

**Interfaces:**
- Consumes: `MissionStore`, `MissionEngine`.
- Produces:
  - `EventBus` — thread-safe pub/sub: `subscribe() -> queue.Queue`, `unsubscribe(q)`, `publish(event: dict)` (non-blocking put to every subscriber).
  - `MissionDaemon(store, engine, poll_interval=0.2)`. The engine's `on_event` must be wired to `self.bus.publish` (the daemon sets `engine.on_event = self.bus.publish` in `__init__`). Methods: `start()` (reconcile, launch worker thread), `stop()` (signal worker to exit, join), `turn_on()`/`turn_off()` (control flag; off = worker stops dispatching new missions, calls `engine.pause()`; on = `engine.resume()`), `is_on() -> bool`, `status() -> dict` (running flag, active mission id or None, counts by state). The worker loop: while running, if ON, pick the oldest `queued` mission and `engine.run_mission(id)`; else sleep `poll_interval`.

- [ ] **Step 1: Write the failing test**

Create `test_nitwit_daemon.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_nitwit_daemon -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nitwit.daemon'`

- [ ] **Step 3: Implement**

Create `nitwit/daemon.py`:

```python
"""MissionDaemon: the always-on worker that runs queued missions one at a time, plus a
thread-safe EventBus fanning engine events out to SSE subscribers."""
from __future__ import annotations

import queue
import threading
import time


class EventBus:
    def __init__(self) -> None:
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # slow consumer drops events rather than blocking the worker


class MissionDaemon:
    def __init__(self, store, engine, poll_interval: float = 0.2) -> None:
        self.store = store
        self.engine = engine
        self.poll_interval = poll_interval
        self.bus = EventBus()
        self.engine.on_event = self.bus.publish
        self._on = threading.Event()          # control flag: dispatch missions when set
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_id: str | None = None

    def start(self) -> None:
        self.engine.reconcile()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.engine.pause()
        if self._thread:
            self._thread.join(timeout=5)

    def turn_on(self) -> None:
        self._on.set()
        self.engine.resume()

    def turn_off(self) -> None:
        self._on.clear()
        self.engine.pause()

    def is_on(self) -> bool:
        return self._on.is_set()

    def status(self) -> dict:
        counts: dict[str, int] = {}
        for m in self.store.list():
            counts[m.state] = counts.get(m.state, 0) + 1
        return {"on": self.is_on(), "active_mission": self._active_id, "counts": counts}

    def _next_queued(self):
        q = self.store.list(state="queued")
        return q[0] if q else None

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._on.is_set():
                time.sleep(self.poll_interval)
                continue
            mission = self._next_queued()
            if mission is None:
                time.sleep(self.poll_interval)
                continue
            self._active_id = mission.id
            try:
                self.engine.run_mission(mission.id)
            except Exception as exc:  # never let one mission kill the worker
                self.bus.publish({"event": "mission_error", "mission_id": mission.id,
                                  "error": str(exc), "time": round(time.time(), 3)})
            finally:
                self._active_id = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_nitwit_daemon -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add nitwit/daemon.py test_nitwit_daemon.py
git commit -m "feat(nitwit): MissionDaemon worker + EventBus"
```

---

## Task 3: HTTP + SSE API

**Files:**
- Create: `nitwit/api.py`
- Test: `test_nitwit_api.py`

**Interfaces:**
- Consumes: `MissionDaemon`, `MissionStore`.
- Produces: `make_server(daemon, port:int=8807, host:str="127.0.0.1") -> ThreadingHTTPServer` whose handler serves JSON REST + SSE. Routes:
  - `GET /status` → `daemon.status()`.
  - `POST /control/on` / `POST /control/off` → toggles, returns status.
  - `GET /missions` → `[mission dicts]`; `GET /missions/{id}` → mission dict or 404.
  - `POST /missions` body `{goal, repos, success_criteria, title?, constraints?}` → creates a mission (queued), returns it.
  - `POST /missions/{id}/pause|resume|cancel` → set_state helper (cancel→"cancelled"); `POST /missions/{id}/answer` body `{answer}` → append to notes + set_state to "queued" (so the worker repicks it).
  - `GET /events` → SSE stream (`text/event-stream`) of `data: <json>\n\n` per event from `daemon.bus.subscribe()`, until the client disconnects.
  - Mission dict = `dataclasses.asdict(mission)`.

- [ ] **Step 1: Write the failing test**

Create `test_nitwit_api.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_nitwit_api -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nitwit.api'`

- [ ] **Step 3: Implement**

Create `nitwit/api.py`:

```python
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

        def do_GET(self):
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
            if self.path == "/control/on":
                daemon.turn_on(); return self._json(200, daemon.status())
            if self.path == "/control/off":
                daemon.turn_off(); return self._json(200, daemon.status())
            if self.path == "/missions":
                d = self._read_json()
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
                        store.append_note(mid, "USER ANSWER: " + str(self._read_json().get("answer", "")))
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_nitwit_api -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add nitwit/api.py test_nitwit_api.py
git commit -m "feat(nitwit): loopback HTTP + SSE API over the daemon"
```

---

## Task 4: `wit` CLI

**Files:**
- Create: `nitwit/cli.py`
- Create: `nitwit/__main__.py`
- Test: `test_nitwit_cli.py`

**Interfaces:**
- Consumes: the HTTP API (via `urllib`).
- Produces:
  - `nitwit/cli.py`: `api_call(base, method, path, body=None) -> (status, obj)`; `cmd_status/cmd_ls/cmd_new/cmd_pause/cmd_resume/cmd_cancel/cmd_answer/cmd_on/cmd_off(args, base)` printing human output; `stream(base, out=print)` (reads `/events` SSE, prints each event line via `humanize_event`); `humanize_event(event:dict) -> str`; `repl(base)` (reads lines; `/`-commands dispatch to the cmd_* funcs, `/tail` streams, `/quit` exits); `main(argv=None)` argparse: subcommands `new|ls|status|tail|pause|resume|cancel|answer|on|off`, flag `-p/--prompt` for one-shot, no args → `repl`. Base URL from `--url` / `NITWIT_URL` env / default `http://127.0.0.1:8807`.
  - `nitwit/__main__.py`: builds the engine (`build_model_engine`) + `MissionStore` (db at `~/.local/share/nitwit/missions.db`), starts a `MissionDaemon` and `make_server`, serves forever. `python3 -m nitwit` runs the daemon.

- [ ] **Step 1: Write the failing test**

Create `test_nitwit_cli.py` (tests parsing/formatting + dispatch against a stub server, no real daemon):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_nitwit_cli -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nitwit.cli'`

- [ ] **Step 3: Implement**

Create `nitwit/cli.py`:

```python
"""`wit` — the CLI client over the loopback daemon API. REPL + one-shot + subcommands."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

DEFAULT_URL = os.environ.get("NITWIT_URL", "http://127.0.0.1:8807")


def api_call(base: str, method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()}
    except urllib.error.URLError as e:
        return 0, {"error": f"cannot reach daemon at {base}: {e.reason}. Is it running? (python3 -m nitwit)"}


def humanize_event(event: dict) -> str:
    kind = event.get("event", "?")
    mid = event.get("mission_id", "")
    if kind == "mission_started":   return f"[{mid}] mission started"
    if kind == "iteration_started": return f"[{mid}] iteration {event.get('iteration')}"
    if kind == "edits_applied":     return f"[{mid}] edited: {', '.join(event.get('paths', []))}"
    if kind == "criteria_evaluated":return f"[{mid}] checked: {event.get('summary','')} -> {'PASS' if event.get('passed') else 'not yet'}"
    if kind == "mission_finished":  return f"[{mid}] finished: {event.get('state')}"
    if kind == "mission_error":     return f"[{mid}] ERROR: {event.get('error')}"
    return f"[{mid}] {kind}"


def cmd_status(args, base):
    _, s = api_call(base, "GET", "/status")
    print(json.dumps(s, indent=2) if isinstance(s, dict) else s)


def cmd_ls(args, base):
    _, missions = api_call(base, "GET", "/missions")
    if not isinstance(missions, list):
        print(missions); return
    for m in missions:
        print(f"{m['id']:14} {m['state']:12} iter={m.get('iteration',0):<3} {m.get('goal','')[:60]}")


def cmd_new(args, base):
    repos = []
    if args.repo:
        repos = [{"path": os.path.abspath(args.repo), "branch": f"agent/{args.branch}",
                  "test_cmd": args.test or "", "checkpoint_commit": ""}]
    crit = []
    if args.test:
        crit.append({"type": "tests", "repo": os.path.abspath(args.repo), "cmd": args.test})
    crit.append({"type": "verifier", "description": "the goal is meaningfully complete"})
    _, m = api_call(base, "POST", "/missions",
                    {"goal": args.goal, "repos": repos, "success_criteria": crit})
    print(f"created {m.get('id')} ({m.get('state')})" if isinstance(m, dict) and m.get("id") else m)


def _simple(action):
    def fn(args, base):
        _, m = api_call(base, "POST", f"/missions/{args.id}/{action}")
        print(f"{args.id}: {m.get('state', m)}" if isinstance(m, dict) else m)
    return fn


def cmd_answer(args, base):
    _, m = api_call(base, "POST", f"/missions/{args.id}/answer", {"answer": args.text})
    print(f"{args.id}: {m.get('state', m)}" if isinstance(m, dict) else m)


def cmd_toggle(on):
    def fn(args, base):
        _, s = api_call(base, "POST", "/control/on" if on else "/control/off")
        print(f"daemon {'ON' if on else 'OFF'}: {s}")
    return fn


def stream(base, out=print):
    req = urllib.request.Request(base + "/events")
    try:
        with urllib.request.urlopen(req) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    out(humanize_event(json.loads(line[6:])))
    except KeyboardInterrupt:
        pass
    except urllib.error.URLError as e:
        out(f"stream ended: {e}")


def cmd_tail(args, base):
    stream(base)


def repl(base):
    print("wit — interactive. /help for commands, /quit to exit. Missions keep running in the daemon.")
    while True:
        try:
            line = input("wit> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            print("/ls /status /tail /new <goal> --repo P --test CMD /pause <id> /resume <id> "
                  "/cancel <id> /answer <id> <text> /on /off /quit")
            continue
        argv = line[1:].split() if line.startswith("/") else ["new"] + [line]
        try:
            main(argv + ["--url", base])
        except SystemExit:
            pass


def build_parser():
    p = argparse.ArgumentParser(prog="wit")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("-p", "--prompt", help="one-shot: create a mission from this goal and exit")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("status")
    sub.add_parser("ls")
    sub.add_parser("tail")
    sub.add_parser("on")
    sub.add_parser("off")
    n = sub.add_parser("new"); n.add_argument("goal"); n.add_argument("--repo"); n.add_argument("--test"); n.add_argument("--branch", default="mission")
    for name in ("pause", "resume", "cancel"):
        s = sub.add_parser(name); s.add_argument("id")
    a = sub.add_parser("answer"); a.add_argument("id"); a.add_argument("text")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    base = args.url
    dispatch = {
        "status": cmd_status, "ls": cmd_ls, "new": cmd_new, "tail": cmd_tail,
        "pause": _simple("pause"), "resume": _simple("resume"), "cancel": _simple("cancel"),
        "answer": cmd_answer, "on": cmd_toggle(True), "off": cmd_toggle(False),
    }
    if args.prompt:
        args.goal, args.repo, args.test, args.branch = args.prompt, None, None, "mission"
        return cmd_new(args, base)
    if not args.cmd:
        return repl(base)
    dispatch[args.cmd](args, base)


if __name__ == "__main__":
    main()
```

Create `nitwit/__main__.py`:

```python
"""`python3 -m nitwit` — run the mission daemon + loopback API server."""
from __future__ import annotations

import os

from nitwit.api import make_server
from nitwit.daemon import MissionDaemon
from nitwit.factory import build_model_engine
from nitwit.missions import MissionStore

DB = os.environ.get("NITWIT_DB", os.path.expanduser("~/.local/share/nitwit/missions.db"))
PORT = int(os.environ.get("NITWIT_PORT", "8807"))


def main() -> None:
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    store = MissionStore(DB)
    engine = build_model_engine(store)
    daemon = MissionDaemon(store, engine)
    daemon.start()
    server = make_server(daemon, port=PORT)
    print(f"nitwit daemon on http://127.0.0.1:{PORT} (db {DB})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        daemon.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_nitwit_cli -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add nitwit/cli.py nitwit/__main__.py test_nitwit_cli.py
git commit -m "feat(nitwit): wit CLI (subcommands + REPL + one-shot) and daemon entrypoint"
```

---

## Task 5: systemd unit + end-to-end integration

**Files:**
- Create: `deploy/nitwit.service`
- Create: `test_nitwit_daemon_integration.py`

**Interfaces:**
- Consumes: everything above.
- Produces: a user systemd unit (delivered as a repo file to `install`/symlink into `~/.config/systemd/user/`), and a full offline end-to-end test: start a daemon+server on an ephemeral port with a **FakeCoder**, create a mission via the HTTP API, turn the daemon on, and assert (by polling `GET /missions/{id}`) it reaches `done` — proving daemon + API + worker + engine compose. No GPU.

- [ ] **Step 1: Write the end-to-end test**

Create `test_nitwit_daemon_integration.py`:

```python
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
```

- [ ] **Step 2: Run it (fails until the pieces compose)**

Run: `python3 -m unittest test_nitwit_daemon_integration -v`
Expected: PASS (daemon + API + worker all wired). If FAIL, fix the composition, not the test.

- [ ] **Step 3: Create the systemd unit**

Create `deploy/nitwit.service`:

```ini
[Unit]
Description=Nitwit autonomous mission daemon
After=default.target

[Service]
Type=simple
WorkingDirectory=/home/nit/qwen-orchestrator
Environment=NITWIT_PORT=8807
ExecStart=/usr/bin/env python3 -m nitwit
Restart=on-failure
Nice=10

[Install]
WantedBy=default.target
```

(Install later, out of band: symlink into `~/.config/systemd/user/`, `systemctl --user daemon-reload && systemctl --user enable --now nitwit`. Not done by this test.)

- [ ] **Step 4: Full offline suite green**

Run: `python3 -m unittest test_nitwit_missions test_nitwit_workspace test_nitwit_coder test_nitwit_engine test_nitwit_integration test_nitwit_model_coder test_nitwit_model_verifier test_nitwit_model_integration test_nitwit_daemon test_nitwit_api test_nitwit_cli test_nitwit_daemon_integration`
Expected: all PASS (live model test skips if endpoints down); `test_orchestrator` still passes.

- [ ] **Step 5: Commit**

```bash
git add deploy/nitwit.service test_nitwit_daemon_integration.py
git commit -m "feat(nitwit): systemd unit + daemon/API/CLI end-to-end integration test"
```

---

## Self-Review

**Spec coverage (Phase 2b scope):**
- Daemon runs missions 24/7, one at a time, reconcile-on-start → Task 2. ✓
- HTTP + SSE API, loopback only → Task 3. ✓
- `wit` CLI: subcommands + one-shot + REPL, control (on/off), mission ops (new/ls/tail/pause/resume/cancel/answer) → Task 4. ✓
- Close-the-REPL-missions-keep-running → the daemon owns the worker; the CLI is a stateless client (Tasks 2+4). ✓
- Structured event streaming for live progress → Task 1 (engine events) + Task 3 (SSE) + Task 4 (tail/humanize). ✓
- systemd service (24/7) + reconcile-on-start → Task 5 + Task 2. ✓
- Deferred (correctly out of 2b scope): token-level streaming of coder output (events are iteration-level); escalation/lane routing (chat/lookup vs mission — Phase 2c); toggle stopping the coder *container* (ops concern); total-prompt token budgeting (carried from 2a); auth on the API (loopback-only mitigates).

**Placeholder scan:** none — every step has full runnable code + exact commands. ✓

**Type consistency:** `MissionDaemon(store, engine)` wires `engine.on_event = self.bus.publish`; the API's `make_server(daemon, port)` reads `daemon.store`/`daemon.bus`/`daemon.status()`; the CLI hits the exact routes the API serves; `humanize_event` keys match the events Task 1 emits (`mission_started/iteration_started/edits_applied/criteria_evaluated/mission_finished/mission_error`). ✓

## Notes for later phases

- **Phase 2c:** escalation + lane routing — chat/lookup answered inline vs escalated to a mission; wire the existing router as the front door; the `wit` REPL's non-`/` input currently maps to `new` (a mission) — 2c makes it route.
- Token-level streaming (coder streams tokens → finer SSE), the UI refit (web client over the same API), and API auth (if ever bound beyond loopback).
