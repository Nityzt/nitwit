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
                # Route the error to a terminal state so the mission doesn't get stuck
                # in `running` forever (the worker only dispatches `queued` missions, so a
                # stuck-`running` mission would sit idle and never retry). Record why.
                self.bus.publish({"event": "mission_error", "mission_id": mission.id,
                                  "error": str(exc), "time": round(time.time(), 3)})
                try:
                    self.store.append_note(mission.id, f"ERROR: {exc}")
                    self.store.set_state(mission.id, "failed")
                except Exception:
                    pass  # store already in a terminal/odd state; nothing more to do
            finally:
                self._active_id = None
