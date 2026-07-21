"""Mission: a first-class structured intent object, plus its durable SQLite store."""
from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict

# List/dict fields are JSON-encoded into single TEXT columns for SQLite.
_JSON_FIELDS = ("constraints", "success_criteria", "repos", "artifacts")


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s or f"mission-{uuid.uuid4().hex[:8]}"


@dataclass
class Mission:
    id: str
    goal: str
    title: str = ""
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[dict] = field(default_factory=list)
    repos: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)  # schema-reserved (Phase 3+)
    notes: str = ""
    question: str = ""  # schema-reserved (needs_input, Phase 4)
    state: str = "queued"
    iteration: int = 0
    created: float = 0.0
    updated: float = 0.0

    def to_row(self) -> dict:
        row = asdict(self)
        for key in _JSON_FIELDS:
            row[key] = json.dumps(row[key])
        return row

    @classmethod
    def from_row(cls, row: dict) -> "Mission":
        data = dict(row)
        for key in _JSON_FIELDS:
            value = data.get(key)
            data[key] = json.loads(value) if isinstance(value, str) else (value or [])
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


VALID_TRANSITIONS: dict[str, set[str]] = {
    "queued": {"running", "cancelled"},
    "running": {"paused", "needs_input", "done", "failed", "queued"},  # ->queued = reconcile rewind
    "paused": {"running", "cancelled"},
    "needs_input": {"running", "cancelled"},
    "done": set(),
    "failed": set(),
    "cancelled": set(),
}


class InvalidTransition(Exception):
    pass


class MissionStore:
    _COLS = list(Mission.__dataclass_fields__.keys())

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        cols = ", ".join(f"{c} TEXT" if c not in ("iteration",) else f"{c} INTEGER" for c in self._COLS)
        # id is the primary key; created/updated stored as REAL via TEXT is fine (json round-trips floats)
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS missions ({cols}, PRIMARY KEY (id))")
        self._conn.commit()

    def create(self, goal, *, title="", constraints=None, success_criteria=None, repos=None) -> Mission:
        now = time.time()
        m = Mission(
            id=uuid.uuid4().hex[:12], goal=goal, title=title or goal[:60],
            constraints=constraints or [], success_criteria=success_criteria or [],
            repos=repos or [], created=now, updated=now,
        )
        self.save(m)
        return m

    def save(self, m: Mission) -> None:
        m.updated = time.time()
        row = m.to_row()
        placeholders = ", ".join("?" for _ in self._COLS)
        self._conn.execute(
            f"INSERT OR REPLACE INTO missions ({', '.join(self._COLS)}) VALUES ({placeholders})",
            [row[c] for c in self._COLS],
        )
        self._conn.commit()

    def get(self, mission_id) -> Mission | None:
        cur = self._conn.execute("SELECT * FROM missions WHERE id = ?", (mission_id,))
        row = cur.fetchone()
        return Mission.from_row(dict(row)) if row else None

    def list(self, state=None) -> list[Mission]:
        if state:
            cur = self._conn.execute("SELECT * FROM missions WHERE state = ? ORDER BY created", (state,))
        else:
            cur = self._conn.execute("SELECT * FROM missions ORDER BY created")
        return [Mission.from_row(dict(r)) for r in cur.fetchall()]

    def set_state(self, mission_id, new_state) -> Mission:
        m = self.get(mission_id)
        if m is None:
            raise InvalidTransition(f"no such mission {mission_id}")
        if new_state != m.state and new_state not in VALID_TRANSITIONS.get(m.state, set()):
            raise InvalidTransition(f"{m.state} -> {new_state} not allowed")
        m.state = new_state
        self.save(m)
        return m

    def bump_iteration(self, mission_id) -> Mission:
        m = self.get(mission_id)
        m.iteration += 1
        self.save(m)
        return m

    def append_note(self, mission_id, text) -> Mission:
        m = self.get(mission_id)
        stamp = time.strftime("%H:%M:%S")
        m.notes = (m.notes + f"\n[{stamp}] {text}").strip()
        self.save(m)
        return m
