"""Mission: a first-class structured intent object, plus its durable SQLite store."""
from __future__ import annotations

import json
import re
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
