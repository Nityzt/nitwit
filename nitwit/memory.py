"""Durable cross-session memory: a tiny SQLite store + a heuristic that spots facts worth
remembering. Nothing is saved without the user's approval (the CLI gates writes)."""
from __future__ import annotations

import os
import re
import sqlite3
import threading
import time

_DEFAULT_DB = os.path.expanduser("~/.local/share/nitwit/memory.db")

_PROPOSE_PATTERNS = [
    r"\bmy name is\s+(.+)",
    r"\bcall me\s+(.+)",
    r"\bi (?:use|prefer|always|never)\b.*",
    r"\bwe use\b.*",
    r"\bthe project uses\b.*",
    r"\bremember(?: that)?\s*:?\s+(.+)",
    r"\bnote that\s+(.+)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PROPOSE_PATTERNS]


def propose_memory(text: str) -> str | None:
    t = (text or "").strip()
    if not t or t.endswith("?"):
        return None
    for pat in _COMPILED:
        m = pat.search(t)
        if m:
            # for "my name is X"/"call me X"/"remember X"/"note that X" prefer the captured span;
            # for "i use/we use/..." keep the whole clause (it carries the fact)
            captured = m.groups()[0] if m.groups() else ""
            fact = captured.strip(" .!") if captured else t
            return fact if fact else t
    return None


class MemoryStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.environ.get("NITWIT_MEMORY_DB", _DEFAULT_DB)
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS memories "
                               "(id INTEGER PRIMARY KEY, text TEXT UNIQUE, created REAL)")
            self._conn.commit()

    def add(self, text: str) -> int:
        text = (text or "").strip()
        if not text:
            return 0
        with self._lock:
            cur = self._conn.execute("SELECT id FROM memories WHERE text = ?", (text,))
            row = cur.fetchone()
            if row:
                return row["id"]
            cur = self._conn.execute("INSERT INTO memories (text, created) VALUES (?, ?)",
                                     (text, time.time()))
            self._conn.commit()
            return cur.lastrowid

    def list(self) -> list[dict]:
        with self._lock:
            cur = self._conn.execute("SELECT id, text, created FROM memories ORDER BY created DESC")
            return [dict(r) for r in cur.fetchall()]

    def facts(self) -> list[str]:
        with self._lock:
            cur = self._conn.execute("SELECT text FROM memories ORDER BY created ASC")
            return [r["text"] for r in cur.fetchall()]

    def delete(self, mem_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            self._conn.commit()
            return cur.rowcount > 0
