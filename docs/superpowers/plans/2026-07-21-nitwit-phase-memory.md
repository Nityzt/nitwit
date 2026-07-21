# Nitwit Phase 4: Persistent memory (propose + approve + recall)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Remember durable facts across sessions. As you tell it things ("I use pnpm", "call me Wit", "the project uses FastAPI"), it proposes to remember them; you approve; they persist to disk and are auto-recalled into context next time — so it just knows.

**Architecture:** A small SQLite `MemoryStore` (`~/.local/share/nitwit/memory.db`). A conservative `propose_memory(user_text)` heuristic spots durable preference/fact statements. The interactive session loads memories at start, injects them into the chat's system prompt (recall), and after each turn offers to save a spotted fact (approve). Slash commands `/remember`, `/memories`, `/forget` for manual control.

**Tech Stack:** Python 3.14, stdlib only (`sqlite3`, `re`, `time`, `os`). No new deps.

## Global Constraints
- Stdlib only; store at `~/.local/share/nitwit/memory.db` (override via `NITWIT_MEMORY_DB`).
- Nothing is saved without approval (interactive prompt) or an explicit `/remember`. `propose_memory` only *suggests*.
- Recall injects known facts into the chat system prompt (and is available for missions later); it must not blow the context — cap the injected block.
- `MemoryStore` must be safe to open from multiple processes (WAL or a lock) since the daemon may also read it later — for Phase 4 (CLI-only writer) a simple connection is fine, but use `check_same_thread=False` + a lock for forward-safety.
- Tests: root-level `test_nitwit_*.py`, `unittest`; temp DB, no network.

## File Structure
- `nitwit/memory.py` — `MemoryStore` (add/list/facts/delete), `propose_memory(text) -> str | None`.
- `nitwit/session.py` — MODIFY `stream_answer`: optional `memories: list[str]` → inject into system prompt.
- `nitwit/cli.py` — MODIFY `interactive`: load memories, pass to chat, propose+approve after each turn, `/remember` `/memories` `/forget`.
- Tests: `test_nitwit_memory.py`, additions to `test_nitwit_session.py`, `test_nitwit_cli.py`.

---

## Task 1: MemoryStore + propose_memory

**Files:** Create `nitwit/memory.py`; Test `test_nitwit_memory.py`.

**Interfaces (Produces):**
- `MemoryStore(db_path=None)` — default `os.environ.get("NITWIT_MEMORY_DB", ~/.local/share/nitwit/memory.db)`; creates a `memories(id INTEGER PK, text TEXT, created REAL)` table. Methods: `add(text: str) -> int` (ignores empty/duplicate text, returns id or the existing id), `list() -> list[dict]` (id/text/created, newest first), `facts() -> list[str]` (just texts, oldest first for stable prompts), `delete(id: int) -> bool`. Thread-safe via an internal lock.
- `propose_memory(text: str) -> str | None` — returns a normalized durable fact if the message states a durable preference/identity/project convention, else None. Fires on patterns like: `i use/prefer/always/never <X>`, `my name is <X>` / `call me <X>`, `we use <X>`, `the project uses <X>`, `remember (that)? <X>`, `note that <X>`. Conservative — returns None for ordinary chat/questions.

- [ ] **Step 1: failing test** — create `test_nitwit_memory.py`:

```python
import os, tempfile, unittest
from nitwit.memory import MemoryStore, propose_memory


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "mem.db")

    def test_add_list_facts_delete_persist(self):
        s = MemoryStore(self.db)
        i = s.add("uses pnpm not npm")
        s.add("prefers tabs")
        self.assertEqual(len(s.list()), 2)
        self.assertIn("uses pnpm not npm", s.facts())
        # dedupe: adding same text again doesn't duplicate
        s.add("uses pnpm not npm")
        self.assertEqual(len(s.list()), 2)
        # persists across instances
        self.assertIn("prefers tabs", MemoryStore(self.db).facts())
        self.assertTrue(s.delete(i))
        self.assertNotIn("uses pnpm not npm", MemoryStore(self.db).facts())

    def test_add_ignores_empty(self):
        s = MemoryStore(self.db)
        s.add("   ")
        self.assertEqual(s.list(), [])


class TestProposeMemory(unittest.TestCase):
    def test_durable_facts_proposed(self):
        for t in ["I use pnpm not npm", "my name is Nit", "call me Wit",
                  "we use FastAPI for the backend", "I always use type hints",
                  "remember that the db is postgres", "I prefer tabs over spaces"]:
            self.assertIsNotNone(propose_memory(t), t)

    def test_ordinary_chat_not_proposed(self):
        for t in ["what does parse() do?", "hi", "thanks", "add a health endpoint",
                  "how does the loop work", "is this thread-safe?"]:
            self.assertIsNone(propose_memory(t), t)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: run, expect FAIL** — ModuleNotFoundError.

- [ ] **Step 3: implement** — create `nitwit/memory.py`:

```python
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
```

- [ ] **Step 4: run, expect PASS** — `python3 -m unittest test_nitwit_memory -v`.

- [ ] **Step 5: commit** — `git add nitwit/memory.py test_nitwit_memory.py && git commit -m "feat(nitwit): durable MemoryStore + propose_memory heuristic"`

---

## Task 2: recall in chat + propose/approve in the session

**Files:** MODIFY `nitwit/session.py`, `nitwit/cli.py`. Test: additions to `test_nitwit_session.py`, `test_nitwit_cli.py`.

**Interfaces:**
- `session.stream_answer(..., memories: list[str] | None = None)` — if `memories`, prepend to the system prompt a block: `"Known facts about the user (honor these):\n- <f1>\n- <f2>..."` (cap ~1500 chars). Unchanged otherwise.
- `cli.interactive` — at start: `mem = MemoryStore()`; pass `memories=mem.facts()` to each `stream_answer`. After a natural-language *answer* turn (not a task, not a slash), call `propose_memory(line)`; if it returns a fact not already in `mem.facts()`, print `remember: "<fact>"? [y saves]` and read one line; on `y`/`yes`, `mem.add(fact)` and refresh the passed facts. Slash: `/remember <text>` (mem.add + confirm), `/memories` (numbered list from `mem.list()`), `/forget <id>` (mem.delete).

- [ ] **Step 1: failing tests** — add to `test_nitwit_session.py`:

```python
class TestStreamAnswerMemory(unittest.TestCase):
    def test_memories_injected_into_system_prompt(self):
        from nitwit import session
        from nitwit.router import Endpoint
        captured = {}
        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                captured["sys"] = messages[0]["content"]
                yield {"type": "chunk", "content": "ok"}; yield {"type": "done"}
        session.stream_answer("hi", None, _endpoint=Endpoint("http://x", "m", {}),
                              out=lambda s: None, _client_factory=lambda u, m, extra_body=None: FakeClient(),
                              memories=["uses pnpm not npm", "call me Wit"])
        self.assertIn("uses pnpm not npm", captured["sys"])
        self.assertIn("call me Wit", captured["sys"])
```

Add to `test_nitwit_cli.py` a memory approve-flow test (drive `interactive` with a scripted stdin: a fact statement, `y` to save, then `/quit`), using a MemoryStore on a temp `NITWIT_MEMORY_DB` and the stub server; assert the fact is persisted:

```python
    def test_interactive_proposes_and_saves_memory(self):
        import io, os, sys, tempfile
        from contextlib import redirect_stdout
        os.environ["NITWIT_MEMORY_DB"] = os.path.join(tempfile.mkdtemp(), "m.db")
        # stream_answer would hit the model; stub it to a no-op so the loop reaches the propose step
        import nitwit.session as S
        orig = S.stream_answer
        S.stream_answer = lambda *a, **k: ""
        old_stdin = sys.stdin
        sys.stdin = io.StringIO("I use pnpm not npm\ny\n/quit\n")
        try:
            from nitwit import cli
            from nitwit.memory import MemoryStore
            with redirect_stdout(io.StringIO()):
                cli.interactive(self.base, os.getcwd())
            self.assertTrue(any("pnpm" in f for f in MemoryStore().facts()))
        finally:
            S.stream_answer = orig
            sys.stdin = old_stdin
            del os.environ["NITWIT_MEMORY_DB"]
```

- [ ] **Step 2: run, expect FAIL**.

- [ ] **Step 3: implement** — in `session.stream_answer`, add `memories=None` and, when building `system`, append (before the "Answer directly" clause or after the identity clause):

```python
    if memories:
        block = "\n".join(f"- {f}" for f in memories)[:1500]
        system += "\nKnown facts about the user (honor these):\n" + block
```

In `nitwit/cli.py`: `from nitwit.memory import MemoryStore, propose_memory` at top. In `interactive`, after `C = _colors()` and daemon/repo setup, add `mem = MemoryStore()`. In the natural-language answer branch, pass `memories=mem.facts()` to `stream_answer`, and after it returns, add the propose/approve:

```python
        else:
            print(C["assist"], end="")
            answer = session.stream_answer(line, repo, memories=mem.facts(), history=history)
            print(C["reset"], end="", flush=True)
            history.append({"role": "user", "content": line})
            history.append({"role": "assistant", "content": answer})
            del history[:-2 * _HISTORY_TURNS]
            cand = propose_memory(line)
            if cand and cand not in mem.facts():
                try:
                    ans = input(f"{C['dim']}remember: \"{cand}\"? [y saves] {C['reset']}").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    ans = ""
                if ans in ("y", "yes"):
                    mem.add(cand)
                    print(f"{C['dim']}(saved){C['reset']}")
```

Add slash handlers in the `/`-command block:

```python
            if cmd == "remember" and len(parts) > 1:
                mem.add(" ".join(parts[1:])); print(f"{C['dim']}(saved){C['reset']}"); continue
            if cmd == "memories":
                items = mem.list()
                print("\n".join(f"{m['id']}. {m['text']}" for m in items) or f"{C['dim']}(no memories yet){C['reset']}"); continue
            if cmd == "forget" and len(parts) > 1 and parts[1].isdigit():
                print(f"{C['dim']}({'forgotten' if mem.delete(int(parts[1])) else 'no such id'}){C['reset']}"); continue
```

Add `/remember`, `/memories`, `/forget` to the `/help` text.

- [ ] **Step 4: run, expect PASS** — `python3 -m unittest test_nitwit_memory test_nitwit_session test_nitwit_cli -v`; `python3 -c "import nitwit.memory, nitwit.session, nitwit.cli"`.

- [ ] **Step 5: commit** — `git add nitwit/session.py nitwit/cli.py test_nitwit_session.py test_nitwit_cli.py && git commit -m "feat(nitwit): recall memories into chat + propose/approve + /remember /memories /forget"`

---

## Self-Review
- Durable across sessions → SQLite `MemoryStore` (Task 1). ✓
- Auto-propose + approve → `propose_memory` + the interactive prompt (Task 2). ✓
- Auto-recall into context → `stream_answer(memories=...)` injection (Task 2). ✓
- Manual control → `/remember` `/memories` `/forget`. ✓
- Conservative proposals (no nagging on ordinary chat) → heuristic returns None for questions/tasks. ✓
- Types: `MemoryStore(db_path=None)`, `.add/list/facts/delete`, `propose_memory(text)->str|None`, `stream_answer(..., memories=None)`. ✓

## Not doing (later)
- Model-based memory extraction (heuristic is v1).
- Injecting memories into MISSION context (chat only for now; a small follow-on wires it into the mission goal/context).
- Scoped/tagged memories, editing.
