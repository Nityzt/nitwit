# Nitwit Engine Core (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the durable mission engine core — a structured Mission object, a SQLite store, a git-backed per-repo workspace, and an iteration loop that drives a task to completion against typed success criteria — proven end-to-end with a deterministic FakeCoder (no GPU/model needed).

**Architecture:** A new self-contained `nitwit/` package. The engine loop is pure orchestration: it depends on an injected `Coder` (proposes edits from context) and `Verifier` (judges `verifier` criteria), so the whole loop is testable offline with fakes. State lives in SQLite (`MissionStore`) and git branches (`Workspace`). No UI, no HTTP, no systemd in this phase — that's Phase 2+.

**Tech Stack:** Python 3.14 (uv), stdlib only — `sqlite3`, `subprocess` (git + test runner), `dataclasses`, `unittest`. No new dependencies.

## Global Constraints

- **Stdlib only** — no new pip dependencies. `sqlite3`, `subprocess`, `dataclasses`, `json`, `unittest`.
- **The Mission schema is complete from day 1** (no future migrations). Phase 1 *implements* `goal / constraints / success_criteria(tests+verifier) / repos(single) / notes / state / iteration`; `artifacts`, `question`, and multi-repo are present in the schema but not exercised.
- **Never push or merge.** The workspace does `branch`, `add`, `commit` only — never `push`, never `merge`, never touches `main`.
- **The coder never executes anything.** It returns proposed edits; the host (`Workspace`) applies and runs. (Matters for the real coder in Phase 2; the FakeCoder here honors the same interface.)
- **Durable state only in SQLite + git.** Nothing the loop needs to resume may live only in memory.
- **A mission is done only when ALL its `success_criteria` are satisfied.** `tests` = a repo's test command exits 0; `verifier` = the injected verifier returns pass.
- **Test convention:** root-level `test_nitwit_*.py`, `unittest`, runnable via `python3 -m unittest`. Follow the existing repo style (see `test_orchestrator.py`).
- **Branch naming:** `agent/<slug>` where slug is derived from the mission title/goal.

---

## File Structure

- `nitwit/__init__.py` — package marker (empty).
- `nitwit/missions.py` — `Mission` dataclass, `slugify`, `MissionStore` (SQLite CRUD + state machine).
- `nitwit/workspace.py` — `FileEdit`, `TestResult`, `Workspace` (git branch/edit/commit + sandboxed test run).
- `nitwit/coder.py` — `MissionContext`, `CoderResponse`, `Coder`/`Verifier` protocols, `FakeCoder`, `FakeVerifier`.
- `nitwit/engine.py` — `MissionEngine` (evaluate criteria, run one iteration, run the loop, reconcile).
- `test_nitwit_missions.py`, `test_nitwit_workspace.py`, `test_nitwit_coder.py`, `test_nitwit_engine.py` — unit tests.
- `test_nitwit_integration.py` — the capstone end-to-end test.

---

## Task 1: Mission object + slugify

**Files:**
- Create: `nitwit/__init__.py`
- Create: `nitwit/missions.py`
- Test: `test_nitwit_missions.py`

**Interfaces:**
- Produces: `Mission` dataclass with fields `id:str, goal:str, title:str, constraints:list[str], success_criteria:list[dict], repos:list[dict], artifacts:list[dict], notes:str, question:str, state:str, iteration:int, created:float, updated:float`; `Mission.to_row() -> dict[str,str|int|float]` (JSON-encodes list fields), `Mission.from_row(dict) -> Mission`; `slugify(text:str) -> str`.

- [ ] **Step 1: Create the package marker**

Create `nitwit/__init__.py` (empty file):

```python
```

- [ ] **Step 2: Write the failing test**

Create `test_nitwit_missions.py`:

```python
import unittest
from nitwit.missions import Mission, slugify


class TestMissionObject(unittest.TestCase):
    def test_defaults(self):
        m = Mission(id="m1", goal="add a health endpoint")
        self.assertEqual(m.state, "queued")
        self.assertEqual(m.iteration, 0)
        self.assertEqual(m.constraints, [])
        self.assertEqual(m.success_criteria, [])
        self.assertEqual(m.repos, [])
        self.assertEqual(m.notes, "")

    def test_row_round_trip_preserves_structured_fields(self):
        m = Mission(
            id="m1", goal="g", title="t",
            constraints=["no new deps"],
            success_criteria=[{"type": "tests", "repo": "/r", "cmd": "pytest"}],
            repos=[{"path": "/r", "branch": "agent/t", "test_cmd": "pytest", "checkpoint_commit": ""}],
            notes="started",
        )
        row = m.to_row()
        # list/dict fields must be JSON strings in the row (SQLite-friendly)
        self.assertIsInstance(row["success_criteria"], str)
        back = Mission.from_row(row)
        self.assertEqual(back.success_criteria, m.success_criteria)
        self.assertEqual(back.repos, m.repos)
        self.assertEqual(back.constraints, m.constraints)
        self.assertEqual(back, m)

    def test_slugify(self):
        self.assertEqual(slugify("Add a /health Endpoint!"), "add-a-health-endpoint")
        self.assertEqual(slugify("  Fix   the BUG  "), "fix-the-bug")
        self.assertTrue(slugify("").startswith("mission"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m unittest test_nitwit_missions -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nitwit.missions'`

- [ ] **Step 4: Write minimal implementation**

Create `nitwit/missions.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m unittest test_nitwit_missions -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add nitwit/__init__.py nitwit/missions.py test_nitwit_missions.py
git commit -m "feat(nitwit): Mission structured object + slugify"
```

---

## Task 2: MissionStore (SQLite CRUD + state machine)

**Files:**
- Modify: `nitwit/missions.py`
- Test: `test_nitwit_missions.py`

**Interfaces:**
- Consumes: `Mission`, `slugify` from Task 1.
- Produces: `MissionStore(db_path:str)`; methods `create(goal, *, title="", constraints=None, success_criteria=None, repos=None) -> Mission` (assigns id + timestamps, state `queued`), `get(id) -> Mission | None`, `list(state=None) -> list[Mission]`, `save(mission) -> None`, `set_state(id, new_state) -> Mission` (raises `InvalidTransition` on illegal moves), `bump_iteration(id) -> Mission`, `append_note(id, text) -> Mission`. Module constant `VALID_TRANSITIONS: dict[str, set[str]]`. Exception `InvalidTransition(Exception)`.

- [ ] **Step 1: Write the failing test**

Append to `test_nitwit_missions.py` (add imports at top: `import tempfile, os`; and `from nitwit.missions import MissionStore, InvalidTransition`):

```python
class TestMissionStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "missions.db"))

    def test_create_assigns_id_and_queued(self):
        m = self.store.create("add health endpoint")
        self.assertTrue(m.id)
        self.assertEqual(m.state, "queued")
        self.assertGreater(m.created, 0)

    def test_get_and_list(self):
        a = self.store.create("task a")
        self.store.create("task b")
        self.assertEqual(self.store.get(a.id).goal, "task a")
        self.assertEqual(len(self.store.list()), 2)
        self.assertEqual(len(self.store.list(state="queued")), 2)
        self.assertEqual(len(self.store.list(state="done")), 0)

    def test_persistence_across_instances(self):
        m = self.store.create("persist me", success_criteria=[{"type": "verifier", "description": "x"}])
        reopened = MissionStore(os.path.join(self.tmp, "missions.db"))
        got = reopened.get(m.id)
        self.assertEqual(got.success_criteria, [{"type": "verifier", "description": "x"}])

    def test_valid_transition(self):
        m = self.store.create("t")
        self.store.set_state(m.id, "running")
        self.assertEqual(self.store.get(m.id).state, "running")

    def test_invalid_transition_raises(self):
        m = self.store.create("t")  # queued
        with self.assertRaises(InvalidTransition):
            self.store.set_state(m.id, "done")  # queued -> done is illegal

    def test_bump_iteration_and_notes(self):
        m = self.store.create("t")
        self.store.bump_iteration(m.id)
        self.assertEqual(self.store.get(m.id).iteration, 1)
        self.store.append_note(m.id, "first")
        self.store.append_note(m.id, "second")
        self.assertIn("first", self.store.get(m.id).notes)
        self.assertIn("second", self.store.get(m.id).notes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_nitwit_missions -v`
Expected: FAIL — `ImportError: cannot import name 'MissionStore'`

- [ ] **Step 3: Write minimal implementation**

Append to `nitwit/missions.py`:

```python
import sqlite3

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_nitwit_missions -v`
Expected: PASS (9 tests total)

- [ ] **Step 5: Commit**

```bash
git add nitwit/missions.py test_nitwit_missions.py
git commit -m "feat(nitwit): MissionStore SQLite CRUD + state machine"
```

---

## Task 3: Workspace git operations

**Files:**
- Create: `nitwit/workspace.py`
- Test: `test_nitwit_workspace.py`

**Interfaces:**
- Produces: `FileEdit(path:str, content:str)` dataclass; `Workspace(repo_path:str)` with `is_clean() -> bool`, `ensure_branch(branch:str) -> None` (create-or-checkout; raises `DirtyRepo` if working tree dirty), `apply_edits(edits:list[FileEdit]) -> None` (writes full file content, creating parent dirs), `commit(message:str) -> str` (stages all, commits, returns short sha; returns "" if nothing to commit). Exception `DirtyRepo(Exception)`. Helper `git(repo_path, *args) -> str` (runs git, returns stdout, raises on nonzero).

- [ ] **Step 1: Write the failing test**

Create `test_nitwit_workspace.py`:

```python
import os
import subprocess
import tempfile
import unittest
from nitwit.workspace import Workspace, FileEdit, DirtyRepo, git


def make_repo() -> str:
    d = tempfile.mkdtemp()
    git(d, "init", "-q")
    git(d, "config", "user.email", "t@t")
    git(d, "config", "user.name", "t")
    with open(os.path.join(d, "README.md"), "w") as fh:
        fh.write("seed\n")
    git(d, "add", "-A")
    git(d, "commit", "-q", "-m", "seed")
    return d


class TestWorkspaceGit(unittest.TestCase):
    def setUp(self):
        self.repo = make_repo()
        self.ws = Workspace(self.repo)

    def test_is_clean(self):
        self.assertTrue(self.ws.is_clean())
        with open(os.path.join(self.repo, "x.txt"), "w") as fh:
            fh.write("dirty")
        self.assertFalse(self.ws.is_clean())

    def test_ensure_branch_creates_and_is_reentrant(self):
        self.ws.ensure_branch("agent/test")
        self.assertEqual(git(self.repo, "branch", "--show-current"), "agent/test")
        self.ws.ensure_branch("agent/test")  # second call must not fail
        self.assertEqual(git(self.repo, "branch", "--show-current"), "agent/test")

    def test_ensure_branch_refuses_dirty(self):
        with open(os.path.join(self.repo, "x.txt"), "w") as fh:
            fh.write("dirty")
        with self.assertRaises(DirtyRepo):
            self.ws.ensure_branch("agent/test")

    def test_apply_edits_and_commit(self):
        self.ws.ensure_branch("agent/test")
        self.ws.apply_edits([FileEdit("src/app.py", "print('hi')\n"),
                             FileEdit("README.md", "changed\n")])
        self.assertTrue(os.path.exists(os.path.join(self.repo, "src/app.py")))
        sha = self.ws.commit("add app")
        self.assertTrue(sha)
        # committing again with no change returns ""
        self.assertEqual(self.ws.commit("noop"), "")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_nitwit_workspace -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nitwit.workspace'`

- [ ] **Step 3: Write minimal implementation**

Create `nitwit/workspace.py`:

```python
"""Workspace: git branch + file edits + sandboxed test runs for one repo. Never push/merge."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


class DirtyRepo(Exception):
    pass


@dataclass
class FileEdit:
    path: str      # repo-relative
    content: str   # full new file content (write_file semantics)


def git(repo_path: str, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", repo_path, *args],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


class Workspace:
    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path

    def is_clean(self) -> bool:
        return git(self.repo_path, "status", "--porcelain") == ""

    def ensure_branch(self, branch: str) -> None:
        if not self.is_clean():
            raise DirtyRepo(f"{self.repo_path} has uncommitted changes; refusing to start")
        existing = git(self.repo_path, "branch", "--list", branch)
        if existing:
            git(self.repo_path, "checkout", "-q", branch)
        else:
            git(self.repo_path, "checkout", "-q", "-b", branch)

    def apply_edits(self, edits: list[FileEdit]) -> None:
        for edit in edits:
            full = os.path.join(self.repo_path, edit.path)
            os.makedirs(os.path.dirname(full) or self.repo_path, exist_ok=True)
            with open(full, "w") as fh:
                fh.write(edit.content)

    def commit(self, message: str) -> str:
        git(self.repo_path, "add", "-A")
        if git(self.repo_path, "status", "--porcelain") == "":
            return ""
        git(self.repo_path, "commit", "-q", "-m", message)
        return git(self.repo_path, "rev-parse", "--short", "HEAD")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_nitwit_workspace -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add nitwit/workspace.py test_nitwit_workspace.py
git commit -m "feat(nitwit): Workspace git branch/edit/commit"
```

---

## Task 4: Workspace.run_tests (sandboxed test runner)

**Files:**
- Modify: `nitwit/workspace.py`
- Test: `test_nitwit_workspace.py`

**Interfaces:**
- Consumes: `Workspace` from Task 3.
- Produces: `TestResult(passed:bool, output:str)` dataclass; `Workspace.run_tests(cmd:str, timeout:int=120) -> TestResult` — runs `cmd` in the repo via the shell, captures combined stdout+stderr, `passed = (exit code == 0)`; on timeout returns `TestResult(False, "TIMEOUT")`.

- [ ] **Step 1: Write the failing test**

Append to `test_nitwit_workspace.py` (add `from nitwit.workspace import TestResult`):

```python
class TestWorkspaceRunTests(unittest.TestCase):
    def setUp(self):
        self.repo = make_repo()
        self.ws = Workspace(self.repo)

    def test_passing_command(self):
        r = self.ws.run_tests("true")
        self.assertTrue(r.passed)

    def test_failing_command_captures_output(self):
        r = self.ws.run_tests("echo boom && false")
        self.assertFalse(r.passed)
        self.assertIn("boom", r.output)

    def test_python_test_file(self):
        with open(os.path.join(self.repo, "check.py"), "w") as fh:
            fh.write("assert 1 + 1 == 2\nprint('ok')\n")
        r = self.ws.run_tests("python3 check.py")
        self.assertTrue(r.passed)
        self.assertIn("ok", r.output)

    def test_timeout(self):
        r = self.ws.run_tests("sleep 5", timeout=1)
        self.assertFalse(r.passed)
        self.assertIn("TIMEOUT", r.output)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_nitwit_workspace -v`
Expected: FAIL — `ImportError: cannot import name 'TestResult'`

- [ ] **Step 3: Write minimal implementation**

Add to `nitwit/workspace.py` (import `import subprocess` already present; add the dataclass near `FileEdit` and the method on `Workspace`):

```python
@dataclass
class TestResult:
    passed: bool
    output: str
```

Add this method to the `Workspace` class:

```python
    def run_tests(self, cmd: str, timeout: int = 120) -> TestResult:
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=self.repo_path,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return TestResult(False, "TIMEOUT")
        output = (proc.stdout + proc.stderr).strip()
        return TestResult(proc.returncode == 0, output)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_nitwit_workspace -v`
Expected: PASS (8 tests total)

- [ ] **Step 5: Commit**

```bash
git add nitwit/workspace.py test_nitwit_workspace.py
git commit -m "feat(nitwit): Workspace.run_tests sandboxed runner"
```

---

## Task 5: Coder/Verifier interfaces + fakes

**Files:**
- Create: `nitwit/coder.py`
- Test: `test_nitwit_coder.py`

**Interfaces:**
- Consumes: `FileEdit` from `nitwit.workspace`.
- Produces:
  - `MissionContext(goal:str, constraints:list[str], notes:str, last_test_output:str, repo_files:dict[str,str])` dataclass.
  - `CoderResponse(edits:list[FileEdit], note:str="", question:str="")` dataclass.
  - `Coder` Protocol: `propose(self, ctx:MissionContext) -> CoderResponse`.
  - `Verifier` Protocol: `judge(self, description:str, ctx:MissionContext) -> bool`.
  - `FakeCoder(scripted:list[CoderResponse])` — returns responses in order, then empty responses; records `calls:int`.
  - `FakeVerifier(verdict:bool=True)` — returns `verdict`; records `calls:int`.

- [ ] **Step 1: Write the failing test**

Create `test_nitwit_coder.py`:

```python
import unittest
from nitwit.workspace import FileEdit
from nitwit.coder import MissionContext, CoderResponse, FakeCoder, FakeVerifier


class TestFakes(unittest.TestCase):
    def ctx(self):
        return MissionContext(goal="g", constraints=[], notes="", last_test_output="", repo_files={})

    def test_fake_coder_returns_scripted_then_empty(self):
        r1 = CoderResponse(edits=[FileEdit("a.py", "x=1\n")])
        coder = FakeCoder([r1])
        out = coder.propose(self.ctx())
        self.assertEqual(out.edits[0].path, "a.py")
        # after the script is exhausted, returns an empty response (no edits)
        self.assertEqual(coder.propose(self.ctx()).edits, [])
        self.assertEqual(coder.calls, 2)

    def test_fake_verifier(self):
        v = FakeVerifier(verdict=False)
        self.assertFalse(v.judge("is it good?", self.ctx()))
        self.assertEqual(v.calls, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_nitwit_coder -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nitwit.coder'`

- [ ] **Step 3: Write minimal implementation**

Create `nitwit/coder.py`:

```python
"""Coder/Verifier interfaces the engine depends on, plus deterministic fakes for testing.

The real model-backed Coder (wrapping the GPU coder + tool loop) is Phase 2; the engine only
ever sees these interfaces, so the whole loop is testable offline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from nitwit.workspace import FileEdit


@dataclass
class MissionContext:
    goal: str
    constraints: list[str]
    notes: str
    last_test_output: str
    repo_files: dict[str, str]  # path -> content snapshot fed to the coder


@dataclass
class CoderResponse:
    edits: list[FileEdit] = field(default_factory=list)
    note: str = ""
    question: str = ""  # non-empty => the coder needs the user (Phase 4 wires needs_input)


class Coder(Protocol):
    def propose(self, ctx: MissionContext) -> CoderResponse: ...


class Verifier(Protocol):
    def judge(self, description: str, ctx: MissionContext) -> bool: ...


class FakeCoder:
    """Returns scripted responses in order; empty responses once exhausted."""

    def __init__(self, scripted: list[CoderResponse]) -> None:
        self._scripted = list(scripted)
        self.calls = 0

    def propose(self, ctx: MissionContext) -> CoderResponse:
        self.calls += 1
        if self._scripted:
            return self._scripted.pop(0)
        return CoderResponse()


class FakeVerifier:
    def __init__(self, verdict: bool = True) -> None:
        self.verdict = verdict
        self.calls = 0

    def judge(self, description: str, ctx: MissionContext) -> bool:
        self.calls += 1
        return self.verdict
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_nitwit_coder -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add nitwit/coder.py test_nitwit_coder.py
git commit -m "feat(nitwit): Coder/Verifier interfaces + fakes"
```

---

## Task 6: MissionEngine — criteria evaluation + one iteration

**Files:**
- Create: `nitwit/engine.py`
- Test: `test_nitwit_engine.py`

**Interfaces:**
- Consumes: `MissionStore`, `Mission` (Tasks 1-2); `Workspace`, `FileEdit`, `TestResult` (Tasks 3-4); `MissionContext`, `CoderResponse`, `Coder`, `Verifier` (Task 5).
- Produces:
  - `MissionEngine(store:MissionStore, coder:Coder, verifier:Verifier, workspace_factory=Workspace, max_iterations:int=20, cooldown_s:float=0.0)`.
  - `evaluate_criteria(mission:Mission, workspaces:dict[str,Workspace]) -> tuple[bool, str]` — returns `(all_passed, summary)`; evaluates each `success_criteria` entry: `tests` → the named repo's `run_tests(cmd)`, `verifier` → `verifier.judge(description, ctx)`.
  - `build_context(mission:Mission, workspaces:dict[str,Workspace], last_test_output:str) -> MissionContext`.
  - `run_iteration(mission:Mission, workspaces:dict[str,Workspace]) -> tuple[Mission, bool]` — builds context, calls coder, applies edits to the first repo's workspace, commits, evaluates criteria, appends a note, bumps iteration; returns `(mission, done)`.

- [ ] **Step 1: Write the failing test**

Create `test_nitwit_engine.py`:

```python
import os
import tempfile
import unittest
from nitwit.missions import MissionStore
from nitwit.workspace import Workspace, FileEdit
from nitwit.coder import CoderResponse, FakeCoder, FakeVerifier
from nitwit.engine import MissionEngine
from test_nitwit_workspace import make_repo


class TestEngineIteration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "m.db"))
        self.repo = make_repo()

    def _mission(self, criteria):
        return self.store.create(
            "make target.txt say ok",
            repos=[{"path": self.repo, "branch": "agent/t", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=criteria,
        )

    def test_evaluate_tests_criterion(self):
        # target.txt must contain 'ok' for the test cmd to pass
        m = self._mission([{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        ws = Workspace(self.repo); ws.ensure_branch("agent/t")
        engine = MissionEngine(self.store, FakeCoder([]), FakeVerifier(True))
        passed, _ = engine.evaluate_criteria(m, {self.repo: ws})
        self.assertFalse(passed)  # target.txt doesn't exist yet
        ws.apply_edits([FileEdit("target.txt", "ok\n")]); ws.commit("add")
        passed, _ = engine.evaluate_criteria(m, {self.repo: ws})
        self.assertTrue(passed)

    def test_run_iteration_applies_edits_and_commits(self):
        m = self._mission([{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        coder = FakeCoder([CoderResponse(edits=[FileEdit("target.txt", "ok\n")], note="wrote target")])
        engine = MissionEngine(self.store, coder, FakeVerifier(True))
        ws = Workspace(self.repo); ws.ensure_branch("agent/t")
        m, done = engine.run_iteration(m, {self.repo: ws})
        self.assertTrue(done)                      # criterion now satisfied
        self.assertEqual(m.iteration, 1)
        self.assertIn("target", m.notes)
        self.assertTrue(os.path.exists(os.path.join(self.repo, "target.txt")))

    def test_verifier_criterion_uses_injected_verifier(self):
        m = self._mission([{"type": "verifier", "description": "is it meaningful?"}])
        engine = MissionEngine(self.store, FakeCoder([]), FakeVerifier(verdict=False))
        ws = Workspace(self.repo); ws.ensure_branch("agent/t")
        passed, _ = engine.evaluate_criteria(m, {self.repo: ws})
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_nitwit_engine -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nitwit.engine'`

- [ ] **Step 3: Write minimal implementation**

Create `nitwit/engine.py`:

```python
"""MissionEngine: the durable iteration loop. Pure orchestration over injected Coder/Verifier."""
from __future__ import annotations

import os
import time

from nitwit.coder import Coder, CoderResponse, MissionContext, Verifier
from nitwit.missions import Mission, MissionStore
from nitwit.workspace import Workspace

# Files bigger than this are skipped in the context snapshot (keep prompt bounded).
_MAX_SNAPSHOT_BYTES = 20000


class MissionEngine:
    def __init__(self, store: MissionStore, coder: Coder, verifier: Verifier,
                 workspace_factory=Workspace, max_iterations: int = 20, cooldown_s: float = 0.0) -> None:
        self.store = store
        self.coder = coder
        self.verifier = verifier
        self.workspace_factory = workspace_factory
        self.max_iterations = max_iterations
        self.cooldown_s = cooldown_s

    def _snapshot(self, repo_path: str) -> dict[str, str]:
        files: dict[str, str] = {}
        for root, dirs, names in os.walk(repo_path):
            if ".git" in dirs:
                dirs.remove(".git")
            for name in names:
                full = os.path.join(root, name)
                try:
                    if os.path.getsize(full) > _MAX_SNAPSHOT_BYTES:
                        continue
                    with open(full, "r", errors="replace") as fh:
                        files[os.path.relpath(full, repo_path)] = fh.read()
                except OSError:
                    continue
        return files

    def build_context(self, mission: Mission, workspaces: dict[str, Workspace], last_test_output: str) -> MissionContext:
        primary = mission.repos[0]["path"] if mission.repos else ""
        repo_files = self._snapshot(primary) if primary else {}
        return MissionContext(
            goal=mission.goal, constraints=mission.constraints, notes=mission.notes,
            last_test_output=last_test_output, repo_files=repo_files,
        )

    def evaluate_criteria(self, mission: Mission, workspaces: dict[str, Workspace]) -> tuple[bool, str]:
        summaries = []
        all_passed = True
        ctx = self.build_context(mission, workspaces, "")
        for crit in mission.success_criteria:
            kind = crit.get("type")
            if kind == "tests":
                ws = workspaces[crit["repo"]]
                result = ws.run_tests(crit["cmd"])
                ok = result.passed
                summaries.append(f"tests({crit['cmd']}): {'pass' if ok else 'fail'}")
            elif kind == "verifier":
                ok = self.verifier.judge(crit.get("description", ""), ctx)
                summaries.append(f"verifier: {'pass' if ok else 'fail'}")
            else:
                ok = False
                summaries.append(f"{kind}: unsupported (phase 1)")
            all_passed = all_passed and ok
        return all_passed, "; ".join(summaries)

    def run_iteration(self, mission: Mission, workspaces: dict[str, Workspace]) -> tuple[Mission, bool]:
        ctx = self.build_context(mission, workspaces, "")
        response: CoderResponse = self.coder.propose(ctx)
        primary_path = mission.repos[0]["path"]
        ws = workspaces[primary_path]
        if response.edits:
            ws.apply_edits(response.edits)
            ws.commit(f"iteration {mission.iteration + 1}: {response.note or 'edits'}")
        mission = self.store.bump_iteration(mission.id)
        if response.note:
            mission = self.store.append_note(mission.id, response.note)
        done, summary = self.evaluate_criteria(mission, workspaces)
        mission = self.store.append_note(mission.id, f"criteria -> {summary}")
        return mission, done
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_nitwit_engine -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add nitwit/engine.py test_nitwit_engine.py
git commit -m "feat(nitwit): MissionEngine criteria eval + single iteration"
```

---

## Task 7: MissionEngine — run loop, stop condition, cap, reconcile

**Files:**
- Modify: `nitwit/engine.py`
- Test: `test_nitwit_engine.py`

**Interfaces:**
- Consumes: everything from Task 6.
- Produces:
  - `run_mission(mission_id:str) -> Mission` — sets `running`, prepares workspaces (`ensure_branch` per repo), loops `run_iteration` until: all criteria pass → `done`; or `iteration >= max_iterations` → `needs_input`; or the paused flag is set → `paused` (stops cleanly, state persisted). Respects `cooldown_s` between iterations.
  - `pause() -> None` / `resume() -> None` — set/clear an internal `threading.Event` the loop checks at each iteration boundary.
  - `reconcile() -> int` — on startup, any mission in `running` is rewound to `queued` (its last commit is the checkpoint); returns count rewound.

- [ ] **Step 1: Write the failing test**

Append to `test_nitwit_engine.py`:

```python
class TestEngineLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "m.db"))
        self.repo = make_repo()

    def _mission(self, criteria):
        return self.store.create(
            "loop to green",
            repos=[{"path": self.repo, "branch": "agent/loop", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=criteria,
        )

    def test_loop_reaches_done_in_two_iterations(self):
        # first response writes the wrong content, second writes the right one
        coder = FakeCoder([
            CoderResponse(edits=[FileEdit("target.txt", "nope\n")], note="attempt 1"),
            CoderResponse(edits=[FileEdit("target.txt", "ok\n")], note="attempt 2"),
        ])
        m = self._mission([{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        engine = MissionEngine(self.store, coder, FakeVerifier(True))
        result = engine.run_mission(m.id)
        self.assertEqual(result.state, "done")
        self.assertEqual(result.iteration, 2)

    def test_loop_hits_cap_and_needs_input(self):
        coder = FakeCoder([])  # never produces a fix
        m = self._mission([{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        engine = MissionEngine(self.store, coder, FakeVerifier(True), max_iterations=3)
        result = engine.run_mission(m.id)
        self.assertEqual(result.state, "needs_input")
        self.assertEqual(result.iteration, 3)

    def test_pause_stops_the_loop(self):
        coder = FakeCoder([CoderResponse(edits=[FileEdit("a.txt", "1\n")], note="x")] * 10)
        m = self._mission([{"type": "tests", "repo": self.repo, "cmd": "grep -q ok target.txt"}])
        engine = MissionEngine(self.store, coder, FakeVerifier(True), max_iterations=10)
        engine.pause()  # paused before it starts
        result = engine.run_mission(m.id)
        self.assertEqual(result.state, "paused")
        self.assertEqual(result.iteration, 0)

    def test_reconcile_rewinds_running(self):
        m = self._mission([{"type": "verifier", "description": "x"}])
        self.store.set_state(m.id, "running")
        engine = MissionEngine(self.store, FakeCoder([]), FakeVerifier(True))
        n = engine.reconcile()
        self.assertEqual(n, 1)
        self.assertEqual(self.store.get(m.id).state, "queued")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_nitwit_engine -v`
Expected: FAIL — `AttributeError: 'MissionEngine' object has no attribute 'run_mission'`

- [ ] **Step 3: Write minimal implementation**

Add `import threading` at the top of `nitwit/engine.py`. Add to `__init__` (end of the method):

```python
        self._paused = threading.Event()
```

Add these methods to `MissionEngine`:

```python
    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def reconcile(self) -> int:
        rewound = 0
        for m in self.store.list(state="running"):
            self.store.set_state(m.id, "queued")
            rewound += 1
        return rewound

    def _prepare_workspaces(self, mission: Mission) -> dict[str, Workspace]:
        workspaces = {}
        for repo in mission.repos:
            ws = self.workspace_factory(repo["path"])
            ws.ensure_branch(repo["branch"])
            workspaces[repo["path"]] = ws
        return workspaces

    def run_mission(self, mission_id: str) -> Mission:
        mission = self.store.get(mission_id)
        if self._paused.is_set():
            return self.store.set_state(mission.id, "paused") if mission.state != "paused" else mission
        mission = self.store.set_state(mission.id, "running")
        workspaces = self._prepare_workspaces(mission)
        while True:
            if self._paused.is_set():
                return self.store.set_state(mission.id, "paused")
            mission, done = self.run_iteration(mission, workspaces)
            if done:
                return self.store.set_state(mission.id, "done")
            if mission.iteration >= self.max_iterations:
                self.store.append_note(mission.id, "hit max_iterations; awaiting input")
                return self.store.set_state(mission.id, "needs_input")
            if self.cooldown_s:
                time.sleep(self.cooldown_s)
```

Note: the `queued → paused` case in `test_pause_stops_the_loop` needs `paused` reachable from `queued`. Update `VALID_TRANSITIONS["queued"]` in `nitwit/missions.py` to include `"paused"`:

```python
    "queued": {"running", "cancelled", "paused"},
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_nitwit_engine -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```bash
git add nitwit/engine.py nitwit/missions.py test_nitwit_engine.py
git commit -m "feat(nitwit): mission run loop, cap, pause, reconcile"
```

---

## Task 8: End-to-end integration test (the capstone)

**Files:**
- Create: `test_nitwit_integration.py`

**Interfaces:**
- Consumes: the whole engine (Tasks 1-7). No new production code — this task proves the phase.

- [ ] **Step 1: Write the end-to-end test**

Create `test_nitwit_integration.py`:

```python
"""End-to-end: a mission with a real failing test in a real git repo, driven to green by a
deterministic coder, must branch, iterate, commit each round, and stop `done`. Also proves
resume: reconcile after an interrupted run, then finish."""
import os
import tempfile
import unittest
from nitwit.missions import MissionStore
from nitwit.workspace import FileEdit, git
from nitwit.coder import CoderResponse, FakeCoder, FakeVerifier
from nitwit.engine import MissionEngine
from test_nitwit_workspace import make_repo


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "m.db"))
        self.repo = make_repo()
        # a real failing test: pytest-free, just a python assert on a module the coder must write
        with open(os.path.join(self.repo, "test_feature.py"), "w") as fh:
            fh.write("from feature import add\nassert add(2, 3) == 5\nprint('PASS')\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "add failing test")

    def test_mission_reaches_green_and_commits(self):
        m = self.store.create(
            "implement feature.add so the test passes",
            repos=[{"path": self.repo, "branch": "agent/feature", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=[{"type": "tests", "repo": self.repo, "cmd": "python3 test_feature.py"},
                              {"type": "verifier", "description": "is the implementation meaningful?"}],
        )
        # iteration 1: a wrong impl (returns 0); iteration 2: the correct impl
        coder = FakeCoder([
            CoderResponse(edits=[FileEdit("feature.py", "def add(a, b):\n    return 0\n")], note="stub"),
            CoderResponse(edits=[FileEdit("feature.py", "def add(a, b):\n    return a + b\n")], note="fix"),
        ])
        engine = MissionEngine(self.store, coder, FakeVerifier(True), max_iterations=5)
        result = engine.run_mission(m.id)

        self.assertEqual(result.state, "done")
        # on the agent branch, with a commit per iteration (2) on top of the seed+test commits
        self.assertEqual(git(self.repo, "branch", "--show-current"), "agent/feature")
        subjects = git(self.repo, "log", "--format=%s").splitlines()
        self.assertTrue(any("iteration 2" in s for s in subjects))
        self.assertTrue(any("iteration 1" in s for s in subjects))
        # the deliverable actually exists and is correct
        with open(os.path.join(self.repo, "feature.py")) as fh:
            self.assertIn("a + b", fh.read())

    def test_resume_after_interruption(self):
        m = self.store.create(
            "resume me",
            repos=[{"path": self.repo, "branch": "agent/resume", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=[{"type": "tests", "repo": self.repo, "cmd": "python3 test_feature.py"}],
        )
        # simulate an engine that died mid-run: mission stuck in 'running'
        self.store.set_state(m.id, "running")

        # a fresh engine reconciles (rewinds running -> queued), then runs to done
        coder = FakeCoder([CoderResponse(edits=[FileEdit("feature.py", "def add(a, b):\n    return a + b\n")], note="fix")])
        engine = MissionEngine(self.store, coder, FakeVerifier(True))
        self.assertEqual(engine.reconcile(), 1)
        self.assertEqual(self.store.get(m.id).state, "queued")
        result = engine.run_mission(m.id)
        self.assertEqual(result.state, "done")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the full nitwit suite**

Run: `python3 -m unittest test_nitwit_missions test_nitwit_workspace test_nitwit_coder test_nitwit_engine test_nitwit_integration -v`
Expected: PASS (all tests, ~23)

- [ ] **Step 3: Confirm the legacy suite still passes**

Run: `python3 -m unittest test_orchestrator -q`
Expected: PASS (no regressions — nitwit is additive)

- [ ] **Step 4: Commit**

```bash
git add test_nitwit_integration.py
git commit -m "test(nitwit): end-to-end mission loop + resume integration"
```

---

## Self-Review

**Spec coverage (Phase 1 scope):**
- Structured Mission object (goal/constraints/success_criteria/repos/artifacts/notes/question/state/iteration) → Task 1. ✓ (`artifacts`/`question` reserved, present in schema.)
- Durable SQLite store + state machine → Task 2. ✓
- Per-repo git workspace, never push/merge → Task 3. ✓
- Sandboxed test runner (the objective oracle) → Task 4. ✓
- Coder/Verifier as injected interfaces (offline-testable loop) → Task 5. ✓
- Typed success-criteria evaluation (`tests` + `verifier`) → Task 6. ✓
- Unbounded-iteration loop, `max_iterations` cap → `needs_input`, pause at boundary, reconcile-on-start → Task 7. ✓
- End-to-end + resume proof → Task 8. ✓
- Deferred to later phases (correctly out of Phase 1 scope): real model-backed Coder (Phase 2), HTTP/SSE API + `wit` CLI (Phase 2), systemd daemon (Phase 3), escalation/lane routing (Phase 4), UI (Phase 5), toggle-stops-coder-container (Phase 3, needs the daemon + `switch_coder.sh`).

**Placeholder scan:** No TBD/TODO; every step has full runnable code and exact commands. ✓

**Type consistency:** `Mission` fields consistent across Tasks 1-7; `FileEdit(path, content)`, `TestResult(passed, output)`, `CoderResponse(edits, note, question)`, `MissionContext(goal, constraints, notes, last_test_output, repo_files)` used identically in producer and consumer tasks; `run_mission(mission_id)` returns a `Mission`; `evaluate_criteria` returns `(bool, str)`. The `queued → paused` transition needed by Task 7's pause test is added to `VALID_TRANSITIONS` in Task 7 Step 3. ✓

## Notes for later phases (not this plan)

- **Phase 2** first task = the real `Coder` wrapping the GPU coder (:8080) with the text-form tool loop (`read_file`/`write_file`/`run_tests`), plus the real `Verifier` calling the CPU Qwen3-4B (:8086); then the HTTP+SSE daemon API and the `wit` REPL client.
- **Phase 3** = `nitwit.service` + toggle that stops the coder container (reuse `switch_coder.sh`).
- The engine already exposes `pause()`/`resume()`/`reconcile()` so the daemon/toggle can drive it without changes.
