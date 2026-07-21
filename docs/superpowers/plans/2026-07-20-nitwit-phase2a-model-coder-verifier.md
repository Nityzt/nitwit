# Nitwit Phase 2a: Real Model-Backed Coder + Verifier

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Phase-1 FakeCoder/FakeVerifier with real implementations that drive the mission loop against the local models — `ModelCoder` on the GPU coder (:8080) and `ModelVerifier` on the CPU Qwen3-4B (:8086) — so a mission can actually write code and self-verify against typed success criteria.

**Architecture:** The Phase-1 engine already depends only on the `Coder`/`Verifier` protocols (`propose(ctx)->CoderResponse`, `judge(description, ctx)->bool`), so this phase adds two protocol implementations plus a factory that wires a `MissionEngine` to the live endpoints — zero engine changes. Each `propose()` is ONE bounded model call per iteration (GPU-safe: the crash envelope forbids sustained GPU tool-spirals; the loop's per-iteration boundary + CPU test/verify work between calls is the safe shape). The coder emits full-file edits as fenced `file:` blocks (deterministic to parse; Qwen reliably emits fenced code — proven in `bench/`). The engine still owns applying edits and running tests — the coder only proposes.

**Tech Stack:** Python 3.14 (uv), stdlib + the existing `orchestrator.OpenAICompatibleClient`/`ModelResponse`/`extract_json`. No new pip deps.

## Global Constraints

- **Stdlib + existing project modules only** — reuse `orchestrator.OpenAICompatibleClient` (`.chat(messages, *, temperature, max_tokens, response_format=None) -> ModelResponse`), `orchestrator.ModelResponse` (fields `content, elapsed_s, usage, timings, raw`), and `orchestrator.extract_json`. No new dependencies.
- **One bounded model call per `propose()`** — no in-`propose` agentic tool loop in this phase (GPU-safety: single bounded prefill+generation per iteration; the mission loop is the outer loop). A read-tool loop is a documented later enhancement, NOT in scope here.
- **The coder only proposes edits; the engine applies + runs tests.** `propose()` must never write files, run git, or run tests.
- **Endpoints (defaults):** coder `http://127.0.0.1:8080` model `qwen2.5-coder-7b`; verifier `http://127.0.0.1:8086` model `qwen3-4b`. Injectable for tests.
- **Edit format:** the coder returns each created/changed file as a fenced block:
  ```
  ```file:relative/path.ext
  <COMPLETE new file content>
  ```
  ```
  Parsed into `FileEdit(path, content)`. Nothing else is treated as an edit.
- **Verifier is lenient on parse failure** (returns `True`) — matching the orchestrator's philosophy that a flaky judge shouldn't sink good work; the `tests` criterion remains the hard AND-gate, so a false verifier-pass can't alone mark a mission done when tests fail.
- **Tests:** root-level `test_nitwit_*.py`, `unittest`. Unit tests use a `FakeClient` (canned `ModelResponse`) — no network. The live-server integration test is **gated** (skipped unless both endpoints are healthy) so the suite stays green offline.

---

## File Structure

- `nitwit/model_coder.py` — `ModelCoder` (Coder impl): prompt build + `file:` block parsing.
- `nitwit/model_verifier.py` — `ModelVerifier` (Verifier impl): verdict prompt + JSON parse.
- `nitwit/factory.py` — `build_model_engine(...)` wiring a `MissionEngine` to live clients.
- `test_nitwit_model_coder.py`, `test_nitwit_model_verifier.py` — unit tests with a `FakeClient`.
- `test_nitwit_model_integration.py` — gated live-server end-to-end (skipped if servers down).

---

## Task 1: ModelCoder — prompt build + edit parsing

**Files:**
- Create: `nitwit/model_coder.py`
- Test: `test_nitwit_model_coder.py`

**Interfaces:**
- Consumes: `orchestrator.ModelResponse`; `nitwit.coder.MissionContext`, `nitwit.coder.CoderResponse`; `nitwit.workspace.FileEdit`.
- Produces: `parse_file_edits(text:str) -> list[FileEdit]`; `build_coder_messages(ctx:MissionContext, max_snippet:int=12000) -> list[dict]`; `ModelCoder(client, max_tokens:int=1600)` with `propose(ctx:MissionContext) -> CoderResponse`. `client` is any object with `.chat(messages, *, temperature, max_tokens, response_format=None) -> ModelResponse`.

- [ ] **Step 1: Write the failing test**

Create `test_nitwit_model_coder.py`:

```python
import unittest
from orchestrator import ModelResponse
from nitwit.coder import MissionContext
from nitwit.model_coder import ModelCoder, parse_file_edits, build_coder_messages


def _resp(content):
    return ModelResponse(content=content, elapsed_s=0.1, usage={}, timings={}, raw={})


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.last_messages = None
        self.last_kwargs = None

    def chat(self, messages, *, temperature, max_tokens, response_format=None):
        self.last_messages = messages
        self.last_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        return _resp(self.content)


def _ctx(**kw):
    base = dict(goal="make add() return a+b", constraints=["python only"], notes="",
               last_test_output="", repo_files={"feature.py": "def add(a,b):\n    return 0\n"})
    base.update(kw)
    return MissionContext(**base)


class TestParseFileEdits(unittest.TestCase):
    def test_parses_single_block(self):
        text = "Here:\n```file:feature.py\ndef add(a, b):\n    return a + b\n```\n"
        edits = parse_file_edits(text)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0].path, "feature.py")
        self.assertEqual(edits[0].content, "def add(a, b):\n    return a + b\n")

    def test_parses_multiple_blocks(self):
        text = ("```file:a.py\nx = 1\n```\n"
                "```file:sub/b.py\ny = 2\n```\n")
        edits = parse_file_edits(text)
        self.assertEqual([e.path for e in edits], ["a.py", "sub/b.py"])

    def test_no_blocks_returns_empty(self):
        self.assertEqual(parse_file_edits("no edits here, just prose"), [])

    def test_strips_think_block(self):
        text = "<think>let me plan</think>\n```file:a.py\nz = 3\n```"
        edits = parse_file_edits(text)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0].content, "z = 3\n")


class TestBuildMessages(unittest.TestCase):
    def test_prompt_includes_goal_constraints_tests_and_files(self):
        msgs = build_coder_messages(_ctx(last_test_output="AssertionError: 0 != 5"))
        joined = "\n".join(m["content"] for m in msgs)
        self.assertIn("make add() return a+b", joined)
        self.assertIn("python only", joined)
        self.assertIn("AssertionError", joined)
        self.assertIn("feature.py", joined)
        self.assertEqual(msgs[0]["role"], "system")


class TestModelCoder(unittest.TestCase):
    def test_propose_returns_parsed_edits(self):
        client = FakeClient("```file:feature.py\ndef add(a, b):\n    return a + b\n```")
        coder = ModelCoder(client)
        out = coder.propose(_ctx())
        self.assertEqual(len(out.edits), 1)
        self.assertEqual(out.edits[0].path, "feature.py")
        self.assertEqual(client.last_kwargs["temperature"], 0.0)

    def test_propose_no_edits_when_model_emits_prose(self):
        coder = ModelCoder(FakeClient("I think this looks fine already."))
        self.assertEqual(coder.propose(_ctx()).edits, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_nitwit_model_coder -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nitwit.model_coder'`

- [ ] **Step 3: Write minimal implementation**

Create `nitwit/model_coder.py`:

```python
"""ModelCoder: the real Coder — one bounded GPU-model call per iteration that proposes
full-file edits as fenced `file:` blocks. The engine applies edits and runs tests; the
coder only proposes."""
from __future__ import annotations

import re

from nitwit.coder import CoderResponse, MissionContext
from nitwit.workspace import FileEdit

_FILE_BLOCK = re.compile(r"```file:(?P<path>[^\n`]+)\n(?P<body>.*?)```", re.DOTALL)

CODER_SYSTEM = (
    "You are an autonomous software engineer working toward a goal in a code repository. "
    "You are given the goal, hard constraints, the current repository files, and the latest "
    "test output. Produce the file edits that make progress toward the goal and make the tests "
    "pass. For EVERY file you create or change, output a fenced block exactly in this form:\n"
    "```file:relative/path.ext\n<the COMPLETE new content of that file>\n```\n"
    "Output ONLY these file blocks — no explanation, no diff syntax, no partial files. "
    "Give the whole file content each time. If no edit is needed, output nothing."
)


def parse_file_edits(text: str) -> list[FileEdit]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    edits: list[FileEdit] = []
    for m in _FILE_BLOCK.finditer(text):
        path = m.group("path").strip()
        body = m.group("body")
        if path:
            edits.append(FileEdit(path=path, content=body))
    return edits


def _repo_snapshot_text(repo_files: dict[str, str], max_snippet: int) -> str:
    if not repo_files:
        return "(empty repository)"
    parts = []
    for path, content in repo_files.items():
        clipped = content[:max_snippet]
        parts.append(f"--- {path} ---\n{clipped}")
    return "\n\n".join(parts)


def build_coder_messages(ctx: MissionContext, max_snippet: int = 12000) -> list[dict]:
    constraints = "\n".join(f"- {c}" for c in ctx.constraints) or "(none)"
    user = (
        f"GOAL:\n{ctx.goal}\n\n"
        f"HARD CONSTRAINTS:\n{constraints}\n\n"
        f"LATEST TEST OUTPUT:\n{ctx.last_test_output or '(no tests run yet)'}\n\n"
        f"NOTES SO FAR:\n{ctx.notes or '(none)'}\n\n"
        f"CURRENT REPOSITORY FILES:\n{_repo_snapshot_text(ctx.repo_files, max_snippet)}"
    )
    return [
        {"role": "system", "content": CODER_SYSTEM},
        {"role": "user", "content": user},
    ]


class ModelCoder:
    def __init__(self, client, max_tokens: int = 1600) -> None:
        self.client = client
        self.max_tokens = max_tokens

    def propose(self, ctx: MissionContext) -> CoderResponse:
        messages = build_coder_messages(ctx)
        response = self.client.chat(messages, temperature=0.0, max_tokens=self.max_tokens)
        edits = parse_file_edits(response.content)
        note = "proposed edits: " + ", ".join(e.path for e in edits) if edits else "no edits proposed"
        return CoderResponse(edits=edits, note=note)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_nitwit_model_coder -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add nitwit/model_coder.py test_nitwit_model_coder.py
git commit -m "feat(nitwit): ModelCoder — real coder proposing fenced file edits"
```

---

## Task 2: ModelVerifier — verdict prompt + JSON parse

**Files:**
- Create: `nitwit/model_verifier.py`
- Test: `test_nitwit_model_verifier.py`

**Interfaces:**
- Consumes: `orchestrator.extract_json`, `orchestrator.ModelResponse`; `nitwit.coder.MissionContext`.
- Produces: `VERDICT_FORMAT` (a json_schema response_format dict forcing `{pass:boolean, reason:string}`); `build_verifier_messages(description:str, ctx:MissionContext) -> list[dict]`; `ModelVerifier(client, max_tokens:int=700)` with `judge(description:str, ctx:MissionContext) -> bool`. On unparseable output, `judge` returns `True` (lenient).

- [ ] **Step 1: Write the failing test**

Create `test_nitwit_model_verifier.py`:

```python
import unittest
from orchestrator import ModelResponse
from nitwit.coder import MissionContext
from nitwit.model_verifier import ModelVerifier, build_verifier_messages


def _resp(content):
    return ModelResponse(content=content, elapsed_s=0.1, usage={}, timings={}, raw={})


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.last_messages = None

    def chat(self, messages, *, temperature, max_tokens, response_format=None):
        self.last_messages = messages
        return _resp(self.content)


def _ctx():
    return MissionContext(goal="g", constraints=[], notes="", last_test_output="all passed",
                          repo_files={"feature.py": "def add(a,b): return a+b"})


class TestModelVerifier(unittest.TestCase):
    def test_pass_true(self):
        v = ModelVerifier(FakeClient('{"pass": true, "reason": "meets the goal"}'))
        self.assertTrue(v.judge("implementation is meaningful", _ctx()))

    def test_pass_false(self):
        v = ModelVerifier(FakeClient('{"pass": false, "reason": "stub only"}'))
        self.assertFalse(v.judge("implementation is meaningful", _ctx()))

    def test_stringy_verdict_normalized(self):
        v = ModelVerifier(FakeClient('{"pass": "yes", "reason": "ok"}'))
        self.assertTrue(v.judge("x", _ctx()))

    def test_unparseable_is_lenient_true(self):
        v = ModelVerifier(FakeClient("I cannot produce JSON here, sorry."))
        self.assertTrue(v.judge("x", _ctx()))

    def test_prompt_includes_description_and_evidence(self):
        client = FakeClient('{"pass": true, "reason": "y"}')
        ModelVerifier(client).judge("the endpoint returns 201", _ctx())
        joined = "\n".join(m["content"] for m in client.last_messages)
        self.assertIn("the endpoint returns 201", joined)
        self.assertIn("all passed", joined)  # last_test_output as evidence


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_nitwit_model_verifier -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nitwit.model_verifier'`

- [ ] **Step 3: Write minimal implementation**

Create `nitwit/model_verifier.py`:

```python
"""ModelVerifier: the real Verifier — asks the CPU Qwen3-4B whether a described success
condition is met by the current work. Lenient on parse failure (the tests criterion is the
hard gate, so a flaky judge shouldn't sink good work)."""
from __future__ import annotations

from orchestrator import extract_json

VERDICT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "schema": {
            "type": "object",
            "properties": {
                "pass": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["pass"],
            "additionalProperties": False,
        },
    },
}

VERIFIER_SYSTEM = (
    "You are a strict but fair verifier. Decide whether the described SUCCESS CONDITION is "
    "genuinely satisfied by the work shown (the repository files and the latest test output). "
    'Answer ONLY as JSON: {"pass": true|false, "reason": "<one sentence>"}. '
    "Pass only if the condition is really met; do not pass a stub, a placeholder, or work that "
    "merely looks plausible."
)


def build_verifier_messages(description: str, ctx) -> list[dict]:
    files = "\n\n".join(f"--- {p} ---\n{c[:8000]}" for p, c in (ctx.repo_files or {}).items()) or "(none)"
    user = (
        f"SUCCESS CONDITION:\n{description}\n\n"
        f"GOAL (for context):\n{ctx.goal}\n\n"
        f"LATEST TEST OUTPUT:\n{ctx.last_test_output or '(none)'}\n\n"
        f"REPOSITORY FILES:\n{files}"
    )
    return [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": user},
    ]


class ModelVerifier:
    def __init__(self, client, max_tokens: int = 700) -> None:
        self.client = client
        self.max_tokens = max_tokens

    def judge(self, description: str, ctx) -> bool:
        messages = build_verifier_messages(description, ctx)
        response = self.client.chat(messages, temperature=0.0, max_tokens=self.max_tokens,
                                    response_format=VERDICT_FORMAT)
        try:
            parsed = extract_json(response.content)
        except ValueError:
            return True  # lenient: don't sink good work on a flaky judge
        if not isinstance(parsed, dict) or "pass" not in parsed:
            return True
        raw = parsed["pass"]
        if isinstance(raw, str):
            return raw.strip().lower() in ("true", "yes", "pass", "ok", "1")
        return bool(raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_nitwit_model_verifier -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add nitwit/model_verifier.py test_nitwit_model_verifier.py
git commit -m "feat(nitwit): ModelVerifier — CPU model judges success criteria"
```

---

## Task 3: Factory + gated live-server integration

**Files:**
- Create: `nitwit/factory.py`
- Test: `test_nitwit_model_integration.py`

**Interfaces:**
- Consumes: `orchestrator.OpenAICompatibleClient`; `nitwit.missions.MissionStore`; `nitwit.engine.MissionEngine`; `nitwit.model_coder.ModelCoder`; `nitwit.model_verifier.ModelVerifier`.
- Produces: `endpoint_healthy(base_url:str, timeout:float=2.0) -> bool`; `build_model_engine(store:MissionStore, *, coder_url="http://127.0.0.1:8080", coder_model="qwen2.5-coder-7b", verifier_url="http://127.0.0.1:8086", verifier_model="qwen3-4b", max_iterations=12, cooldown_s=0.0) -> MissionEngine`.

- [ ] **Step 1: Write the failing test**

Create `test_nitwit_model_integration.py`:

```python
"""Gated end-to-end: runs a REAL small mission against the live coder (:8080) + verifier
(:8086). Skipped automatically when either endpoint is down, so the suite stays green offline.
Bounded (max_iterations small) — GPU-safe per the crash envelope."""
import os
import tempfile
import unittest
from nitwit.missions import MissionStore
from nitwit.workspace import git
from nitwit.factory import build_model_engine, endpoint_healthy
from test_nitwit_workspace import make_repo

CODER_URL = os.environ.get("NITWIT_CODER_URL", "http://127.0.0.1:8080")
VERIFIER_URL = os.environ.get("NITWIT_VERIFIER_URL", "http://127.0.0.1:8086")
LIVE = endpoint_healthy(CODER_URL) and endpoint_healthy(VERIFIER_URL)


@unittest.skipUnless(LIVE, "live coder/verifier endpoints not available")
class TestModelMissionLive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "m.db"))
        self.repo = make_repo()
        with open(os.path.join(self.repo, "test_feature.py"), "w") as fh:
            fh.write("from feature import add\nassert add(2, 3) == 5\nprint('PASS')\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "failing test")

    def test_real_mission_reaches_green(self):
        m = self.store.create(
            "implement feature.add(a, b) so test_feature.py passes",
            repos=[{"path": self.repo, "branch": "agent/feat", "test_cmd": "", "checkpoint_commit": ""}],
            success_criteria=[{"type": "tests", "repo": self.repo, "cmd": "python3 test_feature.py"}],
        )
        engine = build_model_engine(self.store, max_iterations=6)
        result = engine.run_mission(m.id)
        self.assertEqual(result.state, "done", f"mission ended {result.state}; notes:\n{result.notes}")
        with open(os.path.join(self.repo, "feature.py")) as fh:
            self.assertIn("a + b", fh.read().replace(" ", "") or "")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_nitwit_model_integration -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nitwit.factory'`

- [ ] **Step 3: Write minimal implementation**

Create `nitwit/factory.py`:

```python
"""Factory: wire a MissionEngine to the live local models (GPU coder + CPU verifier)."""
from __future__ import annotations

import urllib.request

from orchestrator import OpenAICompatibleClient
from nitwit.engine import MissionEngine
from nitwit.missions import MissionStore
from nitwit.model_coder import ModelCoder
from nitwit.model_verifier import ModelVerifier


def endpoint_healthy(base_url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=timeout) as res:
            return res.status == 200
    except Exception:
        return False


def build_model_engine(store: MissionStore, *,
                       coder_url: str = "http://127.0.0.1:8080",
                       coder_model: str = "qwen2.5-coder-7b",
                       verifier_url: str = "http://127.0.0.1:8086",
                       verifier_model: str = "qwen3-4b",
                       max_iterations: int = 12,
                       cooldown_s: float = 0.0) -> MissionEngine:
    coder = ModelCoder(OpenAICompatibleClient(coder_url, coder_model))
    # The verifier is a thinking model; give it headroom so <think> + JSON both fit.
    verifier = ModelVerifier(OpenAICompatibleClient(verifier_url, verifier_model), max_tokens=1000)
    return MissionEngine(store, coder, verifier, max_iterations=max_iterations, cooldown_s=cooldown_s)
```

- [ ] **Step 4: Run the gated integration test**

First run offline (endpoints may be down) to confirm it SKIPS cleanly:
Run: `python3 -m unittest test_nitwit_model_integration -v`
Expected: `OK (skipped=1)` when endpoints are down, OR PASS when `qwen-llama` (:8080) and `qwen-verifier` (:8086) are up.

If servers are up, confirm the real mission reaches `done` (the coder writes `feature.py`, tests pass).

- [ ] **Step 5: Run the full nitwit suite (offline-safe)**

Run: `python3 -m unittest test_nitwit_missions test_nitwit_workspace test_nitwit_coder test_nitwit_engine test_nitwit_integration test_nitwit_model_coder test_nitwit_model_verifier test_nitwit_model_integration`
Expected: all PASS (the live test skips if endpoints are down); confirm `test_orchestrator` still passes too.

- [ ] **Step 6: Commit**

```bash
git add nitwit/factory.py test_nitwit_model_integration.py
git commit -m "feat(nitwit): factory wiring live coder/verifier + gated integration test"
```

---

## Self-Review

**Spec coverage (Phase 2a scope):**
- Real Coder wrapping the GPU coder :8080 → Task 1 (`ModelCoder`). ✓
- Real Verifier calling CPU Qwen3-4B :8086 → Task 2 (`ModelVerifier`). ✓
- Wire the engine to live endpoints, prove a real mission reaches green → Task 3 (`build_model_engine` + gated live test). ✓
- One bounded model call per iteration (GPU-safety) → `ModelCoder.propose` does exactly one `.chat`. ✓
- Coder only proposes; engine applies/tests → `propose` returns `CoderResponse`, never touches fs/git. ✓
- Deferred (correctly out of 2a scope): in-`propose` read-tool loop for large repos; the HTTP/SSE daemon + `wit` CLI (Phase 2b); multi-repo context routing.

**Placeholder scan:** none — every step has full runnable code and exact commands. ✓

**Type consistency:** `ModelCoder`/`ModelVerifier` implement the exact `Coder.propose(ctx)->CoderResponse` / `Verifier.judge(description, ctx)->bool` protocols from `nitwit/coder.py`; `FileEdit(path, content)` and `MissionContext(goal, constraints, notes, last_test_output, repo_files)` match Phase 1; the `client.chat(...)` signature matches `OpenAICompatibleClient.chat`. The `FakeClient` in tests mirrors that signature exactly. ✓

## Notes for later phases (not this plan)

- **Phase 2b:** the HTTP+SSE daemon (`nitwit.service`) exposing missions over an API, and the `wit` REPL/one-shot CLI + control subcommands, both consuming the same SSE stream.
- The engine + factory built here are the daemon's execution core — the daemon owns a `build_model_engine(...)` and drives `run_mission`/`pause`/`resume`/`reconcile`.
- Revisit: in-`propose` read-tool loop (read_file/list_dir) for repos too big for the context snapshot; per-repo context routing for multi-repo missions.
