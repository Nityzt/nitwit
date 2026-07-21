# Nitwit Interactive CLI (codex/claude-style `wit`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `wit` (run from any directory, on the PATH) open a conversational session like Claude Code / codex / agy: it auto-connects/starts the daemon, uses the current git repo as context, and you interact in natural language — quick questions answered inline (streamed), coding tasks auto-escalated into a durable background mission you can detach from. The current subcommand CLI stays for scripting.

**Architecture:** A new `nitwit/session.py` holds pure helpers (repo detection, test-command detection, intent classification, daemon auto-start). `nitwit/cli.py`'s bare-`wit` path becomes a real interactive loop: classify each input → stream a model answer directly from the coder (chat) OR create a mission via the daemon API and stream its SSE progress inline (task, Ctrl-C = detach). A `wit` shim on `~/.local/bin` makes it launchable anywhere.

**Tech Stack:** Python 3.14, stdlib + existing `orchestrator.OpenAICompatibleClient` (has `stream_chat`). No new deps.

## Global Constraints
- Stdlib + existing project modules only.
- `wit` must work from ANY directory: the shim sets `PYTHONPATH` to the repo so `import nitwit` resolves, but the process CWD stays the user's dir (repo detection uses CWD).
- The session must auto-start the daemon if it isn't running (the user should never have to start it manually), and connect if it is.
- Coding-task edits auto-apply and commit to `agent/<slug>` — never main, never push (unchanged engine behavior).
- Ctrl-C during a streaming mission DETACHES (mission keeps running in the daemon), returning to the prompt — it does not kill the mission or the session.
- Loopback only; the daemon binds 127.0.0.1 (unchanged).
- Tests: root-level `test_nitwit_*.py`, `unittest`. Pure helpers tested directly; streaming/dispatch tested with fakes/stubs (no GPU, no real daemon).

## File Structure
- `nitwit/session.py` — `repo_root`, `detect_test_cmd`, `classify_intent`, `ensure_daemon`, `stream_answer`.
- `nitwit/cli.py` — MODIFY: replace the thin `repl()` with `interactive()`; bare `wit` routes to it.
- `deploy/wit` — executable shim (`python3 -m nitwit.cli` with PYTHONPATH).
- `deploy/install-wit.sh` — installs the shim to `~/.local/bin/wit`.
- Tests: `test_nitwit_session.py`, additions to `test_nitwit_cli.py`.

---

## Task 1: session helpers (`nitwit/session.py`)

**Files:** Create `nitwit/session.py`; Test `test_nitwit_session.py`.

**Interfaces (Produces):**
- `repo_root(cwd: str) -> str | None` — `git -C cwd rev-parse --show-toplevel` stripped, or None if not a repo.
- `detect_test_cmd(repo: str) -> str | None` — by files present: `pyproject.toml`/`setup.py`/`pytest.ini`/any `test_*.py` → `"pytest"`; `package.json` containing a `"test"` script → `"npm test"`; `Cargo.toml` → `"cargo test"`; `go.mod` → `"go test ./..."`; else None.
- `classify_intent(text: str) -> str` — `"task"` if the text starts with (or clearly is) a coding imperative (add/create/implement/write/build/make/fix/refactor/rename/update/change/remove/delete/debug/optimize/generate/set up/wire), else `"answer"`. Ambiguous → `"answer"`.
- `ensure_daemon(url: str, *, spawn: bool = True, timeout: float = 20.0) -> bool` — GET `{url}/status`; if reachable return True; else if `spawn`, launch `python3 -m nitwit` detached (via `subprocess.Popen`, `start_new_session=True`, stdout/stderr to a log under `~/.local/share/nitwit/`) and poll until healthy or timeout.

- [ ] **Step 1: failing test** — create `test_nitwit_session.py`:

```python
import os, tempfile, subprocess, unittest
from nitwit.session import repo_root, detect_test_cmd, classify_intent


def _git(d, *a): subprocess.run(["git", "-C", d, *a], capture_output=True)


class TestRepoRoot(unittest.TestCase):
    def test_detects_repo_and_none(self):
        d = tempfile.mkdtemp(); _git(d, "init")
        self.assertEqual(os.path.realpath(repo_root(d)), os.path.realpath(d))
        sub = os.path.join(d, "a", "b"); os.makedirs(sub)
        self.assertEqual(os.path.realpath(repo_root(sub)), os.path.realpath(d))
        self.assertIsNone(repo_root(tempfile.mkdtemp()))  # fresh non-repo


class TestDetectTestCmd(unittest.TestCase):
    def _mk(self, **files):
        d = tempfile.mkdtemp()
        for name, body in files.items():
            with open(os.path.join(d, name), "w") as fh: fh.write(body)
        return d
    def test_python(self):
        self.assertEqual(detect_test_cmd(self._mk(**{"pyproject.toml": "[tool]"}) ), "pytest")
        self.assertEqual(detect_test_cmd(self._mk(**{"test_x.py": "def test(): pass"})), "pytest")
    def test_node(self):
        self.assertEqual(detect_test_cmd(self._mk(**{"package.json": '{"scripts":{"test":"vitest"}}'})), "npm test")
    def test_node_without_test_script_is_none(self):
        self.assertIsNone(detect_test_cmd(self._mk(**{"package.json": '{"name":"x"}'})))
    def test_rust_go(self):
        self.assertEqual(detect_test_cmd(self._mk(**{"Cargo.toml": ""})), "cargo test")
        self.assertEqual(detect_test_cmd(self._mk(**{"go.mod": ""})), "go test ./...")
    def test_none(self):
        self.assertIsNone(detect_test_cmd(self._mk(**{"README.md": "hi"})))


class TestClassify(unittest.TestCase):
    def test_tasks(self):
        for t in ["add a /health endpoint", "fix the failing test", "refactor parse()",
                  "implement fib", "write a CLI", "make it handle empty input"]:
            self.assertEqual(classify_intent(t), "task", t)
    def test_answers(self):
        for a in ["what does parse() do?", "how does the loop work",
                  "explain this function", "is this thread-safe?"]:
            self.assertEqual(classify_intent(a), "answer", a)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: run, expect FAIL** — `python3 -m unittest test_nitwit_session -v` → ModuleNotFoundError.

- [ ] **Step 3: implement** — create `nitwit/session.py`:

```python
"""Interactive-session helpers: repo/test detection, intent classification, daemon auto-start,
and a streamed chat answer. Pure functions where possible so they're unit-testable offline."""
from __future__ import annotations

import glob
import json
import os
import subprocess
import time
import urllib.request

_TASK_VERBS = ("add", "create", "implement", "write", "build", "make", "fix", "refactor",
               "rename", "update", "change", "remove", "delete", "debug", "optimize",
               "generate", "set up", "wire", "install", "migrate", "convert", "replace")


def repo_root(cwd: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", cwd, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True)
    except Exception:
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def detect_test_cmd(repo: str) -> str | None:
    def has(name): return os.path.exists(os.path.join(repo, name))
    if has("pyproject.toml") or has("setup.py") or has("pytest.ini") or glob.glob(os.path.join(repo, "test_*.py")) \
            or glob.glob(os.path.join(repo, "**", "test_*.py"), recursive=False):
        return "pytest"
    pkg = os.path.join(repo, "package.json")
    if os.path.exists(pkg):
        try:
            data = json.load(open(pkg))
            if isinstance(data.get("scripts"), dict) and "test" in data["scripts"]:
                return "npm test"
        except Exception:
            pass
    if has("Cargo.toml"):
        return "cargo test"
    if has("go.mod"):
        return "go test ./..."
    return None


def classify_intent(text: str) -> str:
    t = (text or "").strip().lower()
    first = t.split()[0] if t.split() else ""
    if first in _TASK_VERBS or any(t.startswith(v + " ") for v in _TASK_VERBS):
        return "task"
    return "answer"


def ensure_daemon(url: str, *, spawn: bool = True, timeout: float = 20.0) -> bool:
    def up():
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/status", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False
    if up():
        return True
    if not spawn:
        return False
    logdir = os.path.expanduser("~/.local/share/nitwit")
    os.makedirs(logdir, exist_ok=True)
    log = open(os.path.join(logdir, "daemon.log"), "a")
    subprocess.Popen(["python3", "-m", "nitwit"], stdout=log, stderr=log, start_new_session=True)
    end = time.time() + timeout
    while time.time() < end:
        if up():
            return True
        time.sleep(0.4)
    return False
```

- [ ] **Step 4: run, expect PASS** — `python3 -m unittest test_nitwit_session -v` (repo_root/detect/classify tests; ensure_daemon covered in Task 3 integration).

- [ ] **Step 5: commit** — `git add nitwit/session.py test_nitwit_session.py && git commit -m "feat(nitwit): session helpers — repo/test detection, intent, daemon autostart"`

---

## Task 2: streamed chat answer + interactive loop

**Files:** MODIFY `nitwit/session.py` (add `stream_answer`) and `nitwit/cli.py` (add `interactive`, route bare `wit`). Test: additions to `test_nitwit_cli.py` and `test_nitwit_session.py`.

**Interfaces (Produces):**
- `session.stream_answer(text, repo, *, coder_url, coder_model, out=print_chunk) -> None` — builds a repo-aware chat prompt (system: "You are a coding assistant in the repo <root>; here are its files: <top-level listing>; answer concisely") and streams the model reply chunk-by-chunk to `out`. Uses `orchestrator.OpenAICompatibleClient(coder_url, coder_model).stream_chat(...)`. Errors print a friendly line, never a traceback.
- `cli.interactive(base, cwd, coder_url, coder_model) -> None` — the session loop: `ensure_daemon`; `repo = repo_root(cwd)`; `test_cmd = detect_test_cmd(repo)`; banner; loop reading input; `/`-commands (`/help /missions /diff <id> /status /on /off /mission <goal> /quit`); natural input → `classify_intent`: `answer` → `stream_answer`; `task` → create mission (repo, branch `agent/<slug>`, `test_cmd` criterion if any, + verifier criterion) via `POST /missions`, `POST /control/on`, announce, then `stream_events_for(base, mission_id)` inline until terminal or KeyboardInterrupt (detach).
- `cli.stream_events_for(base, mission_id, out=print)` — consumes `/events`, prints `humanize_event` for the given mission_id, returns on that mission's `mission_finished`/`mission_error` OR KeyboardInterrupt (detach).

- [ ] **Step 1: failing tests** — add to `test_nitwit_session.py`:

```python
class TestStreamAnswer(unittest.TestCase):
    def test_streams_chunks(self):
        from nitwit import session
        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                for c in ["Hello", " world"]:
                    yield c
        chunks = []
        session.stream_answer("hi", None, coder_url="http://x", coder_model="m",
                              out=chunks.append, _client_factory=lambda u, m: FakeClient())
        self.assertEqual("".join(chunks), "Hello world")
```

Add to `test_nitwit_cli.py` (using the existing stub-server `setUp`):

```python
    def test_bare_interactive_routes_to_session(self):
        # main() with no subcommand and stdin closed should attempt the interactive session,
        # which reads EOF immediately and exits cleanly (no traceback).
        import io, sys
        from contextlib import redirect_stdout
        old = sys.stdin
        sys.stdin = io.StringIO("")  # immediate EOF
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main(["--url", self.base])  # ensure_daemon sees the stub (status 200)
            self.assertIn("nitwit", buf.getvalue().lower())  # printed a banner
        finally:
            sys.stdin = old
```

- [ ] **Step 2: run, expect FAIL** — `stream_answer`/`interactive` missing.

- [ ] **Step 3: implement** — add to `nitwit/session.py`:

```python
def _default_chunk(s):
    import sys
    sys.stdout.write(s); sys.stdout.flush()


def stream_answer(text, repo, *, coder_url, coder_model, out=_default_chunk, _client_factory=None):
    from orchestrator import OpenAICompatibleClient
    factory = _client_factory or (lambda u, m: OpenAICompatibleClient(u, m))
    files = ""
    if repo:
        try:
            files = ", ".join(sorted(os.listdir(repo))[:40])
        except Exception:
            files = ""
    system = ("You are a concise coding assistant working inside a local repository."
              + (f" Repo root: {repo}. Top-level entries: {files}." if repo else "")
              + " Answer the user's question directly and briefly.")
    try:
        client = factory(coder_url, coder_model)
        for chunk in client.stream_chat(
                [{"role": "system", "content": system}, {"role": "user", "content": text}],
                temperature=0.2, max_tokens=800):
            out(chunk)
        out("\n")
    except Exception as exc:
        out(f"\n(couldn't reach the model: {exc})\n")
```

Then rework `nitwit/cli.py` — replace `repl()` with `interactive()` and add `stream_events_for`, and route bare `wit` to it. Add near the top: `from nitwit import session`. Implementation:

```python
def stream_events_for(base, mission_id, out=print):
    import urllib.request
    try:
        with urllib.request.urlopen(base + "/events") as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if ev.get("mission_id") != mission_id:
                    continue
                out(humanize_event(ev))
                if ev.get("event") in ("mission_finished", "mission_error"):
                    return
    except KeyboardInterrupt:
        out("(detached — mission keeps running; `/missions` to check, `wit diff <id>` to review)")
    except Exception as exc:
        out(f"(stream ended: {exc})")


def interactive(base, cwd, coder_url, coder_model):
    if not session.ensure_daemon(base):
        print("could not start the nitwit daemon; check ~/.local/share/nitwit/daemon.log")
        return
    repo = session.repo_root(cwd)
    test_cmd = session.detect_test_cmd(repo) if repo else None
    where = repo or f"{cwd} (not a git repo — tasks need a repo)"
    print(f"nitwit · {where} · tests: {test_cmd or 'none detected'} · /help, /quit")
    while True:
        try:
            line = input("wit ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        if line == "/help":
            print("Talk naturally: a question is answered here; a task ('add ...', 'fix ...') "
                  "runs as a mission on a branch (Ctrl-C detaches).\n"
                  "/missions  /diff <id>  /status  /on  /off  /mission <goal>  /quit")
            continue
        if line.startswith("/"):
            parts = line[1:].split()
            cmd = parts[0] if parts else ""
            if cmd == "missions": cmd_ls(argparse.Namespace(), base); continue
            if cmd == "status":   cmd_status(argparse.Namespace(), base); continue
            if cmd in ("on", "off"): cmd_toggle(cmd == "on")(argparse.Namespace(), base); continue
            if cmd == "diff" and len(parts) > 1:
                _, m = api_call(base, "GET", f"/missions/{parts[1]}")
                print(json.dumps(m, indent=2) if isinstance(m, dict) else m); continue
            if cmd == "mission" and len(parts) > 1:
                _start_mission(base, repo, test_cmd, " ".join(parts[1:])); continue
            print("unknown command; /help"); continue
        if session.classify_intent(line) == "task":
            _start_mission(base, repo, test_cmd, line)
        else:
            session.stream_answer(line, repo, coder_url=coder_url, coder_model=coder_model)


def _start_mission(base, repo, test_cmd, goal):
    if not repo:
        print("this looks like a task, but you're not in a git repo. cd into one and try again.")
        return
    crit = []
    if test_cmd:
        crit.append({"type": "tests", "repo": repo, "cmd": test_cmd})
    crit.append({"type": "verifier", "description": "the request is meaningfully and correctly done"})
    branch = "agent/" + __import__("nitwit.missions", fromlist=["slugify"]).slugify(goal)[:40]
    _, m = api_call(base, "POST", "/missions",
                    {"goal": goal, "repos": [{"path": repo, "branch": branch, "test_cmd": test_cmd or "",
                                              "checkpoint_commit": ""}], "success_criteria": crit})
    if not (isinstance(m, dict) and m.get("id")):
        print(m); return
    api_call(base, "POST", "/control/on")
    print(f"→ mission {m['id']} on {branch} (Ctrl-C to detach; it keeps running)")
    stream_events_for(base, m["id"])
    _, fin = api_call(base, "GET", f"/missions/{m['id']}")
    if isinstance(fin, dict):
        print(f"mission {m['id']}: {fin.get('state')} · review: wit diff {m['id']} (or git -C {repo} diff main..{branch})")
```

Finally, in `main()`, route bare invocation to `interactive` (coder url/model from env with the daemon's defaults):

```python
    if not args.cmd and not args.prompt:
        import os
        return interactive(base, os.getcwd(),
                           os.environ.get("NITWIT_CODER_URL", "http://127.0.0.1:8080"),
                           os.environ.get("NITWIT_CODER_MODEL", "qwen2.5-coder-7b"))
```
(Keep the existing `-p` one-shot and all subcommands.)

- [ ] **Step 4: run, expect PASS** — `python3 -m unittest test_nitwit_session test_nitwit_cli -v`. Also `python3 -c "import nitwit.cli, nitwit.session"`.

- [ ] **Step 5: commit** — `git add nitwit/session.py nitwit/cli.py test_nitwit_session.py test_nitwit_cli.py && git commit -m "feat(nitwit): interactive wit session — chat answers + auto-escalating missions"`

---

## Task 3: `wit` on the PATH + docs + smoke

**Files:** Create `deploy/wit`, `deploy/install-wit.sh`; add a README section. Test: a shell smoke (documented, not in the unittest suite).

**Interfaces (Produces):** a `wit` executable that runs from any directory.

- [ ] **Step 1** — create `deploy/wit` (executable):

```bash
#!/usr/bin/env bash
# `wit` — launch the nitwit CLI from any directory. Keeps your CWD (for repo detection)
# but puts the nitwit package on PYTHONPATH.
exec env PYTHONPATH="/home/nit/qwen-orchestrator${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m nitwit.cli "$@"
```

- [ ] **Step 2** — create `deploy/install-wit.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$HOME/.local/bin"
install -m 0755 "$(dirname "$0")/wit" "$HOME/.local/bin/wit"
echo "installed wit -> $HOME/.local/bin/wit"
case ":$PATH:" in *":$HOME/.local/bin:"*) : ;; *) echo "note: add ~/.local/bin to PATH";; esac
```

- [ ] **Step 3** — `chmod +x deploy/wit deploy/install-wit.sh`; run `bash deploy/install-wit.sh`; verify `command -v wit` and `wit --help` work from a different directory (e.g. `cd /tmp && wit status` reaches/starts the daemon).

- [ ] **Step 4** — add a "Using `wit`" section to `README.md`: `install-wit.sh`, then `cd <repo> && wit`, describe the session (talk naturally; questions answered; tasks run as reviewable missions on a branch; Ctrl-C detaches).

- [ ] **Step 5** — full offline suite green (`python3 -m unittest` across all `test_nitwit_*`), then commit `deploy/wit deploy/install-wit.sh README.md`.

---

## Self-Review
- Open `wit` anywhere → session in CWD repo, daemon auto-started → Task 1 (`ensure_daemon`, `repo_root`, `detect_test_cmd`) + Task 2 (`interactive`) + Task 3 (PATH shim). ✓
- Natural input: question → streamed inline answer; task → auto-escalated durable mission, detachable → Task 2 (`classify_intent` + `stream_answer` + `_start_mission`/`stream_events_for`). ✓
- Auto-apply + review via git → unchanged engine (commits to `agent/<slug>`); session prints the review command. ✓
- Subcommands/`-p` preserved for scripting → `main()` only routes to `interactive` when no subcommand/prompt. ✓
- No placeholders; types consistent (`interactive(base, cwd, coder_url, coder_model)`, `_start_mission(base, repo, test_cmd, goal)`, `stream_events_for(base, mission_id, out)`); reuses `humanize_event`, `api_call`, `slugify`, `OpenAICompatibleClient.stream_chat`. ✓

## Not doing (later)
- Token-level fanciness / TUI (this is a line-streamed REPL, which is the codex/claude baseline).
- Web UI (separate phase, same API).
- Model-based intent classification (heuristic is v1; revisit if it mis-routes).
