# Nitwit Phase 2: Tasks anywhere (scratch workspace + export)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Let a task run even when you're not in a git repo — it works in an isolated scratch workspace (its own git repo under `~/.local/share/nitwit/workspaces/`), and `wit export <id> [dest]` copies the result out. Your current directory is never touched.

**Architecture:** Two small helpers in `nitwit/session.py` — `scratch_workspace(goal)` (create + git-init a workspace, return its path) and `export_workspace(src, dest)` (copy the workspace's files, minus `.git`, to a destination). `_start_mission` uses a scratch workspace when there's no repo (instead of refusing). A `wit export` subcommand + `/export` slash command copy a finished mission's files out.

**Tech Stack:** Python 3.14, stdlib only (`os`, `shutil`, `subprocess`, `uuid`). No new deps.

## Global Constraints
- Stdlib only.
- Scratch workspaces live under `~/.local/share/nitwit/workspaces/<slug>-<short-uuid>/`, each a fresh `git init` repo with an initial empty commit (so `ensure_branch`/commit/resume work).
- In a real git repo, task behavior is UNCHANGED (works on `agent/<slug>` in place).
- `export_workspace` copies files excluding `.git` (and never deletes the source); refuses to overwrite an existing non-empty dest unless told (default dest = a new subdir).
- Never push/merge; loopback only.
- Tests: root-level `test_nitwit_*.py`, `unittest`; use temp dirs (no network).

## File Structure
- `nitwit/session.py` — ADD `scratch_workspace`, `export_workspace`.
- `nitwit/cli.py` — MODIFY `_start_mission` (repo=None → scratch); ADD `cmd_export` + `export` subparser + `/export` in `interactive`.
- Tests: additions to `test_nitwit_session.py`, `test_nitwit_cli.py`.

---

## Task 1: scratch workspace + export helpers + wiring

**Files:** MODIFY `nitwit/session.py`, `nitwit/cli.py`. Test: additions to `test_nitwit_session.py`, `test_nitwit_cli.py`.

**Interfaces (Produces):**
- `session.scratch_workspace(goal: str, *, root=None) -> str` — creates `<root or ~/.local/share/nitwit/workspaces>/<slug>-<uuid8>/`, runs `git init` + `git config user.email/name` + an initial `git commit --allow-empty -m "nitwit workspace"`, returns the absolute path. Uses `nitwit.missions.slugify`.
- `session.export_workspace(src: str, dest: str) -> str` — copies every entry of `src` except `.git` into `dest` (creating `dest`); returns `dest`. Uses `shutil.copytree`/`copy2`. Does not touch `src`.
- `cli._start_mission(base, repo, test_cmd, goal)` — when `repo` is falsy, call `scratch_workspace(goal)` to get a workspace, set `repo=that`, `test_cmd=None`, and print a note that it's a scratch workspace (with an `export` hint); otherwise unchanged.
- `cli.cmd_export(args, base)` — GET `/missions/{args.id}`; read `repos[0]["path"]`; `export_workspace(that, args.dest or os.path.join(os.getcwd(), f"nitwit-{args.id}"))`; print where it went. Wire an `export` subparser (`id`, optional `dest`) into `build_parser`/`main` dispatch, and handle `/export <id> [dest]` in `interactive`.

- [ ] **Step 1: failing tests** — add to `test_nitwit_session.py`:

```python
class TestScratchWorkspace(unittest.TestCase):
    def test_creates_git_repo(self):
        import tempfile, subprocess
        from nitwit import session
        root = tempfile.mkdtemp()
        ws = session.scratch_workspace("build a cli tool", root=root)
        self.assertTrue(ws.startswith(root))
        self.assertTrue(os.path.isdir(os.path.join(ws, ".git")))
        # HEAD exists (initial commit present) so ensure_branch/commit will work
        r = subprocess.run(["git", "-C", ws, "rev-parse", "HEAD"], capture_output=True)
        self.assertEqual(r.returncode, 0)

    def test_export_copies_without_git(self):
        import tempfile
        from nitwit import session
        src = tempfile.mkdtemp()
        os.makedirs(os.path.join(src, ".git"))
        with open(os.path.join(src, ".git", "config"), "w") as fh: fh.write("x")
        with open(os.path.join(src, "app.py"), "w") as fh: fh.write("print(1)")
        os.makedirs(os.path.join(src, "sub"))
        with open(os.path.join(src, "sub", "b.txt"), "w") as fh: fh.write("b")
        dest = os.path.join(tempfile.mkdtemp(), "out")
        session.export_workspace(src, dest)
        self.assertTrue(os.path.exists(os.path.join(dest, "app.py")))
        self.assertTrue(os.path.exists(os.path.join(dest, "sub", "b.txt")))
        self.assertFalse(os.path.exists(os.path.join(dest, ".git")))  # .git excluded
```

Add to `test_nitwit_cli.py` (stub server already returns a mission for GET; extend the stub if needed to include `repos`):

```python
    def test_cmd_export(self):
        import io, tempfile, os
        from contextlib import redirect_stdout
        # the stub returns a mission dict; ensure it has repos[0].path pointing at a real dir
        src = tempfile.mkdtemp()
        with open(os.path.join(src, "f.txt"), "w") as fh: fh.write("hi")
        # monkeypatch api_call to return a mission with that path
        import nitwit.cli as C
        orig = C.api_call
        C.api_call = lambda base, method, path, body=None: (200, {"id": "m1", "repos": [{"path": src}]})
        try:
            dest = os.path.join(tempfile.mkdtemp(), "exported")
            buf = io.StringIO()
            with redirect_stdout(buf):
                C.main(["export", "m1", dest, "--url", self.base])
            self.assertTrue(os.path.exists(os.path.join(dest, "f.txt")))
        finally:
            C.api_call = orig
```

- [ ] **Step 2: run, expect FAIL** — helpers/subcommand missing.

- [ ] **Step 3: implement** — add to `nitwit/session.py`:

```python
def scratch_workspace(goal, *, root=None):
    import subprocess, uuid
    from nitwit.missions import slugify
    root = root or os.path.expanduser("~/.local/share/nitwit/workspaces")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{slugify(goal)[:32]}-{uuid.uuid4().hex[:8]}")
    os.makedirs(path, exist_ok=True)
    subprocess.run(["git", "-C", path, "init", "-q"], check=False)
    subprocess.run(["git", "-C", path, "config", "user.email", "nitwit@localhost"], check=False)
    subprocess.run(["git", "-C", path, "config", "user.name", "nitwit"], check=False)
    subprocess.run(["git", "-C", path, "commit", "-q", "--allow-empty", "-m", "nitwit workspace"], check=False)
    return path


def export_workspace(src, dest):
    import shutil
    os.makedirs(dest, exist_ok=True)
    for name in os.listdir(src):
        if name == ".git":
            continue
        s = os.path.join(src, name)
        d = os.path.join(dest, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    return dest
```

Modify `nitwit/cli.py` `_start_mission` — replace the `if not repo:` refusal:

```python
def _start_mission(base, repo, test_cmd, goal):
    scratch = False
    if not repo:
        repo = session.scratch_workspace(goal)
        test_cmd = None
        scratch = True
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
    if scratch:
        print(f"→ mission {m['id']} in a scratch workspace {repo}\n"
              f"  (Ctrl-C to detach; `wit export {m['id']}` to copy the result out)")
    else:
        print(f"→ mission {m['id']} on {branch} (Ctrl-C to detach; it keeps running)")
    stream_events_for(base, m["id"])
    _, fin = api_call(base, "GET", f"/missions/{m['id']}")
    if isinstance(fin, dict):
        tail = f"wit export {m['id']}" if scratch else f"wit diff {m['id']}"
        print(f"mission {m['id']}: {fin.get('state')} · review: {tail}")
```

Add `cmd_export` and wire it:

```python
def cmd_export(args, base):
    import os
    _, m = api_call(base, "GET", f"/missions/{args.id}")
    if not (isinstance(m, dict) and m.get("repos")):
        print(m if isinstance(m, dict) else f"no such mission {args.id}"); return
    src = m["repos"][0]["path"]
    dest = args.dest or os.path.join(os.getcwd(), f"nitwit-{args.id}")
    session.export_workspace(src, dest)
    print(f"exported {args.id} -> {dest}")
```

In `build_parser`, add: `e = sub.add_parser("export"); e.add_argument("id"); e.add_argument("dest", nargs="?"); e.add_argument("--url", default=argparse.SUPPRESS)`. In `main()`'s dispatch dict add `"export": cmd_export`. In `interactive`, handle `/export`: `if cmd == "export" and len(parts) > 1: cmd_export(argparse.Namespace(id=parts[1], dest=(parts[2] if len(parts) > 2 else None)), base); continue`.

- [ ] **Step 4: run, expect PASS** — `python3 -m unittest test_nitwit_session test_nitwit_cli -v`; `python3 -c "import nitwit.cli, nitwit.session"`.

- [ ] **Step 5: commit** — `git add nitwit/session.py nitwit/cli.py test_nitwit_session.py test_nitwit_cli.py && git commit -m "feat(nitwit): tasks work anywhere via scratch workspaces + wit export"`

---

## Self-Review
- Task works with no git repo → `scratch_workspace` + `_start_mission` scratch branch. ✓
- Result recoverable → `export_workspace` + `wit export`/`/export`. ✓
- In-repo behavior unchanged → the `if not repo` branch only fires when repo is falsy. ✓
- `.git` excluded from export; source untouched; no push/merge. ✓
- Types consistent: `scratch_workspace(goal, *, root=None)->str`, `export_workspace(src, dest)->str`, `cmd_export(args, base)`. ✓

## Not doing (later phases)
- Tool calling (Phase 3), persistent memory (Phase 4).
- Cleanup/GC of old scratch workspaces (a later `wit clean`).
