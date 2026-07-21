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
    if has("pyproject.toml") or has("setup.py") or has("pytest.ini") or glob.glob(os.path.join(repo, "test_*.py")):
        return "pytest"
    pkg = os.path.join(repo, "package.json")
    if os.path.exists(pkg):
        try:
            with open(pkg) as fh:
                data = json.load(fh)
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
    try:
        os.makedirs(logdir, exist_ok=True)
        log = open(os.path.join(logdir, "daemon.log"), "a")
        try:
            subprocess.Popen(["python3", "-m", "nitwit"], stdout=log, stderr=log, start_new_session=True)
        finally:
            log.close()
    except Exception:
        return False
    end = time.time() + timeout
    while time.time() < end:
        if up():
            return True
        time.sleep(0.4)
    return False


def _default_chunk(s):
    import sys
    sys.stdout.write(s); sys.stdout.flush()


def scratch_workspace(goal, *, root=None):
    """Creates an isolated scratch workspace for a mission.
    Returns the absolute path to a new git repo with an initial commit."""
    from nitwit.missions import slugify
    import uuid
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
    """Copies every entry of src except .git into dest (creating dest).
    Returns dest. Does not touch src."""
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


def stream_answer(text, repo, *, history=None, out=_default_chunk, _client_factory=None, _endpoint=None, _route=None):
    """Stream a chat answer on the best-fit CHAT model (router-selected: fast CPU 4B), carrying
    the conversation `history`. Returns the answer text. Never raises."""
    from orchestrator import OpenAICompatibleClient
    from nitwit.router import route as _default_route
    router = _route or _default_route
    ep = _endpoint or router("chat")
    factory = _client_factory or (lambda u, m, extra_body=None: OpenAICompatibleClient(u, m, extra_body=extra_body))
    files = ""
    if repo:
        try:
            files = ", ".join(sorted(os.listdir(repo))[:40])
        except Exception:
            files = ""
    system = ("You are Nitwit, a local, self-hosted coding assistant. You run open-source models "
              "on the user's own machine; you were not created by OpenAI or Anthropic — "
              "if asked, say you are a local self-hosted assistant."
              + (f" You are working in the repository at {repo} (top-level entries: {files})." if repo else "")
              + " Answer directly and briefly, using the conversation so far for context; do not "
                "contradict earlier answers.")
    messages = [{"role": "system", "content": system}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": text})
    parts: list[str] = []

    def emit(s):
        parts.append(s)
        out(s)

    try:
        client = factory(ep.base_url, ep.model, extra_body=ep.extra_body)
        for event in client.stream_chat(messages, temperature=0.2, max_tokens=800):
            if isinstance(event, dict):
                if event.get("type") == "chunk":
                    emit(event.get("content", ""))
            elif isinstance(event, str):
                emit(event)
        out("\n")
    except Exception as exc:
        out(f"\n(couldn't reach the model: {exc})\n")
    return "".join(parts)
