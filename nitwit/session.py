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


def _stream_and_peek(client, messages, out, parts, allow_search):
    """Stream the model reply. If allow_search and the reply begins with 'SEARCH:', return the
    query string (emitting nothing). Otherwise stream the whole reply to out/parts, return None."""
    buf = ""
    committed = False
    for event in client.stream_chat(messages, temperature=0.2, max_tokens=800):
        if isinstance(event, dict):
            chunk = event.get("content", "") if event.get("type") == "chunk" else ""
        elif isinstance(event, str):
            chunk = event
        else:
            chunk = ""
        if not chunk:
            continue
        if committed:
            out(chunk); parts.append(chunk); continue
        buf += chunk
        s = buf.lstrip()
        if allow_search and len(s) >= 7 and s[:7].upper() == "SEARCH:":
            if "\n" in s or len(s) > 160:            # have the full query line
                return s[7:].splitlines()[0].strip()
            continue                                  # keep buffering the query line
        if len(s) >= 7:                               # enough to know it's NOT a SEARCH directive
            committed = True
            out(buf); parts.append(buf); buf = ""
    if not committed and buf:                         # short reply below the 7-char threshold
        s = buf.lstrip()
        if allow_search and s[:7].upper() == "SEARCH:":
            return s[7:].splitlines()[0].strip()
        out(buf); parts.append(buf)
    return None


def stream_answer(text, repo, *, history=None, out=_default_chunk, _client_factory=None,
                  _endpoint=None, _route=None, allow_search=True, _search_fn=None, memories=None):
    """Stream a chat answer on the router-selected CHAT model. The model can request a web search
    (SEARCH: <query>); obvious current-info questions search proactively. Recalls `memories`.
    Never raises; returns the answer text."""
    from orchestrator import OpenAICompatibleClient
    from nitwit.router import route as _default_route
    from nitwit import tools
    router = _route or _default_route
    ep = _endpoint or router("chat")
    factory = _client_factory or (lambda u, m, extra_body=None: OpenAICompatibleClient(u, m, extra_body=extra_body))
    files = ""
    if repo:
        try:
            files = ", ".join(sorted(os.listdir(repo))[:40])
        except Exception:
            files = ""
    system = (
        "You are Nitwit, a local, self-hosted coding assistant running open-source models on the "
        "user's own machine. You were NOT created by OpenAI or Anthropic and you are not GPT-4. "
        "You CAN look things up on the web: whenever answering needs current or external "
        "information (latest versions, news, prices, releases, who currently holds a role, or "
        "anything you are not sure is up to date), reply with EXACTLY `SEARCH: <query>` as your "
        "entire message and nothing else — the system will run the search and give you the "
        "results to answer from. Never claim you lack internet access or cannot look things up."
        + (f" You are working in the repository at {repo} (top-level entries: {files})." if repo else "")
        + " Otherwise answer directly and briefly, using the conversation so far for context; "
          "do not contradict earlier answers."
    )
    if memories:
        block = "\n".join(f"- {f}" for f in memories)[:1500]
        system += "\nKnown facts about the user (honor these):\n" + block

    messages = [{"role": "system", "content": system}]
    messages.extend(history or [])

    def do_search(query):
        out("[searching the web…]\n")
        try:
            return (_search_fn or tools.web_search)(query)
        except Exception:
            return "WEB RESULTS:\n(no results)"

    # proactive fast-path for obvious current-info questions
    if allow_search and tools.needs_web_search(text):
        messages.append({"role": "system",
                         "content": "WEB RESULTS (use these for current facts, cite URLs):\n" + do_search(text)})
        allow_search = False

    messages.append({"role": "user", "content": text})
    parts = []
    try:
        client = factory(ep.base_url, ep.model, extra_body=ep.extra_body)
        query = _stream_and_peek(client, messages, out, parts, allow_search)
        if query is not None:                         # the model asked to search
            results = do_search(query)
            messages.append({"role": "system",
                             "content": "WEB RESULTS (use these, cite URLs):\n" + results})
            parts.clear()                             # discard the SEARCH: directive text
            client2 = factory(ep.base_url, ep.model, extra_body=ep.extra_body)
            _stream_and_peek(client2, messages, out, parts, allow_search=False)  # no more searching
        out("\n")
    except Exception as exc:
        out(f"\n(couldn't reach the model: {exc})\n")
    return "".join(parts)
