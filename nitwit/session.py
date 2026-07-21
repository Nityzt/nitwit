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


def _chunk_text(event) -> str:
    if isinstance(event, dict):
        return event.get("content", "") if event.get("type") == "chunk" else ""
    return event if isinstance(event, str) else ""


def _looks_like_search(s: str) -> bool:
    """True while `s` is (or could still grow into) a leading 'SEARCH:' directive."""
    return "SEARCH:".startswith(s[:7].upper()) if len(s) < 7 else s[:7].upper() == "SEARCH:"


def _stream_and_peek(client, messages, out, parts, allow_search):
    """Stream the model reply.
    - allow_search + reply begins with 'SEARCH:' → return the query (emit nothing).
    - not allow_search + reply begins with 'SEARCH:' → suppress the directive line and stream only
      what follows, so a parroted directive from a weak model never leaks to the user.
    - otherwise stream the whole reply. Returns the search query, or None."""
    buf = ""
    committed = False
    for event in client.stream_chat(messages, temperature=0.2, max_tokens=800):
        chunk = _chunk_text(event)
        if not chunk:
            continue
        if committed:
            out(chunk); parts.append(chunk); continue
        buf += chunk
        s = buf.lstrip()
        if _looks_like_search(s):
            if s[:7].upper() != "SEARCH:":            # still a partial prefix ("SEAR") — wait
                continue
            if "\n" not in s and len(s) <= 200:        # directive line not finished yet
                continue
            query = (s[7:].splitlines() or [""])[0].strip()
            if allow_search:
                return query                           # caller runs the search
            committed = True                           # search disabled: drop the directive line
            rest = s.split("\n", 1)[1] if "\n" in s else ""
            buf = ""
            if rest:
                out(rest); parts.append(rest)
            continue
        if len(s) >= 7:                                # definitely NOT a SEARCH directive
            committed = True
            out(buf); parts.append(buf); buf = ""
    if not committed and buf:                          # stream ended mid-buffer (short/unterminated)
        s = buf.lstrip()
        if s[:7].upper() == "SEARCH:":
            query = (s[7:].splitlines() or [""])[0].strip()
            if allow_search:
                return query
            rest = s.split("\n", 1)[1] if "\n" in s else ""   # strip directive, emit any remainder
            if rest:
                out(rest); parts.append(rest)
            return None
        out(buf); parts.append(buf)
    return None


def _build_system(repo, files, memories, *, can_search):
    """The chat system prompt. `can_search` picks the behaviour clause: in *can_search* mode the
    model is told how to request a search; in *grounded* mode (results already injected) it is told
    to answer from them and NOT to emit a SEARCH: directive — a weak model otherwise parrots it."""
    base = (
        "You are Nitwit, a local, self-hosted coding assistant running open-source models on the "
        "user's own machine. You were NOT created by OpenAI or Anthropic and you are not GPT-4. "
        "You have live web access through the host system and can look things up. Never claim you "
        "lack internet access or cannot look things up."
    )
    if can_search:
        behav = (
            " Whenever answering needs current or external information (latest versions, news, "
            "prices, releases, who currently holds a role, or anything you are not sure is up to "
            "date), reply with EXACTLY `SEARCH: <query>` as your entire message and nothing else — "
            "the system will run the search and give you the results to answer from. Otherwise "
            "answer directly and briefly."
        )
    else:
        behav = (
            " Live web search results are provided in the conversation below — they were fetched "
            "just now for this question and are current. Answer directly from them and cite the "
            "source URLs. The search has already been run, so do NOT reply with a SEARCH: directive. "
            "Do NOT add disclaimers about knowledge cutoffs, training dates, or being unable to "
            "search in real time — treat the results as present fact and just answer."
        )
    repo_clause = (f" You are working in the repository at {repo} (top-level entries: {files})." if repo else "")
    system = base + behav + repo_clause + (
        " Use the conversation so far for context; do not contradict earlier answers."
    )
    if memories:
        block = "\n".join(f"- {f}" for f in memories)[:1500]
        system += "\nKnown facts about the user (honor these):\n" + block
    return system


def stream_answer(text, repo, *, history=None, out=_default_chunk, _client_factory=None,
                  _endpoint=None, _route=None, allow_search=True, _search_fn=None, memories=None):
    """Stream a chat answer on the router-selected CHAT model. The model can request a web search
    (SEARCH: <query>); obvious current-info questions search proactively. Whenever results are
    present the system prompt switches to a *grounded* mode so the model answers from them instead
    of re-emitting SEARCH:. Recalls `memories`. Never raises; returns the answer text."""
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

    def do_search(query):
        out("[searching the web…]\n")
        try:
            return (_search_fn or tools.web_search)(query)
        except Exception:
            return "WEB RESULTS:\n(no results)"

    # A proactive search on obvious current-info questions means results are present up front, so
    # start in grounded mode (no SEARCH: instruction) and disable further searching this turn.
    proactive = allow_search and tools.needs_web_search(text)
    can_search = allow_search and not proactive
    messages = [{"role": "system", "content": _build_system(repo, files, memories, can_search=can_search)}]
    messages.extend(history or [])
    if proactive:
        messages.append({"role": "system",
                         "content": "WEB RESULTS (use these for current facts, cite URLs):\n" + do_search(text)})
        allow_search = False

    messages.append({"role": "user", "content": text})
    parts = []
    try:
        client = factory(ep.base_url, ep.model, extra_body=ep.extra_body)
        query = _stream_and_peek(client, messages, out, parts, allow_search)
        if query is not None:                         # the model asked to search
            results = do_search(query or text)        # empty model query → fall back to the question
            # switch system[0] to grounded so the re-ask answers from results, not another SEARCH:
            messages[0] = {"role": "system",
                           "content": _build_system(repo, files, memories, can_search=False)}
            messages.append({"role": "system", "content": "WEB RESULTS (use these, cite URLs):\n" + results})
            parts.clear()                             # discard the SEARCH: directive text
            client2 = factory(ep.base_url, ep.model, extra_body=ep.extra_body)
            _stream_and_peek(client2, messages, out, parts, allow_search=False)  # no more searching
            if not parts:                             # model only parroted a directive → show results
                out(results); parts.append(results)
        out("\n")
    except Exception as exc:
        out(f"\n(couldn't reach the model: {exc})\n")
    return "".join(parts)
