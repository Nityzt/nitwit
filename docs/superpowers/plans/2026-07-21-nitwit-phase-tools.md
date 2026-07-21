# Nitwit Phase 3: Tool calling — web search in chat

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Give the chat agent web search so it can answer questions about current/external facts (latest versions, releases, prices, news) with real, sourced info instead of stale guesses. When a question needs it, `wit` searches the web (searxng), shows a brief "searching…" note, and answers from the results with source URLs.

**Architecture:** A small `nitwit/tools.py` wraps the existing `web_search` capability (`webui.run_capability` → searxng). `stream_answer` gains a heuristic pre-check: if the question looks like it needs current/external info, run one web search, inject the results into the model's context, and stream a grounded answer; otherwise stream directly (no latency added to normal chat). Reuses the "search only when unsure" discipline that benchmarked well. Missions-side web search + a gated `run_command` are a documented follow-on (Phase 3b).

**Tech Stack:** Python 3.14, stdlib + existing `webui.run_capability`. No new deps.

## Global Constraints
- Stdlib + existing modules only; `web_search` goes through `webui.run_capability("web_search", {"query", "limit"})` (searxng, loopback :8888).
- `tools.web_search(...)` must NEVER raise (returns an empty-ish results string on failure) — chat must not break if searxng is down.
- The search pre-check must NOT fire on ordinary chat (only current/external-info questions) — no latency tax on normal conversation.
- Read-only: web search only; no shell/file tools in this phase.
- Tests: root-level `test_nitwit_*.py`, `unittest`; inject a fake search fn (no network).

## File Structure
- `nitwit/tools.py` — `web_search(query, limit=4) -> str`, `needs_web_search(text) -> bool`.
- `nitwit/session.py` — MODIFY `stream_answer`: optional `allow_search=True` + `_search_fn` seam; when triggered, search + inject results + a "searching…" note.
- Tests: `test_nitwit_tools.py`, additions to `test_nitwit_session.py`.

---

## Task 1: web-search tool + heuristic + chat wiring

**Files:** Create `nitwit/tools.py`; MODIFY `nitwit/session.py`. Test: `test_nitwit_tools.py`, additions to `test_nitwit_session.py`.

**Interfaces (Produces):**
- `tools.web_search(query: str, limit: int = 4, *, _run=None) -> str` — calls `run_capability("web_search", {"query": query, "limit": limit})` (import lazily from `webui`; `_run` injectable for tests), formats up to `limit` results as `"- <title>: <snippet> (<url>)"` lines under a `WEB RESULTS:` header, capped ~1200 chars; returns `"WEB RESULTS:\n(no results)"` on empty/failure. Never raises.
- `tools.needs_web_search(text: str) -> bool` — True if the text signals a need for current/external info: contains any of `latest`, `current`, `today`, `now`, `recent`, `release date`, `released`, `version`, `price`, `news`, `weather`, `who is the`, `how much`, `when is`, `when does`, `stock`, a 4-digit year `>= 2024`, or ends with a question mark AND mentions a proper-noun-ish token — keep it conservative (prefer False on ambiguity so normal chat isn't taxed).
- `session.stream_answer(..., allow_search=True, _search_fn=None)` — after building the messages, if `allow_search and tools.needs_web_search(text)`: call `(_search_fn or tools.web_search)(text)`, emit a dim `"[searching the web…]\n"` note to `out`, and insert a system message `"Web search results for the user's question (use these for current facts, cite URLs):\n<results>"` right before the final user turn. Then stream as before. Unchanged when the heuristic is False. Still never raises; still returns the answer text.

- [ ] **Step 1: failing tests** — create `test_nitwit_tools.py`:

```python
import unittest
from nitwit.tools import web_search, needs_web_search


class TestNeedsWebSearch(unittest.TestCase):
    def test_current_info_true(self):
        for t in ["what's the latest Next.js version?", "current price of an rtx 4090",
                  "who is the current ceo of tesla", "when does the next iphone come out",
                  "any news on the EU AI act in 2026"]:
            self.assertTrue(needs_web_search(t), t)

    def test_ordinary_chat_false(self):
        for t in ["what does parse() do?", "explain closures", "hi", "thanks",
                  "refactor this function", "how does a hash map work"]:
            self.assertFalse(needs_web_search(t), t)


class TestWebSearch(unittest.TestCase):
    def test_formats_results(self):
        fake = {"result": {"results": [
            {"title": "Next.js", "snippet": "v15 is latest", "url": "https://nextjs.org"},
        ]}}
        out = web_search("next.js version", _run=lambda cap, arg: fake)
        self.assertIn("WEB RESULTS", out)
        self.assertIn("Next.js", out)
        self.assertIn("https://nextjs.org", out)

    def test_empty_and_failure_never_raise(self):
        self.assertIn("no results", web_search("x", _run=lambda cap, arg: {"result": {"results": []}}))
        def boom(cap, arg): raise RuntimeError("down")
        self.assertIn("no results", web_search("x", _run=boom))  # must not raise


if __name__ == "__main__":
    unittest.main()
```

Add to `test_nitwit_session.py`:

```python
class TestStreamAnswerSearch(unittest.TestCase):
    def test_search_injected_when_needed(self):
        from nitwit import session
        from nitwit.router import Endpoint
        captured = {}
        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                captured["messages"] = messages
                yield {"type": "chunk", "content": "answer"}; yield {"type": "done"}
        ep = Endpoint("http://x", "m", {})
        session.stream_answer("what is the latest next.js version?", None, _endpoint=ep,
                              out=lambda s: None, _client_factory=lambda u, m, extra_body=None: FakeClient(),
                              _search_fn=lambda q: "WEB RESULTS:\n- Next.js: v15 (https://nextjs.org)")
        joined = "\n".join(m["content"] for m in captured["messages"])
        self.assertIn("WEB RESULTS", joined)
        self.assertIn("nextjs.org", joined)

    def test_no_search_for_ordinary_chat(self):
        from nitwit import session
        from nitwit.router import Endpoint
        captured = {}
        class FakeClient:
            def __init__(self, *a, **k): pass
            def stream_chat(self, messages, *, temperature, max_tokens, response_format=None):
                captured["messages"] = messages
                yield {"type": "chunk", "content": "ok"}; yield {"type": "done"}
        called = {"n": 0}
        def search(q): called["n"] += 1; return "WEB RESULTS:\n(x)"
        session.stream_answer("what does parse() do?", None, _endpoint=Endpoint("http://x", "m", {}),
                              out=lambda s: None, _client_factory=lambda u, m, extra_body=None: FakeClient(),
                              _search_fn=search)
        self.assertEqual(called["n"], 0)  # heuristic False -> no search
        self.assertNotIn("WEB RESULTS", "\n".join(m["content"] for m in captured["messages"]))
```

- [ ] **Step 2: run, expect FAIL** — module/param missing.

- [ ] **Step 3: implement** — create `nitwit/tools.py`:

```python
"""Tools the agent can call. Phase 3: web_search (read-only) via the searxng capability."""
from __future__ import annotations

import re

_CURRENT_SIGNALS = ("latest", "current", "today", "now", "recent", "release date", "released",
                    "version", "price", "news", "weather", "who is the", "how much", "when is",
                    "when does", "stock", "this year", "up to date")


def needs_web_search(text: str) -> bool:
    t = (text or "").lower()
    if any(sig in t for sig in _CURRENT_SIGNALS):
        return True
    for y in re.findall(r"\b(20\d\d)\b", t):
        if int(y) >= 2024:
            return True
    return False


def web_search(query: str, limit: int = 4, *, _run=None) -> str:
    run = _run
    if run is None:
        try:
            from webui import run_capability as run
        except Exception:
            return "WEB RESULTS:\n(no results)"
    try:
        r = run("web_search", {"query": query, "limit": limit})
        results = (r.get("result") or {}).get("results") or []
    except Exception:
        results = []
    if not results:
        return "WEB RESULTS:\n(no results)"
    lines = [f"- {x.get('title','')}: {x.get('snippet','')} ({x.get('url','')})" for x in results[:limit]]
    return "WEB RESULTS:\n" + "\n".join(lines)[:1200]
```

Modify `session.stream_answer` — add params and the injection. Change the signature to include `allow_search=True, _search_fn=None`, and just before appending the final user turn, insert:

```python
    from nitwit import tools
    if allow_search and tools.needs_web_search(text):
        out("[searching the web…]\n")
        results = (_search_fn or tools.web_search)(text)
        messages.append({"role": "system",
                         "content": "Web search results for the user's question — use these for "
                                    "current facts and cite the URLs:\n" + results})
    messages.append({"role": "user", "content": text})
```

(Place the `messages.append({"role":"user",...})` that currently exists AFTER this block; the search system message must come before the user turn.)

- [ ] **Step 4: run, expect PASS** — `python3 -m unittest test_nitwit_tools test_nitwit_session -v`; `python3 -c "import nitwit.tools, nitwit.session"`.

- [ ] **Step 5: commit** — `git add nitwit/tools.py nitwit/session.py test_nitwit_tools.py test_nitwit_session.py && git commit -m "feat(nitwit): chat web search — grounded answers for current-info questions"`

---

## Self-Review
- Web search available in chat, auto when needed → `needs_web_search` + `stream_answer` injection. ✓
- Read-only, never breaks chat → `web_search` never raises; heuristic conservative. ✓
- No latency on normal chat → search only fires when the heuristic is True. ✓
- Sourced answers → results include URLs + the prompt says to cite them. ✓
- Types consistent: `web_search(query, limit=4, *, _run=None)->str`, `needs_web_search(text)->bool`, `stream_answer(..., allow_search=True, _search_fn=None)`. ✓

## Not doing (this plan)
- **Phase 3b:** web_search inside missions (the ModelCoder's bounded tool loop) and a gated `run_command` tool (interactive y/n approval; headless = declared test_cmd only).
- **Phase 4:** persistent memory.
