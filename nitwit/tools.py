"""Tools the agent can call. Phase 3: web_search (read-only) via the searxng capability."""
from __future__ import annotations

import re

# Kept precise: bare high-frequency dev words (current, version, now, today) are OMITTED so
# ordinary chat ("what's the current branch?", "what version of python", "fix this now") does
# NOT trigger a search. Multi-word phrases carry the real "current/external info" intent.
_CURRENT_SIGNALS = ("latest", "newest", "most recent", "release date", "released", "price",
                    "news", "weather", "who is the current", "how much", "when is the next",
                    "when does", "stock price", "up to date", "up-to-date", "as of 20",
                    "search the web", "look it up", "look up", "google", "search for",
                    "find out", "more details about", "any update", "what's happening",
                    "look into", "whats the latest")


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
