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
