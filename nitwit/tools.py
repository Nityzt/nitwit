"""Tools the agent can call. web_search (searxng), fetch_url (page text), gather_context."""
from __future__ import annotations

import html.parser
import re
import urllib.request

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


class _TextExtractor(html.parser.HTMLParser):
    """Pull visible text out of an HTML page, dropping script/style/head noise."""
    _SKIP = {"script", "style", "noscript", "head", "meta", "link", "svg"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0:
            t = data.strip()
            if t:
                self.parts.append(t)


def fetch_url(url: str, *, timeout: int = 6, max_chars: int = 1500, _get=None) -> str:
    """GET a URL and return readable page text (HTML stripped), capped at max_chars. NEVER raises —
    returns "" on any failure. `_get(url) -> html_str` is injectable for tests (no network)."""
    get = _get
    if get is None:
        def get(u):
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (nitwit)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                ct = (r.headers.get("Content-Type") or "").lower()
                if ct and "html" not in ct and "text" not in ct:
                    return ""
                raw = r.read(400_000)
                return raw.decode(r.headers.get_content_charset() or "utf-8", "replace")
    try:
        page = get(url)
    except Exception:
        return ""
    if not page:
        return ""
    try:
        p = _TextExtractor()
        p.feed(page)
        text = " ".join(p.parts)
    except Exception:
        return ""
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def gather_context(query: str, *, k_results: int = 6, k_pages: int = 3, _search=None, _fetch=None) -> dict:
    """search + fetch the top result pages into one grounded CONTEXT block. NEVER raises.
    Returns {"context": str, "sources": list[url], "results": str}."""
    search = _search or web_search
    fetch = _fetch or fetch_url
    try:
        results = search(query, limit=k_results)
    except Exception:
        results = "WEB RESULTS:\n(no results)"
    urls = list(dict.fromkeys(re.findall(r"\((https?://[^)\s]+)\)", results or "")))
    pages = []
    for u in urls[:k_pages]:
        try:
            txt = fetch(u)
        except Exception:
            txt = ""
        if txt:
            pages.append(f"[{u}]\n{txt}")
    context = results or ""
    if pages:
        context += "\n\nPAGE CONTENT (extracted from the top results just now):\n" + "\n\n".join(pages)
    return {"context": context, "sources": urls, "results": results}
