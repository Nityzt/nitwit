#!/usr/bin/env python3
"""Grounded-answer benchmark: give a model a web_search tool and see whether a
well-calibrated small model that searches *when unsure* beats a bigger model that
answers unaided.

Measures, per model + call-cap:
  - answer accuracy (final answer contains the right fact, where we know it)
  - search RECALL  : did it search on questions that need current/external facts?
  - search PRECISION (over-search): did it wastefully search on things it knows?
  - avg tool calls per question

The server must run with --jinja (native tool-calling). Bounded vs unbounded is
--max-calls (a high cap still bounds "unbounded" so a bad model can't spin forever).

  python3 -m bench.tool_loop --base-url http://127.0.0.1:8081 --model minicpm --no-think --max-calls 3
  python3 -m bench.tool_loop --base-url http://127.0.0.1:8081 --model minicpm --no-think --max-calls 0   # unaided baseline
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

from webui import run_capability

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current, external, or uncertain facts. Only call this when you are not confident of the answer from your own knowledge.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
}

# The scaffold is the real test of the "smart, not blind" hypothesis: a system
# prompt that spells out the search-decision policy the raw model got wrong
# (it over-searched known facts 5/5 and under-searched needed ones 1/4).
SCAFFOLD_SYSTEM = (
    "You have a web_search tool, but treat every call as expensive. "
    "Answer directly from your own knowledge for stable, well-known facts "
    "(capitals, chemistry, math, geometry, history, definitions) — do NOT search these. "
    "Call web_search ONLY when the answer depends on current, recent, or fast-changing "
    "information you cannot be confident about: latest software versions, who currently "
    "holds an office, prices, release dates, or live/ongoing events. When you are genuinely "
    "unsure whether your knowledge is current, prefer searching over guessing. "
    "After you have what you need, give a short, direct final answer."
)

# needs_search = ground truth on whether the answer requires current/external info.
# answer = acceptable answer keywords (regex, lowercased); None = grade search behaviour only.
DATASET = [
    {"q": "What is the capital of Japan?", "needs_search": False, "answer": r"tokyo"},
    {"q": "What is the chemical symbol for gold?", "needs_search": False, "answer": r"\bau\b"},
    {"q": "Who wrote the play Romeo and Juliet?", "needs_search": False, "answer": r"shakespeare"},
    {"q": "What is the boiling point of water in Celsius at sea level?", "needs_search": False, "answer": r"\b100\b"},
    {"q": "How many sides does a hexagon have?", "needs_search": False, "answer": r"\b(six|6)\b"},
    {"q": "Who is the current CEO of Tesla?", "needs_search": True, "answer": r"musk"},
    {"q": "What is the latest stable Python 3 release version number?", "needs_search": True, "answer": r"3\.1[0-9]"},
    {"q": "When is the next One Piece manga chapter expected to release?", "needs_search": True, "answer": None},
    {"q": "What was the most recent SpaceX Starship flight number?", "needs_search": True, "answer": None},
]


def chat(base_url, model, messages, tools, max_tokens, no_think):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if tools:
        payload["tools"] = tools
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(f"{base_url}/v1/chat/completions",
                                 data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=240) as res:
        return json.load(res)["choices"][0]["message"]


def search_summary(query: str) -> str:
    run = run_capability("web_search", {"query": query, "limit": 4})
    results = (run.get("result") or {}).get("results") or []
    lines = [f"- {r.get('title','')}: {r.get('snippet','')} ({r.get('url','')})" for r in results[:4]]
    return "SEARCH RESULTS:\n" + ("\n".join(lines)[:1500] or "(no results)")


def content_toolcall_query(content: str) -> str | None:
    """Some models (Qwen2.5-Coder here) emit the tool call as text instead of a
    structured tool_calls field. Pull the web_search query out of the content so the
    comparison stays fair."""
    if not content or "web_search" not in content:
        return None
    m = re.search(r'"query"\s*:\s*"([^"]+)"', content)
    return m.group(1) if m else None


def run_query(base_url, model, question, max_calls, no_think, system=None) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    tools = [WEB_SEARCH_TOOL] if max_calls > 0 else None
    calls = 0
    for _ in range(max_calls + 1):
        msg = chat(base_url, model, messages, tools, 400, no_think)
        tool_calls = msg.get("tool_calls") or []
        structured = bool(tool_calls)
        # normalize structured OR text-form tool calls into a list of (id, query)
        queries = []
        if structured:
            for tc in tool_calls:
                try:
                    queries.append((tc.get("id", "0"), json.loads(tc["function"]["arguments"] or "{}").get("query") or question))
                except Exception:
                    queries.append((tc.get("id", "0"), question))
        elif max_calls > 0:
            cq = content_toolcall_query(msg.get("content") or "")
            if cq:
                queries.append((None, cq))
        if queries and calls < max_calls:
            if structured:
                # Native tool-call turn: assistant(tool_calls) followed by role:tool replies.
                messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
                for tid, query in queries:
                    calls += 1
                    messages.append({"role": "tool", "tool_call_id": tid, "content": search_summary(query)})
            else:
                # Text-form tool call (Qwen2.5-Coder emits the call as prose). Never
                # emit a role:tool reply — with no matching structured tool_calls it
                # hangs the --jinja template (this stalled the earlier GPU run). Feed
                # the results back as an ordinary user turn instead.
                results = "\n\n".join(search_summary(q) for _tid, q in queries)
                calls += len(queries)
                messages.append({"role": "assistant", "content": msg.get("content") or ""})
                messages.append({"role": "user",
                                 "content": results + "\n\nUsing these search results, answer the original question directly."})
            continue
        return {"answer": msg.get("content") or "", "calls": calls}
    # ran out of cap while still wanting to call: force a final answer without tools
    msg = chat(base_url, model, messages + [{"role": "user", "content": "Answer now with what you have."}], None, 400, no_think)
    return {"answer": msg.get("content") or "", "calls": calls}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="minicpm5-1b")
    ap.add_argument("--max-calls", type=int, default=3, help="0 = unaided baseline; small = bounded; large = ~unbounded")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--scaffold", action="store_true", help="add the search-discipline system prompt (tests 'smart, not blind')")
    ap.add_argument("--system", default=None, help="override the system prompt (implies a system message)")
    args = ap.parse_args()

    system = args.system or (SCAFFOLD_SYSTEM if args.scaffold else None)
    correct = graded = searched_when_needed = need = over_searched = total_calls = 0
    print(f"# {args.model} @ {args.base_url}  max_calls={args.max_calls}  no_think={args.no_think}  scaffold={bool(system)}\n")
    for case in DATASET:
        started = time.monotonic()
        out = run_query(args.base_url, args.model, case["q"], args.max_calls, args.no_think, system)
        secs = time.monotonic() - started
        ans = out["answer"].lower()
        total_calls += out["calls"]
        if case["needs_search"]:
            need += 1
            searched_when_needed += int(out["calls"] > 0)
        else:
            over_searched += int(out["calls"] > 0)
        graded_this = case["answer"] is not None
        ok = graded_this and bool(re.search(case["answer"], ans))
        graded += int(graded_this)
        correct += int(ok)
        mark = "OK " if ok else ("-- " if graded_this else "?? ")
        print(f"{mark} calls={out['calls']} {secs:4.0f}s | {case['q'][:44]:44} -> {out['answer'][:70].strip()!r}")

    print(f"\naccuracy (gradeable): {correct}/{graded}"
          f"   search-recall(needed): {searched_when_needed}/{need}"
          f"   over-search(known): {over_searched}/{len(DATASET)-need}"
          f"   avg_calls: {total_calls/len(DATASET):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
