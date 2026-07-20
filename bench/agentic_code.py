#!/usr/bin/env python3
"""Agentic coding benchmark — iterative test-driven repair, with optional web search.

Single-shot pass@1 (coding_eval.py) measures raw code-gen. This measures the *agentic*
SWE loop: the model is given a failing function + the test error and must use that feedback
to converge on a fix over several rounds. With --web it is ALSO handed a web_search tool so
it can look up docs / StackOverflow when the error alone isn't enough — testing "smart web
tool use": does it search only when genuinely stuck, or spam it (the over-search failure the
1B showed on facts)?

Loop per task (multi-turn, message history preserved): show buggy code + latest failure ->
model EITHER calls web_search (docs/SO) OR returns a corrected function -> run hidden tests ->
pass = solved; else feed the new failure back. Bounded by --max-rounds (repair attempts) and
--max-searches. Bounded => GPU-safe (crash analysis proved a bounded loop holds).

  python3 -m bench.agentic_code --base-url http://127.0.0.1:8080 --model qwen2.5-coder-7b --max-rounds 4
  python3 -m bench.agentic_code --base-url http://127.0.0.1:8085 --model probe --web --max-searches 2
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request

from webui import run_capability

SYSTEM_BASE = ("You are a senior engineer fixing a failing function. You are given the current code and "
               "the test failure it produces. Return the COMPLETE corrected function in ONE ```python code "
               "block — no prose outside the block. Read the failure carefully and fix the actual bug.")
SYSTEM_WEB = (" You also have a web_search tool (docs, StackOverflow). Use it ONLY when the failure alone is "
              "not enough to fix the bug — for a clear error, just fix it directly; do not search needlessly.")

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web (documentation, StackOverflow) for help fixing the bug. Only when stuck.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
}

TASKS: list[dict] = [
    {"name": "binary_search_offbyone",
     "buggy": "def search(nums, target):\n    lo, hi = 0, len(nums)\n    while lo < hi:\n        mid = (lo+hi)//2\n        if nums[mid] == target: return mid\n        if nums[mid] < target: lo = mid\n        else: hi = mid\n    return -1",
     "tests": "assert search([1,2,3,4,5],4)==3\nassert search([1,2,3,4,5],1)==0\nassert search([1,3,5],2)==-1\nassert search([],1)==-1"},
    {"name": "flatten_depth",
     "buggy": "def flatten(lst):\n    out = []\n    for x in lst:\n        if isinstance(x, list): out += x\n        else: out.append(x)\n    return out",
     "tests": "assert flatten([1,[2,[3,[4]]]])==[1,2,3,4]\nassert flatten([1,2,3])==[1,2,3]\nassert flatten([])==[]\nassert flatten([[1],[2,[3]]])==[1,2,3]"},
    {"name": "running_median_bug",
     "buggy": "def medians(stream):\n    seen = []\n    res = []\n    for x in stream:\n        seen.append(x)\n        seen.sort()\n        n = len(seen)\n        res.append(seen[n//2])\n    return res",
     "tests": "assert medians([1])==[1]\nassert medians([2,1,3])==[2,1.5,2]\nassert medians([1,2,3,4])==[1,1.5,2,2.5]"},
    {"name": "romanize_bug",
     "buggy": "def to_roman(n):\n    vals=[(1000,'M'),(500,'D'),(100,'C'),(50,'L'),(10,'X'),(5,'V'),(1,'I')]\n    s=''\n    for v,sym in vals:\n        while n>=v:\n            s+=sym; n-=v\n    return s",
     "tests": "assert to_roman(4)=='IV'\nassert to_roman(9)=='IX'\nassert to_roman(58)=='LVIII'\nassert to_roman(1994)=='MCMXCIV'"},
    {"name": "dedupe_stable",
     "buggy": "def dedupe(xs):\n    return list(set(xs))",
     "tests": "assert dedupe([3,1,3,2,1])==[3,1,2]\nassert dedupe([])==[]\nassert dedupe([1,1,1])==[1]"},
    {"name": "parse_ints_negatives",
     "buggy": "def parse_ints(s):\n    import re\n    return [int(x) for x in re.findall(r'\\d+', s)]",
     "tests": "assert parse_ints('a1 b-2 c3')==[1,-2,3]\nassert parse_ints('-5 to 5')==[-5,5]\nassert parse_ints('none')==[]"},
    # Classic Python gotcha — mutable default argument shared across calls.
    {"name": "mutable_default",
     "buggy": "def collect(x, acc=[]):\n    acc.append(x)\n    return acc",
     "tests": "assert collect(1)==[1]\nassert collect(2)==[2]\nassert collect(3, [0])==[0,3]\nassert collect(4)==[4]"},
    # Rewards knowing strptime directives (a docs-lookup candidate).
    {"name": "date_parse_fmt",
     "buggy": "def to_iso(s):\n    from datetime import datetime\n    return datetime.strptime(s, '%d-%m-%y').strftime('%Y-%m-%d')",
     "tests": "assert to_iso('2023-07-04')=='2023-07-04'\nassert to_iso('1999-12-31')=='1999-12-31'"},
]


def chat(base_url, model, messages, tools, think, max_tokens=900):
    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    if tools:
        payload["tools"] = tools
    if think is True:
        payload["chat_template_kwargs"] = {"enable_thinking": True}; payload["max_tokens"] = 2500
    elif think is False:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(f"{base_url}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as res:
        return json.load(res)["choices"][0]["message"]


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return (blocks[-1] if blocks else "").strip()


def web_summary(query: str) -> str:
    try:
        run = run_capability("web_search", {"query": query, "limit": 4})
        results = (run.get("result") or {}).get("results") or []
    except Exception:
        results = []
    lines = [f"- {r.get('title','')}: {r.get('snippet','')}" for r in results[:4]]
    return "WEB SEARCH RESULTS:\n" + ("\n".join(lines)[:1100] or "(no results)")


def content_query(content: str):
    if not content or "web_search" not in content:
        return None
    m = re.search(r'"query"\s*:\s*"([^"]+)"', content)
    return m.group(1) if m else None


def run_tests(code: str, tests: str, timeout: int = 12):
    if not code:
        return False, "no code submitted"
    prog = code + "\n\n" + textwrap.dedent(tests) + "\nprint('__OK__')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as fh:
        fh.write(prog); fh.flush()
        try:
            r = subprocess.run([sys.executable, "-I", fh.name], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "TimeoutError"
    if "__OK__" in r.stdout:
        return True, ""
    return False, (r.stderr.strip() or "no output")[-400:]


def solve(base_url, model, task, max_rounds, max_searches, think, web) -> dict:
    ok, err = run_tests(task["buggy"], task["tests"])
    if ok:
        return {"solved": True, "rounds": 0, "searches": 0}
    sys_prompt = SYSTEM_BASE + (SYSTEM_WEB if web else "")
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Current code:\n```python\n{task['buggy']}\n```\n\n"
                                             f"Tests fail with:\n```\n{err}\n```\nFix it."}]
    tools = [WEB_SEARCH_TOOL] if web else None
    code = task["buggy"]; rounds = 0; searches = 0
    for _ in range(max_rounds + max_searches + 2):
        msg = chat(base_url, model, messages, tools, think)
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        # --- did it choose to search? ---
        query = None; structured = False
        if tool_calls:
            structured = True
            try:
                query = json.loads(tool_calls[0]["function"]["arguments"] or "{}").get("query")
            except Exception:
                query = None
        elif web:
            query = content_query(content)
        if query and searches < max_searches:
            searches += 1
            if structured:
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                messages.append({"role": "tool", "tool_call_id": tool_calls[0].get("id", "0"),
                                 "content": web_summary(query)})
            else:
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": web_summary(query) +
                                 "\n\nNow return the complete corrected function."})
            continue
        # --- otherwise treat as a code submission ---
        newcode = extract_code(content)
        if newcode:
            code = newcode
        ok, err = run_tests(code, task["tests"])
        rounds += 1
        if ok:
            return {"solved": True, "rounds": rounds, "searches": searches}
        if rounds >= max_rounds:
            break
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Still failing:\n```\n{err}\n```\nFix it."})
    return {"solved": False, "rounds": rounds, "searches": searches, "err": (err or "").splitlines()[-1][:60] if err else ""}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen2.5-coder-7b")
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--max-searches", type=int, default=2)
    ap.add_argument("--web", action="store_true", help="offer a web_search tool (docs/StackOverflow)")
    ap.add_argument("--think", dest="think", action="store_true")
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.set_defaults(think=None)
    args = ap.parse_args()

    solved = total_rounds = total_searches = 0; t0 = time.monotonic()
    print(f"# agentic repair — {args.model} @ {args.base_url}  rounds<={args.max_rounds} "
          f"web={args.web}(<= {args.max_searches})  think={args.think}\n")
    for task in TASKS:
        out = solve(args.base_url, args.model, task, args.max_rounds, args.max_searches, args.think, args.web)
        solved += int(out["solved"]); total_rounds += out["rounds"]; total_searches += out["searches"]
        mark = "SOLVED" if out["solved"] else "FAIL  "
        print(f"{mark} {task['name']:24} rounds={out['rounds']} searches={out['searches']}  {out.get('err','')}")
    n = len(TASKS)
    print(f"\nsolved@{args.max_rounds}: {solved}/{n} ({100*solved//n}%)   avg_rounds={total_rounds/n:.1f}"
          f"   searches={total_searches}   total={time.monotonic()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
