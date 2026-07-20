#!/usr/bin/env python3
"""Stack-relevant executable coding benchmark — TypeScript (the user's primary language).

coding_eval.py is Python algorithms; this targets the actual stack: TypeScript, the shared
language of React / Next.js / Astro / React Native / Expo, plus backend TS patterns. Each task
is a real frontend/backend shape (data massaging, a reducer, async retry, pagination, snake->
camel from a Supabase/Postgres row, query-string parsing). The model's TS is executed with
Node 24's native type-stripping (`node file.ts`) against hidden asserts — correctness by
execution, not vibes.

  python3 -m bench.stack_eval --base-url http://127.0.0.1:8080 --model qwen2.5-coder-7b
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
import urllib.request

SYSTEM = ("You are an expert TypeScript engineer. Implement exactly what is asked as self-contained "
          "TypeScript (no imports, no external packages, no `export`). Return ONE ```typescript code "
          "block with the required function(s)/types. No explanation, no tests.")

ASSERT = ('function assert(c:any,m?:string){ if(!c) throw new Error(m||"assert failed"); }\n'
          'function eq(a:any,b:any){ return JSON.stringify(a)===JSON.stringify(b); }\n')

# Real stack shapes. `tests` is TS run after the model's code; must end clean for __OK__.
PROBLEMS: list[dict] = [
    {"name": "groupBy_generic",
     "instruction": "Write a generic `groupBy<T>(arr: T[], key: (x: T) => string): Record<string, T[]>` "
                    "that groups array items by the string key, preserving order within each group.",
     "tests": "const r=groupBy([{t:'a',v:1},{t:'b',v:2},{t:'a',v:3}],x=>x.t);\n"
              "assert(eq(r,{a:[{t:'a',v:1},{t:'a',v:3}],b:[{t:'b',v:2}]}));\n"
              "assert(eq(groupBy([],()=>'x'),{}));"},
    {"name": "reducer_counter",
     "instruction": "Write a React-style reducer `reducer(state: number, action: {type:'inc'|'dec'|'set', "
                    "payload?: number}): number`. 'inc' adds 1, 'dec' subtracts 1, 'set' sets to payload.",
     "tests": "let s=0;s=reducer(s,{type:'inc'});s=reducer(s,{type:'inc'});s=reducer(s,{type:'dec'});\n"
              "assert(s===1,'got '+s);\ns=reducer(s,{type:'set',payload:10});assert(s===10);"},
    {"name": "snake_to_camel_keys",
     "instruction": "Write `camelKeys(row: Record<string, any>): Record<string, any>` converting snake_case "
                    "keys to camelCase (one level, e.g. created_at -> createdAt, user_id -> userId).",
     "tests": "assert(eq(camelKeys({user_id:1,created_at:'x',name:'a'}),{userId:1,createdAt:'x',name:'a'}));\n"
              "assert(eq(camelKeys({}),{}));"},
    {"name": "parse_query_string",
     "instruction": "Write `parseQuery(qs: string): Record<string,string>` parsing a URL query string "
                    "(may start with '?') into an object, URL-decoding values. Later duplicates win.",
     "tests": "assert(eq(parseQuery('?a=1&b=two&a=3'),{a:'3',b:'two'}));\n"
              "assert(eq(parseQuery('x=hello%20world'),{x:'hello world'}));\nassert(eq(parseQuery(''),{}));"},
    {"name": "paginate",
     "instruction": "Write `paginate<T>(items: T[], page: number, size: number): T[]` returning the 1-indexed "
                    "page slice (page 1 = first `size` items). Out-of-range pages return [].",
     "tests": "assert(eq(paginate([1,2,3,4,5],1,2),[1,2]));\nassert(eq(paginate([1,2,3,4,5],3,2),[5]));\n"
              "assert(eq(paginate([1,2,3,4,5],9,2),[]));"},
    {"name": "deep_merge",
     "instruction": "Write `deepMerge(a: any, b: any): any` deeply merging plain objects (b wins on "
                    "conflicts), recursing into nested objects but replacing arrays and primitives.",
     "tests": "assert(eq(deepMerge({a:1,n:{x:1,y:2}},{n:{y:3,z:4},b:2}),{a:1,n:{x:1,y:3,z:4},b:2}));\n"
              "assert(eq(deepMerge({a:[1,2]},{a:[3]}),{a:[3]}));"},
    {"name": "retry_async",
     "instruction": "Write `async function retry<T>(fn: () => Promise<T>, tries: number): Promise<T>` that "
                    "calls fn, retrying on rejection up to `tries` total attempts, throwing the last error.",
     "tests": "let n=0;const flaky=async()=>{n++;if(n<3)throw new Error('x');return 42;};\n"
              "retry(flaky,5).then(v=>{assert(v===42&&n===3,'v='+v+' n='+n);console.log('__OK__');})"
              ".catch(e=>{throw e;});RANDOM_ASYNC_MARK"},
    {"name": "debounce_calls",
     "instruction": "Write `chunk<T>(arr: T[], size: number): T[][]` splitting an array into consecutive "
                    "chunks of at most `size` (last chunk may be shorter; size<=0 returns []).",
     "tests": "assert(eq(chunk([1,2,3,4,5],2),[[1,2],[3,4],[5]]));\nassert(eq(chunk([],3),[]));\n"
              "assert(eq(chunk([1,2],0),[]));"},
]


def chat(base_url, model, instruction, think):
    payload = {"model": model, "temperature": 0.0, "max_tokens": 1200,
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}]}
    if think is True:
        payload["chat_template_kwargs"] = {"enable_thinking": True}; payload["max_tokens"] = 3000
    elif think is False:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as res:
        data = json.load(res)
    return data["choices"][0]["message"]["content"] or "", (data.get("usage") or {}).get("completion_tokens")


def extract_ts(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    blocks = re.findall(r"```(?:typescript|ts|tsx|javascript|js)?\s*\n(.*?)```", text, flags=re.DOTALL)
    code = (blocks[-1] if blocks else text).strip()
    return re.sub(r"^\s*export\s+", "", code, flags=re.MULTILINE)  # strip stray exports


def run_ts(code: str, tests: str, timeout: int = 20):
    # async task prints __OK__ itself; sync tasks get a trailing marker.
    if "RANDOM_ASYNC_MARK" in tests:
        body = code + "\n" + ASSERT + tests.replace("RANDOM_ASYNC_MARK", "")
    else:
        body = code + "\n" + ASSERT + tests + '\nconsole.log("__OK__");\n'
    with tempfile.NamedTemporaryFile("w", suffix=".ts", delete=True) as fh:
        fh.write(body); fh.flush()
        try:
            r = subprocess.run(["node", "--experimental-strip-types", "--no-warnings", fh.name],
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if "__OK__" in r.stdout:
        return True, ""
    err = (r.stderr.strip().splitlines() or ["(no output)"])
    line = next((l for l in err if "Error" in l or "assert" in l), err[-1])
    return False, line[:90]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen2.5-coder-7b")
    ap.add_argument("--think", dest="think", action="store_true")
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.set_defaults(think=None)
    args = ap.parse_args()

    passed = 0; total_lat = 0.0
    print(f"# TypeScript stack — {args.model} @ {args.base_url}  think={args.think}\n")
    for p in PROBLEMS:
        t0 = time.monotonic()
        try:
            text, _ = chat(args.base_url, args.model, p["instruction"], args.think)
        except Exception as exc:
            print(f"FAIL  {p['name']:20} gen-error {str(exc)[:40]}"); continue
        secs = time.monotonic() - t0; total_lat += secs
        ok, why = run_ts(extract_ts(text), p["tests"])
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {p['name']:20} {secs:5.1f}s  {'' if ok else why}")
    n = len(PROBLEMS)
    print(f"\nTS pass@1: {passed}/{n} ({100*passed//n}%)   avg_latency={total_lat/n:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
