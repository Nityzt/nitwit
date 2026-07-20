#!/usr/bin/env python3
"""Backend stack benchmark — FastAPI + pydantic v2, executed via TestClient.

Targets the user's backend: FastAPI, REST, pydantic validation. The model returns a complete
FastAPI app bound to `app`; hidden tests drive it with starlette's TestClient (real request/
response cycle) and assert status codes + JSON. Correctness by execution.

Runs in a uv venv that has fastapi/httpx/pydantic (set --python to it):
  python3 -m bench.api_eval --model qwen2.5-coder-7b \
      --python /tmp/.../scratchpad/fastapi-venv/bin/python
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request

SYSTEM = ("You are an expert FastAPI engineer. Return a COMPLETE, runnable FastAPI application "
          "assigned to a module-level variable named `app` (pydantic v2). Include all imports. "
          "Return ONE ```python code block, no explanation, no `if __name__` block, no uvicorn call.")

PROBLEMS: list[dict] = [
    {"name": "item_response_model",
     "instruction": "FastAPI app with GET /items/{item_id} returning JSON {\"id\": item_id, \"name\": "
                    "\"item-<id>\"} using a pydantic response_model with fields id:int and name:str.",
     "tests": "r=c.get('/items/7'); assert r.status_code==200, r.status_code\n"
              "assert r.json()=={'id':7,'name':'item-7'}, r.json()"},
    {"name": "create_user_validation",
     "instruction": "FastAPI app with POST /users accepting JSON {email:str, age:int}. age must be >= 0 "
                    "(use pydantic validation). On success return the same data with status 201.",
     "tests": "r=c.post('/users',json={'email':'a@b.com','age':30}); assert r.status_code==201, r.status_code\n"
              "assert r.json()['email']=='a@b.com'\n"
              "r2=c.post('/users',json={'email':'x','age':-5}); assert r2.status_code==422, r2.status_code"},
    {"name": "query_pagination",
     "instruction": "FastAPI app with GET /search taking query params q:str (required) and limit:int=10. "
                    "Return {\"q\": q, \"limit\": limit}. Missing q must yield 422.",
     "tests": "r=c.get('/search',params={'q':'hi','limit':5}); assert r.json()=={'q':'hi','limit':5}, r.json()\n"
              "assert c.get('/search').status_code==422\n"
              "assert c.get('/search',params={'q':'x'}).json()['limit']==10"},
    {"name": "not_found_httpexception",
     "instruction": "FastAPI app with an in-memory dict of users {1:'alice'}. GET /users/{uid} returns "
                    "{\"name\": ...} if present, else raises HTTPException 404 with detail 'not found'.",
     "tests": "assert c.get('/users/1').json()=={'name':'alice'}\n"
              "r=c.get('/users/99'); assert r.status_code==404, r.status_code; assert r.json()['detail']=='not found'"},
    {"name": "dependency_auth",
     "instruction": "FastAPI app with GET /me that requires header 'x-token'. Use a dependency that raises "
                    "HTTPException 401 if the header is missing or != 'secret'; otherwise return {\"ok\": true}.",
     "tests": "assert c.get('/me',headers={'x-token':'secret'}).json()=={'ok':True}\n"
              "assert c.get('/me').status_code==401\n"
              "assert c.get('/me',headers={'x-token':'nope'}).status_code==401"},
]

HARNESS = """
import sys, json
try:
    from starlette.testclient import TestClient
    c = TestClient(app)
{TESTS}
    print("__OK__")
except Exception as e:
    import traceback; traceback.print_exc()
    sys.exit(1)
"""


def chat(base_url, model, instruction, think):
    payload = {"model": model, "temperature": 0.0, "max_tokens": 1400,
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}]}
    if think is True:
        payload["chat_template_kwargs"] = {"enable_thinking": True}; payload["max_tokens"] = 3000
    elif think is False:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as res:
        return json.load(res)["choices"][0]["message"]["content"] or ""


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return (blocks[-1] if blocks else text).strip()


def run_case(python: str, code: str, tests: str, timeout: int = 25):
    indented = "\n".join("    " + ln for ln in tests.splitlines())
    prog = code + "\n" + HARNESS.replace("{TESTS}", indented)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as fh:
        fh.write(prog); fh.flush()
        try:
            r = subprocess.run([python, fh.name], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if "__OK__" in r.stdout:
        return True, ""
    err = (r.stderr.strip().splitlines() or ["(no output)"])
    line = next((l for l in reversed(err) if "Error" in l or "assert" in l.lower()), err[-1])
    return False, line[:90]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen2.5-coder-7b")
    ap.add_argument("--python", required=True, help="path to a venv python with fastapi/httpx/pydantic")
    ap.add_argument("--think", dest="think", action="store_true")
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.set_defaults(think=None)
    args = ap.parse_args()

    passed = 0; total_lat = 0.0
    print(f"# FastAPI stack — {args.model} @ {args.base_url}  think={args.think}\n")
    for p in PROBLEMS:
        t0 = time.monotonic()
        try:
            text = chat(args.base_url, args.model, p["instruction"], args.think)
        except Exception as exc:
            print(f"FAIL  {p['name']:24} gen-error {str(exc)[:40]}"); continue
        secs = time.monotonic() - t0; total_lat += secs
        ok, why = run_case(args.python, extract_code(text), p["tests"])
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {p['name']:24} {secs:5.1f}s  {'' if ok else why}")
    n = len(PROBLEMS)
    print(f"\nFastAPI pass@1: {passed}/{n} ({100*passed//n}%)   avg_latency={total_lat/n:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
