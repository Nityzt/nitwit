#!/usr/bin/env python3
"""Executable coding benchmark (pass@1) — the real target for SWE/agentic optimization.

Unlike the verifier grid (judgment on canned text), this measures whether a model can
WRITE CORRECT CODE: generate a solution, run it against hidden tests in a sandboxed
subprocess, pass/fail by execution. That's the metric that matters for a coding/synthesis
model. Prompts are short (problem statements) and output is generation-heavy — GPU-safe
(generation doesn't spike the RX 580; prefill does, and these prefills are tiny).

  python3 -m bench.coding_eval --base-url http://127.0.0.1:8080 --model qwen2.5-coder-7b
  python3 -m bench.coding_eval --base-url http://127.0.0.1:8084 --model probe --think

Each problem: an instruction + a required entry-point function; the model's fenced code is
executed with hidden asserts (edge cases included) under a wall-clock timeout.
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

SYSTEM = ("You are an expert Python engineer. Implement exactly what is asked. "
          "Return ONE self-contained ```python code block with the required function(s) and "
          "any imports. No explanation, no tests, no example calls.")

# Each case: name, instruction (what to build), entry (must be defined), tests (assert body).
# Tests include edge cases so a sloppy solution fails. `extra` seeds buggy code for fix tasks.
PROBLEMS: list[dict] = [
    {"name": "two_sum", "entry": "two_sum",
     "instruction": "Write `two_sum(nums, target)` returning indices [i, j] (i<j) of the two "
                    "distinct entries summing to target, or None if none exist.",
     "tests": "assert sorted(two_sum([2,7,11,15],9))==[0,1]\nassert two_sum([3,3],6)==[0,1]\n"
              "assert two_sum([1,2,3],7) is None\nassert two_sum([],1) is None"},
    {"name": "valid_parens", "entry": "is_valid",
     "instruction": "Write `is_valid(s)` returning True iff the string of brackets ()[]{} is "
                    "correctly matched and nested.",
     "tests": "assert is_valid('()[]{}')\nassert is_valid('([{}])')\nassert not is_valid('(]')\n"
              "assert not is_valid('([)]')\nassert is_valid('')\nassert not is_valid('(')"},
    {"name": "merge_sorted", "entry": "merge",
     "instruction": "Write `merge(a, b)` merging two ascending lists into one ascending list.",
     "tests": "assert merge([1,3,5],[2,4,6])==[1,2,3,4,5,6]\nassert merge([],[1])==[1]\n"
              "assert merge([1,1],[1])==[1,1,1]\nassert merge([],[])==[]"},
    {"name": "roman", "entry": "to_int",
     "instruction": "Write `to_int(s)` converting a valid Roman numeral (I,V,X,L,C,D,M) to int.",
     "tests": "assert to_int('III')==3\nassert to_int('IV')==4\nassert to_int('IX')==9\n"
              "assert to_int('LVIII')==58\nassert to_int('MCMXCIV')==1994"},
    {"name": "lcp", "entry": "longest_common_prefix",
     "instruction": "Write `longest_common_prefix(strs)` returning the longest common prefix of a "
                    "list of strings, '' if none.",
     "tests": "assert longest_common_prefix(['flower','flow','flight'])=='fl'\n"
              "assert longest_common_prefix(['dog','car'])==''\nassert longest_common_prefix(['a'])=='a'\n"
              "assert longest_common_prefix([])==''"},
    {"name": "binsearch", "entry": "search",
     "instruction": "Write `search(nums, target)` doing binary search on an ascending list, "
                    "returning the index or -1.",
     "tests": "assert search([-1,0,3,5,9,12],9)==4\nassert search([-1,0,3,5,9,12],2)==-1\n"
              "assert search([],1)==-1\nassert search([5],5)==0"},
    {"name": "anagram_groups", "entry": "group_anagrams",
     "instruction": "Write `group_anagrams(words)` grouping words that are anagrams. Return a list "
                    "of groups; order within/among groups does not matter.",
     "tests": "r=group_anagrams(['eat','tea','tan','ate','nat','bat'])\n"
              "s=sorted(sorted(g) for g in r)\n"
              "assert s==sorted([sorted(x) for x in [['ate','eat','tea'],['nat','tan'],['bat']]])"},
    {"name": "merge_intervals", "entry": "merge_intervals",
     "instruction": "Write `merge_intervals(intervals)` merging overlapping [start,end] intervals, "
                    "returning them sorted by start.",
     "tests": "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\n"
              "assert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]\n"
              "assert merge_intervals([[1,4],[0,4]])==[[0,4]]"},
    {"name": "word_count", "entry": "top_k_frequent",
     "instruction": "Write `top_k_frequent(words, k)` returning the k most frequent words, ties "
                    "broken alphabetically, most frequent first.",
     "tests": "assert top_k_frequent(['i','love','leetcode','i','love','coding'],2)==['i','love']\n"
              "assert top_k_frequent(['the','day','is','sunny','the','the','the','sunny','is','is'],4)"
              "==['the','is','sunny','day']"},
    {"name": "lru_cache", "entry": "LRUCache",
     "instruction": "Implement class `LRUCache` with `__init__(self, capacity)`, `get(self, key)` "
                    "(returns value or -1) and `put(self, key, value)`, evicting the least-recently-used "
                    "key when over capacity. get and put both count as uses.",
     "tests": "c=LRUCache(2)\nc.put(1,1)\nc.put(2,2)\nassert c.get(1)==1\nc.put(3,3)\n"
              "assert c.get(2)==-1\nc.put(4,4)\nassert c.get(1)==-1\nassert c.get(3)==3\nassert c.get(4)==4"},
    {"name": "fix_bug", "entry": "is_palindrome",
     "instruction": "The following function is meant to test whether a string is a palindrome "
                    "considering only alphanumeric characters and ignoring case, but it is buggy. "
                    "Return a corrected version.\n\n"
                    "def is_palindrome(s):\n    s = [c for c in s if c.isalpha()]\n"
                    "    return s == s[::-1]",
     "tests": "assert is_palindrome('A man, a plan, a canal: Panama')\nassert is_palindrome('0P')==False\n"
              "assert is_palindrome('  ')\nassert is_palindrome('ab_a')"},
    {"name": "edit_distance", "entry": "min_distance",
     "instruction": "Write `min_distance(a, b)` returning the Levenshtein edit distance "
                    "(insert/delete/replace) between two strings.",
     "tests": "assert min_distance('horse','ros')==3\nassert min_distance('intention','execution')==5\n"
              "assert min_distance('','abc')==3\nassert min_distance('same','same')==0"},
]


# Harder set — DP, graphs, stack parsing, a stateful O(1) class. Separates strong coders.
HARD_PROBLEMS: list[dict] = [
    {"name": "coin_change", "entry": "coin_change",
     "instruction": "Write `coin_change(coins, amount)` returning the fewest coins summing to "
                    "amount, or -1 if impossible.",
     "tests": "assert coin_change([1,2,5],11)==3\nassert coin_change([2],3)==-1\n"
              "assert coin_change([1],0)==0\nassert coin_change([186,419,83,408],6249)==20"},
    {"name": "lis", "entry": "length_of_lis",
     "instruction": "Write `length_of_lis(nums)` returning the length of the longest strictly "
                    "increasing subsequence.",
     "tests": "assert length_of_lis([10,9,2,5,3,7,101,18])==4\nassert length_of_lis([0,1,0,3,2,3])==4\n"
              "assert length_of_lis([7,7,7,7])==1\nassert length_of_lis([])==0"},
    {"name": "word_break", "entry": "word_break",
     "instruction": "Write `word_break(s, words)` returning True iff s can be segmented into a "
                    "space-separated sequence of one or more words from the list `words`.",
     "tests": "assert word_break('leetcode',['leet','code'])\nassert word_break('applepenapple',['apple','pen'])\n"
              "assert not word_break('catsandog',['cats','dog','sand','and','cat'])"},
    {"name": "decode_string", "entry": "decode_string",
     "instruction": "Write `decode_string(s)` decoding strings like '3[a2[c]]' -> 'accaccacc' "
                    "(k[encoded] repeats encoded k times; may nest).",
     "tests": "assert decode_string('3[a]2[bc]')=='aaabcbc'\nassert decode_string('3[a2[c]]')=='accaccacc'\n"
              "assert decode_string('2[abc]3[cd]ef')=='abcabccdcdcdef'"},
    {"name": "course_schedule", "entry": "can_finish",
     "instruction": "Write `can_finish(num_courses, prerequisites)` where prerequisites[i]=[a,b] "
                    "means b must precede a. Return True iff all courses can be finished (no cycle).",
     "tests": "assert can_finish(2,[[1,0]])\nassert not can_finish(2,[[1,0],[0,1]])\n"
              "assert can_finish(3,[[1,0],[2,1]])\nassert not can_finish(3,[[0,1],[1,2],[2,0]])"},
    {"name": "trap_water", "entry": "trap",
     "instruction": "Write `trap(height)` returning units of rain water trapped between the bars.",
     "tests": "assert trap([0,1,0,2,1,0,1,3,2,1,2,1])==6\nassert trap([4,2,0,3,2,5])==9\n"
              "assert trap([])==0\nassert trap([1,2,3])==0"},
    {"name": "min_stack", "entry": "MinStack",
     "instruction": "Implement class `MinStack` with push(x), pop(), top() and get_min(), all O(1), "
                    "returning min of current elements from get_min().",
     "tests": "s=MinStack()\ns.push(-2)\ns.push(0)\ns.push(-3)\nassert s.get_min()==-3\ns.pop()\n"
              "assert s.top()==0\nassert s.get_min()==-2"},
    {"name": "kth_largest", "entry": "find_kth_largest",
     "instruction": "Write `find_kth_largest(nums, k)` returning the kth largest element (kth in "
                    "sorted-descending order, duplicates counted).",
     "tests": "assert find_kth_largest([3,2,1,5,6,4],2)==5\nassert find_kth_largest([3,2,3,1,2,4,5,5,6],4)==4\n"
              "assert find_kth_largest([1],1)==1"},
]


def chat(base_url, model, problem, think):
    payload = {"model": model, "temperature": 0.0, "max_tokens": 1400,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": problem["instruction"]}]}
    if think is True:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
        payload["max_tokens"] = 3000
    elif think is False:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(f"{base_url}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as res:
        data = json.load(res)
    msg = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage") or {}
    return msg, usage.get("completion_tokens")


def extract_code(text: str) -> str:
    # strip a <think> block if present, then take the last fenced python block, else raw.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return (blocks[-1] if blocks else text).strip()


def run_case(code: str, tests: str, timeout: int = 12) -> tuple[bool, str]:
    prog = code + "\n\n" + textwrap.dedent(tests) + "\nprint('__OK__')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as fh:
        fh.write(prog); fh.flush()
        try:
            r = subprocess.run([sys.executable, "-I", fh.name], capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if "__OK__" in r.stdout:
        return True, ""
    err = (r.stderr.strip().splitlines() or ["(no output)"])[-1]
    return False, err[:80]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen2.5-coder-7b")
    ap.add_argument("--think", dest="think", action="store_true", help="enable_thinking=true")
    ap.add_argument("--no-think", dest="think", action="store_false", help="enable_thinking=false")
    ap.add_argument("--set", choices=["easy", "hard", "all"], default="easy")
    ap.set_defaults(think=None)
    args = ap.parse_args()

    problems = {"easy": PROBLEMS, "hard": HARD_PROBLEMS, "all": PROBLEMS + HARD_PROBLEMS}[args.set]
    passed = 0
    total_lat = 0.0
    total_tok = 0
    print(f"# {args.model} @ {args.base_url}  think={args.think}  set={args.set}\n")
    for p in problems:
        t0 = time.monotonic()
        try:
            text, tok = chat(args.base_url, args.model, p, args.think)
        except Exception as exc:
            print(f"FAIL  {p['name']:16} gen-error: {str(exc)[:50]}"); continue
        secs = time.monotonic() - t0
        total_lat += secs; total_tok += tok or 0
        ok, why = run_case(extract_code(text), p["tests"])
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {p['name']:16} {secs:5.1f}s tok={tok or '?':>4}  {'' if ok else why}")
    n = len(problems)
    print(f"\npass@1: {passed}/{n} ({100*passed/n:.0f}%)   avg_latency={total_lat/n:.1f}s   avg_tokens={total_tok//n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
