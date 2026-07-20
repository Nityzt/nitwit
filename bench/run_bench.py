#!/usr/bin/env python3
"""Objective per-stage benchmark for the orchestrator.

Runs each utility stage over the labeled dataset and reports accuracy + telemetry
(latency, prompt/completion tokens) so we can compare providers (Qwen-GPU vs a
MiniCPM-CPU utility model) on identical inputs and adopt per-stage by the numbers.

  python3 -m bench.run_bench                      # heuristic stages only (no model)
  python3 -m bench.run_bench --model-stages       # + planner/verifier on the default server
  python3 -m bench.run_bench --model-stages --base-url http://127.0.0.1:8081 --model minicpm5-1b
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

from bench import dataset
from orchestrator import Orchestrator, OpenAICompatibleClient, Subtask, WorkerResult
from webui import (
    classify_request,
    extract_memory_suggestions_from_text,
    extract_tool_requests_from_text,
    web_search_query_from_request,
)


class RecordingClient:
    """Wraps a model client and remembers the last call's usage/timings, so the
    harness can read per-stage tokens/latency without touching Orchestrator."""

    def __init__(self, inner, strip_format: bool = False) -> None:
        self.inner = inner
        self.last: dict = {}
        # Thinking models can't use the json_schema grammar (it blocks the <think>
        # block), so when reasoning is on we drop the grammar and lean on extract_json.
        self.strip_format = strip_format

    def chat(self, messages, *, temperature, max_tokens, response_format=None):
        if self.strip_format:
            response_format = None
        r = self.inner.chat(messages, temperature=temperature, max_tokens=max_tokens, response_format=response_format)
        usage = r.usage or {}
        self.last = {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
        return r

    def stream_chat(self, *args, **kwargs):
        return self.inner.stream_chat(*args, **kwargs)


def _pct(n_correct: int, n: int) -> float:
    return round(100.0 * n_correct / n, 1) if n else 0.0


def bench_routing() -> dict:
    correct = 0
    for case in dataset.ROUTING_CASES:
        got = classify_request(case["prompt"], "").get("mode")
        correct += int(got == case["mode"])
    return {"stage": "routing (heuristic)", "n": len(dataset.ROUTING_CASES),
            "accuracy_pct": _pct(correct, len(dataset.ROUTING_CASES))}


def bench_query_rewrite() -> dict:
    correct = 0
    for case in dataset.QUERY_REWRITE_CASES:
        q = web_search_query_from_request(case["request"]).lower()
        ok = all(t in q for t in case["must_include"]) and not any(t in q.split() for t in case["must_exclude"])
        correct += int(ok)
    return {"stage": "query-rewrite (heuristic)", "n": len(dataset.QUERY_REWRITE_CASES),
            "accuracy_pct": _pct(correct, len(dataset.QUERY_REWRITE_CASES))}


def bench_memory_extraction() -> dict:
    correct = 0
    for case in dataset.MEMORY_CASES:
        got = extract_memory_suggestions_from_text(case["answer"])
        key = got[0]["key"] if got else None
        correct += int(key == case["expect_key"])
    return {"stage": "memory-extract", "n": len(dataset.MEMORY_CASES),
            "accuracy_pct": _pct(correct, len(dataset.MEMORY_CASES))}


def bench_tool_extraction() -> dict:
    correct = 0
    for case in dataset.TOOL_CASES:
        got = extract_tool_requests_from_text(case["answer"])
        cap = got[0]["capability"] if got else None
        correct += int(cap == case["expect_capability"])
    return {"stage": "tool-extract", "n": len(dataset.TOOL_CASES),
            "accuracy_pct": _pct(correct, len(dataset.TOOL_CASES))}


def _telemetry(rows: list[dict]) -> dict:
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    ptok = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
    ctok = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
    return {
        "avg_latency_ms": round(statistics.mean(lat), 0) if lat else None,
        "avg_prompt_tokens": round(statistics.mean(ptok), 0) if ptok else None,
        "avg_completion_tokens": round(statistics.mean(ctok), 0) if ctok else None,
    }


def bench_planner(orch: Orchestrator) -> dict:
    prompts = [c["prompt"] for c in dataset.ROUTING_CASES if c["mode"] == "plan"]
    rows, valid = [], 0
    for prompt in prompts:
        started = time.monotonic()
        tasks = orch.plan(prompt)
        latency = (time.monotonic() - started) * 1000
        valid += int(bool(tasks))               # schema guarantees JSON; measure it produced tasks
        rows.append({"latency_ms": latency, **orch.client.last})
    return {"stage": "planner (model)", "n": len(prompts), "accuracy_pct": _pct(valid, len(prompts)), **_telemetry(rows)}


def bench_verifier(orch: Orchestrator) -> dict:
    rows, agree = [], 0
    for case in dataset.VERIFIER_CASES:
        results = [
            WorkerResult(task=Subtask(id=w["id"], title=w["id"], prompt=""), answer=w["answer"],
                         elapsed_s=0.0, call={}, compact={})
            for w in case["worker_results"]
        ]
        started = time.monotonic()
        verdict = orch.verify(case["request"], results)
        latency = (time.monotonic() - started) * 1000
        agree += int(bool(verdict.get("pass")) == case["pass"])
        rows.append({"latency_ms": latency, **orch.client.last})
    return {"stage": "verifier (model)", "n": len(dataset.VERIFIER_CASES),
            "accuracy_pct": _pct(agree, len(dataset.VERIFIER_CASES)), **_telemetry(rows)}


def print_table(results: list[dict]) -> None:
    cols = ["stage", "n", "accuracy_pct", "avg_latency_ms", "avg_prompt_tokens", "avg_completion_tokens"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in results)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-stages", action="store_true", help="also run planner/verifier (needs a model server)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen2.5-coder-7b")
    ap.add_argument("--no-think", action="store_true", help="disable reasoning (MiniCPM: enable_thinking=false) for direct utility output")
    ap.add_argument("--think", action="store_true", help="reasoning ON (MiniCPM: enable_thinking=true) with bigger token budgets so <think> completes")
    args = ap.parse_args()

    results = [bench_routing(), bench_query_rewrite(), bench_memory_extraction(), bench_tool_extraction()]
    if args.model_stages:
        extra, orch_kwargs = None, {}
        if args.think:
            extra = {"chat_template_kwargs": {"enable_thinking": True}}
            orch_kwargs = {"planner_tokens": 2600, "verifier_tokens": 1600, "compactor_tokens": 1200}
        elif args.no_think:
            extra = {"chat_template_kwargs": {"enable_thinking": False}}
        client = RecordingClient(OpenAICompatibleClient(args.base_url, args.model, extra_body=extra), strip_format=args.think)
        orch = Orchestrator(client, max_workers=1, **orch_kwargs)
        print(f"# model stages via {args.base_url} ({args.model})", file=sys.stderr)
        results += [bench_planner(orch), bench_verifier(orch)]

    print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
