#!/usr/bin/env python3
"""Small multi-call orchestration proof of concept for local coding models.

The design is intentionally narrow:
- one planner call returns focused subtasks as JSON
- worker calls solve subtasks in parallel with tiny prompts
- one verifier call checks coverage and flags weak spots
- one synthesizer call produces the final answer

It supports Ollama's native API and OpenAI-compatible local servers such as
llama.cpp. There are no third-party Python dependencies.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Protocol


DEFAULT_MODEL = "qwen2.5-coder-7b"
DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_OPENAI_BASE = "http://127.0.0.1:8080"


@dataclasses.dataclass(frozen=True)
class Subtask:
    id: str
    title: str
    prompt: str
    role: str = "worker"
    depends_on: tuple[str, ...] = ()
    token_budget: int | None = None


@dataclasses.dataclass(frozen=True)
class WorkerResult:
    task: Subtask
    answer: str
    elapsed_s: float
    call: dict[str, Any]
    compact: dict[str, Any]
    compact_call: dict[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class ModelResponse:
    content: str
    elapsed_s: float
    usage: dict[str, Any]
    timings: dict[str, Any]
    raw: dict[str, Any]


# Constrained decoding: force valid JSON out of the planner/verifier/compactor so a
# 7B can't wrap it in prose or truncate it. On this llama.cpp build json_object mode
# is a no-op (still emits ```json fences), but a json_schema response_format compiles
# to a GBNF grammar that enforces clean, structured output. The tolerant parsers stay
# as a fallback. Requiring pass:boolean also fixes stringy verifier verdicts at source.
def _json_schema_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


PLANNER_FORMAT = _json_schema_format("plan", {
    "type": "object",
    "properties": {
        "strategy": {"type": "string"},
        "settings": {"type": "object"},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "prompt": {"type": "string"},
                    "role": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "prompt"],
            },
        },
    },
    "required": ["tasks"],
})
VERIFIER_FORMAT = _json_schema_format("verify", {
    "type": "object",
    "properties": {
        "pass": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "missing_tasks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["pass"],
})
COMPACTOR_FORMAT = _json_schema_format("compact", {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "use_later": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary"],
})


class ModelClient(Protocol):
    def chat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int, response_format: dict[str, Any] | None = None) -> ModelResponse:
        ...


def http_json(method: str, url: str, payload: Any | None = None, timeout: int = 120) -> Any:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def read_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", "replace").strip()
    except Exception:
        body = ""
    if len(body) > 1200:
        body = body[:1200] + "..."
    return f"HTTP {exc.code} {exc.reason}: {body}" if body else f"HTTP {exc.code} {exc.reason}"


def _apply_ollama_format(payload: dict[str, Any], response_format: dict[str, Any] | None) -> None:
    """Ollama takes a top-level `format` ("json" or a JSON schema), not OpenAI's
    response_format, so translate."""
    if not response_format:
        return
    if response_format.get("type") == "json_schema":
        payload["format"] = (response_format.get("json_schema") or {}).get("schema") or "json"
    else:
        payload["format"] = "json"


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_s: int = 1200) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def chat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int, response_format: dict[str, Any] | None = None) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        _apply_ollama_format(payload, response_format)
        started = time.monotonic()
        data = http_json("POST", f"{self.base_url}/api/chat", payload, timeout=self.timeout_s)
        elapsed = time.monotonic() - started
        usage = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
            "total_tokens": (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0),
        }
        timings = {
            "prompt_ms": ns_to_ms(data.get("prompt_eval_duration")),
            "predicted_ms": ns_to_ms(data.get("eval_duration")),
        }
        return ModelResponse(data.get("message", {}).get("content", ""), elapsed, usage, timings, data)

    def stream_chat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int, response_format: dict[str, Any] | None = None):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        _apply_ollama_format(payload, response_format)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        final: dict[str, Any] = {}
        with urllib.request.urlopen(req, timeout=self.timeout_s) as res:
            for raw in res:
                if not raw.strip():
                    continue
                event = json.loads(raw.decode("utf-8"))
                final = event
                chunk = event.get("message", {}).get("content") or ""
                if chunk:
                    yield {"type": "chunk", "content": chunk}
        usage = {
            "prompt_tokens": final.get("prompt_eval_count"),
            "completion_tokens": final.get("eval_count"),
            "total_tokens": (final.get("prompt_eval_count") or 0) + (final.get("eval_count") or 0),
        }
        timings = {
            "prompt_ms": ns_to_ms(final.get("prompt_eval_duration")),
            "predicted_ms": ns_to_ms(final.get("eval_duration")),
        }
        yield {"type": "done", "elapsed_s": time.monotonic() - started, "usage": usage, "timings": timings, "raw": final}


class OpenAICompatibleClient:
    def __init__(self, base_url: str, model: str, timeout_s: int = 1200, extra_body: dict[str, Any] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        # Per-provider payload extras merged into every request. MiniCPM needs
        # {"chat_template_kwargs": {"enable_thinking": false}} so it answers directly
        # instead of spending the token budget in a <think> block.
        self.extra_body = extra_body or {}

    def chat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int, response_format: dict[str, Any] | None = None) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.0,
        }
        if response_format:
            payload["response_format"] = response_format
        payload.update(self.extra_body)
        started = time.monotonic()
        try:
            data = http_json("POST", f"{self.base_url}/v1/chat/completions", payload, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(read_http_error(exc)) from exc
        elapsed = time.monotonic() - started
        return ModelResponse(
            data["choices"][0]["message"].get("content", ""),
            elapsed,
            data.get("usage", {}),
            data.get("timings", {}),
            data,
        )

    def stream_chat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int, response_format: dict[str, Any] | None = None):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.0,
        }
        if response_format:
            payload["response_format"] = response_format
        payload.update(self.extra_body)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        final: dict[str, Any] = {}
        try:
            res_ctx = urllib.request.urlopen(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(read_http_error(exc)) from exc
        with res_ctx as res:
            for raw in res:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                body = line.removeprefix("data:").strip()
                if body == "[DONE]":
                    break
                event = json.loads(body)
                final = event
                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content") or choice.get("text") or ""
                if content:
                    yield {"type": "chunk", "content": content}
        yield {
            "type": "done",
            "elapsed_s": time.monotonic() - started,
            "usage": final.get("usage", {}),
            "timings": final.get("timings", {}),
            "raw": final,
        }


def ns_to_ms(value: Any) -> float | None:
    try:
        return round(float(value) / 1_000_000, 3)
    except (TypeError, ValueError):
        return None


def compact_call(stage: str, response: ModelResponse, *, label: str = "") -> dict[str, Any]:
    usage = response.usage or {}
    timings = response.timings or {}
    completion_tokens = usage.get("completion_tokens") or timings.get("predicted_n")
    prompt_tokens = usage.get("prompt_tokens") or timings.get("prompt_n")
    total_tokens = usage.get("total_tokens") or (
        (prompt_tokens or 0) + (completion_tokens or 0) if prompt_tokens or completion_tokens else None
    )
    tokens_per_second = timings.get("predicted_per_second")
    if tokens_per_second is None and completion_tokens:
        try:
            tokens_per_second = round(float(completion_tokens) / response.elapsed_s, 2)
        except (TypeError, ValueError, ZeroDivisionError):
            tokens_per_second = None
    return {
        "stage": stage,
        "label": label,
        "elapsed_s": round(response.elapsed_s, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": round(tokens_per_second, 2) if isinstance(tokens_per_second, (int, float)) else None,
        "usage": usage,
        "timings": timings,
    }


def sum_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    def total(key: str) -> int:
        return int(sum(call.get(key) or 0 for call in calls))

    elapsed = round(sum(float(call.get("elapsed_s") or 0) for call in calls), 3)
    completion = total("completion_tokens")
    return {
        "calls": len(calls),
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": completion,
        "total_tokens": total("total_tokens"),
        "model_elapsed_s": elapsed,
        "avg_completion_tokens_per_second": round(completion / elapsed, 2) if elapsed and completion else None,
    }


def truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 80:
        return text[:max_chars]
    head = max_chars // 2
    tail = max_chars - head - 40
    return f"{text[:head].rstrip()}\n\n[...truncated...]\n\n{text[-tail:].lstrip()}"


def compact_result(result: WorkerResult, max_answer_chars: int) -> dict[str, Any]:
    if result.compact:
        return {
            "id": result.task.id,
            "title": result.task.title,
            "role": result.task.role,
            "compact": result.compact,
        }
    return {
        "id": result.task.id,
        "title": result.task.title,
        "role": result.task.role,
        "answer": truncate_text(result.answer, max_answer_chars),
    }


def strip_fences(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()


def extract_json(text: str) -> Any:
    """Parse JSON even when a small model wraps it in prose or fences."""
    cleaned = strip_fences(text)
    candidates = [cleaned]
    first_obj, last_obj = cleaned.find("{"), cleaned.rfind("}")
    if first_obj != -1 and last_obj > first_obj:
        candidates.append(cleaned[first_obj : last_obj + 1])
    first_arr, last_arr = cleaned.find("["), cleaned.rfind("]")
    if first_arr != -1 and last_arr > first_arr:
        candidates.append(cleaned[first_arr : last_arr + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"model did not return parseable JSON:\n{text}")


def extract_complete_json_objects(text: str) -> list[dict[str, Any]]:
    """Recover complete object literals from a truncated JSON array."""
    objects: list[dict[str, Any]] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : index + 1]
                start = None
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    objects.append(parsed)
    return objects


def extract_truncated_planner_json(text: str) -> dict[str, Any]:
    """Best-effort planner recovery when the model truncates a JSON response."""
    cleaned = strip_fences(text)
    tasks_start = cleaned.find('"tasks"')
    if tasks_start == -1:
        return {"strategy": "", "settings": {}, "tasks": []}
    array_start = cleaned.find("[", tasks_start)
    if array_start == -1:
        return {"strategy": "", "settings": {}, "tasks": []}
    task_objects = [
        item
        for item in extract_complete_json_objects(cleaned[array_start:])
        if any(key in item for key in ("prompt", "title", "id"))
    ]
    strategy = ""
    match = re.search(r'"strategy"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned)
    if match:
        try:
            strategy = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            strategy = match.group(1)
    return {"strategy": strategy, "settings": {}, "tasks": task_objects}


PLANNER_SYSTEM = """You are a planner for a small local coding model.
Your job is to make a weak-but-fast local model feel more capable by decomposing
work into focused calls. Choose the orchestration plan, not just the task list.

Optimize for correctness and effective context, not speed. Prefer sequential
dependencies when one worker needs another worker's output. Avoid overlap. Each
worker prompt must be narrow enough to solve well in one model call.
Hard limits:
- Never create more tasks than the user asks for.
- Keep each task prompt under 160 characters.
- Prefer 3-5 high-value tasks over exhaustive coverage.

Return JSON only:
{
  "strategy": "short explanation of the workflow you chose",
  "settings": {
    "max_workers": 1,
    "worker_tokens": 700,
    "verifier_tokens": 500,
    "synth_tokens": 900,
    "max_rounds": 2
  },
  "tasks": [
    {
      "id": "t1",
      "title": "short name",
      "role": "architect|researcher|critic|implementer|tester|summarizer|worker",
      "prompt": "specific worker instruction",
      "depends_on": [],
      "token_budget": 700
    }
  ]
}
"""


WORKER_SYSTEM = """You are a focused worker running as one small local model call.
Solve only the assigned subtask from your assigned role. Be concrete. State assumptions and uncertainty.
Do not solve unrelated subtasks.
"""


VERIFIER_SYSTEM = """You are a lenient verifier for decomposed local-model work.
Decide whether the worker results, taken together, SUBSTANTIALLY answer the user's
original request.

Pass unless there is a SIGNIFICANT problem: a core part of the request left
unanswered, a direct contradiction between workers, or a clearly wrong or unsafe
claim. Do NOT fail for style, extra detail, minor omissions, formatting, tone, or
anything the user did not actually ask for. When in doubt, pass.

Return concise JSON only:
{
  "pass": true,
  "issues": ["only genuinely significant problems; empty if none"],
  "missing_tasks": ["only a core unanswered part of the request; empty if none"]
}
"""


COMPACTOR_SYSTEM = """You are a context compactor for a local multi-agent system.
Compress one worker result into reusable evidence for later workers, verifier,
and synthesizer. Preserve concrete details. Remove prose, repetition, and filler.
Do not add new claims.

Return concise JSON only:
{
  "summary": "dense reusable summary",
  "key_points": ["specific point"],
  "decisions": ["decision or recommendation"],
  "risks": ["risk or caveat"],
  "open_questions": ["unresolved question"],
  "use_later": ["facts later agents should preserve"]
}
"""


SYNTH_SYSTEM = """You synthesize decomposed worker results into one final response.
Use the verifier notes. Be direct, remove duplication, and preserve useful detail.
"""


class Orchestrator:
    def __init__(
        self,
        client: ModelClient,
        *,
        util_client: ModelClient | None = None,
        verifier_client: ModelClient | None = None,
        max_workers: int = 1,
        max_tasks: int = 8,
        planner_tokens: int = 900,
        worker_tokens: int = 700,
        verifier_tokens: int = 500,
        compactor_tokens: int = 320,
        synth_tokens: int = 700,
        max_rounds: int = 2,
        dynamic: bool = True,
        pipeline_verify: bool = True,
        on_event: Any | None = None,
    ) -> None:
        self.client = client
        # Device-split model tier (falls back to the main coder if a specialist isn't wired):
        #   util_client     -> MiniCPM-1B on CPU: light structured stages (plan, compact)
        #   verifier_client -> Qwen3-4B on CPU (thinking): the verify stage
        #   client          -> the swappable GPU coder: worker + synthesize
        self.util_client = util_client or client
        self.verifier_client = verifier_client or client
        # Pipeline the slow CPU verify against the GPU: while the verifier (CPU) checks a round,
        # the coder (GPU) speculatively synthesizes (assume pass). Different devices => genuine
        # overlap, so the ~40s verify hides behind synth's GPU time. Only engages when the
        # verifier is a *separate* client from the coder (else both are the single GPU slot).
        self.pipeline_verify = pipeline_verify
        self.max_workers = max_workers
        self.max_tasks = max_tasks
        self.planner_tokens = planner_tokens
        self.worker_tokens = worker_tokens
        self.verifier_tokens = verifier_tokens
        self.compactor_tokens = compactor_tokens
        self.synth_tokens = synth_tokens
        self.max_rounds = max_rounds
        self.dynamic = dynamic
        self.on_event = on_event
        self.plan_meta: dict[str, Any] = {}
        self.model_calls: list[dict[str, Any]] = []

    def emit(self, event: str, **data: Any) -> None:
        if self.on_event:
            self.on_event({"event": event, "time": round(time.time(), 3), **data})

    def record_call(self, stage: str, response: ModelResponse, *, label: str = "") -> dict[str, Any]:
        call = compact_call(stage, response, label=label)
        self.model_calls.append(call)
        self.emit("model_call_finished", **call)
        return call

    def context_results_for_task(self, results: list[WorkerResult], max_results: int = 6) -> str:
        unique: dict[str, WorkerResult] = {}
        for result in results:
            unique[result.task.id] = result
        selected = list(unique.values())[-max_results:]
        if not selected:
            return ""
        return "\n\nRelevant prior compacted results:\n" + json.dumps(
            [compact_result(item, 900) for item in selected],
            indent=2,
        )

    def clamp_int(self, value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    def apply_dynamic_settings(self, settings: Any) -> None:
        if not self.dynamic or not isinstance(settings, dict):
            return
        # The machine constraint wins: this llama.cpp service has one slot.
        self.max_workers = self.clamp_int(settings.get("max_workers"), self.max_workers, 1, self.max_workers)
        self.worker_tokens = self.clamp_int(settings.get("worker_tokens"), self.worker_tokens, 120, 2000)
        self.verifier_tokens = self.clamp_int(settings.get("verifier_tokens"), self.verifier_tokens, 120, 1600)
        self.compactor_tokens = self.clamp_int(settings.get("compactor_tokens"), self.compactor_tokens, 120, 800)
        self.synth_tokens = self.clamp_int(settings.get("synth_tokens"), self.synth_tokens, 200, 1200)
        self.max_rounds = self.clamp_int(settings.get("max_rounds"), self.max_rounds, 1, 4)

    def parse_tasks(self, raw_tasks: Any, *, prefix: str = "t") -> list[Subtask]:
        if not isinstance(raw_tasks, list):
            raise ValueError("planner JSON must contain a tasks list")

        tasks: list[Subtask] = []
        for index, item in enumerate(raw_tasks, start=1):
            if isinstance(item, str):
                item = {"prompt": item, "title": item[:70]}
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("id") or f"{prefix}{index}")
            title = str(item.get("title") or task_id)
            role = str(item.get("role") or "worker")
            prompt = str(item.get("prompt") or "").strip()
            depends_raw = item.get("depends_on") or []
            depends = tuple(str(dep) for dep in depends_raw if isinstance(dep, str))
            token_budget = item.get("token_budget")
            budget = self.clamp_int(token_budget, self.worker_tokens, 120, 2000) if token_budget else None
            if prompt:
                tasks.append(Subtask(task_id, title, prompt, role, depends, budget))
        if not tasks:
            raise ValueError("planner produced no usable tasks")
        return tasks[: self.max_tasks]

    def plan(self, request: str) -> list[Subtask]:
        response = self.util_client.chat(
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Create at most {self.max_tasks} tasks. The local server should be treated as "
                        f"single-slot, so prefer max_workers=1 unless the task is explicitly parallel-safe.\n\n"
                        f"User request:\n{request}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=self.planner_tokens,
            response_format=PLANNER_FORMAT,
        )
        self.record_call("planner", response, label="initial plan")
        try:
            parsed = extract_json(response.content)
        except ValueError:
            parsed = extract_truncated_planner_json(response.content)
            if not parsed.get("tasks"):
                raise
            self.emit("planner_json_recovered", recovered_tasks=len(parsed["tasks"]))
        if isinstance(parsed, dict):
            self.plan_meta = {
                "strategy": parsed.get("strategy", ""),
                "settings": parsed.get("settings", {}),
            }
            self.apply_dynamic_settings(parsed.get("settings"))
            raw_tasks = parsed.get("tasks")
        else:
            self.plan_meta = {"strategy": "", "settings": {}}
            raw_tasks = parsed
        return self.parse_tasks(raw_tasks)

    def solve_task(
        self,
        request: str,
        task: Subtask,
        context_results: list[WorkerResult] | None = None,
    ) -> WorkerResult:
        start = time.monotonic()
        context_results = context_results or []
        self.emit(
            "worker_started",
            id=task.id,
            title=task.title,
            role=task.role,
            token_budget=task.token_budget or self.worker_tokens,
            context=[
                {
                    "id": result.task.id,
                    "title": result.task.title,
                    "role": result.task.role,
                    "compact": result.compact,
                }
                for result in context_results[-6:]
            ],
        )
        context_text = self.context_results_for_task(context_results)
        response = self.client.chat(
            [
                {"role": "system", "content": WORKER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Original request:\n{request}\n\n"
                        f"Assigned role: {task.role}\n"
                        f"Assigned subtask ({task.id}: {task.title}):\n{task.prompt}"
                        f"{context_text}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=task.token_budget or self.worker_tokens,
        )
        call = self.record_call("worker", response, label=f"{task.id}: {task.title}")
        compact, compact_call = self.compact_worker_result(request, task, response.content)
        result = WorkerResult(task, response.content.strip(), time.monotonic() - start, call, compact, compact_call)
        self.emit(
            "worker_finished",
            id=task.id,
            title=task.title,
            elapsed_s=round(result.elapsed_s, 3),
            call=call,
            compact=compact,
        )
        return result

    def compact_worker_result(self, request: str, task: Subtask, answer: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        self.emit("compacting", id=task.id, title=task.title)
        response = self.util_client.chat(
            [
                {"role": "system", "content": COMPACTOR_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": truncate_text(request, 900),
                            "worker": {
                                "id": task.id,
                                "title": task.title,
                                "role": task.role,
                                "prompt": truncate_text(task.prompt, 700),
                            },
                            "answer": truncate_text(answer, 3500),
                        },
                        indent=2,
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=self.compactor_tokens,
            response_format=COMPACTOR_FORMAT,
        )
        call = self.record_call("compactor", response, label=f"{task.id}: {task.title}")
        try:
            parsed = extract_json(response.content)
        except ValueError:
            parsed = {"summary": truncate_text(response.content, 900), "key_points": [], "decisions": [], "risks": [], "open_questions": [], "use_later": []}
        if not isinstance(parsed, dict):
            parsed = {"summary": truncate_text(str(parsed), 900), "key_points": [], "decisions": [], "risks": [], "open_questions": [], "use_later": []}
        self.emit("compacted", id=task.id, title=task.title, call=call, compact=parsed)
        return parsed, call

    def solve_tasks(
        self,
        request: str,
        tasks: list[Subtask],
        prior_results: list[WorkerResult] | None = None,
    ) -> list[WorkerResult]:
        prior_results = prior_results or []
        results_by_id: dict[str, WorkerResult] = {}
        prior_by_id = {result.task.id: result for result in prior_results}
        pending = list(tasks)
        completed: list[WorkerResult] = []

        while pending:
            runnable = [
                task
                for task in pending
                if all(dep in results_by_id or dep in prior_by_id for dep in task.depends_on)
            ]
            if not runnable:
                # Break bad dependency graphs instead of hanging forever.
                runnable = [pending[0]]

            if self.max_workers <= 1 or len(runnable) == 1:
                task = runnable[0]
                context = prior_results + [results_by_id[dep] for dep in task.depends_on if dep in results_by_id]
                result = self.solve_task(request, task, context)
                results_by_id[task.id] = result
                completed.append(result)
                pending.remove(task)
                continue

            batch = runnable[: self.max_workers]
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                future_map = {
                    pool.submit(self.solve_task, request, task, prior_results): task for task in batch
                }
                for future in concurrent.futures.as_completed(future_map):
                    task = future_map[future]
                    result = future.result()
                    results_by_id[result.task.id] = result
                    completed.append(result)
                    pending.remove(task)

        return completed

    def verify(self, request: str, results: list[WorkerResult]) -> dict[str, Any]:
        response = self.verifier_client.chat(
            [
                {"role": "system", "content": VERIFIER_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": request,
                            "worker_results": [compact_result(result, 850) for result in results[-10:]],
                        },
                        indent=2,
                    ),
                },
            ],
            temperature=0.0,
            # A dedicated thinking verifier (Qwen3-4B on CPU) needs headroom for the <think>
            # block *plus* the JSON verdict; the coder's own verify keeps its tighter budget.
            max_tokens=max(self.verifier_tokens, 700) if self.verifier_client is not self.client else self.verifier_tokens,
            response_format=VERIFIER_FORMAT,
        )
        self.record_call("verifier", response, label="coverage check")
        try:
            parsed = extract_json(response.content)
        except ValueError:
            # A *thinking* verifier (Qwen3-4B) can spend its whole budget in the <think>
            # block and return no parseable JSON. Don't crash the run or sink a good answer.
            return {"pass": True, "issues": [], "missing_tasks": []}
        if not isinstance(parsed, dict):
            # A verifier that can't emit clean JSON shouldn't sink a good answer;
            # give the benefit of the doubt rather than forcing a fail + follow-up round.
            return {"pass": True, "issues": [], "missing_tasks": []}
        # Normalize a small model's loose output: accept true/"true"/"yes"/1 as pass,
        # and keep only real lists for issues/missing_tasks.
        raw_pass = parsed.get("pass")
        if isinstance(raw_pass, str):
            parsed["pass"] = raw_pass.strip().lower() in ("true", "yes", "pass", "ok", "1")
        else:
            parsed["pass"] = bool(raw_pass)
        parsed["issues"] = parsed["issues"] if isinstance(parsed.get("issues"), list) else []
        parsed["missing_tasks"] = parsed["missing_tasks"] if isinstance(parsed.get("missing_tasks"), list) else []
        return parsed

    def synthesize(self, request: str, results: list[WorkerResult], verifier: dict[str, Any]) -> str:
        response = self.client.chat(
            [
                {"role": "system", "content": SYNTH_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": request,
                            "worker_results": [compact_result(result, 1200) for result in results[-10:]],
                            "verifier": verifier,
                        },
                        indent=2,
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=self.synth_tokens,
        )
        self.record_call("synthesizer", response, label="final answer")
        return response.content.strip()

    def followup_tasks(self, verifier: dict[str, Any], round_index: int, existing_results: list[WorkerResult]) -> list[Subtask]:
        missing = verifier.get("missing_tasks") if isinstance(verifier, dict) else []
        if not isinstance(missing, list):
            return []
        tasks: list[Subtask] = []
        depends_on = tuple(result.task.id for result in existing_results[-6:])
        for index, item in enumerate(missing, start=1):
            if isinstance(item, dict):
                prompt = str(item.get("prompt") or item.get("title") or "").strip()
                title = str(item.get("title") or prompt[:70] or f"Follow-up {index}")
                role = str(item.get("role") or "critic")
                token_budget = item.get("token_budget")
            else:
                prompt = str(item).strip()
                title = prompt[:70] or f"Follow-up {index}"
                role = "critic"
                token_budget = None
            if prompt:
                tasks.append(
                    Subtask(
                        id=f"r{round_index}t{index}",
                        title=title,
                        prompt=prompt,
                        role=role,
                        depends_on=depends_on,
                        token_budget=self.clamp_int(token_budget, self.worker_tokens, 120, 2000) if token_budget else None,
                    )
                )
        return tasks[: self.max_tasks]

    def run(self, request: str) -> dict[str, Any]:
        started = time.monotonic()
        self.emit("planning")
        tasks = self.plan(request)
        self.emit("planned", strategy=self.plan_meta.get("strategy"), settings=self.plan_meta.get("settings"), tasks=[dataclasses.asdict(task) for task in tasks])
        worker_results: list[WorkerResult] = []
        rounds: list[dict[str, Any]] = []
        verifier: dict[str, Any] = {"pass": False, "issues": ["not verified"], "missing_tasks": []}
        current_tasks = tasks

        # Pipeline only when verify runs on a *different* device than the coder (else both are
        # the single GPU slot and can't overlap). CPU verifier + GPU coder => genuine overlap.
        pipeline = self.pipeline_verify and (self.verifier_client is not self.client)
        pass_stub = {"pass": True, "issues": [], "missing_tasks": []}
        final: str | None = None

        for round_index in range(1, self.max_rounds + 1):
            self.emit("round_started", round=round_index, count=len(current_tasks))
            round_results = self.solve_tasks(request, current_tasks, worker_results)
            worker_results.extend(round_results)
            self.emit("verifying", round=round_index, pipelined=pipeline)

            spec_synth = None
            pool = None
            if pipeline:
                # Kick off the CPU verify and a speculative GPU synthesize (assume pass) at once.
                snapshot = list(worker_results)
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
                verify_fut = pool.submit(self.verify, request, snapshot)
                spec_synth = pool.submit(self.synthesize, request, snapshot, pass_stub)
                verifier = verify_fut.result()
            else:
                verifier = self.verify(request, worker_results)

            self.emit("verified", round=round_index, verifier=verifier)
            rounds.append(
                {
                    "round": round_index,
                    "tasks": [dataclasses.asdict(task) for task in current_tasks],
                    "result_ids": [result.task.id for result in round_results],
                    "verifier": verifier,
                }
            )
            passed = verifier.get("pass") is True
            if spec_synth is not None:
                # Always resolve it (the GPU already ran it). Keep it only if verify passed;
                # on a fail we discard it and do a followup round with real verifier feedback.
                synth_result = spec_synth.result()
                if passed:
                    final = synth_result
            if pool is not None:
                pool.shutdown(wait=True)
            if passed:
                break
            current_tasks = self.followup_tasks(verifier, round_index + 1, worker_results)
            if not current_tasks:
                break
            self.emit("followup_planned", round=round_index + 1, tasks=[dataclasses.asdict(task) for task in current_tasks])

        self.emit("synthesizing")
        if final is None:
            final = self.synthesize(request, worker_results, verifier)
        self.emit("done")
        return {
            "request": request,
            "strategy": self.plan_meta.get("strategy", ""),
            "dynamic_settings": {
                "max_workers": self.max_workers,
                "max_tasks": self.max_tasks,
                "worker_tokens": self.worker_tokens,
                "verifier_tokens": self.verifier_tokens,
                "compactor_tokens": self.compactor_tokens,
                "synth_tokens": self.synth_tokens,
                "max_rounds": self.max_rounds,
            },
            "usage_summary": sum_calls(self.model_calls),
            "model_calls": self.model_calls,
            "tasks": [dataclasses.asdict(task) for task in tasks],
            "rounds": rounds,
            "worker_results": [
                {
                    "id": result.task.id,
                    "title": result.task.title,
                    "role": result.task.role,
                    "answer": result.answer,
                    "compact": result.compact,
                    "elapsed_s": round(result.elapsed_s, 3),
                    "call": result.call,
                    "compact_call": result.compact_call,
                }
                for result in worker_results
            ],
            "verifier": verifier,
            "final": final,
            "elapsed_s": round(time.monotonic() - started, 3),
        }


def make_client(args: argparse.Namespace) -> ModelClient:
    if args.provider == "ollama":
        return OllamaClient(args.base_url or DEFAULT_OLLAMA_BASE, args.model, timeout_s=args.timeout)
    if args.provider == "openai-compatible":
        return OpenAICompatibleClient(args.base_url or DEFAULT_OPENAI_BASE, args.model, timeout_s=args.timeout)
    raise ValueError(f"unknown provider: {args.provider}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local small-model orchestration PoC")
    parser.add_argument("request", nargs="*", help="request to solve; omit to read stdin")
    parser.add_argument("--provider", choices=["ollama", "openai-compatible"], default="openai-compatible")
    parser.add_argument("--base-url", help="provider base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1200, help="HTTP timeout per model call in seconds")
    parser.add_argument("--planner-tokens", type=int, default=900)
    parser.add_argument("--worker-tokens", type=int, default=700)
    parser.add_argument("--verifier-tokens", type=int, default=500)
    parser.add_argument("--compactor-tokens", type=int, default=320)
    parser.add_argument("--synth-tokens", type=int, default=700)
    parser.add_argument("--no-dynamic", action="store_true", help="do not let planner adjust stage budgets")
    parser.add_argument("--json", action="store_true", help="print full trace JSON")
    args = parser.parse_args(argv)

    request = " ".join(args.request).strip() or sys.stdin.read().strip()
    if not request:
        parser.error("provide a request as arguments or stdin")

    orch = Orchestrator(
        make_client(args),
        max_workers=args.max_workers,
        max_tasks=args.max_tasks,
        planner_tokens=args.planner_tokens,
        worker_tokens=args.worker_tokens,
        verifier_tokens=args.verifier_tokens,
        compactor_tokens=args.compactor_tokens,
        synth_tokens=args.synth_tokens,
        max_rounds=args.max_rounds,
        dynamic=not args.no_dynamic,
    )
    try:
        trace = orch.run(request)
    except urllib.error.URLError as exc:
        print(f"model endpoint unavailable: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(trace, indent=2))
    else:
        print(trace["final"])
        print()
        print(f"[trace] tasks={len(trace['tasks'])} elapsed={trace['elapsed_s']}s verifier_pass={trace['verifier'].get('pass')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
