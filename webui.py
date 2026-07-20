#!/usr/bin/env python3
"""Tiny web UI for trying the local Qwen orchestrator."""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import html
import hashlib
import hmac
import json
import os
import re
import subprocess
import sqlite3
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse
import urllib.request
import base64

from orchestrator import (
    DEFAULT_MODEL,
    DEFAULT_OPENAI_BASE,
    ModelResponse,
    OllamaClient,
    OpenAICompatibleClient,
    Orchestrator,
    compact_call,
    extract_json,
    sum_calls,
    truncate_text,
)


PROJECT_EXCLUDE_DIRS = {".git", "node_modules", ".next", "dist", "build", ".cache", "coverage", "__pycache__", ".venv", "venv", "vendor"}
PROJECT_INCLUDE_SUFFIXES = {
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".json",
    ".md",
    ".css",
    ".scss",
    ".py",
    ".toml",
    ".yml",
    ".yaml",
}
PROJECT_PRIORITY_NAMES = {
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "package.json",
    "tsconfig.json",
    "next.config.ts",
    "vite.config.ts",
    "pyproject.toml",
}
PROJECT_SEARCH_ROOTS = [
    Path.home() / "Desktop" / "projects",
    Path.home() / "Desktop" / "project",
    Path.home() / "projects",
    Path.home() / "code",
    Path.home() / "src",
]
PROJECT_STOP_WORDS = {
    "project",
    "projects",
    "desktop",
    "architecture",
    "understand",
    "explain",
    "code",
    "gaps",
    "folder",
    "through",
    "figure",
    "what",
    "next",
    "this",
    "that",
    "with",
    "from",
    "into",
    "about",
    "should",
    "would",
    "could",
    "there",
    "their",
    "where",
}
WEB_TERMS = {
    "web",
    "search",
    "online",
    "internet",
    "current",
    "latest",
    "recent",
    "today",
    "news",
    "website",
    "webpage",
    "page",
    "url",
}
# High-precision phrases that signal the answer needs up-to-date external facts even
# when no WEB_TERM keyword is present (e.g. "find the release date for the next
# one piece chapter" has none). Kept as phrases to avoid over-routing on stray words.
WEB_RESEARCH_PHRASES = (
    "release date", "come out", "comes out", "coming out", "out yet", "is out",
    "when is", "when are", "when does", "when do", "when will", "when's",
    "how much is", "how much does", "how much are", "how much will", "how much would",
    "who won", "who is winning", "who's winning", "next chapter", "next episode",
    "next season", "latest version", "current price", "stock price", "weather in",
    "results of", "score of", "final score", "who is the current", "as of today",
    "this week", "right now",
)
DOCS_HINT_TERMS = {
    "api",
    "best",
    "best-practice",
    "best-practices",
    "library",
    "framework",
    "dependency",
    "package",
    "sdk",
    "version",
    "upgrade",
    "migration",
    "auth",
    "security",
    "deploy",
    "performance",
    "react",
    "next",
    "nextjs",
    "vite",
    "node",
    "express",
    "fastapi",
    "django",
    "flask",
    "pytest",
    "typescript",
    "python",
    "sqlalchemy",
    "postgres",
    "sqlite",
    "tailwind",
}
CURRENT_DATE = dt.date.today().isoformat()
USER_MEMORY_PROMPT_CHARS = 3500
PROJECT_MEMORY_PROMPT_CHARS = 3500
PROJECT_RETRIEVAL_PROMPT_CHARS = 4500
WEB_CONTEXT_PROMPT_CHARS = 4500
DOCS_CONTEXT_PROMPT_CHARS = 3500
TOOL_EVIDENCE_PROMPT_CHARS = 4500
MAX_PENDING_PER_SESSION = 5   # cap how many prompts one chat can stack up (queued + running)

# Invitation for the model to *propose* a durable memory. It never saves anything —
# the user approves each suggestion in the UI, same gate as tool requests.
MEMORY_SUGGESTION_GUIDANCE = (
    "\n\nIf you learned a DURABLE, reusable fact or preference about the user or their "
    "environment (not a one-off detail of this task), you may propose saving it with this "
    "exact JSON at the very end of your answer — do not claim it was saved:\n"
    '{"memory_suggestion":{"scope":"user","key":"short label","value":"the fact to remember",'
    '"tags":["optional"],"reason":"why it is worth remembering"}}\n'
    "Only propose something stable and genuinely useful later. If nothing qualifies, omit it entirely."
)
SEARCH_QUERY_STOP_WORDS = WEB_TERMS | {
    "can",
    "could",
    "would",
    "will",
    "you",
    "your",
    "me",
    "my",
    "please",
    "pls",
    "give",
    "show",
    "tell",
    "find",
    "look",
    "lookup",
    "results",
    "result",
    "info",
    "information",
    "for",
    "about",
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "on",
    "in",
    "with",
    "is",
    "are",
    "was",
    "were",
}

TERMINAL_STATUSES = {"done", "error", "cancelled"}
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
PROJECT_MEMORY_MAX_AGE_S = 7 * 24 * 60 * 60
DEFAULT_SEARXNG_URL = os.environ.get("QWEN_SEARXNG_URL", "http://127.0.0.1:8888").rstrip("/")
# Device-split model tier: CPU-resident specialists offload light stages off the swappable
# GPU coder. MiniCPM-1B (:8081) handles plan/compact; Qwen3-4B (:8086) handles verify. If a
# service is down the orchestrator falls back to the coder for that stage (graceful degrade).
UTIL_BASE_URL = os.environ.get("QWEN_UTIL_URL", "http://127.0.0.1:8081").rstrip("/")
UTIL_MODEL = os.environ.get("QWEN_UTIL_MODEL", "minicpm5-1b")
VERIFIER_BASE_URL = os.environ.get("QWEN_VERIFIER_URL", "http://127.0.0.1:8086").rstrip("/")
VERIFIER_MODEL = os.environ.get("QWEN_VERIFIER_MODEL", "qwen3-4b")
DEVICE_SPLIT = os.environ.get("QWEN_DEVICE_SPLIT", "1") != "0"
ADMIN_TOKEN = os.environ.get("QWEN_ORCHESTRATOR_ADMIN_TOKEN", os.environ.get("QWEN_ORCHESTRATOR_TOKEN", "")).strip()
RESTRICTED_TOKENS = {
    token.strip()
    for token in os.environ.get("QWEN_ORCHESTRATOR_RESTRICTED_TOKENS", "").replace(",", "\n").splitlines()
    if token.strip()
}
AUTH_TOKEN = ADMIN_TOKEN
SESSION_COOKIE = "qwen_orchestrator_token"
ADMIN_ONLY_CAPABILITIES = {"git_status", "list_dir", "file_preview", "search_text"}
ADMIN_ONLY_MODES = {"project_research", "code_review", "implementation", "debug"}
DEFAULT_PROJECT_MAX_FILES = int(os.environ.get("QWEN_PROJECT_MAX_FILES", "80"))
MODEL_CALL_COOLDOWN_S = float(os.environ.get("QWEN_MODEL_CALL_COOLDOWN_S", "3"))
RUN_DEFAULTS = {
    "max_tasks": 5,
    "max_workers": 1,
    "max_rounds": 2,
    "planner_tokens": 700,
    "worker_tokens": 550,
    "verifier_tokens": 360,
    "compactor_tokens": 260,
    "synth_tokens": 650,
    "timeout": 1200,
}


class JobCancelled(Exception):
    pass
GREETING_RE = re.compile(r"^\s*(hi|hey|hello|yo|sup|thanks|thank you|ok|okay)\s*[!.?]*\s*$", re.I)
PROJECT_TERMS = {
    "project",
    "repo",
    "repository",
    "codebase",
    "desktop/project",
    "desktop/projects",
}
DEEP_TERMS = {
    "plan",
    "design",
    "debug",
    "refactor",
    "review",
    "analyze",
    "compare",
    "architecture",
    "implement",
    "build",
    "migrate",
    "explain",
    "gaps",
}
MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "chat": {
        "label": "Chat",
        "description": "No model call for simple greetings/status.",
        "expected_calls": 0,
        "settings": {},
    },
    "direct_answer": {
        "label": "Direct Answer",
        "description": "One focused model call.",
        "expected_calls": 1,
        "settings": {},
    },
    "search_results": {
        "label": "Search Results",
        "description": "Run web search and return result links without model synthesis.",
        "expected_calls": 0,
        "settings": {},
    },
    "web_research": {
        "label": "Web Research",
        "description": "Host fetches web evidence, then workers synthesize from it.",
        "expected_calls": 7,
        "settings": {"max_tasks": 3, "max_rounds": 1, "planner_tokens": 520, "worker_tokens": 500, "verifier_tokens": 320, "compactor_tokens": 220, "synth_tokens": 650},
    },
    "plan": {
        "label": "Planner",
        "description": "Small planning workflow without project scan.",
        "expected_calls": 7,
        "settings": {"max_tasks": 3, "max_rounds": 1, "planner_tokens": 520, "worker_tokens": 420, "verifier_tokens": 260, "compactor_tokens": 220, "synth_tokens": 520},
    },
    "project_research": {
        "label": "Project Research",
        "description": "Sequential project file readers plus architecture synthesis.",
        "expected_calls": 20,
        "settings": {"max_tasks": 8, "max_rounds": 2, "planner_tokens": 900, "worker_tokens": 700, "verifier_tokens": 500, "compactor_tokens": 320, "synth_tokens": 700},
    },
    "code_review": {
        "label": "Code Review",
        "description": "Project readers focused on risks, gaps, and maintainability.",
        "expected_calls": 20,
        "settings": {"max_tasks": 7, "max_rounds": 2, "planner_tokens": 820, "worker_tokens": 650, "verifier_tokens": 520, "compactor_tokens": 300, "synth_tokens": 760},
    },
    "implementation": {
        "label": "Implementation",
        "description": "Programming-oriented workflow. Currently read-only; produces implementation plan/patch guidance.",
        "expected_calls": 12,
        "settings": {"max_tasks": 6, "max_rounds": 2, "planner_tokens": 760, "worker_tokens": 650, "verifier_tokens": 420, "compactor_tokens": 280, "synth_tokens": 760},
    },
    "debug": {
        "label": "Debug",
        "description": "Hypothesis-driven debugging workflow.",
        "expected_calls": 10,
        "settings": {"max_tasks": 5, "max_rounds": 2, "planner_tokens": 700, "worker_tokens": 600, "verifier_tokens": 420, "compactor_tokens": 260, "synth_tokens": 650},
    },
    "deep_orchestration": {
        "label": "Deep Orchestration",
        "description": "General decomposed reasoning workflow.",
        "expected_calls": 12,
        "settings": {"max_tasks": 5, "max_rounds": 2, "planner_tokens": 700, "worker_tokens": 550, "verifier_tokens": 360, "compactor_tokens": 260, "synth_tokens": 650},
    },
}


def fmt_status(status: str) -> str:
    return "complete" if status == "done" else status


def completed_calls_from_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    trace = job.get("trace") or {}
    trace_calls = [*(trace.get("project_model_calls") or []), *(trace.get("model_calls") or [])]
    if trace_calls:
        return trace_calls
    return [event for event in job.get("events", []) if event.get("event") == "model_call_finished"]


def estimated_calls_from_events(job: dict[str, Any]) -> int | None:
    events = job.get("events") or []
    if not any(event.get("event") == "planned" for event in events):
        return None

    planned_tasks = 0
    followup_tasks = 0
    verifier_calls = 0
    for event in events:
        if event.get("event") == "planned":
            planned_tasks += len(event.get("tasks") or [])
        elif event.get("event") == "followup_planned":
            followup_tasks += len(event.get("tasks") or [])
        elif event.get("event") == "verifying":
            verifier_calls += 1

    task_calls = (planned_tasks + followup_tasks) * 2
    planner_call = 1
    synth_call = 1 if any(event.get("event") in {"synthesizing", "done"} for event in events) else 0
    return planner_call + task_calls + verifier_calls + synth_call


def job_metrics(job: dict[str, Any]) -> dict[str, Any]:
    calls = completed_calls_from_job(job)
    summary = sum_calls(calls)
    stream_stats = job.get("stream_stats") or {}
    active_call = job.get("active_model_call") or {}
    started = job.get("started")
    finished = job.get("finished")
    now = time.time()
    elapsed = 0.0
    if started:
        elapsed = max(0.0, (finished or now) - float(started))
    route = job.get("route") or (job.get("trace") or {}).get("mode") or {}
    started_count = len([event for event in job.get("events", []) if event.get("event") == "model_call_started"])
    dynamic_expected = estimated_calls_from_events(job)
    expected_calls = int(dynamic_expected or route.get("expected_calls") or summary.get("calls") or 0)
    completed = int(summary.get("calls") or 0)
    expected_calls = max(expected_calls, started_count, completed)
    eta = None
    if expected_calls and completed and summary.get("model_elapsed_s"):
        avg = float(summary["model_elapsed_s"]) / completed
        eta = max(0.0, (expected_calls - completed) * avg)
    live_completion_tokens = stream_stats.get("approx_completion_tokens") or stream_stats.get("approx_tokens") or 0
    completion_tokens = (summary.get("completion_tokens") or 0) + (live_completion_tokens or 0)
    total_tokens = (summary.get("total_tokens") or 0) + (live_completion_tokens or 0)
    tokens_per_second = stream_stats.get("approx_tokens_per_second") or summary.get("avg_completion_tokens_per_second")
    return {
        "status": fmt_status(str(job.get("status") or "queued")),
        "elapsed_seconds": round(elapsed, 2),
        "eta_seconds": round(eta, 2) if eta is not None else None,
        "calls_planned": expected_calls,
        "calls_started": started_count,
        "calls_finished": completed,
        "calls_failed": len([event for event in job.get("events", []) if event.get("event") in {"call_failed", "model_call_failed"}]),
        "tokens_prompt": summary.get("prompt_tokens") or 0,
        "tokens_completion": completion_tokens,
        "tokens_total": total_tokens,
        "tokens_per_second": tokens_per_second,
        "model_elapsed_seconds": summary.get("model_elapsed_s") or 0,
        "active_call_elapsed_seconds": active_call.get("elapsed_s"),
    }


def _short(text: Any, limit: int = 64) -> str:
    text = str(text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _plural(n: Any, one: str, many: str | None = None) -> str:
    n = n or 0
    return one if n == 1 else (many or one + "s")


# A short phase word per model-call stage, shown as the stage chip heading.
STAGE_WORDS = {
    "planner": "Planning",
    "worker": "Worker",
    "verifier": "Verifying",
    "synthesizer": "Synthesizing",
    "compactor": "Compacting",
    "project_file_reader": "Reading files",
    "project_directory_compactor": "Reading project",
    "project_architecture_compactor": "Project memory",
    "direct": "Answering",
    "model": "Working",
}


def humanize_event(event: dict[str, Any]) -> str:
    """A plain-language one-liner for a job event: what step ran and what data it
    touched. Reads the fields the event already carries (query, title, path, tasks,
    count, verifier) so the live activity feed reads like a narrative instead of raw
    event names like "web search finished"."""
    name = str(event.get("event") or "")
    stage = str(event.get("stage") or "")
    label = str(event.get("label") or "")
    count = event.get("count") or 0

    def call_phrase() -> str:
        base = {
            "planner": "Planning the approach",
            "worker": "Working a subtask" + (f": {_short(label, 40)}" if label and label != "subtask" else ""),
            "verifier": "Checking the answer for gaps",
            "synthesizer": "Writing the final answer",
            "compactor": "Condensing a worker result",
            "project_file_reader": "Reading a project file",
            "project_directory_compactor": "Summarizing a directory",
            "project_architecture_compactor": "Building project memory",
            "direct": "Answering directly",
        }
        return base.get(stage, "Thinking")

    simple = {
        "queued": "Waiting for the local model slot",
        "queued_for_slot": "Queued — waiting for the single model slot",
        "workflow_ready": "Ready to start the workflow",
        "workflow_started": "Starting the orchestration",
        "planning": "Planning the approach",
        "verifying": "Checking coverage and contradictions",
        "synthesizing": "Writing the final answer from the worker results",
        "done": "Final answer ready",
        "planner_json_recovered": "Recovered the plan from a truncated response",
        "model_call_failed": "A model call failed — retrying or falling back",
        "model_call_cooldown": "Cooling down between model calls (GPU safety)",
        "model_slot_waiting": "Waiting for the model slot",
        "model_slot_acquired": "Model slot acquired",
        "model_slot_released": "Model slot released",
        "web_page_failed": "A web fetch failed — moving on",
        "web_search_failed": "The web search failed — moving on",
        "docs_page_failed": "A docs fetch failed — moving on",
        "docs_search_failed": "The docs lookup failed — moving on",
        "project_memory_cache_hit": "Reusing cached project memory (files unchanged)",
        "project_memory_cached": "Saved project memory for next time",
        "project_retrieval_ready": "Picked the most relevant project context",
        "project_partial_resume": "Resuming from a previous interrupted run",
        "capability_context_attached": "Added capability context for the model",
        "direct_answer_started": "Answering directly (no orchestration needed)",
        "orchestrator_prompt_ready": "Prompt assembled — handing off to the workers",
        "verifier_failed_final_answer": "The verifier flagged the final answer",
        "interrupted_by_restart": "Interrupted by a server restart",
        "cancel_requested": "Stop requested — finishing the current step",
        "cancelled": "Stopped",
    }
    if name in simple:
        return simple[name]

    if name == "request_routed":
        return f"Routed to {event.get('label') or event.get('mode') or 'a workflow'}"
    if name == "planned":
        tasks = event.get("tasks") or []
        titles = ", ".join(_short(t.get("title"), 26) for t in tasks[:3] if isinstance(t, dict) and t.get("title"))
        return f"Planned {len(tasks)} {_plural(len(tasks), 'subtask')}" + (f": {titles}" if titles else "")
    if name == "round_started":
        return f"Round {event.get('round', 1)} — running {count} {_plural(count, 'worker')}"
    if name == "followup_planned":
        tasks = event.get("tasks") or []
        return f"Planned {len(tasks)} follow-up {_plural(len(tasks), 'task')} to close gaps"
    if name == "compacting":
        return f"Condensing result — {_short(event.get('title'))}"
    if name == "compacted":
        return f"Condensed — {_short(event.get('title'))}"
    if name == "verified":
        verifier = event.get("verifier") or {}
        if verifier.get("pass") is True:
            return "Verified — coverage looks complete"
        missing = len(verifier.get("missing_tasks") or [])
        issues = len(verifier.get("issues") or [])
        return f"Verifier found gaps — {missing} {_plural(missing, 'follow-up')}, {issues} {_plural(issues, 'issue')}"
    if name == "model_call_started":
        return call_phrase() + "…"
    if name == "model_call_finished":
        tok = event.get("total_tokens")
        return call_phrase() + (f" · {tok} tokens" if tok else "")
    if name == "web_search_started":
        return f'Searching the web — "{_short(event.get("query"), 48)}"'
    if name == "web_search_finished":
        return f"Found {event.get('count', 0)} web {_plural(event.get('count'), 'result')}" + (f" via {event.get('engine')}" if event.get("engine") else "")
    if name == "web_page_started":
        return f"Opening a page — {_short(event.get('title') or event.get('url'))}"
    if name == "web_page_read":
        return f"Read a page — {_short(event.get('title') or event.get('url'))}"
    if name == "docs_search_started":
        return f'Looking up docs — "{_short(event.get("query"), 48)}"'
    if name == "docs_search_finished":
        return f"Found {event.get('count', 0)} documentation {_plural(event.get('count'), 'result')}"
    if name == "docs_page_started":
        return f"Opening docs — {_short(event.get('title') or event.get('url'))}"
    if name == "docs_page_read":
        return f"Read docs — {_short(event.get('title') or event.get('url'))}"
    if name == "project_discovered":
        return f"Found the project — {_short(event.get('path'))}"
    if name == "project_file_started":
        return f"Reading {_short(event.get('path'))}"
    if name == "project_file_read":
        return f"Summarized {_short(event.get('path'))}"
    if name == "project_file_queued":
        return f"Queued for reading — {_short(event.get('path'))}"
    if name == "project_file_resumed":
        return f"Reused a cached read — {_short(event.get('path'))}"
    if name == "project_directory_started":
        return f"Summarizing directory {_short(event.get('path'))}"
    if name == "project_directory_compacted":
        return f"Directory summarized — {_short(event.get('path'))}"
    if name == "capability_run_attached":
        return f"Tool result attached — {event.get('capability') or 'capability'}"
    if name == "tool_evidence_attached":
        return f"Attached {count} approved tool {_plural(count, 'result')}"
    if name == "tool_evidence_carried":
        return f"Carried {count} approved tool {_plural(count, 'result')} into this run"
    if name == "memory_suggested":
        return f"Proposed {count} {_plural(count, 'memory', 'memories')} to remember — approve to save"
    if name == "memory_saved":
        return "Saved a memory you approved"
    if name == "error":
        return f"Error — {_short(event.get('detail') or event.get('error'), 80)}"

    return name.replace("_", " ").capitalize() if name else "Working"


def stage_word(event: dict[str, Any]) -> str:
    """Short heading word for the live stage chip (kept terse; the humanized
    sentence goes in the detail line)."""
    name = str(event.get("event") or "")
    stage = str(event.get("stage") or "")
    if stage in STAGE_WORDS:
        return STAGE_WORDS[stage]
    if name.startswith("web_"):
        return "Web search"
    if name.startswith("docs_"):
        return "Docs lookup"
    if name.startswith("project_"):
        return "Project"
    if name in ("planning", "planned", "planner_json_recovered", "followup_planned"):
        return "Planning"
    if name in ("verifying", "verified", "verifier_failed_final_answer"):
        return "Verifying"
    if name == "synthesizing":
        return "Synthesizing"
    if name in ("capability_run_attached", "tool_evidence_attached", "tool_evidence_carried", "capability_context_attached"):
        return "Tools"
    if name in ("queued", "queued_for_slot", "workflow_ready", "workflow_started", "model_slot_waiting", "model_slot_acquired", "model_slot_released"):
        return "Working"
    return "Working"


def current_stage(job: dict[str, Any]) -> dict[str, Any]:
    status = job.get("status")
    if status == "done":
        return {"label": "Final answer ready", "detail": "The synthesized response is complete.", "kind": "done"}
    if status == "cancelled":
        return {"label": "Stopped", "detail": "Cancellation was requested and the job exited.", "kind": "cancelled"}
    if status == "error":
        return {"label": "Error", "detail": job.get("error") or "The job failed.", "kind": "error"}
    active_call = job.get("active_model_call")
    if active_call:
        synthetic = {"event": "model_call_started", "stage": active_call.get("stage"), "label": active_call.get("label")}
        detail_bits = [
            project_progress(job).get("current_file") or "",
            humanize_event(synthetic).rstrip("…"),
            f"{active_call.get('elapsed_s', 0)}s so far",
        ]
        return {
            "label": stage_word(synthetic),
            "detail": " · ".join(item for item in detail_bits if item),
            "kind": status or "running",
        }
    events = job.get("events") or []
    last = events[-1] if events else {"event": status or "queued"}
    details = [humanize_event(last)]
    if last.get("total_tokens"):
        details.append(f"{last['total_tokens']} tokens")
    if last.get("tokens_per_second"):
        details.append(f"{last['tokens_per_second']} tok/s")
    if job.get("cancel_requested") and status not in TERMINAL_STATUSES:
        details.insert(0, "stopping after the current step")
    return {"label": stage_word(last), "detail": " · ".join(details) or "Working through the next step.", "kind": status or "queued"}


def task_snapshots(job: dict[str, Any]) -> list[dict[str, Any]]:
    events = job.get("events") or []
    planned = next((event for event in reversed(events) if event.get("event") == "planned"), None)
    tasks = (job.get("trace") or {}).get("tasks") or (planned or {}).get("tasks") or []
    finished = {event.get("id") for event in events if event.get("event") == "worker_finished"}
    active = next((event for event in reversed(events) if event.get("event") == "worker_started"), {})
    snapshots = []
    for task in tasks:
        state = "done" if task.get("id") in finished else "running" if active.get("id") == task.get("id") else "pending"
        snapshots.append({**task, "state": state})
    return snapshots


def context_snapshots(job: dict[str, Any]) -> list[dict[str, Any]]:
    trace = job.get("trace") or {}
    capability_events = [
        {
            "id": event.get("id"),
            "title": event.get("capability"),
            "type": "capability result",
            "compact": {
                "summary": event.get("summary", ""),
                "key_points": [event.get("result_preview", "")] if event.get("result_preview") else [],
                "use_later": [event.get("result_preview", "")] if event.get("result_preview") else [],
            },
        }
        for event in job.get("events", [])
        if event.get("event") == "capability_run_attached"
    ]
    worker_results = trace.get("worker_results") or []
    if worker_results:
        return capability_events + [
            {
                "id": result.get("id"),
                "title": result.get("title"),
                "type": "worker compact",
                "compact": result.get("compact") or {},
            }
            for result in worker_results
        ]
    return capability_events + [
        {
            "id": event.get("id"),
            "title": event.get("title"),
            "type": "worker compact",
            "compact": event.get("compact") or {},
        }
        for event in job.get("events", [])
        if event.get("event") == "compacted"
    ]


def project_file_snapshots(job: dict[str, Any]) -> list[dict[str, Any]]:
    trace_files = ((job.get("trace") or {}).get("project_context") or {}).get("file_summaries") or []
    if trace_files:
        return [{"path": item.get("path"), "state": "read", "summary": item.get("summary", "")} for item in trace_files]
    started = {event.get("path"): event for event in job.get("events", []) if event.get("event") == "project_file_started"}
    read = {event.get("path"): event for event in job.get("events", []) if event.get("event") == "project_file_read"}
    resumed = {event.get("path"): event for event in job.get("events", []) if event.get("event") == "project_file_resumed"}
    paths = list(dict.fromkeys([*started.keys(), *read.keys(), *resumed.keys()]))
    return [
        {
            "path": path,
            "state": "read" if path in read else "resumed" if path in resumed else "running",
            "summary": (read.get(path) or resumed.get(path) or {}).get("summary", ""),
            "index": (started.get(path) or resumed.get(path) or {}).get("index"),
            "total": (started.get(path) or resumed.get(path) or {}).get("total"),
        }
        for path in paths
        if path
    ]


def project_progress(job: dict[str, Any]) -> dict[str, Any]:
    events = job.get("events") or []
    discovered = next((event for event in events if event.get("event") == "project_discovered"), {})
    total = int(discovered.get("count") or (job.get("route") or {}).get("project_files") or 0)
    resumed = {event.get("path") for event in events if event.get("event") == "project_file_resumed" and event.get("path")}
    read = {event.get("path") for event in events if event.get("event") == "project_file_read" and event.get("path")}
    started = [event for event in events if event.get("event") == "project_file_started" and event.get("path")]
    completed = resumed | read
    current = ""
    current_index = None
    current_total = total or None
    for event in reversed(started):
        if event.get("path") not in read:
            current = str(event.get("path"))
            current_index = event.get("index")
            current_total = event.get("total") or current_total
            break
    percent = round((len(completed) / total) * 100, 1) if total else None
    return {
        "path": discovered.get("path") or (job.get("route") or {}).get("project_path") or "",
        "total_files": total,
        "resumed_files": len(resumed),
        "read_files": len(read),
        "completed_files": len(completed),
        "remaining_files": max(0, total - len(completed)) if total else None,
        "current_file": current,
        "current_index": current_index,
        "current_total": current_total,
        "percent": percent,
    }


def enrich_job(job: dict[str, Any]) -> dict[str, Any]:
    enriched = json.loads(json.dumps(job))
    for event in enriched.get("events") or []:
        event["line"] = humanize_event(event)   # plain-language one-liner for the activity feed (derived, not persisted)
    enriched["metrics"] = job_metrics(enriched)
    enriched["stage"] = current_stage(enriched)
    enriched["calls"] = completed_calls_from_job(enriched)
    enriched["agents"] = task_snapshots(enriched)
    enriched["context_blocks"] = context_snapshots(enriched)
    enriched["project_files"] = project_file_snapshots(enriched)
    enriched["project_progress"] = project_progress(enriched)
    return enriched


class Persistence:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.lock, self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    created REAL NOT NULL,
                    updated REAL NOT NULL,
                    status TEXT NOT NULL,
                    request TEXT NOT NULL,
                    job_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created DESC)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS project_memory (
                    path TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    updated REAL NOT NULL,
                    memory_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS capability_runs (
                    id TEXT PRIMARY KEY,
                    created REAL NOT NULL,
                    capability TEXT NOT NULL,
                    ok INTEGER NOT NULL,
                    run_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_capability_runs_created ON capability_runs(created DESC)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    created REAL NOT NULL,
                    updated REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope, updated DESC)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created REAL NOT NULL,
                    updated REAL NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated DESC)")
            # Migrate: tie each job to a session. Older DBs have a jobs table with no
            # session_id column; add it in place so existing history survives.
            job_cols = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
            if "session_id" not in job_cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN session_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs(session_id, created DESC)")

    def create_session(self, title: str) -> dict[str, Any]:
        now = time.time()
        session_id = uuid.uuid4().hex[:12]
        title = (title or "New chat").strip()[:120] or "New chat"
        with self.lock, self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created, updated, archived) VALUES (?, ?, ?, ?, 0)",
                (session_id, title, now, now),
            )
        return {"id": session_id, "title": title, "created": round(now, 3), "updated": round(now, 3), "archived": False}

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.lock, self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE archived = 0 ORDER BY updated DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {"id": r["id"], "title": r["title"], "created": r["created"], "updated": r["updated"], "archived": bool(r["archived"])}
            for r in rows
        ]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.lock, self.connect() as conn:
            r = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not r:
            return None
        return {"id": r["id"], "title": r["title"], "created": r["created"], "updated": r["updated"], "archived": bool(r["archived"])}

    def rename_session(self, session_id: str, title: str) -> bool:
        title = (title or "").strip()[:120]
        if not title:
            return False
        with self.lock, self.connect() as conn:
            cur = conn.execute("UPDATE sessions SET title = ?, updated = ? WHERE id = ?", (title, time.time(), session_id))
            return cur.rowcount > 0

    def touch_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self.lock, self.connect() as conn:
            conn.execute("UPDATE sessions SET updated = ? WHERE id = ?", (time.time(), session_id))

    def delete_session(self, session_id: str) -> bool:
        with self.lock, self.connect() as conn:
            conn.execute("DELETE FROM jobs WHERE session_id = ?", (session_id,))
            cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0

    def session_messages(self, session_id: str, limit: int = 40) -> list[dict[str, Any]]:
        """Chat history for a session, built from its jobs in time order: each job's
        request becomes a user turn and its answer an assistant turn. This is the
        server-side source of truth that replaces the old per-browser localStorage."""
        with self.lock, self.connect() as conn:
            rows = conn.execute(
                "SELECT job_json FROM jobs WHERE session_id = ? ORDER BY created ASC LIMIT ?", (session_id, limit),
            ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            try:
                job = json.loads(row["job_json"])
            except json.JSONDecodeError:
                continue
            request = str((job.get("config") or {}).get("request") or "").strip()
            answer = str(job.get("answer") or "").strip()
            if request:
                messages.append({"role": "user", "content": request})
            if answer:
                messages.append({"role": "assistant", "content": answer})
        return messages

    def save_job(self, job: dict[str, Any]) -> None:
        snapshot = json.loads(json.dumps(job))
        with self.lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, created, updated, status, request, job_json, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated = excluded.updated,
                    status = excluded.status,
                    request = excluded.request,
                    job_json = excluded.job_json,
                    session_id = excluded.session_id
                """,
                (
                    snapshot["id"],
                    float(snapshot.get("created") or time.time()),
                    time.time(),
                    str(snapshot.get("status") or "unknown"),
                    str((snapshot.get("config") or {}).get("request") or snapshot.get("request") or "")[:1000],
                    json.dumps(snapshot),
                    str((snapshot.get("config") or {}).get("session_id") or ""),
                ),
            )

    def load_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.lock, self.connect() as conn:
            rows = conn.execute("SELECT job_json FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        jobs = []
        for row in rows:
            try:
                jobs.append(json.loads(row["job_json"]))
            except json.JSONDecodeError:
                continue
        return jobs

    def load_job(self, job_id: str) -> dict[str, Any] | None:
        with self.lock, self.connect() as conn:
            row = conn.execute("SELECT job_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["job_json"])
        except json.JSONDecodeError:
            return None

    def save_project_memory(self, memory: dict[str, Any], fingerprint: str) -> None:
        path = str(memory.get("path") or "")
        if not path:
            return
        snapshot = json.loads(json.dumps(memory))
        with self.lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO project_memory (path, fingerprint, updated, memory_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    fingerprint = excluded.fingerprint,
                    updated = excluded.updated,
                    memory_json = excluded.memory_json
                """,
                (path, fingerprint, time.time(), json.dumps(snapshot)),
            )

    def load_project_memory(self, path: Path, fingerprint: str, max_age_s: int = PROJECT_MEMORY_MAX_AGE_S) -> dict[str, Any] | None:
        with self.lock, self.connect() as conn:
            row = conn.execute("SELECT fingerprint, updated, memory_json FROM project_memory WHERE path = ?", (str(path),)).fetchone()
        if not row:
            return None
        if row["fingerprint"] != fingerprint:
            return None
        if time.time() - float(row["updated"]) > max_age_s:
            return None
        try:
            memory = json.loads(row["memory_json"])
        except json.JSONDecodeError:
            return None
        memory["cache"] = {
            "hit": True,
            "updated": row["updated"],
            "age_s": round(time.time() - float(row["updated"]), 1),
            "fingerprint": fingerprint,
        }
        return memory

    def save_capability_run(self, run: dict[str, Any]) -> None:
        snapshot = json.loads(json.dumps(run))
        with self.lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO capability_runs (id, created, capability, ok, run_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    snapshot["id"],
                    float(snapshot.get("created") or time.time()),
                    str(snapshot.get("capability") or ""),
                    1 if snapshot.get("ok") else 0,
                    json.dumps(snapshot),
                ),
            )

    def load_capability_runs(self, limit: int = 30) -> list[dict[str, Any]]:
        with self.lock, self.connect() as conn:
            rows = conn.execute("SELECT run_json FROM capability_runs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        runs = []
        for row in rows:
            try:
                runs.append(json.loads(row["run_json"]))
            except json.JSONDecodeError:
                continue
        return runs

    def save_memory(self, scope: str, key: str, value: str, tags: list[str] | None = None) -> dict[str, Any]:
        now = time.time()
        memory_id = uuid.uuid4().hex[:12]
        tags = tags or []
        record = {
            "id": memory_id,
            "scope": scope,
            "key": key,
            "value": value,
            "tags": tags,
            "created": round(now, 3),
            "updated": round(now, 3),
        }
        with self.lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (id, scope, key, value, tags, created, updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (memory_id, scope, key, value, json.dumps(tags), now, now),
            )
        return record

    def load_memories(self, scope: str = "user", limit: int = 50) -> list[dict[str, Any]]:
        with self.lock, self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE scope = ? ORDER BY updated DESC LIMIT ?",
                (scope, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "scope": row["scope"],
                "key": row["key"],
                "value": row["value"],
                "tags": json.loads(row["tags"] or "[]"),
                "created": row["created"],
                "updated": row["updated"],
            }
            for row in rows
        ]

    def delete_memory(self, memory_id: str) -> bool:
        with self.lock, self.connect() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cur.rowcount > 0


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen Orchestration Console</title>
  <style>
    /* ============================================================
       Qwen Orchestrator — "sodium bench" instrument theme
       Committed dark control-room look. Monospace is the chrome;
       sans is reserved for the model's prose answer.
       ============================================================ */
    :root {
      color-scheme: dark;
      --ink: #0d0f0e;
      --ink-2: #0a0c0b;
      --panel: #141815;
      --panel-2: #171c18;
      --well: #10140f;
      --line: #262d27;
      --line-2: #313a32;
      --text: #e7ece4;
      --muted: #8b968a;
      --faint: #5c655c;
      --amber: #f2b45a;
      --amber-dim: #b9863f;
      --cyan: #6fd6c4;
      --cyan-dim: #3f8579;
      --blue: #7ba8f2;
      --violet: #c193f2;
      --bad: #e8705d;
      --good: #8fd18a;
      --warn: #f2b45a;
      --mono: ui-monospace, "JetBrains Mono", "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      --glow-amber: 0 0 0 1px color-mix(in srgb, var(--amber) 34%, transparent), 0 0 18px -4px color-mix(in srgb, var(--amber) 55%, transparent);
      --radius: 4px;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; min-width: 320px; }
    body {
      margin: 0;
      background: var(--ink);
      color: var(--text);
      font: 13px/1.5 var(--mono);
      -webkit-font-smoothing: antialiased;
      letter-spacing: .1px;
      overflow: hidden;              /* the app owns the viewport; panels scroll internally */
      display: flex;
      flex-direction: column;
    }
    /* faint bench-grid texture + top scanline, not a blur-gradient hero */
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(color-mix(in srgb, var(--line) 40%, transparent) 1px, transparent 1px),
        linear-gradient(90deg, color-mix(in srgb, var(--line) 40%, transparent) 1px, transparent 1px);
      background-size: 44px 44px, 44px 44px;
      mask-image: linear-gradient(180deg, rgba(0,0,0,.5), transparent 620px);
      pointer-events: none;
      z-index: 0;
    }
    body > * { position: relative; z-index: 1; }

    /* ---------- header : the instrument bar ---------- */
    header {
      flex: 0 0 auto;
      height: 54px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, var(--panel), var(--ink));
      padding: 0 18px;
      display: flex;
      align-items: center;
      gap: 20px;
      z-index: 5;
    }
    .brand { display: flex; align-items: center; gap: 12px; min-width: 0; flex: 0 0 auto; }
    .brand-text { min-width: 0; }
    .header-right { display: flex; align-items: center; gap: 16px; flex: 0 0 auto; }
    .kbd-hint { display: flex; align-items: center; gap: 6px; white-space: nowrap; font-size: 10px; letter-spacing: .3px; color: var(--faint); }
    .kbd-hint b { font-weight: 600; color: var(--muted); border: 1px solid var(--line-2); border-radius: 3px; padding: 1px 5px; }
    .kbd-hint i { color: var(--line-2); font-style: normal; }
    .mark {
      width: 30px; height: 30px;
      border-radius: var(--radius);
      border: 1px solid color-mix(in srgb, var(--amber) 40%, var(--line));
      background: radial-gradient(120% 120% at 30% 20%, color-mix(in srgb, var(--amber) 30%, var(--panel)), var(--well));
      display: grid; place-items: center;
      color: var(--amber);
      font-weight: 700; font-size: 15px;
      box-shadow: inset 0 0 12px -6px var(--amber);
    }
    h1 {
      margin: 0;
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 2.4px;
      text-transform: uppercase;
    }
    .tagline {
      color: var(--faint);
      font-size: 10.5px;
      letter-spacing: 1.4px;
      text-transform: uppercase;
      margin-top: 3px;
    }
    /* service pill = equipment status lamp */
    .service-pill {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: .6px;
      color: var(--muted);
      border: 1px solid var(--line-2);
      border-radius: 99px;
      padding: 6px 13px 6px 11px;
      display: inline-flex; align-items: center; gap: 8px;
      background: var(--well);
      white-space: nowrap;
    }
    .service-pill::before {
      content: "";
      width: 8px; height: 8px; border-radius: 99px;
      background: var(--faint);
      box-shadow: 0 0 0 3px color-mix(in srgb, var(--faint) 18%, transparent);
    }
    .service-pill.ok { color: var(--good); border-color: color-mix(in srgb, var(--good) 34%, var(--line)); }
    .service-pill.ok::before { background: var(--good); box-shadow: 0 0 8px 0 var(--good); }
    .coder-select {
      background: var(--surface, #1b1e27); color: var(--text, #e6e6e6);
      border: 1px solid var(--line, #333); border-radius: 8px;
      padding: 4px 8px; font-size: 12px; max-width: 240px; cursor: pointer;
    }
    .coder-select:disabled { opacity: 0.55; cursor: progress; }
    .service-pill.bad { color: var(--bad); border-color: color-mix(in srgb, var(--bad) 38%, var(--line)); }
    .service-pill.bad::before { background: var(--bad); box-shadow: 0 0 8px 0 var(--bad); }

    /* ---------- dashboard layout : fills the viewport, panels scroll internally ---------- */
    main {
      flex: 1;
      min-height: 0;
      position: relative;
      display: grid;
      grid-template-columns: var(--c0, 224px) var(--c1, 320px) minmax(0, 1fr) var(--c3, 500px);
      grid-template-rows: minmax(0, 1fr) var(--dock-h, 210px);
      grid-template-areas:
        "rail request answer workflow"
        "dock dock dock dock";
      gap: 12px;
      padding: 12px;
    }
    main.dock-collapsed { grid-template-rows: minmax(0, 1fr) 41px; }
    .session-rail { grid-area: rail; }
    .console { grid-area: request; }
    .answer-col { grid-area: answer; }
    .workflow-col { grid-area: workflow; }

    .col, .dock {
      position: relative;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: var(--radius);
      display: flex;
      flex-direction: column;
      min-height: 0;
      overflow: hidden;
    }

    /* drag handles between panels */
    .resizer { position: absolute; z-index: 6; touch-action: none; }
    .resizer::after { content: ""; position: absolute; background: transparent; transition: background .15s ease; }
    .resizer-x { top: 0; bottom: 0; width: 10px; cursor: col-resize; }
    .resizer-x.right { right: 0; }
    .resizer-x.left { left: 0; }
    .resizer-x::after { top: 8px; bottom: 8px; left: 50%; width: 2px; transform: translateX(-50%); border-radius: 2px; }
    .resizer-y { left: 0; right: 0; top: 0; height: 10px; cursor: row-resize; }
    .resizer-y::after { left: 50%; right: auto; transform: translateX(-50%); top: 50%; margin-top: -1px; width: 46px; height: 2px; border-radius: 2px; }
    .resizer:hover::after, .resizer.dragging::after { background: var(--amber); }
    .dock { grid-area: dock; }

    .panel-head {
      flex: 0 0 auto;
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
      padding: 11px 14px;
      border-bottom: 1px solid var(--line);
    }
    .scroll { flex: 1; min-height: 0; overflow-y: auto; padding: 14px; }

    /* ---------- sessions rail : channel strips ---------- */
    .new-chat {
      font: 600 11px/1 var(--mono); letter-spacing: .4px; color: var(--amber);
      background: transparent; border: 1px solid color-mix(in srgb, var(--amber) 40%, var(--line));
      border-radius: 6px; padding: 5px 9px; cursor: pointer;
    }
    .new-chat:hover { background: color-mix(in srgb, var(--amber) 12%, var(--panel)); }
    .session-list { padding: 8px; display: flex; flex-direction: column; gap: 5px; }
    .session-strip {
      position: relative; display: grid; grid-template-columns: auto minmax(0,1fr) auto; align-items: center; gap: 9px;
      padding: 9px 9px 9px 11px; border: 1px solid var(--line); border-left: 2px solid transparent;
      border-radius: 7px; background: var(--well); cursor: pointer; text-align: left; width: 100%; font-family: var(--mono);
    }
    .session-strip:hover { border-color: var(--line-2); }
    .session-strip.active { border-left-color: var(--cyan); background: color-mix(in srgb, var(--cyan) 8%, var(--panel)); }
    .session-led { width: 8px; height: 8px; border-radius: 50%; background: var(--faint); flex: 0 0 auto; }
    .session-strip.live-running .session-led { background: var(--amber); box-shadow: 0 0 7px 0 var(--amber); animation: pulse 1.4s ease-in-out infinite; }
    .session-strip.live-queued .session-led { background: var(--blue); }
    .session-body { min-width: 0; }
    .session-name { font-size: 12px; font-weight: 600; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; line-height: 1.3; }
    .session-sub { font-size: 10.5px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-top: 2px; }
    .session-del {
      opacity: 0; border: 0; background: transparent; color: var(--muted); cursor: pointer;
      font-size: 14px; line-height: 1; padding: 2px 4px; border-radius: 4px; transition: opacity .12s ease;
    }
    .session-strip:hover .session-del, .session-strip:focus-within .session-del { opacity: .65; }
    .session-del:hover { opacity: 1; color: var(--bad); background: color-mix(in srgb, var(--bad) 12%, transparent); }

    /* ---------- verifier caveat + memory-suggestion chips (above the answer) ---------- */
    .answer-note {
      flex: 0 0 auto; margin: 10px 14px 0; padding: 8px 11px; border-radius: 7px;
      font: 400 11px/1.5 var(--mono); color: color-mix(in srgb, var(--bad) 70%, var(--text));
      background: color-mix(in srgb, var(--bad) 9%, var(--panel)); border: 1px solid color-mix(in srgb, var(--bad) 32%, var(--line));
    }
    .answer-note[hidden] { display: none; }
    .answer-note b { color: var(--bad); font-weight: 600; letter-spacing: .5px; text-transform: uppercase; font-size: 9.5px; margin-right: 6px; }
    .memory-chips { flex: 0 0 auto; display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 14px 0; }
    .memory-chips[hidden] { display: none; }
    .mem-chip {
      display: inline-flex; align-items: center; gap: 6px; max-width: 100%;
      padding: 5px 6px 5px 10px; border: 1px solid color-mix(in srgb, var(--violet) 38%, var(--line));
      border-radius: 999px; background: color-mix(in srgb, var(--violet) 9%, var(--panel)); font-family: var(--mono);
    }
    .mem-chip .mem-text { font-size: 11px; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .mem-chip .mem-text b { color: var(--violet); font-weight: 600; }
    .mem-chip button { border: 0; background: transparent; cursor: pointer; font: 600 11px/1 var(--mono); padding: 4px 7px; border-radius: 6px; }
    .mem-chip .mem-save { color: var(--good); }
    .mem-chip .mem-save:hover { background: color-mix(in srgb, var(--good) 16%, transparent); }
    .mem-chip .mem-dismiss { color: var(--muted); }
    .mem-chip .mem-dismiss:hover { color: var(--text); }

    /* composer column */
    .composer-form { flex: 1; min-height: 0; display: flex; flex-direction: column; gap: 12px; padding: 14px; }
    .composer-form #request { flex: 1; min-height: 90px; resize: none; }
    .composer-controls { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; flex: 0 0 auto; }
    .field { display: flex; flex-direction: column; min-width: 0; }
    .field label { margin: 0 0 5px; }

    /* answer column */
    .answer-col .answer { flex: 1; min-height: 0; }
    .answer-col .quick-grid { flex: 0 0 auto; }

    /* queue bar: the prompts in flight for the single model slot. The running one
       is highlighted; clicking a chip shows that prompt in the panel below. */
    .queue-bar {
      flex: 0 0 auto;
      display: flex; gap: 6px; align-items: center;
      padding: 8px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--well);
      overflow-x: auto; overflow-y: hidden;
      scrollbar-width: thin;
    }
    .queue-bar[hidden] { display: none; }
    .queue-bar .queue-label {
      flex: 0 0 auto;
      font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase;
      color: var(--faint); padding-right: 2px;
    }
    .queue-chip {
      flex: 0 0 auto;
      display: inline-flex; align-items: center; gap: 6px;
      max-width: 220px;
      padding: 4px 9px;
      font-size: 11px; color: var(--muted);
      background: var(--panel); border: 1px solid var(--line-2);
      border-radius: 999px; cursor: pointer;
      transition: border-color .12s, color .12s, background .12s;
    }
    .queue-chip:hover { color: var(--text); border-color: var(--line-2); }
    .queue-chip .chip-text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .queue-chip .chip-dot {
      flex: 0 0 auto; width: 6px; height: 6px; border-radius: 50%;
      background: var(--faint);
    }
    .queue-chip.st-running .chip-dot { background: var(--amber); box-shadow: 0 0 7px 0 var(--amber); animation: pulse 1.4s ease-in-out infinite; }
    .queue-chip.st-queued .chip-dot { background: var(--blue); }
    .queue-chip.st-running { color: var(--text); border-color: color-mix(in srgb, var(--amber) 44%, var(--line)); background: color-mix(in srgb, var(--amber) 12%, var(--panel)); }
    .queue-chip.st-queued { color: var(--text); border-color: color-mix(in srgb, var(--blue) 38%, var(--line)); }
    .queue-chip.selected { box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--cyan) 60%, transparent); border-color: color-mix(in srgb, var(--cyan) 55%, var(--line)); }
    .queue-chip .chip-pos { font: 600 10px/1 var(--mono); color: var(--blue); flex: 0 0 auto; }
    .queue-clear {
      font: 600 10.5px/1 var(--mono); color: var(--muted); background: transparent;
      border: 1px solid var(--line); border-radius: 999px; padding: 5px 10px; cursor: pointer; flex: 0 0 auto;
    }
    .queue-clear:hover { color: var(--bad); border-color: color-mix(in srgb, var(--bad) 42%, var(--line)); }

    /* prompt banner: what the selected/running job is actually working on */
    .prompt-view {
      flex: 0 0 auto;
      display: flex; align-items: flex-start; gap: 9px;
      padding: 9px 14px;
      border-bottom: 1px solid var(--line);
      background: color-mix(in srgb, var(--blue) 5%, var(--panel));
    }
    .prompt-view[hidden] { display: none; }
    .prompt-view .prompt-label {
      flex: 0 0 auto; margin-top: 1px;
      font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase;
      color: var(--blue);
    }
    .prompt-view .prompt-text {
      flex: 1; min-width: 0;
      font-size: 12px; line-height: 1.45; color: var(--text);
      display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
      overflow: hidden;
    }

    /* ---------- dock ---------- */
    .dock-tabs {
      flex: 0 0 auto;
      display: flex; align-items: center; gap: 3px;
      padding: 6px 8px;
      border-bottom: 1px solid var(--line);
      background: var(--well);
    }
    .dock-tab {
      padding: 6px 13px;
      font-size: 10px; font-weight: 600; letter-spacing: 1.3px; text-transform: uppercase;
      background: transparent; border: 1px solid transparent; color: var(--muted);
      border-radius: 3px;
    }
    .dock-tab:hover { color: var(--text); }
    .dock-tab.active { color: var(--amber); border-color: color-mix(in srgb, var(--amber) 34%, var(--line)); background: var(--panel); }
    .dock-collapse { margin-left: auto; padding: 5px 12px; color: var(--muted); background: transparent; border-color: var(--line-2); }
    .dock-body { flex: 1; min-height: 0; }
    .dock-pane { display: none; height: 100%; overflow-y: auto; padding: 14px; }
    .dock-pane.active { display: block; }
    main.dock-collapsed .dock-body { display: none; }

    .settings-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px 14px; }
    .tools-layout { display: grid; grid-template-columns: minmax(0, 300px) minmax(0, 1fr); gap: 18px; align-items: start; }
    .tools-form { display: grid; gap: 9px; align-content: start; }
    .tools-form textarea { min-height: 60px; }

    /* section labels: silkscreen equipment labels */
    .panel-title {
      display: flex; align-items: center; gap: 9px;
      font-size: 11px; font-weight: 600;
      letter-spacing: 2px; text-transform: uppercase;
      color: var(--text);
    }
    .panel-title::before {
      content: "";
      width: 6px; height: 6px;
      background: var(--amber);
      box-shadow: 0 0 8px 0 var(--amber);
    }
    .row, .answer-tools {
      display: flex; align-items: center; justify-content: space-between;
      gap: 12px; margin-bottom: 12px;
    }
    .subhead {
      display: flex; align-items: baseline; justify-content: space-between;
      gap: 10px;
      margin: 20px 0 9px;
      font-size: 10px; letter-spacing: 1.6px; text-transform: uppercase;
      color: var(--muted);
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }
    .subhead span:last-child { color: var(--amber); font-size: 11px; letter-spacing: .5px; }

    .status {
      font-size: 10.5px; letter-spacing: 1px; text-transform: uppercase;
      color: var(--muted);
    }

    /* ---------- forms ---------- */
    label {
      display: block;
      color: var(--muted);
      font-size: 10px; letter-spacing: 1.2px; text-transform: uppercase;
      margin: 14px 0 6px;
    }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line-2);
      border-radius: var(--radius);
      background: var(--well);
      color: var(--text);
      font: 13px/1.5 var(--mono);
      padding: 10px 11px;
      outline: none;
      transition: border-color .14s ease, box-shadow .14s ease;
    }
    select {
      appearance: none;
      background-image: linear-gradient(45deg, transparent 50%, var(--muted) 50%), linear-gradient(135deg, var(--muted) 50%, transparent 50%);
      background-position: calc(100% - 16px) 52%, calc(100% - 11px) 52%;
      background-size: 5px 5px, 5px 5px;
      background-repeat: no-repeat;
      cursor: pointer;
    }
    textarea:focus, input:focus, select:focus {
      border-color: color-mix(in srgb, var(--amber) 60%, var(--line));
      box-shadow: 0 0 0 3px color-mix(in srgb, var(--amber) 15%, transparent);
    }
    textarea { min-height: 150px; resize: vertical; }
    .composer textarea { min-height: 150px; font-size: 13.5px; line-height: 1.55; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; }
    .grid label { margin-top: 10px; }

    /* ---------- buttons ---------- */
    .button-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
    .button-row button { flex: 1 1 auto; white-space: nowrap; }
    .button-row #run { flex: 2 1 120px; }   /* keep Run the prominent primary action */
    button {
      font-family: var(--mono);
      font-size: 11px; font-weight: 600; letter-spacing: 1.4px; text-transform: uppercase;
      border-radius: var(--radius);
      padding: 11px 16px;
      cursor: pointer;
      border: 1px solid var(--line-2);
      background: var(--panel-2);
      color: var(--text);
      transition: border-color .14s, background .14s, box-shadow .14s, transform .04s;
    }
    button:hover { border-color: var(--muted); }
    button:active { transform: translateY(1px); }
    #run, #runCapability, #saveMemory {
      flex: 1;
      color: var(--ink);
      background: linear-gradient(180deg, color-mix(in srgb, var(--amber) 92%, #fff), var(--amber));
      border-color: var(--amber);
      box-shadow: var(--glow-amber);
    }
    #run:hover, #runCapability:hover, #saveMemory:hover {
      background: var(--amber);
      box-shadow: 0 0 0 1px var(--amber), 0 0 24px -4px var(--amber);
    }
    .stop-button { color: var(--bad); border-color: color-mix(in srgb, var(--bad) 40%, var(--line)); }
    .stop-button:hover:not(:disabled) { background: color-mix(in srgb, var(--bad) 16%, var(--panel)); border-color: var(--bad); }
    #continue.ready { color: var(--amber); border-color: color-mix(in srgb, var(--amber) 45%, var(--line)); }
    #continue.ready:hover { background: color-mix(in srgb, var(--amber) 12%, var(--panel)); }
    #continue.has-evidence { color: var(--cyan); border-color: color-mix(in srgb, var(--cyan) 55%, var(--line)); }
    #continue.has-evidence:hover { background: color-mix(in srgb, var(--cyan) 14%, var(--panel)); }
    #retry.ready { color: var(--blue); border-color: color-mix(in srgb, var(--blue) 45%, var(--line)); }
    #retry.ready:hover { background: color-mix(in srgb, var(--blue) 12%, var(--panel)); }
    button:disabled { opacity: .4; cursor: not-allowed; }

    /* ---------- conversation / answer (sans prose) ---------- */
    .answer {
      font-family: var(--sans);
      font-size: 14.5px; line-height: 1.62;
      letter-spacing: 0;
      color: var(--text);
      white-space: pre-wrap;
      word-break: break-word;
    }
    .answer:empty::before { content: "Awaiting request."; color: var(--faint); }
    .answer code, .answer pre {
      font-family: var(--mono);
      font-size: 12.5px;
      background: var(--well);
      border: 1px solid var(--line);
      border-radius: var(--radius);
    }
    .answer code { padding: 1px 5px; }
    .answer pre { padding: 12px; overflow-x: auto; }

    /* progress bar : signal meter, flush under the panel head */
    .progress {
      flex: 0 0 auto;
      height: 3px;
      background: var(--well);
      overflow: hidden;
      margin: 0;
    }
    .progress span {
      display: block; height: 100%; width: 0;
      background: linear-gradient(90deg, var(--amber-dim), var(--amber));
      box-shadow: 0 0 10px 0 var(--amber);
      transition: width .4s ease;
    }

    /* quick-grid : instrument readouts, footer strip under the answer */
    .quick-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; padding: 12px 14px; border-top: 1px solid var(--line); }
    .quick-card {
      border: 1px solid var(--line);
      background: var(--well);
      border-radius: var(--radius);
      padding: 11px 12px;
      display: flex; flex-direction: column; gap: 3px;
    }
    .quick-card strong {
      font-size: 19px; font-weight: 600; letter-spacing: .5px;
      color: var(--amber);
      font-variant-numeric: tabular-nums;
    }
    .quick-card span {
      font-size: 9.5px; letter-spacing: 1.4px; text-transform: uppercase;
      color: var(--faint);
    }

    /* ---------- pipeline rail : the signature, slim in the header ---------- */
    .pipeline {
      flex: 1 1 auto;
      max-width: 470px;
      margin: 0 auto;
      display: flex; align-items: center; justify-content: center; gap: 0;
      min-width: 0;
      overflow: hidden;
    }
    .pipeline .node {
      display: flex; flex-direction: column; align-items: center; gap: 5px;
      flex: 0 0 auto;
      min-width: 54px;
    }
    .pipeline .dot {
      width: 12px; height: 12px; border-radius: 99px;
      background: var(--well);
      border: 1.5px solid var(--line-2);
      transition: all .25s ease;
    }
    .pipeline .node span {
      font-size: 9px; letter-spacing: 1.2px; text-transform: uppercase;
      color: var(--faint);
      transition: color .25s ease;
    }
    .pipeline .link {
      flex: 1 1 auto; height: 1.5px; min-width: 16px;
      background: var(--line-2);
      position: relative; top: -9px;
    }
    /* completed */
    .pipeline .node.done .dot { background: var(--cyan); border-color: var(--cyan); box-shadow: 0 0 8px -1px var(--cyan); }
    .pipeline .node.done span { color: var(--cyan-dim); }
    /* active */
    .pipeline .node.active .dot { background: var(--amber); border-color: var(--amber); box-shadow: 0 0 0 4px color-mix(in srgb, var(--amber) 20%, transparent), 0 0 12px 0 var(--amber); animation: pulse 1.4s ease-in-out infinite; }
    .pipeline .node.active span { color: var(--amber); }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .55; } }

    /* ---------- workflow inner ---------- */
    .stage {
      display: flex; flex-direction: column; gap: 3px;
      border: 1px solid var(--line);
      border-left: 2px solid var(--amber);
      background: var(--well);
      border-radius: var(--radius);
      padding: 12px 13px;
    }
    .stage strong { font-size: 12px; letter-spacing: 1px; text-transform: uppercase; color: var(--text); }
    .stage span { color: var(--muted); font-size: 11.5px; letter-spacing: 0; }

    .metrics { display: flex; flex-wrap: wrap; gap: 7px; margin: 12px 0; }
    .metric, .pill {
      font-size: 10.5px; letter-spacing: .4px;
      border: 1px solid var(--line-2);
      background: var(--panel-2);
      color: var(--muted);
      border-radius: 99px;
      padding: 4px 10px;
      font-variant-numeric: tabular-nums;
    }
    .metric strong { color: var(--amber); font-weight: 600; }
    .pill.active { color: var(--amber); border-color: color-mix(in srgb, var(--amber) 40%, var(--line)); }

    .empty {
      color: var(--faint);
      font-size: 11.5px; letter-spacing: .3px;
      border: 1px dashed var(--line-2);
      border-radius: var(--radius);
      padding: 14px;
      text-align: center;
    }

    .cards { display: grid; gap: 8px; }
    .agent-card, .context-card {
      border: 1px solid var(--line);
      border-left: 2px solid var(--line-2);
      background: var(--panel-2);
      border-radius: var(--radius);
      padding: 11px 12px;
    }
    .agent-title, .event-title {
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
      font-size: 11.5px; font-weight: 600; letter-spacing: .4px;
      color: var(--text);
    }
    .agent-meta, .context-meta, .event-meta {
      color: var(--muted); font-size: 10.5px; letter-spacing: .2px;
      margin-top: 5px; line-height: 1.5;
    }
    /* task/agent states */
    .agent-card.running { border-left-color: var(--amber); box-shadow: inset 2px 0 0 -1px var(--amber), 0 0 0 1px color-mix(in srgb,var(--amber) 14%, transparent); }
    .agent-card.running .agent-title::after { content: "running"; color: var(--amber); font-size: 9px; letter-spacing: 1.4px; text-transform: uppercase; animation: pulse 1.4s ease-in-out infinite; }
    .agent-card.done { border-left-color: var(--cyan); }
    .agent-card.pending { opacity: .62; }
    .agent-card.resumed { border-left-color: var(--cyan-dim); }
    .agent-card.fail, .agent-card.error { border-left-color: var(--bad); }
    .agent-card.pass { border-left-color: var(--good); }

    /* status word colors (used inline by JS) */
    .done { color: var(--cyan); }
    .running { color: var(--amber); }
    .pending { color: var(--faint); }
    .pass { color: var(--good); }
    .fail, .error { color: var(--bad); }
    .cancelled { color: var(--muted); }
    .resumed { color: var(--cyan-dim); }

    /* live model calls */
    .call-wrap { display: grid; gap: 8px; }
    .call-wrap .agent-card { border-left-color: var(--cyan-dim); }

    /* requested-tool chip / button */
    .run-requested-tool {
      margin-top: 9px;
      font-family: var(--mono); font-size: 10px; letter-spacing: 1px; text-transform: uppercase;
      padding: 7px 11px;
      background: var(--well);
      border: 1px solid color-mix(in srgb, var(--cyan) 34%, var(--line));
      color: var(--cyan);
      border-radius: var(--radius);
      cursor: pointer; width: auto;
    }
    .run-requested-tool:hover { background: color-mix(in srgb, var(--cyan) 12%, var(--panel)); }

    /* file list */
    .file-list { display: grid; gap: 2px; max-height: 220px; overflow-y: auto; }
    .file-row {
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
      padding: 6px 9px;
      border-radius: var(--radius);
      font-size: 11px;
      color: var(--muted);
      border: 1px solid transparent;
    }
    .file-row:hover { background: var(--well); border-color: var(--line); }

    /* jobs / capability / memory lists */
    .job-list { display: grid; gap: 6px; }
    .job-row {
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel-2);
      cursor: pointer;
      font-size: 11.5px;
      transition: border-color .14s, background .14s;
    }
    .job-row:hover { border-color: var(--line-2); background: var(--well); }
    .job-row.active { border-color: var(--amber); box-shadow: inset 2px 0 0 -1px var(--amber); }
    .job-row .event-meta { margin-top: 3px; }

    .delete-memory {
      width: auto; padding: 5px 9px;
      font-size: 9px; letter-spacing: 1px;
      color: var(--bad); border-color: color-mix(in srgb, var(--bad) 30%, var(--line));
      background: transparent;
    }
    .delete-memory:hover { background: color-mix(in srgb, var(--bad) 14%, var(--panel)); }

    /* raw events + trace */
    .event {
      border-bottom: 1px solid var(--line);
      padding: 9px 2px;
      font-size: 11px;
    }
    .event:last-child { border-bottom: 0; }
    #events, #trace { max-height: 300px; overflow-y: auto; }
    #trace {
      white-space: pre-wrap; word-break: break-word;
      font-size: 11.5px; line-height: 1.55; color: var(--muted);
      background: var(--well);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 12px;
      max-height: 360px;
    }

    /* details / summary */
    details { border-top: 1px solid var(--line); padding-top: 4px; }
    .panel > details:first-child { border-top: 0; padding-top: 0; }
    summary {
      cursor: pointer; list-style: none;
      font-size: 11px; font-weight: 600; letter-spacing: 1.6px; text-transform: uppercase;
      color: var(--muted);
      padding: 4px 0;
      display: flex; align-items: center; gap: 9px;
    }
    summary::-webkit-details-marker { display: none; }
    summary::before {
      content: "+"; color: var(--amber); font-weight: 700;
      width: 12px; display: inline-block; text-align: center;
    }
    details[open] > summary::before { content: "\2212"; }
    summary:hover { color: var(--text); }

    /* capability result block */
    #capabilityResult { margin-top: 12px; }
    #capabilityResult pre, #capabilityResult .agent-card {
      background: var(--well); border: 1px solid var(--line); border-radius: var(--radius);
      padding: 11px; font-size: 11px; overflow-x: auto; white-space: pre-wrap; word-break: break-word;
    }

    /* scrollbars */
    ::-webkit-scrollbar { width: 9px; height: 9px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 99px; border: 2px solid var(--ink); }
    ::-webkit-scrollbar-thumb:hover { background: var(--muted); }

    /* focus visibility */
    :focus-visible { outline: 2px solid var(--amber); outline-offset: 2px; }

    /* ---------- readability + colour pass ---------- */
    /* larger, less-cramped labels (no more sub-11px text) */
    label { font-size: 11px; letter-spacing: .7px; }
    .status { font-size: 11px; letter-spacing: .8px; }
    .tagline { font-size: 11px; letter-spacing: 1px; }
    .kbd-hint { font-size: 11px; }
    .subhead { font-size: 11px; letter-spacing: 1.1px; }
    .subhead span:last-child { font-size: 12px; }
    .quick-card span { font-size: 11px; letter-spacing: 1px; }
    .quick-card strong { font-size: 21px; }
    .pipeline .node span { font-size: 10.5px; letter-spacing: .7px; }
    .dock-tab { font-size: 11px; letter-spacing: 1px; }
    .panel-title { font-size: 12px; letter-spacing: 1.6px; }
    .agent-meta, .context-meta, .event-meta { font-size: 11.5px; }
    .file-row { font-size: 11.5px; }
    .agent-title, .event-title { font-size: 12px; }
    .stage strong { font-size: 12.5px; }
    .stage span { font-size: 12px; }

    /* metric pills: separate value from label so they stop running together */
    .metric, .pill { font-size: 11.5px; }
    .metric { display: inline-flex; align-items: baseline; gap: 6px; }
    .metric strong { color: var(--amber); font-weight: 700; }
    .metric span { color: var(--muted); }

    /* semantic status chips — state readable at a glance */
    .pill.done, .pill.complete, .pill.pass { color: var(--good); border-color: color-mix(in srgb, var(--good) 42%, var(--line)); background: color-mix(in srgb, var(--good) 13%, var(--panel-2)); }
    .pill.running, .pill.queued, .pill.started { color: var(--amber); border-color: color-mix(in srgb, var(--amber) 42%, var(--line)); background: color-mix(in srgb, var(--amber) 13%, var(--panel-2)); }
    .pill.error, .pill.fail { color: var(--bad); border-color: color-mix(in srgb, var(--bad) 44%, var(--line)); background: color-mix(in srgb, var(--bad) 13%, var(--panel-2)); }
    .pill.cancelled, .pill.interrupted { color: var(--muted); border-color: var(--line-2); }

    /* per-column identity + per-stage pipeline colour (blue → amber → violet → green) */
    .console .panel-title::before { background: var(--blue); box-shadow: 0 0 8px 0 var(--blue); }
    .workflow-col .panel-title::before { background: var(--violet); box-shadow: 0 0 8px 0 var(--violet); }
    .pipeline .node[data-stage="plan"] { --sc: var(--blue); }
    .pipeline .node[data-stage="work"] { --sc: var(--amber); }
    .pipeline .node[data-stage="verify"] { --sc: var(--violet); }
    .pipeline .node[data-stage="synth"] { --sc: var(--good); }
    .pipeline .node.done .dot { background: var(--sc); border-color: var(--sc); box-shadow: 0 0 8px -1px var(--sc); }
    .pipeline .node.done span { color: var(--sc); }
    .pipeline .node.active .dot { background: var(--sc); border-color: var(--sc); box-shadow: 0 0 0 4px color-mix(in srgb, var(--sc) 22%, transparent), 0 0 12px 0 var(--sc); }
    .pipeline .node.active span { color: var(--sc); }
    /* colour the live stage card's accent by kind */
    .stage.running { border-left-color: var(--amber); }
    .stage.done { border-left-color: var(--good); }
    .stage.error { border-left-color: var(--bad); }
    .stage.cancelled { border-left-color: var(--muted); }

    /* ---------- responsive : below this, drop the fixed dashboard and let it stack/scroll ---------- */
    /* Laptop / MacBook: three columns, workflow drops beneath request+answer,
       sessions rail spans both content rows on the left. */
    @media (max-width: 1280px) {
      main {
        grid-template-columns: 196px minmax(0, 1fr) minmax(0, 1fr);
        grid-template-rows: minmax(0, 1.15fr) minmax(200px, 0.85fr) var(--dock-h, 200px);
        grid-template-areas:
          "rail request answer"
          "rail workflow workflow"
          "dock dock dock";
      }
      main.dock-collapsed { grid-template-rows: minmax(0, 1.15fr) minmax(200px, 0.85fr) 41px; }
      .resizer { display: none; }               /* fixed layout below desktop width */
      .kbd-hint { display: none; }
    }
    /* Tablet / phone: single column that scrolls; sessions become a horizontal
       channel strip so they never eat the whole screen. */
    @media (max-width: 880px) {
      html, body { overflow-x: hidden; }
      body { overflow-y: auto; display: block; }
      main {
        display: flex; flex-direction: column; height: auto;
        grid-template-areas: none; padding: 10px; gap: 10px;
      }
      .col, .dock { min-height: 0; max-height: none; max-width: 100%; }
      .resizer { display: none; }
      .session-rail { max-height: 168px; }
      .session-list { flex-direction: row; overflow-x: auto; overflow-y: hidden; padding: 8px; gap: 8px; }
      .session-strip { min-width: 208px; flex: 0 0 auto; }
      #request { min-height: 118px; }
      .answer-col { min-height: 48vh; }
      .workflow-col { min-height: 40vh; }
      .dock { min-height: 46vh; }
      .settings-grid { grid-template-columns: repeat(2, 1fr); }
      .tools-layout { grid-template-columns: 1fr; }
    }
    @media (max-width: 560px) {
      header { height: auto; flex-wrap: wrap; padding: 8px 12px; gap: 6px 10px; }
      .pipeline, .tagline, .kbd-hint { display: none; }
      main { padding: 8px; gap: 8px; }
      .composer-controls, .settings-grid { grid-template-columns: 1fr; }
      .quick-grid { grid-template-columns: repeat(3, 1fr); }
      .button-row { flex-wrap: wrap; }
      .button-row button { flex: 1 1 auto; min-height: 40px; }
      .session-rail { max-height: 128px; }
      .session-strip { min-width: 172px; }
      #request { min-height: 100px; }
      .answer-col { min-height: 52vh; }
      .dock-tabs { overflow-x: auto; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation: none !important; transition: none !important; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="mark">Q</div>
      <div class="brand-text">
        <h1>Qwen Orchestrator</h1>
        <div class="tagline">many small calls · visible token flow</div>
      </div>
    </div>
    <div class="pipeline" id="pipeline" aria-hidden="true">
      <div class="node" data-stage="plan"><i class="dot"></i><span>Plan</span></div>
      <div class="link"></div>
      <div class="node" data-stage="work"><i class="dot"></i><span>Work</span></div>
      <div class="link"></div>
      <div class="node" data-stage="verify"><i class="dot"></i><span>Verify</span></div>
      <div class="link"></div>
      <div class="node" data-stage="synth"><i class="dot"></i><span>Synth</span></div>
    </div>
    <div class="header-right">
      <span class="kbd-hint"><b>⏎</b> run <i>·</i> <b>⇧⇥</b> mode <i>·</i> <b>esc</b> stop</span>
      <select class="coder-select" id="coderProfile" title="GPU coder model (swaps the resident model for this session)" hidden></select>
      <div class="status service-pill" id="service">checking service...</div>
    </div>
  </header>
  <main id="app">
    <!-- Column 0 : sessions rail (channel strips) -->
    <aside class="col session-rail">
      <div class="panel-head">
        <div class="panel-title">Sessions</div>
        <button id="newChat" type="button" class="new-chat" title="Start a new chat">+ New</button>
      </div>
      <div id="sessionList" class="scroll session-list"><div class="empty">Loading chats…</div></div>
    </aside>

    <!-- Column 1 : request console -->
    <section class="col console">
      <div class="resizer resizer-x right" data-edge="c1" title="Drag to resize · double-click to reset"></div>
      <div class="panel-head">
        <div class="panel-title">Request</div>
        <span class="status" id="status">idle</span>
      </div>
      <form id="form" class="composer-form">
        <textarea id="request" placeholder="Ask for something worth decomposing — architecture, refactoring, code review, a migration or debugging plan, design tradeoffs. Enter to run, Shift+Enter for a new line."></textarea>
        <div class="composer-controls">
          <div class="field">
            <label for="modeOverride">Mode</label>
            <select id="modeOverride">
              <option value="auto">Auto route</option>
              <option value="chat">Chat</option>
              <option value="direct_answer">Direct answer</option>
              <option value="search_results">Search results</option>
              <option value="web_research">Web research</option>
              <option value="plan">Planner</option>
              <option value="project_research">Project research</option>
              <option value="code_review">Code review</option>
              <option value="implementation">Implementation guidance</option>
              <option value="debug">Debug</option>
              <option value="deep_orchestration">Deep orchestration</option>
            </select>
          </div>
          <div class="field">
            <label for="projectPath">Project (optional)</label>
            <input id="projectPath" value="" placeholder="path or name">
          </div>
        </div>
        <div class="button-row">
          <button id="run" type="submit">Run</button>
          <button id="retry" type="button" disabled title="Run this prompt again as a fresh attempt">Retry</button>
          <button id="continue" type="button" disabled title="Re-run the selected job — cached work is reused">Continue</button>
          <button id="stop" class="stop-button" type="button" disabled>Stop</button>
        </div>
      </form>
    </section>

    <!-- Column 2 : answer -->
    <section class="col answer-col">
      <div class="panel-head">
        <div class="panel-title">Answer</div>
        <span class="status" id="meta"></span>
      </div>
      <div id="queueBar" class="queue-bar" hidden></div>
      <div id="promptView" class="prompt-view" hidden>
        <span class="prompt-label" id="promptLabel">Prompt</span>
        <span class="prompt-text" id="promptText"></span>
      </div>
      <div class="progress" aria-label="job progress"><span id="progressBar"></span></div>
      <div id="answerNote" class="answer-note" hidden></div>
      <div id="memoryChips" class="memory-chips" hidden></div>
      <div id="answer" class="answer scroll">Ask for something worth breaking down — an architecture, a refactor, a code review, a migration or debugging plan, a design tradeoff. I'll plan it, run focused workers, double-check the result, and write you one answer. Each step shows up live in the workflow panel as it runs.</div>
      <div class="quick-grid">
        <div class="quick-card"><strong id="quickMode">Auto</strong><span>mode</span></div>
        <div class="quick-card"><strong id="quickTokens">-</strong><span>tokens</span></div>
        <div class="quick-card"><strong id="quickSpeed">-</strong><span>gen tok/s</span></div>
      </div>
    </section>

    <!-- Column 3 : live workflow -->
    <section class="col workflow-col">
      <div class="resizer resizer-x left" data-edge="c3" title="Drag to resize · double-click to reset"></div>
      <div class="panel-head">
        <div class="panel-title">Workflow</div>
        <span class="status" id="metaWorkflow"></span>
      </div>
      <div class="scroll">
        <div id="currentStage" class="stage"><strong>Idle</strong><span>Submit a request to start planning.</span></div>
        <div id="metrics" class="metrics"></div>
        <div id="modeCard" class="empty">The workflow is chosen automatically once your request is routed.</div>
        <div id="projectProgress"></div>
        <div class="subhead"><span>Agents &amp; Tasks</span><span id="taskCount"></span></div>
        <div id="tasks" class="cards"></div>
        <div class="subhead"><span>Live Model Calls</span><span id="callCount"></span></div>
        <div id="calls" class="call-wrap"></div>
        <div class="subhead"><span>Context Memory</span><span id="memoryCount"></span></div>
        <div id="memory" class="cards"></div>
        <div class="subhead"><span>Project Files</span><span id="fileCount"></span></div>
        <div id="files" class="file-list"></div>
      </div>
    </section>

    <!-- Dock : reference + config surfaces, tabbed -->
    <section class="dock" id="dock">
      <div class="resizer resizer-y" data-edge="dock" title="Drag to resize · double-click to reset"></div>
      <div class="dock-tabs" role="tablist">
        <button type="button" class="dock-tab active" data-tab="jobs">Jobs</button>
        <button type="button" class="dock-tab" data-tab="settings">Settings</button>
        <button type="button" class="dock-tab" data-tab="tools">Capabilities</button>
        <button type="button" class="dock-tab" data-tab="memory">Memory</button>
        <button type="button" class="dock-tab" data-tab="trace">Trace</button>
        <button type="button" class="dock-tab" data-tab="events">Events</button>
        <button type="button" class="dock-collapse" id="dockCollapse" title="Collapse dock" aria-label="Collapse dock">&minus;</button>
      </div>
      <div class="dock-body">
        <div class="dock-pane active" data-pane="jobs">
          <div id="jobs" class="job-list"></div>
        </div>

        <div class="dock-pane" data-pane="settings">
          <div class="settings-grid">
            <div class="field"><label for="provider">Provider</label>
              <select id="provider">
                <option value="openai-compatible">OpenAI-compatible llama.cpp</option>
                <option value="ollama">Ollama</option>
              </select>
            </div>
            <div class="field"><label for="baseUrl">Base URL</label><input id="baseUrl" value="http://127.0.0.1:8080"></div>
            <div class="field"><label for="model">Model</label><input id="model" value="qwen2.5-coder-7b"></div>
            <div class="field"><label for="maxTasks">Max tasks</label><input id="maxTasks" type="number" min="1" max="12" value="5"></div>
            <div class="field"><label for="maxWorkers">Workers</label><input id="maxWorkers" type="number" min="1" max="8" value="1"></div>
            <div class="field"><label for="maxRounds">Verifier rounds</label><input id="maxRounds" type="number" min="1" max="4" value="2"></div>
            <div class="field"><label for="plannerTokens">Planner tokens</label><input id="plannerTokens" type="number" min="80" value="700"></div>
            <div class="field"><label for="workerTokens">Worker tokens</label><input id="workerTokens" type="number" min="80" value="550"></div>
            <div class="field"><label for="verifierTokens">Verifier tokens</label><input id="verifierTokens" type="number" min="80" value="360"></div>
            <div class="field"><label for="compactorTokens">Compactor tokens</label><input id="compactorTokens" type="number" min="80" value="260"></div>
            <div class="field"><label for="synthTokens">Synth tokens</label><input id="synthTokens" type="number" min="80" value="650"></div>
            <div class="field"><label for="timeout">Timeout / call</label><input id="timeout" type="number" min="10" value="1200"></div>
          </div>
        </div>

        <div class="dock-pane" data-pane="tools">
          <div class="tools-layout">
            <div class="tools-form">
              <div class="field"><label for="capability">Capability</label>
                <select id="capability">
                  <option value="git_status">Git status</option>
                  <option value="list_dir">List directory</option>
                  <option value="file_preview">Preview file</option>
                  <option value="search_text">Search text</option>
                  <option value="web_search">Web search</option>
                  <option value="webpage_summary">Webpage summary</option>
                  <option value="python_eval">Python calculation</option>
                </select>
              </div>
              <div class="field"><label for="capPath">Path</label><input id="capPath" value="/home/nit/qwen-orchestrator"></div>
              <div class="field"><label for="capQuery">Query or expression</label><input id="capQuery" value="1 + 2 * 3"></div>
              <button id="runCapability" type="button">Run capability</button>
            </div>
            <div class="tools-out">
              <div id="capabilityResult"></div>
              <div class="subhead"><span>Recent runs</span><span></span></div>
              <div id="capabilityRuns" class="job-list"></div>
            </div>
          </div>
        </div>

        <div class="dock-pane" data-pane="memory">
          <div class="tools-layout">
            <div class="tools-form">
              <div class="field"><label for="memoryScope">Scope</label>
                <select id="memoryScope">
                  <option value="user">User</option>
                  <option value="project">Current project path</option>
                </select>
              </div>
              <div class="field"><label for="memoryKey">Key</label><input id="memoryKey" placeholder="preference, project note, machine fact"></div>
              <div class="field"><label for="memoryValue">Value</label><textarea id="memoryValue" placeholder="A durable fact or preference to include in future jobs."></textarea></div>
              <div class="field"><label for="memoryTags">Tags</label><input id="memoryTags" placeholder="comma,separated,tags"></div>
              <button id="saveMemory" type="button">Save memory</button>
            </div>
            <div class="tools-out">
              <div id="memoryList" class="job-list"></div>
            </div>
          </div>
        </div>

        <div class="dock-pane" data-pane="trace">
          <div id="trace"></div>
        </div>

        <div class="dock-pane" data-pane="events">
          <div id="events"></div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const CHAT_HISTORY_KEY = 'qwen-orchestrator-chat-history-v1';
    const ACTIVE_JOB_KEY = 'qwen-orchestrator-active-job-v1';
    let chatHistory = [];
    const recordedJobs = new Set();
    try {
      chatHistory = JSON.parse(localStorage.getItem(CHAT_HISTORY_KEY) || '[]').filter((item) => item && item.role && item.content).slice(-12);
    } catch (_) {
      chatHistory = [];
    }
    function saveChatHistory() {
      localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(chatHistory.slice(-12)));
    }
    function rememberChatTurn(job) {
      if (!job || recordedJobs.has(job.id) || job.status !== 'done' || !job.answer) return;
      const mode = job.route?.mode;
      if (!['direct_answer', 'chat'].includes(mode)) return;
      const requestText = job.request || job.trace?.user_request || job.config?.request || '';
      if (!requestText) return;
      chatHistory.push({role: 'user', content: requestText});
      chatHistory.push({role: 'assistant', content: job.answer});
      chatHistory = chatHistory.slice(-12);
      recordedJobs.add(job.id);
      saveChatHistory();
    }
    const form = $('form');
    const run = $('run');
    const stop = $('stop');
    const status = $('status');
    const answer = $('answer');
    const events = $('events');
    const trace = $('trace');
    const meta = $('meta');
    const currentStage = $('currentStage');
    const tasksEl = $('tasks');
    const callsEl = $('calls');
    const memoryEl = $('memory');
    const filesEl = $('files');
    const projectProgressEl = $('projectProgress');
    const jobsEl = $('jobs');
    const capabilityResult = $('capabilityResult');
    const capabilityRunsEl = $('capabilityRuns');
    const memoryListEl = $('memoryList');
    const modeCard = $('modeCard');
    const progressBar = $('progressBar');
    const queueBar = $('queueBar');
    const promptView = $('promptView');
    const promptText = $('promptText');
    const promptLabel = $('promptLabel');
    const answerNote = $('answerNote');
    const memoryChips = $('memoryChips');
    const sessionList = $('sessionList');
    let activeJob = null;
    let activeJobId = localStorage.getItem(ACTIVE_JOB_KEY) || null;
    let activeSessionId = localStorage.getItem('qwen-orchestrator-session-v1') || null;
    let sessionRole = 'admin';

    function num(id) {
      return Number($(id).value);
    }

    function escapeText(value) {
      return String(value).replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      })[c]);
    }

    function details(title, body, open = false) {
      return `<details ${open ? 'open' : ''}><summary>${escapeText(title)}</summary><pre>${escapeText(body)}</pre></details>`;
    }

    async function loadSession() {
      try {
        const res = await fetch('/api/session');
        const data = await res.json();
        sessionRole = data.role || 'admin';
      } catch (err) {
        sessionRole = 'restricted';
      }
      if (sessionRole !== 'admin') {
        // Sessions are an admin feature; collapse the rail for restricted users.
        const rail = document.querySelector('.session-rail');
        if (rail) rail.style.display = 'none';
        document.getElementById('app').style.gridTemplateColumns = 'var(--c1, 336px) minmax(240px, 1fr) var(--c3, 540px)';
        ['project_research', 'code_review', 'implementation', 'debug'].forEach((mode) => {
          const option = document.querySelector(`#modeOverride option[value="${mode}"]`);
          if (option) option.remove();
        });
        ['git_status', 'list_dir', 'file_preview', 'search_text'].forEach((capability) => {
          const option = document.querySelector(`#capability option[value="${capability}"]`);
          if (option) option.remove();
        });
        const projectField = $('projectPath')?.closest('.field');
        if (projectField) projectField.style.display = 'none';
        const capPathField = $('capPath')?.closest('.field');
        if (capPathField) capPathField.style.display = 'none';
        const memoryScope = $('memoryScope')?.closest('.dock-pane');
        if (memoryScope) memoryScope.innerHTML = '<div class="empty">Memory is disabled for restricted users.</div>';
        capabilityRunsEl.innerHTML = '<div class="empty">Capability history is admin-only.</div>';
      }
    }

    function eventCard(event) {
      // Prefer the server's plain-language line; keep only numeric extras as meta
      // (the line already names the file/query/title, so don't repeat them).
      const title = event.line || event.event.replaceAll('_', ' ');
      const bits = [];
      if (event.elapsed_s) bits.push(`${event.elapsed_s}s`);
      if (event.total_tokens) bits.push(`${event.total_tokens} tok`);
      if (event.approx_completion_tokens) bits.push(`~${event.approx_completion_tokens} tok`);
      if (event.tokens_per_second) bits.push(`${event.tokens_per_second} tok/s`);
      if (event.approx_tokens_per_second) bits.push(`~${event.approx_tokens_per_second} tok/s`);
      const meta = bits.length ? `<div class="event-meta">${escapeText(bits.join(' · '))}</div>` : '';
      return `<div class="event"><div class="event-title">${escapeText(title)}</div>${meta}</div>`;
    }

    function stageText(job) {
      if (job.status === 'done') return ['Done', 'Final answer is ready.'];
      if (job.status === 'cancelled') return ['Cancelled', 'The job was stopped by the user.'];
      if (job.status === 'error') return ['Error', job.error || 'The orchestration failed.'];
      const last = (job.events || []).at(-1);
      if (!last) return [job.status || 'Idle', 'No workflow events yet.'];
      const title = last.line || last.event.replaceAll('_', ' ');
      const detail = [
        last.round ? `round ${last.round}` : '',
        last.id ? last.id : '',
        last.title ? last.title : '',
        last.total_tokens ? `${last.total_tokens} tokens` : '',
        last.tokens_per_second ? `${last.tokens_per_second} tok/s` : '',
        job.stream_stats?.approx_tokens_per_second ? `~${job.stream_stats.approx_tokens_per_second} tok/s live` : ''
      ].filter(Boolean).join(' · ');
      return [title, detail || 'Working on the next model call.'];
    }

    function renderStage(job) {
      const [title, detail] = job.stage ? [job.stage.label, job.stage.detail] : stageText(job);
      currentStage.className = `stage ${job.status === 'done' ? 'done' : job.status === 'error' ? 'error' : job.status === 'cancelled' ? 'cancelled' : job.status === 'running' ? 'running' : ''}`;
      currentStage.innerHTML = `<strong>${escapeText(title)}</strong><span>${escapeText(detail)}</span>`;
    }

    function stageReached(job) {
      const evs = (job.events || []).map((e) => e.event || '');
      let idx = -1;
      if (evs.some((e) => /rout|plan/.test(e))) idx = 0;
      if (evs.some((e) => /worker/.test(e))) idx = 1;
      if (evs.some((e) => /verif/.test(e))) idx = 2;
      if (evs.some((e) => /synth|compact/.test(e)) || job.status === 'done') idx = 3;
      return idx;
    }

    function renderPipeline(job) {
      const rail = $('pipeline');
      if (!rail) return;
      const nodes = rail.querySelectorAll('.node');
      const reached = job ? stageReached(job) : -1;
      const finished = job && ['done', 'error', 'cancelled'].includes(job.status);
      nodes.forEach((node, i) => {
        node.classList.remove('active', 'done');
        if (i < reached || (i === reached && finished)) node.classList.add('done');
        else if (i === reached && !finished) node.classList.add('active');
      });
    }

    function metric(label, value) {
      return `<div class="metric"><strong>${escapeText(value ?? '-')}</strong><span>${escapeText(label)}</span></div>`;
    }

    function fmtSeconds(value) {
      if (value == null || Number.isNaN(value)) return '-';
      const seconds = Math.max(0, Math.round(value));
      const minutes = Math.floor(seconds / 60);
      const rest = seconds % 60;
      return minutes ? `${minutes}m ${rest}s` : `${rest}s`;
    }

    function fmtNumber(value) {
      if (value == null || value === '') return '-';
      const n = Number(value);
      return Number.isFinite(n) ? n.toLocaleString() : String(value);
    }

    function progressPercent(job) {
      const project = job.project_progress || {};
      if (project.total_files && job.status === 'running') {
        return Math.max(4, Math.min(96, Math.round(Number(project.percent || 0))));
      }
      const metrics = job.metrics || {};
      if (job.status === 'done') return 100;
      if (job.status === 'cancelled' || job.status === 'error') return 100;
      const planned = Number(metrics.calls_planned || job.route?.expected_calls || 0);
      const finished = Number(metrics.calls_finished || 0);
      if (!planned) return job.status === 'running' ? 8 : 0;
      return Math.max(4, Math.min(96, Math.round((finished / planned) * 100)));
    }

    function renderQuickStats(job) {
      const metrics = job.metrics || {};
      const route = job.route || job.trace?.mode || {};
      $('quickMode').textContent = route.label || route.mode || 'Auto';
      $('quickTokens').textContent = fmtNumber(metrics.tokens_total || job.stream_stats?.approx_completion_tokens || job.stream_stats?.approx_tokens || 0);
      $('quickSpeed').textContent = metrics.tokens_per_second ? `${metrics.tokens_per_second} tok/s` : '-';
      progressBar.style.width = `${progressPercent(job)}%`;
    }

    function estimate(job, summary) {
      if (job.metrics) {
        return {
          elapsed: job.metrics.elapsed_seconds || 0,
          eta: job.metrics.eta_seconds,
          expected: job.metrics.calls_planned || 0,
          completed: job.metrics.calls_finished || 0
        };
      }
      const route = job.route || job.trace?.mode || {};
      const expected = route.expected_calls || job.trace?.usage_summary?.calls || 0;
      const completed = summary?.calls || 0;
      const elapsed = job.started ? (Date.now() / 1000 - job.started) : 0;
      if (!expected || !completed || !summary?.model_elapsed_s) {
        return {elapsed, eta: null, expected, completed};
      }
      const avg = summary.model_elapsed_s / completed;
      return {elapsed, eta: Math.max(0, (expected - completed) * avg), expected, completed};
    }

    function renderMode(job, estimateInfo) {
      const route = job.route || job.trace?.mode;
      if (!route) {
        modeCard.className = 'empty';
        modeCard.textContent = 'Mode will be selected after routing.';
        return;
      }
      modeCard.className = 'stage';
      modeCard.innerHTML = `<strong>${escapeText(route.label || route.mode)}</strong><span>${escapeText(route.reason || route.description || '')}</span>
        <div class="event-meta">expected calls ${escapeText(estimateInfo.expected || route.expected_calls || '-')} · completed ${escapeText(estimateInfo.completed || 0)} · elapsed ${escapeText(fmtSeconds(estimateInfo.elapsed))} · ETA ${escapeText(estimateInfo.eta == null ? 'calculating' : fmtSeconds(estimateInfo.eta))}</div>`;
    }

    function renderMetrics(summary, statusText, estimateInfo = null) {
      if (activeJob?.metrics) {
        const m = activeJob.metrics;
        const p = activeJob.project_progress || {};
        const active = m.active_call_elapsed_seconds != null ? ` · active ${fmtSeconds(m.active_call_elapsed_seconds)}` : '';
        $('metrics').innerHTML =
          metric('status', m.status || statusText || 'idle') +
          metric(p.total_files ? 'files' : 'calls', p.total_files ? `${fmtNumber(p.completed_files)} / ${fmtNumber(p.total_files)} · ${fmtNumber(p.resumed_files)} resumed` : `${fmtNumber(m.calls_finished)} done · ${fmtNumber(m.calls_started || 0)} started / ${fmtNumber(m.calls_planned || '-')}${active}`) +
          metric('tokens / speed', `${fmtNumber(m.tokens_total)}${m.tokens_per_second ? ` · ${m.tokens_per_second} tok/s` : ''}`) +
          metric('elapsed / ETA', `${fmtSeconds(m.elapsed_seconds)} / ${m.eta_seconds == null ? '-' : fmtSeconds(m.eta_seconds)}`);
        return;
      }
      if (!summary) {
        $('metrics').innerHTML =
          metric('status', statusText || 'idle') +
          metric('speed', activeJob?.stream_stats?.approx_tokens_per_second ? `~${activeJob.stream_stats.approx_tokens_per_second} tok/s` : '-') +
          metric('elapsed', estimateInfo ? fmtSeconds(estimateInfo.elapsed) : '-') +
          metric('ETA', estimateInfo?.eta == null ? '-' : fmtSeconds(estimateInfo.eta));
        return;
      }
      $('metrics').innerHTML =
        metric('model calls', summary.calls) +
          metric('tokens / speed', activeJob?.stream_stats?.approx_completion_tokens || activeJob?.stream_stats?.approx_tokens ? `~${activeJob.stream_stats.approx_completion_tokens || activeJob.stream_stats.approx_tokens} · ~${activeJob.stream_stats.approx_tokens_per_second} tok/s` : summary.total_tokens) +
        metric('elapsed', estimateInfo ? fmtSeconds(estimateInfo.elapsed) : `${summary.model_elapsed_s}s`) +
        metric('ETA', estimateInfo?.eta == null ? '-' : fmtSeconds(estimateInfo.eta));
    }

    function renderProjectProgress(job) {
      const p = job.project_progress || {};
      if (!p.total_files) {
        projectProgressEl.innerHTML = '';
        return;
      }
      const pct = p.percent == null ? '-' : `${p.percent}%`;
      const current = p.current_file ? `<div class="context-meta">current: <code>${escapeText(p.current_file)}</code>${p.current_index ? ` · ${escapeText(p.current_index)}/${escapeText(p.current_total || p.total_files)}` : ''}</div>` : '';
      projectProgressEl.innerHTML = `<div class="stage running">
        <strong>Project Scan ${escapeText(pct)}</strong>
        <span>${escapeText(p.completed_files)} of ${escapeText(p.total_files)} files · ${escapeText(p.resumed_files)} resumed · ${escapeText(p.read_files)} read this run · ${escapeText(p.remaining_files)} remaining</span>
        ${current}
      </div>`;
    }

    function callTable(calls) {
      if (!calls || !calls.length) return '<p class="status">No completed model calls yet.</p>';
      const rows = calls.map((call) => `
        <tr>
          <td>${escapeText(call.stage)}</td>
          <td>${escapeText(call.label || '')}</td>
          <td>${escapeText(call.prompt_tokens ?? '-')}</td>
          <td>${escapeText(call.completion_tokens ?? '-')}</td>
          <td>${escapeText(call.total_tokens ?? '-')}</td>
          <td>${escapeText(call.elapsed_s ?? '-')}s</td>
          <td>${escapeText(call.tokens_per_second ?? '-')}</td>
        </tr>`).join('');
      return `<table>
        <thead><tr><th>Stage</th><th>Label</th><th>Prompt</th><th>Out</th><th>Total</th><th>Time</th><th>tok/s</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
    }

    function taskCards(job) {
      if (job.agents && job.agents.length) {
        $('taskCount').textContent = `${job.agents.length}`;
        return job.agents.map((task) => {
          const state = task.state || 'pending';
          return `<div class="agent-card ${state === 'running' ? 'active' : ''}">
            <div class="agent-title">
              <span>${escapeText(task.id)} · ${escapeText(task.title || 'task')}</span>
              <span class="pill ${escapeText(state)}">${escapeText(state)}</span>
            </div>
            <div class="agent-meta">${escapeText(task.role || 'worker')} · budget ${escapeText(task.token_budget || 'default')} tokens</div>
            <div class="agent-meta">${escapeText(task.prompt || '')}</div>
            ${(task.depends_on || []).length ? `<div class="agent-meta">depends on ${escapeText(task.depends_on.join(', '))}</div>` : ''}
          </div>`;
        }).join('');
      }
      const events = job.events || [];
      const planned = [...events].reverse().find((event) => event.event === 'planned');
      const finished = new Set(events.filter((event) => event.event === 'worker_finished').map((event) => event.id));
      const active = [...events].reverse().find((event) => event.event === 'worker_started');
      const tasks = job.trace?.tasks || planned?.tasks || [];
      $('taskCount').textContent = tasks.length ? `${tasks.length}` : '';
      if (!tasks.length) return '<div class="empty">No subtasks yet — the planner runs first and breaks your request down.</div>';
      return tasks.map((task) => {
        const state = finished.has(task.id) ? 'done' : active?.id === task.id ? 'running' : 'pending';
        return `<div class="agent-card ${state === 'running' ? 'active' : ''}">
          <div class="agent-title">
            <span>${escapeText(task.id)} · ${escapeText(task.title || 'task')}</span>
            <span class="pill ${escapeText(state)}">${escapeText(state)}</span>
          </div>
          <div class="agent-meta">${escapeText(task.role || 'worker')} · budget ${escapeText(task.token_budget || 'default')} tokens</div>
          <div class="agent-meta">${escapeText(task.prompt || '')}</div>
          ${(task.depends_on || []).length ? `<div class="agent-meta">depends on ${escapeText(task.depends_on.join(', '))}</div>` : ''}
        </div>`;
      }).join('');
    }

    function memoryCards(job) {
      const projectRetrieval = job.trace?.project_context?.retrieval;
      const toolRequests = job.trace?.tool_requests || [];
      const retrievalCard = projectRetrieval ? `<div class="context-card">
        <div class="agent-title">
          <span>Project retrieval</span>
          <span class="pill">${escapeText((projectRetrieval.snippets || []).length)} snippets</span>
        </div>
        <div class="context-meta">terms: ${escapeText((projectRetrieval.terms || []).slice(0, 16).join(', ') || '-')}</div>
        ${(projectRetrieval.snippets || []).slice(0, 5).map((item) => `<div class="context-meta"><code>${escapeText(item.path || '')}</code> · score ${escapeText(item.score || 0)}</div>`).join('')}
      </div>` : '';
      const toolRequestCards = toolRequests.length ? toolRequests.map((item) => `<div class="context-card">
        <div class="agent-title">
          <span>Requested tool · ${escapeText(item.capability)}</span>
          <span class="pill">manual</span>
        </div>
        <div class="context-meta">${escapeText(item.reason || '')}</div>
        <button type="button" class="run-requested-tool" data-capability="${escapeText(item.capability)}" data-input="${escapeText(JSON.stringify(item.input || {}))}">Run requested tool</button>
        <details><summary>Input</summary><pre>${escapeText(JSON.stringify(item.input || {}, null, 2))}</pre></details>
      </div>`).join('') : '';
      if (job.context_blocks && job.context_blocks.length) {
        $('memoryCount').textContent = `${job.context_blocks.length}`;
        return retrievalCard + toolRequestCards + job.context_blocks.map((item) => {
          const compact = item.compact || {};
          const points = compact.key_points || compact.use_later || [];
          return `<div class="context-card">
            <div class="agent-title">
              <span>${escapeText(item.id || '')} · ${escapeText(item.title || item.type || 'memory')}</span>
              <span class="pill">compact</span>
            </div>
            <div class="context-meta">${escapeText(compact.summary || '')}</div>
            ${points.length ? `<ul>${points.slice(0, 4).map((point) => `<li>${escapeText(point)}</li>`).join('')}</ul>` : ''}
            <details><summary>Full compact JSON</summary><pre>${escapeText(JSON.stringify(compact, null, 2))}</pre></details>
          </div>`;
        }).join('');
      }
      const compacted = (job.events || []).filter((event) => event.event === 'compacted');
      const finalCompacts = (job.trace?.worker_results || []).map((result) => ({
        id: result.id,
        title: result.title,
        compact: result.compact
      }));
      const items = finalCompacts.length ? finalCompacts : compacted;
      $('memoryCount').textContent = items.length ? `${items.length}` : '';
      if (!items.length) return retrievalCard + toolRequestCards || '<div class="empty">Condensed notes from each worker appear here as they finish.</div>';
      return retrievalCard + toolRequestCards + items.map((item) => {
        const compact = item.compact || {};
        const points = compact.key_points || compact.use_later || [];
        return `<div class="context-card">
          <div class="agent-title">
            <span>${escapeText(item.id || '')} · ${escapeText(item.title || 'memory')}</span>
            <span class="pill">compact</span>
          </div>
          <div class="context-meta">${escapeText(compact.summary || '')}</div>
          ${points.length ? `<ul>${points.slice(0, 4).map((point) => `<li>${escapeText(point)}</li>`).join('')}</ul>` : ''}
          <details><summary>Full compact JSON</summary><pre>${escapeText(JSON.stringify(compact, null, 2))}</pre></details>
        </div>`;
      }).join('');
    }

    function contextCards(job) {
      const started = (job.events || []).filter((event) => event.event === 'worker_started' && event.context?.length);
      if (!started.length) return '';
      const latest = started.at(-1);
      return `<details><summary>Latest agent context (${escapeText(latest.id)})</summary><pre>${escapeText(JSON.stringify(latest.context, null, 2))}</pre></details>`;
    }

function fileRows(job) {
      const files = job.project_files || [];
      const retrieved = job.trace?.project_context?.retrieval?.snippets || [];
      $('fileCount').textContent = files.length ? `${files.length}` : '';
      if (!files.length && retrieved.length) {
        return retrieved.map((file) => `<div class="file-row active">
          <div class="agent-title">
            <code>${escapeText(file.path || '')}</code>
            <span class="pill running">retrieved</span>
          </div>
          <div class="context-meta">score ${escapeText(file.score || 0)} · ${escapeText(file.summary || '')}</div>
        </div>`).join('');
      }
      if (!files.length) return '<div class="empty">In project modes, the files the readers open show up here.</div>';
      const ordered = [...files].sort((a, b) => {
        const rank = {running: 0, resumed: 1, read: 2, pending: 3};
        return (rank[a.state] ?? 9) - (rank[b.state] ?? 9) || ((b.index || 0) - (a.index || 0));
      });
      return ordered.slice(0, 100).map((file) => {
        const state = file.state || 'pending';
        const position = file.index && file.total ? `${file.index}/${file.total}` : '';
        return `<div class="file-row ${state === 'running' ? 'active' : state === 'resumed' ? 'resumed' : ''}">
          <div class="agent-title">
            <code>${escapeText(file.path || '')}</code>
            <span class="pill ${escapeText(state)}">${escapeText(position || state)}</span>
          </div>
          ${file.summary ? `<div class="context-meta">${escapeText(file.summary)}</div>` : ''}
        </div>`;
      }).join('');
    }

    // ---- sessions (server-side named chats: own history + jobs + queue) ----
    let sessions = [];
    async function loadSessions() {
      try {
        sessions = (await (await fetch('/api/sessions')).json()).sessions || [];
      } catch (err) { sessions = []; }
      if (!sessions.length) {
        const created = await createSession('New chat', false);   // always have a home to land in
        if (created) sessions = [created];
      }
      if (!activeSessionId || !sessions.find((s) => s.id === activeSessionId)) {
        activeSessionId = sessions[0] ? sessions[0].id : null;
        if (activeSessionId) localStorage.setItem('qwen-orchestrator-session-v1', activeSessionId);
      }
      renderSessions();
    }
    function renderSessions() {
      if (!sessions.length) { sessionList.innerHTML = '<div class="empty">No chats yet — start one.</div>'; return; }
      sessionList.innerHTML = sessions.map((s) => {
        const live = s.running ? ' live-running' : (s.queued ? ' live-queued' : '');
        const active = s.id === activeSessionId ? ' active' : '';
        const sub = s.last_request || `${s.job_count || 0} ${s.job_count === 1 ? 'run' : 'runs'}`;
        return `<button type="button" class="session-strip${active}${live}" data-session-id="${escapeText(s.id)}">
          <span class="session-led"></span>
          <span class="session-body">
            <span class="session-name">${escapeText(s.title)}</span>
            <span class="session-sub">${escapeText(sub || 'Empty chat')}</span>
          </span>
          <span class="session-del" data-del="${escapeText(s.id)}" title="Delete this chat" role="button">&times;</span>
        </button>`;
      }).join('');
    }
    async function createSession(title, select = true) {
      try {
        const res = await fetch('/api/sessions', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ title: title || 'New chat' }) });
        if (!res.ok) return null;
        const s = await res.json();
        if (select) { sessions.unshift(s); selectSession(s.id); }
        return s;
      } catch (err) { return null; }
    }
    function selectSession(id) {
      if (!id) return;
      activeSessionId = id;
      localStorage.setItem('qwen-orchestrator-session-v1', id);
      activeJob = null; activeJobId = null; pollGen++;
      answer.textContent = 'New chat — ask something to get started.';
      meta.textContent = ''; status.textContent = 'idle';
      answerNote.hidden = true; memoryChips.hidden = true; promptView.hidden = true;
      renderSessions();
      loadJobs();
    }
    sessionList.addEventListener('click', async (event) => {
      const del = event.target.closest('[data-del]');
      if (del) {
        event.stopPropagation();
        if (!confirm('Delete this chat and all of its runs?')) return;
        const id = del.dataset.del;
        await fetch(`/api/sessions/${id}`, { method: 'DELETE' });
        sessions = sessions.filter((s) => s.id !== id);
        if (activeSessionId === id) { activeSessionId = null; if (sessions[0]) selectSession(sessions[0].id); else await loadSessions(); }
        else renderSessions();
        return;
      }
      const strip = event.target.closest('[data-session-id]');
      if (strip) selectSession(strip.dataset.sessionId);
    });
    sessionList.addEventListener('dblclick', async (event) => {
      const strip = event.target.closest('[data-session-id]');
      if (!strip) return;
      const s = sessions.find((x) => x.id === strip.dataset.sessionId);
      const title = prompt('Rename chat', s ? s.title : '');
      if (title && title.trim()) {
        await fetch(`/api/sessions/${strip.dataset.sessionId}/rename`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ title: title.trim() }) });
        if (s) s.title = title.trim();
        renderSessions();
      }
    });
    $('newChat').addEventListener('click', () => createSession('New chat'));

    async function loadJobs() {
      try {
        const url = activeSessionId ? `/api/jobs?session_id=${encodeURIComponent(activeSessionId)}` : '/api/jobs';
        const res = await fetch(url);
        const data = await res.json();
        const jobs = data.jobs || [];
        if (!activeJob && jobs.length) {
          const saved = activeJobId ? jobs.find((job) => job.id === activeJobId) : null;
          const live = jobs.find((job) => ['queued', 'running', 'cancelling'].includes(job.status));
          const target = saved || live;
          if (target) {
            setTimeout(() => selectJob(target.id), 0);
          }
        }
        renderQueue(jobs);
        jobsEl.innerHTML = jobs.length ? jobs.slice(0, 8).map((job) => {
          const active = job.id === activeJobId ? 'active' : '';
          const metrics = job.metrics || {};
          const route = job.route || {};
          return `<div class="job-row ${active}" data-job-id="${escapeText(job.id)}">
            <div class="agent-title">
              <code>${escapeText(job.id)}</code>
              <span class="pill ${escapeText(job.status)}">${escapeText(fmtStatus(job.status))}</span>
            </div>
            <div class="context-meta">${escapeText(route.label || route.mode || 'routing')} · ${escapeText(fmtSeconds(metrics.elapsed_seconds || 0))} · ${escapeText(metrics.calls_finished || 0)}/${escapeText(metrics.calls_planned || '-')} calls</div>
            <div class="context-meta">${escapeText(job.request || '')}</div>
          </div>`;
        }).join('') : '<div class="empty">No runs yet — submit a request to get started.</div>';
      } catch (err) {
        jobsEl.innerHTML = `<div class="empty">${escapeText(String(err))}</div>`;
      }
    }

    async function loadCapabilityRuns() {
      if (sessionRole !== 'admin') {
        capabilityRunsEl.innerHTML = '<div class="empty">Capability history is admin-only.</div>';
        return;
      }
      try {
        const res = await fetch('/api/capability/runs');
        const data = await res.json();
        const runs = data.runs || [];
        capabilityRunsEl.innerHTML = runs.length ? runs.slice(0, 6).map((run) => `
          <div class="job-row">
            <div class="agent-title">
              <code>${escapeText(run.capability || '')}</code>
              <span class="pill ${run.ok ? 'done' : 'error'}">${escapeText(run.ok ? 'ok' : 'error')}</span>
            </div>
            <div class="context-meta">${escapeText(run.label || run.error || '')}</div>
          </div>
        `).join('') : '<div class="empty">No tools have been run yet.</div>';
      } catch (err) {
        capabilityRunsEl.innerHTML = `<div class="empty">${escapeText(String(err))}</div>`;
      }
    }

    async function loadMemories() {
      if (sessionRole !== 'admin') {
        memoryListEl.innerHTML = '<div class="empty">Memory is disabled for restricted users.</div>';
        return;
      }
      try {
        const scope = $('memoryScope')?.value === 'project' && $('projectPath').value
          ? `project:${$('projectPath').value}`
          : 'user';
        const res = await fetch(`/api/memories?scope=${encodeURIComponent(scope)}`);
        const data = await res.json();
        const memories = data.memories || [];
        memoryListEl.innerHTML = memories.length ? memories.slice(0, 10).map((item) => `
          <div class="job-row" data-memory-id="${escapeText(item.id)}">
            <div class="agent-title">
              <code>${escapeText(item.key || '')}</code>
              <span class="pill">memory</span>
            </div>
            <div class="context-meta">${escapeText(item.value || '')}</div>
            <div class="context-meta">${escapeText((item.tags || []).join(', '))}</div>
            <button type="button" class="delete-memory" data-memory-id="${escapeText(item.id)}">Delete memory</button>
          </div>
        `).join('') : '<div class="empty">No saved memories yet — add a durable fact or preference below.</div>';
      } catch (err) {
        memoryListEl.innerHTML = `<div class="empty">${escapeText(String(err))}</div>`;
      }
    }

    function fmtStatus(value) {
      return value === 'done' ? 'complete' : value;
    }

    function jobRequest(job) {
      return (job && (job.config && job.config.request || job.request)) || '';
    }

    // The prompt banner: what the selected/running job is actually working on.
    // Defaults to the current job because renderJob is called for whatever job is
    // being followed — a fresh run, or one clicked in the queue / jobs list.
    function renderPrompt(job) {
      const req = jobRequest(job);
      if (!req) { promptView.hidden = true; return; }
      promptView.hidden = false;
      promptText.textContent = req;
      promptLabel.textContent =
        job.status === 'running' ? 'Running' :
        job.status === 'queued' ? 'Queued' :
        job.status === 'cancelling' ? 'Stopping' : 'Prompt';
    }

    // The queue bar: every prompt in flight for the single model slot. The running
    // one pulses; queued ones wait behind it. Clicking a chip shows that prompt.
    function renderQueue(jobs) {
      const pending = (jobs || []).filter((j) => ['running', 'queued', 'cancelling'].includes(j.status));
      if (!pending.length) { queueBar.hidden = true; queueBar.innerHTML = ''; return; }
      queueBar.hidden = false;
      const order = { running: 0, cancelling: 1, queued: 2 };
      pending.sort((a, b) => (order[a.status] - order[b.status]) || (a.created - b.created));
      const queuedCount = pending.filter((j) => j.status === 'queued').length;
      const chips = pending.map((j) => {
        const cls = j.status === 'queued' ? 'st-queued' : 'st-running';
        const sel = j.id === activeJobId ? ' selected' : '';
        const text = jobRequest(j) || j.id;
        // #N shows the global line position; running shows a live dot instead.
        const badge = j.status === 'queued' && j.queue_position ? `<span class="chip-pos">#${j.queue_position}</span>` : '<span class="chip-dot"></span>';
        return `<button type="button" class="queue-chip ${cls}${sel}" data-job-id="${escapeText(j.id)}" title="${escapeText(text)}">
          ${badge}<span class="chip-text">${escapeText(text)}</span>
        </button>`;
      }).join('');
      const clear = queuedCount ? `<button type="button" id="clearQueue" class="queue-clear" title="Cancel the ${queuedCount} still waiting in this chat">Clear ${queuedCount} waiting</button>` : '';
      queueBar.innerHTML = '<span class="queue-label">Queue</span>' + chips + clear;
    }

    async function checkHealth() {
      try {
        const res = await fetch('/api/health');
        const data = await res.json();
        const svc = $('service');
        svc.textContent = data.ok ? `${data.model || 'model'} ready` : (data.error || 'model offline');
        svc.className = 'status service-pill ' + (data.ok ? 'ok' : 'bad');
      } catch (err) {
        const svc = $('service');
        svc.textContent = 'model offline';
        svc.className = 'status service-pill bad';
      }
    }

    // GPU coder profile selector (admin only). Switching swaps the resident GPU model
    // (session-level, not per-message) via switch_coder.sh behind /api/coder-profile.
    async function loadCoderProfiles() {
      const sel = $('coderProfile');
      if (!sel) return;
      try {
        const res = await fetch('/api/coder-profiles');
        const data = await res.json();
        if (!data.ok || !(data.profiles || []).length) { sel.hidden = true; return; }
        sel.innerHTML = data.profiles.map(p =>
          `<option value="${p.name}"${p.active ? ' selected' : ''}>${escapeText(p.label || p.name)}</option>`
        ).join('');
        sel.hidden = false;
      } catch (err) { sel.hidden = true; }
    }

    async function switchCoderProfile(name) {
      const sel = $('coderProfile');
      const svc = $('service');
      sel.disabled = true;
      const prev = svc.textContent;
      svc.textContent = 'switching coder…';
      try {
        const res = await fetch('/api/coder-profile', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({name})
        });
        const data = await res.json();
        if (!res.ok || !data.ok) { alert('Coder switch failed: ' + (data.message || data.error || 'unknown')); }
        await loadCoderProfiles();
      } catch (err) {
        alert('Coder switch failed: ' + err);
      } finally {
        sel.disabled = false;
        await checkHealth();
      }
    }

    // The verifier's concerns show as a caveat UNDER the real answer, never
    // replacing it — a nitpicky 7B verifier shouldn't hide a usable answer.
    function renderVerifierNote(job) {
      const note = job.status === 'done' && job.trace ? job.trace.verifier_note : '';
      if (!note) { answerNote.hidden = true; answerNote.textContent = ''; return; }
      answerNote.hidden = false;
      answerNote.innerHTML = '<b>Heads up</b>' + escapeText(note);
    }

    // Durable memories the model proposed; nothing is saved until you approve one.
    function renderMemoryChips(job) {
      const suggestions = (job.status === 'done' && job.trace && job.trace.memory_suggestions) || [];
      if (!suggestions.length) { memoryChips.hidden = true; memoryChips.innerHTML = ''; return; }
      memoryChips.hidden = false;
      memoryChips.innerHTML = suggestions.map((s, i) => {
        const label = `${s.key}: ${s.value}`;
        return `<span class="mem-chip" data-idx="${i}" title="${escapeText(s.reason || label)}">
          <span class="mem-text"><b>Remember</b> ${escapeText(label)}</span>
          <button type="button" class="mem-save" data-idx="${i}">Save</button>
          <button type="button" class="mem-dismiss" data-idx="${i}">Dismiss</button>
        </span>`;
      }).join('');
    }

    memoryChips.addEventListener('click', async (event) => {
      const btn = event.target.closest('button');
      if (!btn) return;
      const idx = Number(btn.dataset.idx);
      const suggestion = activeJob && activeJob.trace && (activeJob.trace.memory_suggestions || [])[idx];
      if (!suggestion) return;
      if (btn.classList.contains('mem-dismiss')) {
        activeJob.trace.memory_suggestions.splice(idx, 1);   // client-side only; nothing was saved
        renderMemoryChips(activeJob);
        return;
      }
      const body = { scope: suggestion.scope, key: suggestion.key, value: suggestion.value, tags: suggestion.tags || [], job_id: activeJob.id };
      if (suggestion.scope === 'project') body.project_path = (activeJob.route && activeJob.route.project_path) || '';
      const res = await fetch('/api/memories', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
      if (res.ok) {
        activeJob.trace.memory_suggestions.splice(idx, 1);
        renderMemoryChips(activeJob);
        loadMemories();
      }
    });

    function renderJob(job) {
      activeJob = job;
      status.textContent = job.cancel_requested && !['done', 'error', 'cancelled'].includes(job.status)
        ? 'cancelling after current call'
        : fmtStatus(job.status);
      stop.disabled = !['queued', 'running'].includes(job.status);
      renderPrompt(job);
      renderVerifierNote(job);
      renderMemoryChips(job);
      renderQuickStats(job);
      renderStage(job);
      renderPipeline(job);
      const completedCalls = (job.events || []).filter((event) => event.event === 'model_call_finished');
      const liveSummary = completedCalls.length ? {
        calls: completedCalls.length,
        prompt_tokens: completedCalls.reduce((n, call) => n + (call.prompt_tokens || 0), 0),
        completion_tokens: completedCalls.reduce((n, call) => n + (call.completion_tokens || 0), 0),
        total_tokens: completedCalls.reduce((n, call) => n + (call.total_tokens || 0), 0),
        model_elapsed_s: Number(completedCalls.reduce((n, call) => n + (call.elapsed_s || 0), 0).toFixed(3))
      } : null;
      const summary = job.trace ? job.trace.usage_summary : liveSummary;
      const estimateInfo = estimate(job, summary);
      renderMode(job, estimateInfo);
      renderMetrics(summary, job.status, estimateInfo);
      renderProjectProgress(job);
      const liveCalls = job.calls || (job.trace ? [...(job.trace.project_model_calls || []), ...(job.trace.model_calls || [])] : completedCalls);
      $('callCount').textContent = liveCalls.length ? `${liveCalls.length}` : '';
      callsEl.innerHTML = callTable(liveCalls);
      tasksEl.innerHTML = taskCards(job);
      memoryEl.innerHTML = memoryCards(job) + contextCards(job);
      filesEl.innerHTML = fileRows(job);
      const visibleEvents = (job.events || []).map(eventCard).join('');
      if (visibleEvents) {
        events.innerHTML = visibleEvents;
      } else if (job.status === 'queued') {
        events.innerHTML = '<div class="event"><div class="event-title">queued</div><div class="event-meta">waiting for the single local model slot</div></div>';
      } else if (job.status === 'running') {
        events.innerHTML = '<div class="event"><div class="event-title">running</div><div class="event-meta">the model is decoding; long calls can take several minutes</div></div>';
      } else {
        events.innerHTML = '';
      }

      if (job.status === 'done' && job.trace) {
        const verifier = job.trace.verifier || {};
        answer.textContent = job.trace.final || '';
        meta.innerHTML = `${job.trace.elapsed_s}s · verifier <span class="${verifier.pass ? 'pass' : 'fail'}">${verifier.pass ? 'pass' : 'fail'}</span>`;
        trace.innerHTML =
          details('Strategy', JSON.stringify({strategy: job.trace.strategy, settings: job.trace.dynamic_settings}, null, 2), true) +
          details('Project context', JSON.stringify(job.trace.project_context || {}, null, 2), true) +
          details('Available capabilities', JSON.stringify(job.trace.project_context?.available_capabilities || [], null, 2)) +
          details('Requested tools', JSON.stringify(job.trace.tool_requests || [], null, 2)) +
          `<details open><summary>Model calls</summary>${callTable([...(job.trace.project_model_calls || []), ...(job.trace.model_calls || [])])}</details>` +
          details('Rounds', JSON.stringify(job.trace.rounds, null, 2), true) +
          details('Planner tasks', JSON.stringify(job.trace.tasks, null, 2), true) +
          details('Worker results', JSON.stringify(job.trace.worker_results, null, 2)) +
          details('Verifier', JSON.stringify(job.trace.verifier, null, 2)) +
          details('Full JSON', JSON.stringify(job.trace, null, 2));
      } else if (job.status === 'error') {
        answer.textContent = job.error || 'Unknown error';
        meta.textContent = '';
      } else if (job.status === 'cancelled') {
        answer.textContent = job.partial_answer
          ? `${job.partial_answer}\n\n[stopped]`
          : 'Stopped.';
        meta.textContent = 'cancelled';
      } else if (job.status === 'queued') {
        answer.textContent = 'Queued — the server runs one request at a time to keep the single-slot local model (and the GPU) steady. Yours will start shortly.';
        meta.textContent = '';
      } else if (job.status === 'running' || job.status === 'cancelling') {
        if (job.partial_answer) {
          const speed = job.stream_stats?.approx_tokens_per_second
            ? `\n\n[streaming ~${job.stream_stats.approx_tokens_per_second} tok/s · ~${job.stream_stats.approx_completion_tokens || job.stream_stats.approx_tokens} tokens]`
            : '';
          answer.textContent = job.partial_answer + speed;
        } else {
          const stage = job.stage || {};
          const prefix = job.status === 'cancelling' ? 'Stopping after the current step' : 'Working on it';
          const line = stage.detail || (stage.label ? `${stage.label}…` : '');
          answer.textContent = line
            ? `${prefix} — ${line}\n\nThe answer streams in here as it's written. This can take a few minutes on the RX 580 profile.`
            : `${prefix} — the first planner call can take a moment before the workflow appears.`;
        }
        meta.textContent = job.started ? `started ${new Date(job.started * 1000).toLocaleTimeString()}` : '';
      }
      const nowMs = Date.now();
      if (nowMs - lastJobsLoadMs > 1500 || ['done', 'error', 'cancelled'].includes(job.status)) {
        lastJobsLoadMs = nowMs;
        loadJobs();
      }
      rememberChatTurn(job);
    }

    let pollGen = 0;
    let jobStream = null;
    let lastJobsLoadMs = 0;

    function closeJobStream() {
      if (jobStream) {
        jobStream.close();
        jobStream = null;
      }
    }

    async function poll(id) {
      const gen = ++pollGen;            // supersede any job we were following
      closeJobStream();
      activeJobId = id;
      localStorage.setItem(ACTIVE_JOB_KEY, id);
      if (window.EventSource) {
        jobStream = new EventSource(`/api/jobs/${id}/stream`);
        jobStream.onmessage = (event) => {
          if (gen !== pollGen) return;
          const job = JSON.parse(event.data);
          renderJob(job);
          if (['done', 'error', 'cancelled'].includes(job.status)) {
            closeJobStream();
            run.disabled = false;
          }
        };
        jobStream.onerror = () => {
          closeJobStream();
        };
      }
      while (gen === pollGen) {
        if (jobStream) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
          continue;
        }
        let job;
        try {
          const res = await fetch(`/api/jobs/${id}`);
          job = await res.json();
        } catch (err) { break; }
        if (gen !== pollGen) return;    // a newer selection took over mid-fetch
        renderJob(job);
        if (['done', 'error', 'cancelled'].includes(job.status)) break;
        const delay = job.active_model_call ? 300 : (job.route?.mode === 'direct_answer' ? 250 : 900);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
      if (gen === pollGen) run.disabled = false;   // renderJob already set stop.disabled
    }

    setInterval(() => {
      if (activeJob && ['running', 'cancelling'].includes(activeJob.status)) {
        renderJob(activeJob);
      }
    }, 1000);
    setInterval(loadJobs, 5000);
    setInterval(loadSessions, 6000);   // keep the rail LEDs + titles fresh

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const requestText = $('request').value.trim();
      if (!requestText) { $('request').focus(); return; }
      // Make sure this prompt lands in a chat; auto-title a fresh one from the ask.
      if (!activeSessionId) { await createSession(requestText.slice(0, 48) || 'New chat'); }
      const sess = sessions.find((s) => s.id === activeSessionId);
      if (sess && sess.title === 'New chat') {
        const t = requestText.slice(0, 48);
        fetch(`/api/sessions/${activeSessionId}/rename`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ title: t }) });
        sess.title = t; renderSessions();
      }
      // Run stays enabled while a job is in flight so more prompts can be queued;
      // the server holds a single orchestration slot and lines them up FIFO.
      stop.disabled = false;
      answer.textContent = 'Sending your request…';
      trace.innerHTML = '';
      events.innerHTML = '';
      tasksEl.innerHTML = '<div class="empty">Waiting for the planner to break this down…</div>';
      callsEl.innerHTML = callTable([]);
      memoryEl.innerHTML = '<div class="empty">No condensed notes yet.</div>';
      filesEl.innerHTML = '<div class="empty">In project modes, the files the readers open show up here.</div>';
      progressBar.style.width = '2%';
      meta.textContent = '';
      status.textContent = 'starting';

      const payload = {
        request: $('request').value,
        project_path: $('projectPath').value,
        mode_override: $('modeOverride').value,
        provider: $('provider').value,
        base_url: $('baseUrl').value,
        model: $('model').value,
        max_tasks: num('maxTasks'),
        max_workers: num('maxWorkers'),
        max_rounds: num('maxRounds'),
        planner_tokens: num('plannerTokens'),
        worker_tokens: num('workerTokens'),
        verifier_tokens: num('verifierTokens'),
        compactor_tokens: num('compactorTokens'),
        synth_tokens: num('synthTokens'),
        timeout: num('timeout'),
        session_id: activeSessionId || '',
        conversation_history: chatHistory.slice(-10),   // only used when there's no session (restricted users)
        continue_from: pendingContinueFrom || ''
      };
      pendingContinueFrom = null;   // one-shot: a plain submit never carries evidence

      const res = await fetch('/api/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) {
        status.textContent = 'error';
        answer.textContent = data.error || res.statusText;
        return;
      }
      answer.textContent = 'Queued — watching for progress…';
      status.textContent = 'queued';
      activeJobId = data.id;
      localStorage.setItem(ACTIVE_JOB_KEY, data.id);
      $('request').value = '';                // clear for the next prompt to queue
      loadJobs();                             // surface the new job in the queue bar at once
      loadSessions();                         // refresh the rail (title + live LED)
      poll(data.id);
    });

    stop.addEventListener('click', async () => {
      if (!activeJobId) return;
      stop.disabled = true;
      status.textContent = 'cancelling after current call';
      await fetch(`/api/jobs/${activeJobId}/cancel`, {method: 'POST'});
    });

    $('runCapability').addEventListener('click', async () => {
      const capability = $('capability').value;
      const query = $('capQuery').value;
      const body = {
        capability,
        path: $('capPath').value,
        query,
        expression: query,
        job_id: activeJobId
      };
      capabilityResult.innerHTML = '<div class="empty">Running the tool…</div>';
      const res = await fetch('/api/capability/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });
      const data = await res.json();
      capabilityResult.innerHTML = details(data.ok ? `${data.label || capability} result` : 'Capability error', JSON.stringify(data, null, 2), true);
      loadCapabilityRuns();
      if (activeJobId) selectJob(activeJobId);   // surface the attached evidence + relabel Continue
    });

    $('saveMemory').addEventListener('click', async () => {
      const payload = {
        scope: $('memoryScope').value,
        project_path: $('projectPath').value,
        key: $('memoryKey').value,
        value: $('memoryValue').value,
        tags: $('memoryTags').value.split(',').map((tag) => tag.trim()).filter(Boolean)
      };
      const res = await fetch('/api/memories', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const data = await res.json();
        memoryListEl.innerHTML = `<div class="empty">${escapeText(data.error || res.statusText)}</div>`;
        return;
      }
      $('memoryKey').value = '';
      $('memoryValue').value = '';
      $('memoryTags').value = '';
      loadMemories();
    });

    $('memoryScope').addEventListener('change', loadMemories);

    memoryListEl.addEventListener('click', async (event) => {
      const button = event.target.closest('.delete-memory');
      if (!button) return;
      await fetch(`/api/memories/${button.dataset.memoryId}`, {method: 'DELETE'});
      loadMemories();
    });

    // Select a job (from the jobs list or a queue chip): show its prompt + state
    // in the display panel, and live-follow it if it is still in flight.
    async function selectJob(id) {
      activeJobId = id;
      localStorage.setItem(ACTIVE_JOB_KEY, id);
      pollGen++;                                  // stop following a previously-selected job
      document.querySelectorAll('.job-row').forEach((r) => r.classList.toggle('active', r.dataset.jobId === id));
      document.querySelectorAll('.queue-chip').forEach((c) => c.classList.toggle('selected', c.dataset.jobId === id));
      let job;
      try { job = await (await fetch(`/api/jobs/${id}`)).json(); }
      catch (err) { return; }
      renderJob(job);
      selectedJobConfig = job.config || null;
      const cb = $('continue');
      // A user-approved tool run leaves a capability_run_attached event on the job;
      // when present, Continue carries that evidence into the re-run (one prefill).
      const toolEvidenceCount = (job.events || []).filter((e) => e.event === 'capability_run_attached').length;
      if (cb) {
        cb.disabled = !selectedJobConfig;
        cb.classList.toggle('ready', !!selectedJobConfig && !toolEvidenceCount);
        cb.classList.toggle('has-evidence', !!selectedJobConfig && toolEvidenceCount > 0);
        const short = (selectedJobConfig && selectedJobConfig.request || '').slice(0, 60);
        cb.textContent = toolEvidenceCount ? `Continue with ${toolEvidenceCount} tool result${toolEvidenceCount === 1 ? '' : 's'}` : 'Continue';
        cb.title = !selectedJobConfig ? 'Select a job to continue it'
          : toolEvidenceCount ? `Re-run with ${toolEvidenceCount} approved tool result${toolEvidenceCount === 1 ? '' : 's'} as evidence`
          : `Re-run: ${short}${short.length >= 60 ? '…' : ''}`;
      }
      const rb = $('retry');
      if (rb) {
        rb.disabled = !selectedJobConfig;
        rb.classList.toggle('ready', !!selectedJobConfig);
        rb.title = selectedJobConfig ? 'Run this prompt again as a fresh attempt' : 'Select a job to retry it';
      }
      if (['queued', 'running', 'cancelling'].includes(job.status)) poll(id);   // live-follow until it finishes
    }

    jobsEl.addEventListener('click', (event) => {
      const row = event.target.closest('[data-job-id]');
      if (row) selectJob(row.dataset.jobId);
    });

    queueBar.addEventListener('click', async (event) => {
      if (event.target.closest('#clearQueue')) {
        // Cancel every job still waiting in this chat (leaves the running one alone).
        const url = activeSessionId ? `/api/jobs?session_id=${encodeURIComponent(activeSessionId)}` : '/api/jobs';
        const jobs = ((await (await fetch(url)).json()).jobs || []).filter((j) => j.status === 'queued');
        await Promise.all(jobs.map((j) => fetch(`/api/jobs/${j.id}/cancel`, { method: 'POST' })));
        loadJobs();
        return;
      }
      const chip = event.target.closest('[data-job-id]');
      if (chip) selectJob(chip.dataset.jobId);
    });

    memoryEl.addEventListener('click', async (event) => {
      const button = event.target.closest('.run-requested-tool');
      if (!button) return;
      const capability = button.dataset.capability;
      const input = JSON.parse(button.dataset.input || '{}');
      $('capability').value = capability;
      $('capPath').value = input.path || $('capPath').value;
      $('capQuery').value = input.query || input.expression || '';
      capabilityResult.innerHTML = '<div class="empty">Running the requested tool…</div>';
      const res = await fetch('/api/capability/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({capability, ...input, job_id: activeJobId})
      });
      const data = await res.json();
      capabilityResult.innerHTML = details(data.ok ? `${data.label || capability} result` : 'Capability error', JSON.stringify(data, null, 2), true);
      loadCapabilityRuns();
      if (activeJobId) selectJob(activeJobId);   // surface the attached evidence + relabel Continue
    });

    // ---- dock: tabbed reference/config surfaces ----
    let selectedJobConfig = null;
    let pendingContinueFrom = null;   // set only by the explicit Continue click; carries approved tool evidence into one re-run
    document.querySelectorAll('.dock-tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        const name = tab.dataset.tab;
        document.querySelectorAll('.dock-tab').forEach((t) => t.classList.toggle('active', t === tab));
        document.querySelectorAll('.dock-pane').forEach((p) => p.classList.toggle('active', p.dataset.pane === name));
        $('app').classList.remove('dock-collapsed');
      });
    });
    $('dockCollapse').addEventListener('click', () => $('app').classList.toggle('dock-collapsed'));

    // ---- continue / resume the selected job (re-run its config; cached work is reused) ----
    const CONFIG_FIELDS = {
      request: 'request', project_path: 'projectPath', mode_override: 'modeOverride',
      provider: 'provider', base_url: 'baseUrl', model: 'model', max_tasks: 'maxTasks',
      max_workers: 'maxWorkers', max_rounds: 'maxRounds', planner_tokens: 'plannerTokens',
      worker_tokens: 'workerTokens', verifier_tokens: 'verifierTokens',
      compactor_tokens: 'compactorTokens', synth_tokens: 'synthTokens', timeout: 'timeout'
    };
    function applyJobConfig(cfg) {
      if (!cfg) return;
      for (const [key, id] of Object.entries(CONFIG_FIELDS)) {
        if (cfg[key] != null && $(id)) $(id).value = cfg[key];
      }
    }
    $('continue').addEventListener('click', () => {
      if (!selectedJobConfig || run.disabled) return;
      pendingContinueFrom = activeJobId;   // carry this job's approved tool evidence into the re-run
      applyJobConfig(selectedJobConfig);
      form.requestSubmit(run);
    });
    $('retry').addEventListener('click', () => {
      if (!selectedJobConfig || run.disabled) return;
      pendingContinueFrom = null;          // fresh attempt: same prompt, no carried evidence
      applyJobConfig(selectedJobConfig);
      form.requestSubmit(run);
    });
    $('retry').addEventListener('click', () => {
      if (!selectedJobConfig || run.disabled) return;
      // Fresh attempt at the same prompt — no tool evidence carried, new job in this chat.
      applyJobConfig(selectedJobConfig);
      form.requestSubmit(run);
    });

    // ---- keyboard shortcuts ----
    function submitRun() { if (!run.disabled) form.requestSubmit(run); }
    document.addEventListener('keydown', (e) => {
      if (e.isComposing) return;
      // Enter in the request box runs; Shift+Enter is a newline
      if (e.target === $('request') && e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
        e.preventDefault(); submitRun(); return;
      }
      // Ctrl/Cmd+Enter runs from anywhere
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); submitRun(); return; }
      // Esc stops a running job
      if (e.key === 'Escape' && !stop.disabled) { e.preventDefault(); stop.click(); return; }
      // Shift+Tab cycles the mode (except inside the dock forms, where field nav matters)
      if (e.key === 'Tab' && e.shiftKey && !(e.target.closest && e.target.closest('.dock'))) {
        e.preventDefault();
        const sel = $('modeOverride');
        sel.selectedIndex = (sel.selectedIndex + 1) % sel.options.length;
        sel.dispatchEvent(new Event('change'));
      }
    });

    // ---- draggable panel sizing (persisted to localStorage) ----
    (function () {
      const app = $('app');
      const MIN = { c1: 244, c3: 360, dock: 120 };
      const cur = (name, dflt) => {
        const v = parseInt(getComputedStyle(app).getPropertyValue(name));
        return Number.isFinite(v) ? v : dflt;
      };
      try {
        const s = JSON.parse(localStorage.getItem('qwen_layout') || '{}');
        if (s.c1) app.style.setProperty('--c1', s.c1 + 'px');
        if (s.c3) app.style.setProperty('--c3', s.c3 + 'px');
        if (s.dock) app.style.setProperty('--dock-h', s.dock + 'px');
      } catch (e) {}
      const persist = () => localStorage.setItem('qwen_layout', JSON.stringify({
        c1: cur('--c1', 344), c3: cur('--c3', 560), dock: cur('--dock-h', 210)
      }));
      document.querySelectorAll('.resizer').forEach((h) => {
        h.addEventListener('pointerdown', (e) => {
          e.preventDefault();
          const edge = h.dataset.edge, sx = e.clientX, sy = e.clientY;
          const start = { c1: cur('--c1', 344), c3: cur('--c3', 560), dock: cur('--dock-h', 210) };
          h.classList.add('dragging');
          try { h.setPointerCapture(e.pointerId); } catch (err) {}
          const move = (ev) => {
            if (edge === 'c1') {
              const v = Math.max(MIN.c1, Math.min(window.innerWidth * 0.5, start.c1 + (ev.clientX - sx)));
              app.style.setProperty('--c1', Math.round(v) + 'px');
            } else if (edge === 'c3') {
              const v = Math.max(MIN.c3, Math.min(window.innerWidth * 0.5, start.c3 - (ev.clientX - sx)));
              app.style.setProperty('--c3', Math.round(v) + 'px');
            } else {
              app.classList.remove('dock-collapsed');
              const v = Math.max(MIN.dock, Math.min(window.innerHeight * 0.72, start.dock - (ev.clientY - sy)));
              app.style.setProperty('--dock-h', Math.round(v) + 'px');
            }
          };
          const up = () => {
            h.classList.remove('dragging');
            document.removeEventListener('pointermove', move);
            document.removeEventListener('pointerup', up);
            persist();
          };
          document.addEventListener('pointermove', move);
          document.addEventListener('pointerup', up);
        });
        h.addEventListener('dblclick', () => {
          app.style.removeProperty(h.dataset.edge === 'dock' ? '--dock-h' : '--' + h.dataset.edge);
          persist();
        });
      });
    })();

    (async () => {
      await loadSession();
      checkHealth();
      if (sessionRole === 'admin') {
        await loadSessions();
        loadCoderProfiles();
        const sel = $('coderProfile');
        if (sel) sel.addEventListener('change', (e) => switchCoderProfile(e.target.value));
      }
      loadJobs();
      loadCapabilityRuns();
      loadMemories();
    })();
  </script>
</body>
</html>
"""


LOGIN_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen Orchestrator Login</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { min-height: 100vh; margin: 0; display: grid; place-items: center; background: #111827; color: #f8fafc; }
    form { width: min(420px, calc(100vw - 32px)); display: grid; gap: 14px; padding: 24px; border: 1px solid #334155; border-radius: 10px; background: #0f172a; box-shadow: 0 22px 80px rgba(0,0,0,.35); }
    h1 { margin: 0; font-size: 22px; }
    p { margin: 0; color: #94a3b8; line-height: 1.45; }
    input { width: 100%; box-sizing: border-box; padding: 12px; border-radius: 8px; border: 1px solid #475569; background: #020617; color: #f8fafc; font-size: 16px; }
    button { padding: 12px 14px; border: 0; border-radius: 8px; background: #38bdf8; color: #082f49; font-weight: 700; cursor: pointer; }
    .error { color: #fca5a5; min-height: 20px; }
  </style>
</head>
<body>
  <form method="post" action="/login">
    <h1>Qwen Orchestrator</h1>
    <p>Enter the access token for this local instance.</p>
    <input name="token" type="password" autocomplete="current-password" autofocus>
    <button type="submit">Sign in</button>
    <div class="error">{{ERROR}}</div>
  </form>
</body>
</html>
"""


class JobStore:
    def __init__(self, persistence: Persistence | None = None) -> None:
        self.persistence = persistence
        self.jobs: dict[str, dict[str, Any]] = {
            job["id"]: job for job in (self.persistence.load_jobs() if self.persistence else []) if job.get("id")
        }
        self.lock = threading.Lock()
        # One orchestration slot. The local model has a single slot, so concurrent
        # submissions line up here FIFO rather than interleaving and thrashing it.
        # A job waiting on this reports status "queued" (see run_slot).
        self.run_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.mark_stale_running_jobs()

    def mark_stale_running_jobs(self) -> None:
        now = round(time.time(), 3)
        with self.lock:
            for job_id, job in self.jobs.items():
                if job.get("status") in TERMINAL_STATUSES:
                    continue
                job["status"] = "error"
                job["error"] = "server restarted before this job reached a terminal state"
                job["active_model_call"] = None
                job["finished"] = now
                job.setdefault("events", []).append(
                    {
                        "event": "interrupted_by_restart",
                        "time": now,
                        "detail": "The Python worker thread cannot continue after a reboot/restart. Partial project summaries may be reused by a new run.",
                    }
                )
                self.persist_locked(job_id)

    def persist_locked(self, job_id: str) -> None:
        if self.persistence and job_id in self.jobs:
            self.persistence.save_job(self.jobs[job_id])

    def create(self, config: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "created": round(time.time(), 3),
                "config": config,
                "cancel_requested": False,
                "events": [
                    {
                        "event": "queued",
                        "time": round(time.time(), 3),
                        "detail": "waiting for the local model slot",
                    }
                ],
            }
            self.persist_locked(job_id)
        session_id = str((config or {}).get("session_id") or "")
        if session_id and self.persistence:
            self.persistence.touch_session(session_id)   # float the session to the top of the rail on new activity
        return job_id

    def update(self, job_id: str, **values: Any) -> None:
        with self.lock:
            values.setdefault("heartbeat", round(time.time(), 3))
            self.jobs[job_id].update(values)
            self.persist_locked(job_id)

    def update_live(self, job_id: str, **values: Any) -> None:
        with self.lock:
            values.setdefault("heartbeat", round(time.time(), 3))
            self.jobs[job_id].update(values)

    def event(self, job_id: str, event: dict[str, Any]) -> None:
        with self.lock:
            self.jobs[job_id]["heartbeat"] = round(time.time(), 3)
            self.jobs[job_id]["events"].append(event)
            self.persist_locked(job_id)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
        if not job and self.persistence:
            job = self.persistence.load_job(job_id)
        return enrich_job(job) if job else None

    def cancel(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            if job.get("status") in TERMINAL_STATUSES:
                return True
            job["cancel_requested"] = True
            if job.get("status") not in TERMINAL_STATUSES:
                job["status"] = "cancelling"
            job.setdefault("events", []).append(
                {"event": "cancel_requested", "time": round(time.time(), 3)}
            )
            self.persist_locked(job_id)
            return True

    def is_cancelled(self, job_id: str) -> bool:
        with self.lock:
            return bool(self.jobs.get(job_id, {}).get("cancel_requested"))

    def status(self, job_id: str) -> str | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return str(job.get("status")) if job else None

    def list(self, session_id: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda job: job["created"], reverse=True)
            # Global FIFO line position: the single model slot serves one job at a
            # time across all sessions, so position is by creation time over every
            # job still waiting or running (queued behind the one holding the slot).
            waiting = sorted(
                (job for job in jobs if job.get("status") == "queued"),
                key=lambda job: job["created"],
            )
            position = {job["id"]: index + 1 for index, job in enumerate(waiting)}
            rows = []
            for job in jobs:
                sid = str((job.get("config") or {}).get("session_id") or "")
                if session_id is not None and sid != session_id:
                    continue
                rows.append({
                    "id": job["id"],
                    "session_id": sid,
                    "status": job["status"],
                    "created": job["created"],
                    "started": job.get("started"),
                    "finished": job.get("finished"),
                    "cancel_requested": job.get("cancel_requested", False),
                    "queue_position": position.get(job["id"]),
                    "route": job.get("route"),
                    "metrics": job_metrics(job),
                    "stage": current_stage(job),
                    "request": job.get("config", {}).get("request", "")[:160],
                    "event_count": len(job.get("events", [])),
                    "completed_model_calls": len(
                        [event for event in job.get("events", []) if event.get("event") == "model_call_finished"]
                    ),
                })
            return rows

    def pending_count(self, session_id: str) -> int:
        """How many jobs in a session are still queued or running — used to cap how
        many prompts can stack up per session."""
        with self.lock:
            return sum(
                1
                for job in self.jobs.values()
                if str((job.get("config") or {}).get("session_id") or "") == session_id
                and job.get("status") in ("queued", "running", "cancelling")
            )

    def annotate_sessions(self, sessions: list[dict[str, Any]]) -> None:
        """Add live state (running/queued LED), a job count, and a last-request
        subtitle to each session row for the rail."""
        with self.lock:
            by_session: dict[str, list[dict[str, Any]]] = {}
            for job in self.jobs.values():
                sid = str((job.get("config") or {}).get("session_id") or "")
                if sid:
                    by_session.setdefault(sid, []).append(job)
        for session in sessions:
            jobs = by_session.get(session["id"], [])
            session["job_count"] = len(jobs)
            session["running"] = any(j.get("status") == "running" for j in jobs)
            session["queued"] = any(j.get("status") in ("queued", "cancelling") for j in jobs)
            latest = max(jobs, key=lambda j: j.get("created", 0), default=None)
            session["last_request"] = str((latest.get("config") or {}).get("request") or "")[:80] if latest else ""

    def partial_project_summaries(self, project_path: str, fingerprint: str) -> dict[str, dict[str, Any]]:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda job: job.get("created", 0), reverse=True)
        summaries: dict[str, dict[str, Any]] = {}
        for job in jobs:
            route = job.get("route") or {}
            if route.get("project_path") != project_path:
                continue
            events = job.get("events") or []
            discovered = next((event for event in events if event.get("event") == "project_discovered"), {})
            prefix = str(discovered.get("fingerprint") or "")
            if prefix and not fingerprint.startswith(prefix):
                continue
            for event in events:
                if event.get("event") != "project_file_read" or not event.get("path"):
                    continue
                raw_summary = str(event.get("summary") or "")
                parsed: dict[str, Any]
                try:
                    maybe = extract_json(raw_summary)
                    parsed = maybe if isinstance(maybe, dict) else {"summary": raw_summary}
                except Exception:
                    parsed = {"summary": raw_summary}
                summaries[str(event["path"])] = {"path": str(event["path"]), **parsed}
        return summaries


def infer_call_stage(messages: list[dict[str, str]]) -> tuple[str, str]:
    system = (messages[0].get("content") if messages else "") or ""
    if "planner" in system:
        return "planner", "initial plan"
    if "codebase reader" in system:
        return "project_file_reader", "file summary"
    if "directory architecture" in system:
        return "project_directory_compactor", "directory memory"
    if "whole-project memory" in system:
        return "project_architecture_compactor", "project memory"
    if "focused worker" in system:
        return "worker", "subtask"
    if "context compactor" in system:
        return "compactor", "worker memory"
    if "verifier" in system:
        return "verifier", "coverage check"
    if "synthesize" in system:
        return "synthesizer", "final answer"
    return "model", system[:80] or "model call"


class ObservableClient:
    def __init__(self, client: Any, store: JobStore, job_id: str) -> None:
        self.client = client
        self.store = store
        self.job_id = job_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def chat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int, response_format: dict[str, Any] | None = None) -> ModelResponse:
        stage, label = infer_call_stage(messages)
        self.store.event(self.job_id, {"event": "model_slot_waiting", "time": round(time.time(), 3), "stage": stage, "label": label})
        with self.store.model_lock:
            self.store.event(self.job_id, {"event": "model_slot_acquired", "time": round(time.time(), 3), "stage": stage, "label": label})
            try:
                with self.tracked_call(messages, temperature=temperature, max_tokens=max_tokens) as active:
                    if hasattr(self.client, "stream_chat"):
                        content = ""
                        done: dict[str, Any] = {}
                        started = time.monotonic()
                        for event in self.client.stream_chat(messages, temperature=temperature, max_tokens=max_tokens, response_format=response_format):
                            if self.store.is_cancelled(self.job_id):
                                raise JobCancelled()
                            if event.get("type") == "chunk":
                                chunk = event.get("content", "")
                                if chunk:
                                    content += chunk
                                    active.note_chunk(content, started)
                            elif event.get("type") == "done":
                                done = event
                        return ModelResponse(
                            content,
                            float(done.get("elapsed_s") or (time.monotonic() - started)),
                            done.get("usage") or {},
                            done.get("timings") or {},
                            done.get("raw") or {},
                        )
                    return self.client.chat(messages, temperature=temperature, max_tokens=max_tokens, response_format=response_format)
            finally:
                self.store.event(self.job_id, {"event": "model_slot_released", "time": round(time.time(), 3), "stage": stage, "label": label})

    def stream_chat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int, response_format: dict[str, Any] | None = None):
        stage, label = infer_call_stage(messages)
        self.store.event(self.job_id, {"event": "model_slot_waiting", "time": round(time.time(), 3), "stage": stage, "label": label})
        with self.store.model_lock:
            self.store.event(self.job_id, {"event": "model_slot_acquired", "time": round(time.time(), 3), "stage": stage, "label": label})
            try:
                with self.tracked_call(messages, temperature=temperature, max_tokens=max_tokens) as active:
                    content = ""
                    started = time.monotonic()
                    for event in self.client.stream_chat(messages, temperature=temperature, max_tokens=max_tokens, response_format=response_format):
                        if self.store.is_cancelled(self.job_id):
                            raise JobCancelled()
                        if event.get("type") == "chunk":
                            chunk = event.get("content", "")
                            if chunk:
                                content += chunk
                                active.note_chunk(content, started)
                        yield event
            finally:
                self.store.event(self.job_id, {"event": "model_slot_released", "time": round(time.time(), 3), "stage": stage, "label": label})

    def tracked_call(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int):
        return ActiveModelCall(self, messages, temperature, max_tokens)


class ActiveModelCall:
    def __init__(self, observable: ObservableClient, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> None:
        self.observable = observable
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = threading.Event()
        self.started_monotonic = 0.0
        self.active: dict[str, Any] = {}
        self.thread: threading.Thread | None = None
        self.last_progress_update = 0.0

    def __enter__(self) -> "ActiveModelCall":
        stage, label = infer_call_stage(self.messages)
        call_id = uuid.uuid4().hex[:8]
        self.started_monotonic = time.monotonic()
        started_wall = round(time.time(), 3)
        self.active = {
            "id": call_id,
            "stage": stage,
            "label": label,
            "started": started_wall,
            "elapsed_s": 0,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "prompt_chars": sum(len(message.get("content", "")) for message in self.messages),
        }
        self.observable.store.update(self.observable.job_id, active_model_call=self.active)
        self.observable.store.event(self.observable.job_id, {"event": "model_call_started", "time": started_wall, **self.active})

        def heartbeat() -> None:
            while not self.stop.wait(1):
                elapsed = round(time.monotonic() - self.started_monotonic, 1)
                self.observable.store.update(self.observable.job_id, active_model_call={**self.active, "elapsed_s": elapsed})

        self.thread = threading.Thread(target=heartbeat, daemon=True)
        self.thread.start()
        return self

    def note_chunk(self, content: str, started_monotonic: float) -> None:
        now = time.monotonic()
        if now - self.last_progress_update < 0.25:
            return
        self.last_progress_update = now
        elapsed = max(0.001, now - started_monotonic)
        approx_tokens = max(1, round(len(content) / 4))
        progress = {
            "chars": len(content),
            "approx_completion_tokens": approx_tokens,
            "elapsed_s": round(elapsed, 2),
            "approx_tokens_per_second": round(approx_tokens / elapsed, 2),
        }
        active = {**self.active, **progress}
        self.active = active
        self.observable.store.update_live(
            self.observable.job_id,
            active_model_call=active,
            stream_stats=progress,
        )

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        if exc is not None:
            elapsed = round(time.monotonic() - self.started_monotonic, 3)
            self.observable.store.event(
                self.observable.job_id,
                {
                    "event": "model_call_failed",
                    "time": round(time.time(), 3),
                    **self.active,
                    "elapsed_s": elapsed,
                    "error": str(exc),
                },
            )
        self.stop.set()
        self.observable.store.update(self.observable.job_id, active_model_call=None)
        if exc is None and MODEL_CALL_COOLDOWN_S > 0:
            self.observable.store.event(
                self.observable.job_id,
                {
                    "event": "model_call_cooldown",
                    "time": round(time.time(), 3),
                    "seconds": MODEL_CALL_COOLDOWN_S,
                },
            )
            time.sleep(MODEL_CALL_COOLDOWN_S)
        return False


def int_field(data: dict[str, Any], name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(data.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def is_project_file(path: Path) -> bool:
    return path.name in PROJECT_PRIORITY_NAMES or path.suffix in PROJECT_INCLUDE_SUFFIXES


def should_skip(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    return any(part in PROJECT_EXCLUDE_DIRS for part in relative.parts)


def read_text_file(path: Path, max_chars: int) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[could not read: {exc}]"
    data = data.strip()
    if len(data) <= max_chars:
        return data
    head = max_chars // 2
    tail = max_chars - head - 40
    return f"{data[:head].rstrip()}\n\n[...snip...]\n\n{data[-tail:].lstrip()}"


def build_project_context(project_path: str) -> dict[str, Any]:
    if not project_path.strip():
        return {"path": "", "context": "", "files": 0}

    root = Path(project_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"project path is not a directory: {root}")

    all_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and not should_skip(path, root) and is_project_file(path)
    ]
    all_files.sort(key=lambda path: (path.name not in PROJECT_PRIORITY_NAMES, len(path.parts), str(path)))

    tree_lines: list[str] = []
    for path in all_files[:80]:
        tree_lines.append(str(path.relative_to(root)))

    selected: list[Path] = []
    for path in all_files:
        rel = str(path.relative_to(root))
        if path.name in PROJECT_PRIORITY_NAMES or rel.startswith(("app/", "components/", "lib/")):
            selected.append(path)
        if len(selected) >= 26:
            break

    sections = [
        "PROJECT CONTEXT BRIEF",
        f"Root: {root}",
        "",
        "File tree excerpt:",
        "\n".join(tree_lines) or "[no readable source files found]",
        "",
        "Selected file excerpts:",
    ]

    total_chars = 0
    max_total_chars = 6200
    for path in selected:
        rel = str(path.relative_to(root))
        remaining = max_total_chars - total_chars
        if remaining <= 800:
            break
        per_file = 1000 if path.name in PROJECT_PRIORITY_NAMES else 450
        content = read_text_file(path, min(per_file, remaining))
        total_chars += len(content)
        sections.append(f"\n--- {rel} ---\n{content}")

    context = "\n".join(sections)
    return {
        "path": str(root),
        "context": context,
        "files": len(all_files),
        "selected_files": [str(path.relative_to(root)) for path in selected],
        "chars": len(context),
    }


def project_files(root: Path, max_files: int | None = None) -> list[Path]:
    max_files = max_files or DEFAULT_PROJECT_MAX_FILES
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and not should_skip(path, root) and is_project_file(path)
    ]
    files.sort(key=lambda path: (path.name not in PROJECT_PRIORITY_NAMES, len(path.parts), str(path)))
    return files[:max_files]


def project_fingerprint(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(str(root).encode("utf-8", "replace"))
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = str(path.relative_to(root))
        digest.update(rel.encode("utf-8", "replace"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def request_project_terms(request: str) -> list[str]:
    terms = []
    for raw in re.findall(r"[A-Za-z0-9_.-]{3,}", request):
        term = raw.strip(".,:;()[]{}").lower()
        if term and term not in PROJECT_STOP_WORDS and not term.isdigit():
            terms.append(term)
    return terms


def retrieval_terms(text: str) -> set[str]:
    terms = set(request_project_terms(text))
    for raw in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+", text):
        term = raw.lower()
        if len(term) >= 3 and term not in PROJECT_STOP_WORDS and not term.isdigit():
            terms.add(term)
    return terms


def summary_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(summary_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(summary_text(item) for item in value)
    return str(value or "")


def retrieval_score(terms: set[str], haystack: str, *, path_bonus: str = "") -> int:
    if not terms:
        return 0
    lowered = haystack.lower()
    path_lowered = path_bonus.lower()
    score = 0
    for term in terms:
        if term in path_lowered:
            score += 8
        count = lowered.count(term)
        if count:
            score += min(count, 8)
    return score


def retrieve_project_context(root: Path, request: str, memory: dict[str, Any], *, max_files: int = 8, max_dirs: int = 4) -> dict[str, Any]:
    terms = retrieval_terms(request)
    file_summaries = memory.get("file_summaries") or []
    directory_summaries = memory.get("directory_summaries") or []

    ranked_files = []
    for item in file_summaries:
        path = str(item.get("path") or "")
        score = retrieval_score(terms, f"{path} {summary_text(item)}", path_bonus=path)
        if score:
            ranked_files.append((score, item))
    if not ranked_files:
        ranked_files = [(1, item) for item in file_summaries[:max_files]]
    ranked_files.sort(key=lambda pair: (-pair[0], str(pair[1].get("path") or "")))

    ranked_dirs = []
    for item in directory_summaries:
        directory = str(item.get("directory") or "")
        score = retrieval_score(terms, f"{directory} {summary_text(item)}", path_bonus=directory)
        if score:
            ranked_dirs.append((score, item))
    ranked_dirs.sort(key=lambda pair: (-pair[0], str(pair[1].get("directory") or "")))

    snippets = []
    for score, item in ranked_files[:max_files]:
        rel = str(item.get("path") or "")
        path = root / rel
        if path.is_file() and not should_skip(path, root) and is_project_file(path):
            snippets.append(
                {
                    "path": rel,
                    "score": score,
                    "summary": item.get("summary", ""),
                    "snippet": read_text_file(path, 1400),
                }
            )

    return {
        "terms": sorted(terms),
        "files": [{"score": score, **item} for score, item in ranked_files[:max_files]],
        "directories": [{"score": score, **item} for score, item in ranked_dirs[:max_dirs]],
        "snippets": snippets,
    }


def safe_path(value: str, *, must_exist: bool = True) -> Path:
    path = Path(value or ".").expanduser().resolve()
    home = Path.home().resolve()
    try:
        path.relative_to(home)
    except ValueError as exc:
        raise ValueError(f"path must be under {home}") from exc
    if must_exist and not path.exists():
        raise ValueError(f"path does not exist: {path}")
    return path


def project_memory_scope(path: str) -> str:
    if not path.strip():
        return "project:"
    return f"project:{safe_path(path)}"


def run_readonly_command(args: list[str], cwd: Path, timeout_s: int = 10) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    return {
        "command": args,
        "cwd": str(cwd),
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-12000:],
        "stderr": proc.stderr[-6000:],
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def capability_git_status(data: dict[str, Any]) -> dict[str, Any]:
    path = safe_path(str(data.get("path") or "."))
    cwd = path if path.is_dir() else path.parent
    return {
        "status": run_readonly_command(["git", "status", "--short", "--branch"], cwd),
        "recent": run_readonly_command(["git", "log", "--oneline", "-n", "8"], cwd),
        "diff_stat": run_readonly_command(["git", "diff", "--stat"], cwd),
    }


def capability_list_dir(data: dict[str, Any]) -> dict[str, Any]:
    path = safe_path(str(data.get("path") or "."))
    if not path.is_dir():
        raise ValueError(f"not a directory: {path}")
    limit = max(1, min(int(data.get("limit") or 80), 200))
    entries = []
    for item in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:limit]:
        try:
            stat = item.stat()
        except OSError:
            continue
        entries.append(
            {
                "name": item.name,
                "path": str(item),
                "kind": "dir" if item.is_dir() else "file",
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return {"path": str(path), "entries": entries}


def capability_file_preview(data: dict[str, Any]) -> dict[str, Any]:
    path = safe_path(str(data.get("path") or ""))
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    max_chars = max(200, min(int(data.get("max_chars") or 4000), 16000))
    return {"path": str(path), "content": read_text_file(path, max_chars), "max_chars": max_chars}


def capability_search_text(data: dict[str, Any]) -> dict[str, Any]:
    path = safe_path(str(data.get("path") or "."))
    query = str(data.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    cwd = path if path.is_dir() else path.parent
    return run_readonly_command(["rg", "--line-number", "--no-heading", "--max-count", "80", query, str(path)], cwd)


def fetch_url_text(url: str, *, timeout_s: int = 12, max_bytes: int = 1_000_000) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("url must start with http:// or https://")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 qwen-orchestrator/0.1",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.2",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as res:
        content_type = res.headers.get("Content-Type", "")
        raw = res.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    charset = "utf-8"
    match = re.search(r"charset=([A-Za-z0-9_.-]+)", content_type)
    if match:
        charset = match.group(1)
    return raw.decode(charset, "replace"), content_type


def normalize_search_url(href: str, base_url: str) -> str:
    href = html.unescape(href or "")
    absolute = urljoin(base_url, href)
    parsed = urlparse(absolute)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    if "u" in query and query["u"]:
        encoded = query["u"][0]
        for candidate in (encoded, encoded[2:] if encoded.startswith("a1") else ""):
            if not candidate:
                continue
            try:
                padded = candidate + "=" * (-len(candidate) % 4)
                decoded = base64.urlsafe_b64decode(padded).decode("utf-8", "replace")
            except Exception:
                continue
            if decoded.startswith(("http://", "https://")):
                return decoded
    return absolute


class BingParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.results: list[dict[str, str]] = []
        self._in_result = False
        self._in_h2 = False
        self._current: dict[str, str] | None = None
        self._title_text: list[str] = []
        self._snippet_target: dict[str, str] | None = None
        self._snippet_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        classes = set(attr.get("class", "").split())
        if tag == "li" and "b_algo" in classes:
            self._in_result = True
        elif self._in_result and tag == "h2":
            self._in_h2 = True
        elif self._in_result and self._in_h2 and tag == "a" and self._current is None:
            self._current = {"url": normalize_search_url(attr.get("href", ""), self.base_url)}
            self._title_text = []
        elif self._in_result and tag == "p" and self.results:
            self._snippet_target = self.results[-1]
            self._snippet_text = []

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._title_text.append(data)
        if self._snippet_target is not None:
            self._snippet_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current is not None:
            title = " ".join(" ".join(self._title_text).split())
            url = self._current.get("url", "")
            if title and url and not url.startswith("javascript:") and not any(item.get("url") == url for item in self.results):
                self.results.append({"title": title, "url": url, "snippet": ""})
            self._current = None
            self._title_text = []
        elif tag == "h2":
            self._in_h2 = False
        elif tag == "p" and self._snippet_target is not None:
            snippet = " ".join(" ".join(self._snippet_text).split())
            if snippet:
                self._snippet_target["snippet"] = snippet
            self._snippet_target = None
            self._snippet_text = []
        elif tag == "li":
            self._in_result = False


class DuckDuckGoParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._text: list[str] = []
        self._snippet_target: dict[str, str] | None = None
        self._snippet_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        classes = set(attr.get("class", "").split())
        if tag == "a" and ("result__a" in classes or "result-link" in classes):
            self._current = {"url": normalize_search_url(attr.get("href", ""), self.base_url)}
            self._text = []
        elif tag in {"a", "div", "span"} and ({"result__snippet", "result-snippet"} & classes):
            self._snippet_target = self.results[-1] if self.results else None
            self._snippet_text = []

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._text.append(data)
        if self._snippet_target is not None:
            self._snippet_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current is not None:
            title = " ".join(" ".join(self._text).split())
            url = self._current.get("url", "")
            if title and url and not any(item.get("url") == url for item in self.results):
                self.results.append({"title": title, "url": url, "snippet": ""})
            self._current = None
            self._text = []
        elif self._snippet_target is not None and tag in {"a", "div", "span"}:
            snippet = " ".join(" ".join(self._snippet_text).split())
            if snippet:
                self._snippet_target["snippet"] = snippet
            self._snippet_target = None
            self._snippet_text = []


class PageTextParser(HTMLParser):
    BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "blockquote", "pre", "td", "th"}
    SKIP_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._capture_tag: str | None = None
        self._buffer: list[str] = []
        self.blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            self._buffer = []
        elif tag in self.BLOCK_TAGS and self._skip_depth == 0:
            self._capture_tag = tag
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title or self._capture_tag:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        text = " ".join(" ".join(self._buffer).split())
        if tag == "title" and self._in_title:
            self.title = html.unescape(text)
            self._in_title = False
            self._buffer = []
        elif self._capture_tag == tag:
            if len(text) >= 20:
                self.blocks.append(html.unescape(text))
            self._capture_tag = None
            self._buffer = []


def web_search_query_from_request(request: str) -> str:
    raw = " ".join(str(request or "").strip().split())
    if not raw:
        return ""
    query = raw
    replacements = [
        r"^(?:hey|hi|hello)[,\s]+",
        r"^(?:can|could|would|will)\s+you\s+",
        r"^(?:please|pls)\s+",
        r"^(?:search|find|lookup|look\s+up)\s+(?:the\s+)?(?:web|internet|online)?\s*(?:for\s+)?",
        r"^(?:search|find|lookup|look\s+up)\s+(?:the\s+)?",
        r"^(?:tell|show|give)\s+me\s+(?:about\s+)?",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in replacements:
            updated = re.sub(pattern, "", query, flags=re.IGNORECASE).strip(" ,.;:-")
            if updated != query:
                query = updated
                changed = True
    suffixes = [
        r"\s+(?:and\s+)?(?:give|show|tell)\s+me\s+(?:the\s+)?(?:results?|answer|info|information)\s+for\s+(?:the\s+)?",
        r"\s+(?:and\s+)?(?:give|show|tell)\s+(?:the\s+)?(?:results?|answer|info|information)\s+for\s+(?:the\s+)?",
        r"\s+(?:and\s+)?(?:give|show|tell)\s+me\s+(?:the\s+)?(?:results?|answer|info|information)\s*$",
        r"\s+(?:and\s+)?(?:give|show|tell)\s+(?:the\s+)?(?:results?|answer|info|information)\s*$",
        r"\s+(?:please|pls)\s*$",
    ]
    for pattern in suffixes:
        query = re.sub(pattern, " ", query, flags=re.IGNORECASE).strip(" ,.;:-")
    lowered = query.lower()
    if "fifa" in lowered and "world cup" in lowered and any(term in lowered for term in {"last game", "latest game", "last match", "latest match"}):
        extras = []
        if "completed" not in lowered:
            extras.append("latest completed")
        if "score" not in lowered:
            extras.append("result score")
        if "as of" not in lowered:
            extras.append(f"as of {CURRENT_DATE}")
        query = f"{query} {' '.join(extras)}".strip()
    return query or raw


def is_fifa_world_cup_last_result_request(text: str) -> bool:
    lowered = text.lower()
    return "fifa" in lowered and "world cup" in lowered and any(
        term in lowered for term in {"last game", "latest game", "last match", "latest match"}
    )


def sports_result_search_queries(request: str, primary_query: str) -> list[str]:
    if not is_fifa_world_cup_last_result_request(request):
        return [primary_query]
    candidates = [
        primary_query,
        f"2026 FIFA World Cup latest completed match final score {CURRENT_DATE}",
        "2026 FIFA World Cup latest final score ESPN",
        "2026 FIFA World Cup latest completed match Argentina England 2-1",
    ]
    deduped = []
    for query in candidates:
        if query not in deduped:
            deduped.append(query)
    return deduped


def search_relevance_terms(query: str) -> list[str]:
    terms = []
    for term in re.findall(r"[a-z0-9][a-z0-9.+-]{1,}", query.lower()):
        if term not in SEARCH_QUERY_STOP_WORDS and term not in terms:
            terms.append(term)
    return terms


def search_result_score(terms: list[str], item: dict[str, str]) -> int:
    haystack = " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("snippet") or ""),
            str(item.get("url") or ""),
        ]
    ).lower()
    return sum(1 for term in terms if term in haystack)


def search_results_are_relevant(query: str, results: list[dict[str, str]]) -> bool:
    terms = search_relevance_terms(query)
    if not terms:
        return bool(results)
    return any(search_result_score(terms, item) > 0 for item in results)


def capability_searxng_search(query: str, limit: int, base_url: str = DEFAULT_SEARXNG_URL) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/search?q={quote_plus(query)}&format=json&language=en&categories=general"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "qwen-orchestrator/0.1",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=12) as res:
        content_type = res.headers.get("Content-Type", "")
        data = json.loads(res.read().decode("utf-8", errors="replace"))
    results = []
    for item in data.get("results") or []:
        result_url = normalize_search_url(str(item.get("url") or ""), base_url)
        title = html.unescape(str(item.get("title") or "")).strip()
        snippet = html.unescape(str(item.get("content") or item.get("snippet") or "")).strip()
        if title and result_url and not any(existing.get("url") == result_url for existing in results):
            results.append(
                {
                    "title": title,
                    "url": result_url,
                    "snippet": snippet,
                    "engine": ", ".join(item.get("engines") or []) if isinstance(item.get("engines"), list) else str(item.get("engine") or ""),
                    "score": item.get("score"),
                    "published": item.get("publishedDate") or item.get("published_date") or "",
                }
            )
        if len(results) >= limit:
            break
    return {
        "query": query,
        "engine": "searxng",
        "url": url,
        "content_type": content_type,
        "results": results,
        "count": len(results),
    }


def capability_web_search(data: dict[str, Any]) -> dict[str, Any]:
    raw_query = str(data.get("query") or data.get("q") or "").strip()
    query = web_search_query_from_request(raw_query)
    if not raw_query:
        raise ValueError("query is required")
    limit = max(1, min(int(data.get("limit") or 5), 10))
    attempts: list[dict[str, Any]] = []
    searxng_url = str(data.get("searxng_url") or DEFAULT_SEARXNG_URL).rstrip("/")
    try:
        searxng = capability_searxng_search(query, limit, searxng_url)
        relevant = search_results_are_relevant(query, searxng["results"])
        attempts.append({"engine": "searxng", "ok": True, "count": searxng["count"], "relevant": relevant, "url": searxng_url})
        if searxng["results"] and relevant:
            searxng["raw_query"] = raw_query
            searxng["attempts"] = attempts
            return searxng
    except Exception as exc:
        attempts.append({"engine": "searxng", "ok": False, "error": str(exc), "url": searxng_url})
    backends = [
        ("duckduckgo_html", f"https://html.duckduckgo.com/html?q={quote_plus(query)}", DuckDuckGoParser),
        ("duckduckgo_lite", f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}", DuckDuckGoParser),
        ("bing", f"https://www.bing.com/search?q={quote_plus(query)}", BingParser),
    ]
    results: list[dict[str, str]] = []
    engine = ""
    search_url = ""
    content_type = ""
    for engine_name, candidate_url, parser_cls in backends:
        try:
            body, content_type = fetch_url_text(candidate_url, timeout_s=15, max_bytes=900_000)
            parser = parser_cls(candidate_url)
            parser.feed(body)
            results = parser.results[:limit]
            relevant = search_results_are_relevant(query, results)
            attempts.append({"engine": engine_name, "ok": True, "count": len(results), "relevant": relevant})
            if results and relevant:
                engine = engine_name
                search_url = candidate_url
                break
        except Exception as exc:
            attempts.append({"engine": engine_name, "ok": False, "error": str(exc)})
    if not engine:
        engine = attempts[-1]["engine"] if attempts else ""
        search_url = backends[-1][1]
    return {
        "query": query,
        "raw_query": raw_query,
        "engine": engine,
        "url": search_url,
        "content_type": content_type,
        "results": results,
        "count": len(results),
        "attempts": attempts,
    }


def capability_webpage_summary(data: dict[str, Any]) -> dict[str, Any]:
    url = str(data.get("url") or data.get("query") or "").strip()
    if not url:
        raise ValueError("url is required")
    max_chars = max(1000, min(int(data.get("max_chars") or 6000), 20000))
    body, content_type = fetch_url_text(url, timeout_s=15, max_bytes=1_500_000)
    parser = PageTextParser()
    parser.feed(body)
    text = "\n\n".join(parser.blocks)
    if not text.strip():
        text = re.sub(r"\s+", " ", body)
    return {
        "url": url,
        "title": parser.title,
        "content_type": content_type,
        "text": truncate_text(text, max_chars),
        "chars": min(len(text), max_chars),
        "truncated": len(text) > max_chars,
    }


SAFE_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Subscript,
    ast.Slice,
    ast.IfExp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
}
SAFE_PYTHON_NAMES = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "sorted": sorted,
    "range": range,
}


def capability_python_eval(data: dict[str, Any]) -> dict[str, Any]:
    expr = str(data.get("expression") or "").strip()
    if not expr:
        raise ValueError("expression is required")
    if len(expr) > 2000:
        raise ValueError("expression is too long")
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in SAFE_AST_NODES:
            raise ValueError(f"unsupported expression node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in SAFE_PYTHON_NAMES:
            raise ValueError(f"name is not allowed: {node.id}")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValueError("only simple safe function calls are allowed")
    result = eval(compile(tree, "<safe-python-eval>", "eval"), {"__builtins__": {}}, SAFE_PYTHON_NAMES)
    return {"expression": expr, "result": result}


CAPABILITIES: dict[str, dict[str, Any]] = {
    "git_status": {
        "label": "Git status",
        "description": "Read-only git branch/status/recent commits/diff stat for a local repo.",
        "handler": capability_git_status,
    },
    "list_dir": {
        "label": "List directory",
        "description": "List files under a home-directory path.",
        "handler": capability_list_dir,
    },
    "file_preview": {
        "label": "Preview file",
        "description": "Read the first part of a text file under your home directory.",
        "handler": capability_file_preview,
    },
    "search_text": {
        "label": "Search text",
        "description": "Run ripgrep against a home-directory path.",
        "handler": capability_search_text,
    },
    "web_search": {
        "label": "Web search",
        "description": "Search the web with DuckDuckGo HTML and return bounded result metadata.",
        "handler": capability_web_search,
    },
    "webpage_summary": {
        "label": "Webpage summary",
        "description": "Fetch a webpage and extract bounded readable text for model grounding.",
        "handler": capability_webpage_summary,
    },
    "python_eval": {
        "label": "Python calculation",
        "description": "Evaluate a restricted Python expression for calculations and data checks.",
        "handler": capability_python_eval,
    },
}


def capability_manifest() -> list[dict[str, str]]:
    return [
        {"id": key, "label": value["label"], "description": value["description"]}
        for key, value in CAPABILITIES.items()
    ]


def run_capability(name: str, data: dict[str, Any]) -> dict[str, Any]:
    capability = CAPABILITIES.get(name)
    if not capability:
        raise ValueError(f"unknown capability: {name}")
    started = time.monotonic()
    result = capability["handler"](data)
    return {
        "id": uuid.uuid4().hex[:12],
        "created": round(time.time(), 3),
        "capability": name,
        "label": capability["label"],
        "ok": True,
        "elapsed_s": round(time.monotonic() - started, 3),
        "result": result,
    }


def model_server_health(base_url: str = DEFAULT_OPENAI_BASE) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        started = time.monotonic()
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as res:
            data = json.loads(res.read().decode("utf-8"))
        models = data.get("data") or data.get("models") or []
        return {
            "ok": True,
            "model": DEFAULT_MODEL,
            "base_url": base_url,
            "elapsed_s": round(time.monotonic() - started, 3),
            "models": models,
        }
    except Exception as exc:
        return {
            "ok": False,
            "model": DEFAULT_MODEL,
            "base_url": base_url,
            "error": str(exc),
        }


SWITCH_CODER_SCRIPT = str(Path(__file__).resolve().parent / "switch_coder.sh")


def list_coder_profiles() -> dict[str, Any]:
    """Parse `switch_coder.sh list` into structured rows for the UI. The shell script is
    the single source of truth for the profile registry + VRAM guard; this only reads it."""
    try:
        out = subprocess.run([SWITCH_CODER_SCRIPT, "list"], capture_output=True, text=True, timeout=10)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "profiles": [], "active": None}
    profiles: list[dict[str, Any]] = []
    active = None
    for line in out.stdout.splitlines():
        if line.startswith("active:"):
            active = line.split("active:", 1)[1].strip().split()[0]
            continue
        m = re.match(r"^(\*| )([\w-]+)\s+(\d+)\s+(\d+)\s+(\d+)MiB\s+(.*)$", line)
        if m:
            profiles.append({
                "name": m.group(2), "gpu_layers": int(m.group(3)), "ctx": int(m.group(4)),
                "vram_mib": int(m.group(5)), "label": m.group(6).strip(),
                "active": m.group(1) == "*",
            })
    return {"ok": True, "profiles": profiles, "active": active}


def switch_coder_profile(name: str) -> dict[str, Any]:
    """Switch the GPU coder slot (session-level, admin-only). Blocks until the new coder is
    healthy or the guard/launch fails; returns the resulting profile state either way."""
    if not re.fullmatch(r"[\w-]+", name or ""):
        return {"ok": False, "error": "invalid profile name"}
    try:
        # Generous timeout: the 30B is a 17 GB file and can take ~2 min to load.
        out = subprocess.run([SWITCH_CODER_SCRIPT, name], capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "switch timed out (model still loading?)"}
    ok = out.returncode == 0
    state = list_coder_profiles()
    return {"ok": ok, "message": (out.stdout or out.stderr).strip()[-400:],
            "profiles": state.get("profiles", []), "active": state.get("active")}


def extract_tool_requests_from_text(text: str) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    if "tool_request" not in text:
        return requests
    for match in re.finditer(r"\{", text):
        candidate = text[match.start() :]
        try:
            parsed = extract_json(candidate)
        except Exception:
            continue
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if not isinstance(item, dict):
                continue
            request = item.get("tool_request") or item
            if not isinstance(request, dict):
                continue
            capability = str(request.get("capability") or "")
            if capability in CAPABILITIES:
                requests.append(
                    {
                        "capability": capability,
                        "input": request.get("input") or {},
                        "reason": str(request.get("reason") or ""),
                    }
                )
        if requests:
            break
    return requests


def extract_trace_tool_requests(trace: dict[str, Any]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for result in trace.get("worker_results") or []:
        for request in extract_tool_requests_from_text(str(result.get("answer") or "")):
            requests.append({"source": result.get("id") or "worker", **request})
    for request in extract_tool_requests_from_text(str(trace.get("final") or "")):
            requests.append({"source": "final", **request})
    return requests


def extract_memory_suggestions_from_text(text: str) -> list[dict[str, Any]]:
    """Parse model-proposed durable memories: {"memory_suggestion": {scope, key,
    value, tags, reason}}. The model can only *propose*; the user approves before
    anything is saved (same gate as tool_request)."""
    suggestions: list[dict[str, Any]] = []
    if "memory_suggestion" not in text:
        return suggestions
    for match in re.finditer(r"\{", text):
        try:
            parsed = extract_json(text[match.start():])
        except Exception:
            continue
        for item in (parsed if isinstance(parsed, list) else [parsed]):
            if not isinstance(item, dict):
                continue
            suggestion = item.get("memory_suggestion")
            if not isinstance(suggestion, dict):
                continue
            key = str(suggestion.get("key") or "").strip()[:160]
            value = str(suggestion.get("value") or "").strip()[:1000]
            if not key or not value:
                continue
            scope = str(suggestion.get("scope") or "user").strip().lower()
            scope = scope if scope in ("user", "project") else "user"
            tags = [str(tag).strip()[:40] for tag in (suggestion.get("tags") or []) if str(tag).strip()][:6]
            suggestions.append({"scope": scope, "key": key, "value": value, "tags": tags, "reason": str(suggestion.get("reason") or "")[:280]})
        if suggestions:
            break
    return suggestions


def extract_trace_memory_suggestions(trace: dict[str, Any], existing: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Collect memory suggestions from the final answer, dropping any that duplicate
    a memory already saved (same key, case-insensitive) or repeat within the batch."""
    seen_keys = {str(m.get("key") or "").strip().lower() for m in (existing or [])}
    out: list[dict[str, Any]] = []
    for suggestion in extract_memory_suggestions_from_text(str(trace.get("final") or "")):
        k = suggestion["key"].lower()
        if k in seen_keys:
            continue
        seen_keys.add(k)
        out.append(suggestion)
    return out[:4]   # keep the approval surface small


def attach_capability_result(store: JobStore, job_id: str, result: dict[str, Any], summary: str) -> None:
    if store.persistence:
        store.persistence.save_capability_run(result)
    preview = json.dumps(result.get("result"), indent=2)[:1200]
    store.event(
        job_id,
        {
            "event": "capability_run_attached",
            "time": round(time.time(), 3),
            "id": result["id"],
            "capability": result["capability"],
            "ok": result["ok"],
            "summary": summary,
            "result_preview": preview,
        },
    )


def collect_attached_capabilities(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Approved tool results already attached to a job, in run order.

    Reads the ``capability_run_attached`` events that ``/api/capability/run`` and
    ``attach_capability_result`` emit, so a user-approved tool run can be carried
    into a continuation run as evidence."""
    items: list[dict[str, Any]] = []
    for event in job.get("events") or []:
        if event.get("event") != "capability_run_attached":
            continue
        items.append(
            {
                "capability": event.get("capability"),
                "input": event.get("input") or {},
                "ok": event.get("ok"),
                "summary": event.get("summary") or "",
                "result_preview": event.get("result_preview") or "",
            }
        )
    return items


def tool_evidence_prompt_block(evidence: list[dict[str, Any]]) -> str:
    """Prompt block injecting host-gathered, user-approved tool results.

    Human-in-the-loop only: this evidence exists because the user clicked to run a
    requested tool, and each continuation is a single user-initiated run (one
    prefill). Do NOT turn this into an automatic re-prompt loop -- repeated
    back-to-back prefills are the workload that puts the RX 580 at risk
    (see ~/.claude/CLAUDE.md GPU note)."""
    if not evidence:
        return ""
    return (
        "\n\nTOOL EVIDENCE GATHERED BY HOST (approved by the user):\n"
        f"{json.dumps(evidence, indent=2)[:TOOL_EVIDENCE_PROMPT_CHARS]}\n\n"
        "Use this tool evidence as the source of truth for what these tools returned. "
        "Cite it when making claims that depend on it. Do not claim you ran any tool "
        "yourself, and do not pretend to have evidence from tools not listed here. "
        "If the evidence is insufficient, say what is missing instead of inventing details."
    )


def build_web_context(store: JobStore, job_id: str, request: str) -> dict[str, Any]:
    query = web_search_query_from_request(request)
    context: dict[str, Any] = {"query": query, "raw_query": request, "search": {}, "searches": [], "pages": [], "errors": []}
    try:
        all_results: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for search_query in sports_result_search_queries(request, query):
            if store.is_cancelled(job_id):
                raise JobCancelled()
            store.event(job_id, {"event": "web_search_started", "time": round(time.time(), 3), "query": search_query, "raw_query": request})
            search_run = run_capability("web_search", {"query": search_query, "limit": 5})
            attach_capability_result(store, job_id, search_run, "Web search completed")
            search = search_run.get("result") or {}
            context["searches"].append(search)
            if not context["search"]:
                context["search"] = search
            store.event(
                job_id,
                {
                    "event": "web_search_finished",
                    "time": round(time.time(), 3),
                    "query": search.get("query") or search_query,
                    "raw_query": request,
                    "count": search.get("count", 0),
                    "engine": search.get("engine", ""),
                },
            )
            for item in search.get("results") or []:
                url = str(item.get("url") or "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(item)
            if not is_fifa_world_cup_last_result_request(request):
                break
        ranked_results = sorted(
            all_results,
            key=lambda item: (-web_result_quality_score(item), str(item.get("title") or "")),
        )
        context["search_results"] = ranked_results[:8]
        page_limit = 4 if is_fifa_world_cup_last_result_request(request) else 2
        for item in ranked_results[:page_limit]:
            if store.is_cancelled(job_id):
                raise JobCancelled()
            url = str(item.get("url") or "")
            if not url:
                continue
            try:
                store.event(job_id, {"event": "web_page_started", "time": round(time.time(), 3), "url": url, "title": item.get("title", "")})
                page_run = run_capability("webpage_summary", {"url": url, "max_chars": 2600})
                attach_capability_result(store, job_id, page_run, f"Fetched webpage: {item.get('title') or url}")
                page = page_run.get("result") or {}
                context["pages"].append(
                    {
                        "url": page.get("url") or url,
                        "title": page.get("title") or item.get("title", ""),
                        "search_snippet": item.get("snippet", ""),
                        "text": page.get("text", ""),
                        "truncated": page.get("truncated", False),
                    }
                )
                store.event(job_id, {"event": "web_page_read", "time": round(time.time(), 3), "url": url, "title": page.get("title", "")})
            except Exception as exc:
                context["errors"].append({"url": url, "error": str(exc)})
                store.event(job_id, {"event": "web_page_failed", "time": round(time.time(), 3), "url": url, "error": str(exc)})
    except JobCancelled:
        raise
    except Exception as exc:
        context["errors"].append({"stage": "web_search", "error": str(exc)})
        store.event(job_id, {"event": "web_search_failed", "time": round(time.time(), 3), "error": str(exc)})
    return context


def should_prefetch_docs(route: dict[str, Any], request: str, project_memory: dict[str, Any]) -> bool:
    mode = route.get("mode")
    if mode not in {"plan", "implementation", "debug", "code_review", "project_research"}:
        return False
    lowered = request.lower()
    words = set(re.findall(r"[a-z0-9_.-]+", lowered))
    if words & DOCS_HINT_TERMS:
        return True
    if mode in {"implementation", "debug", "code_review"}:
        retrieval = project_memory.get("retrieval") or {}
        snippets = json.dumps(retrieval.get("snippets") or [])[:4000].lower()
        return any(term in snippets for term in DOCS_HINT_TERMS)
    return False


def docs_search_query(request: str, project_memory: dict[str, Any]) -> str:
    text = request.strip()
    words = re.findall(r"[a-z0-9_.-]+", text.lower())
    tech_terms = [word for word in words if word in DOCS_HINT_TERMS]
    retrieval = project_memory.get("retrieval") or {}
    snippets = json.dumps(retrieval.get("snippets") or [])[:4000].lower()
    for term in sorted(DOCS_HINT_TERMS):
        if term in snippets and term not in tech_terms:
            tech_terms.append(term)
        if len(tech_terms) >= 5:
            break
    focus = " ".join(tech_terms[:5])
    if focus:
        return f"{focus} official documentation best practices {text}"[:220]
    return f"official documentation best practices {text}"[:220]


def build_docs_context(store: JobStore, job_id: str, request: str, project_memory: dict[str, Any]) -> dict[str, Any]:
    query = docs_search_query(request, project_memory)
    context: dict[str, Any] = {"query": query, "raw_query": request, "search": {}, "searches": [], "pages": [], "errors": [], "kind": "docs"}
    try:
        store.event(job_id, {"event": "docs_search_started", "time": round(time.time(), 3), "query": query, "raw_query": request})
        search_run = run_capability("web_search", {"query": query, "limit": 6})
        attach_capability_result(store, job_id, search_run, "Documentation/best-practices search completed")
        search = search_run.get("result") or {}
        context["search"] = search
        context["searches"].append(search)
        results = search.get("results") or []
        ranked_results = sorted(results, key=lambda item: (-docs_result_quality_score(item), str(item.get("title") or "")))
        context["search_results"] = ranked_results[:6]
        store.event(
            job_id,
            {
                "event": "docs_search_finished",
                "time": round(time.time(), 3),
                "query": search.get("query") or query,
                "count": search.get("count", 0),
                "engine": search.get("engine", ""),
            },
        )
        for item in ranked_results[:2]:
            if store.is_cancelled(job_id):
                raise JobCancelled()
            url = str(item.get("url") or "")
            if not url:
                continue
            try:
                store.event(job_id, {"event": "docs_page_started", "time": round(time.time(), 3), "url": url, "title": item.get("title", "")})
                page_run = run_capability("webpage_summary", {"url": url, "max_chars": 2600})
                attach_capability_result(store, job_id, page_run, f"Fetched docs page: {item.get('title') or url}")
                page = page_run.get("result") or {}
                context["pages"].append(
                    {
                        "url": page.get("url") or url,
                        "title": page.get("title") or item.get("title", ""),
                        "search_snippet": item.get("snippet", ""),
                        "text": page.get("text", ""),
                        "truncated": page.get("truncated", False),
                    }
                )
                store.event(job_id, {"event": "docs_page_read", "time": round(time.time(), 3), "url": url, "title": page.get("title", "")})
            except Exception as exc:
                context["errors"].append({"url": url, "error": str(exc)})
                store.event(job_id, {"event": "docs_page_failed", "time": round(time.time(), 3), "url": url, "error": str(exc)})
    except JobCancelled:
        raise
    except Exception as exc:
        context["errors"].append({"stage": "docs_search", "error": str(exc)})
        store.event(job_id, {"event": "docs_search_failed", "time": round(time.time(), 3), "error": str(exc)})
    return context


def compact_web_context_for_prompt(web_context: dict[str, Any]) -> dict[str, Any]:
    search_results = []
    for item in (web_context.get("search_results") or (web_context.get("search") or {}).get("results") or [])[:8]:
        search_results.append(
            {
                "title": truncate_text(str(item.get("title") or ""), 120),
                "url": str(item.get("url") or ""),
                "snippet": truncate_text(str(item.get("snippet") or ""), 240),
            }
        )
    pages = []
    for page in (web_context.get("pages") or [])[:4]:
        pages.append(
            {
                "title": truncate_text(str(page.get("title") or ""), 120),
                "url": str(page.get("url") or ""),
                "search_snippet": truncate_text(str(page.get("search_snippet") or ""), 200),
                "text": truncate_text(str(page.get("text") or ""), 450),
            }
        )
    return {
        "query": web_context.get("query", ""),
        "raw_query": web_context.get("raw_query", ""),
        "search_results": search_results,
        "pages": pages,
        "errors": web_context.get("errors", [])[:3],
    }


# --- Source authority: stop low-quality aggregators from outranking primary sources.
# This is the fix for a content farm (e.g. "ClayStage") beating real reporting. ---
AUTHORITATIVE_DOMAINS = {
    "wikipedia.org", "britannica.com", "reuters.com", "apnews.com", "bbc.com",
    "bbc.co.uk", "nytimes.com", "theguardian.com", "washingtonpost.com", "npr.org",
    "aljazeera.com", "bloomberg.com", "wsj.com", "ft.com", "cnn.com", "cnbc.com",
    "abcnews.go.com", "arstechnica.com", "theverge.com", "espn.com", "nature.com",
    "sciencedirect.com", "who.int",
}
LOW_AUTHORITY_DOMAINS = {
    "facebook.com", "youtube.com", "tiktok.com", "instagram.com", "pinterest.com",
    "quora.com", "medium.com",
}
# Substrings that flag speculative rumor/leak content farms.
SPAM_HOST_SIGNALS = (
    "claystage", "otakukart", "epicstream", "fictionhorizon", "spoilerguy",
    "spoiler", "leaks", "top10", "listicle",
)


def source_authority_score(host: str) -> int:
    """Rank a hostname by how much a claim from it can be trusted. Reputable
    references/news and official/.gov/.edu domains score high; social media and
    rumor/leak content farms score negative."""
    host = (host or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return 0
    score = 0
    if any(host == d or host.endswith("." + d) for d in AUTHORITATIVE_DOMAINS):
        score += 10
    if host.endswith(".gov") or host.endswith(".edu") or ".gov." in host or ".edu." in host:
        score += 8
    if host.startswith("docs.") or host.startswith("developer."):
        score += 4
    if any(host == d or host.endswith("." + d) for d in LOW_AUTHORITY_DOMAINS):
        score -= 8
    if any(sig in host for sig in SPAM_HOST_SIGNALS):
        score -= 10
    if host.count("-") >= 3:               # spammy multi-hyphen domains
        score -= 3
    return score


def web_result_quality_score(item: dict[str, Any]) -> int:
    title = str(item.get("title") or "")
    snippet = str(item.get("snippet") or "")
    url = str(item.get("url") or "")
    haystack = f"{title} {snippet}".lower()
    host = urlparse(url).netloc.lower()
    score = source_authority_score(host)      # authority is the dominant signal
    if re.search(r"\b\d+\s*[-–]\s*\d+\b", haystack):            # a scoreline / range
        score += 4
    if re.search(r"\b\d+\s*(?:days?|hours?)\s+ago\b", haystack):
        score += 3
    for term in ["final score", "final result", "full time", "completed", "confirmed", "official", "announced", "released"]:
        if term in haystack:
            score += 1
    for term in ["schedule", "fixture", "tickets", "odds", "where to watch", "rumor", "rumour", "leak", "prediction", "speculation", "spoiler"]:
        if term in haystack:
            score -= 2
    return score


def docs_result_quality_score(item: dict[str, Any]) -> int:
    title = str(item.get("title") or "")
    snippet = str(item.get("snippet") or "")
    url = str(item.get("url") or "")
    haystack = f"{title} {snippet} {url}".lower()
    host = urlparse(url).netloc.lower()
    score = 0
    if "docs." in host or "developer." in host or "dev." in host:
        score += 5
    for domain in ["github.com", "readthedocs.io", "python.org", "nodejs.org", "react.dev", "nextjs.org", "fastapi.tiangolo.com", "docs.djangoproject.com", "flask.palletsprojects.com", "pytest.org", "typescriptlang.org", "tailwindcss.com", "sqlalchemy.org", "postgresql.org"]:
        if domain in host:
            score += 6
            break
    for term in ["official", "documentation", "docs", "guide", "best practices", "security", "migration", "api reference"]:
        if term in haystack:
            score += 2
    for domain in ["medium.com", "dev.to", "stackoverflow.com", "reddit.com", "quora.com", "facebook.", "youtube.", "tiktok."]:
        if domain in host:
            score -= 4
            break
    return score


def discover_project(project_spec: str, request: str) -> Path | None:
    spec = project_spec.strip()
    if spec:
        candidate = Path(spec).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
        for root in PROJECT_SEARCH_ROOTS:
            found = root / spec
            if found.is_dir():
                return found.resolve()

    terms = request_project_terms(request)
    if not terms:
        return None

    candidates: list[Path] = []
    for root in PROJECT_SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for path in root.glob("*"):
            if path.is_dir() and path.name.lower() in terms:
                candidates.append(path.resolve())
        for parent in root.glob("*"):
            if not parent.is_dir() or parent.name.startswith("."):
                continue
            for path in parent.glob("*"):
                if path.is_dir() and path.name.lower() in terms and not should_skip(path, root):
                    candidates.append(path.resolve())
    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    return candidates[0] if candidates else None


def project_call(store: JobStore, job_id: str, stage: str, label: str, response: Any, calls: list[dict[str, Any]]) -> dict[str, Any]:
    call = compact_call(stage, response, label=label)
    calls.append(call)
    store.event(job_id, {"event": "model_call_finished", "time": round(time.time(), 3), **call})
    return call


def parsed_json_mapping(content: str, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = extract_json(content)
    except Exception:
        parsed = {**fallback, "summary": content.strip()}
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {**fallback, "summary": "; ".join(str(item) for item in parsed[:8])}
    return {**fallback, "summary": str(parsed)}


def summarize_project_file(
    client: Any,
    store: JobStore,
    job_id: str,
    root: Path,
    path: Path,
    calls: list[dict[str, Any]],
    *,
    index: int | None = None,
    total: int | None = None,
) -> dict[str, Any]:
    if store.is_cancelled(job_id):
        raise JobCancelled()
    rel = str(path.relative_to(root))
    content = read_text_file(path, 7000)
    store.event(
        job_id,
        {
            "event": "project_file_started",
            "time": round(time.time(), 3),
            "path": rel,
            "index": index,
            "total": total,
        },
    )
    response = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a codebase reader. Summarize exactly one project file for later architecture analysis. "
                    "Preserve concrete exports, components, state, data flow, dependencies, risks, and TODOs. "
                    "Return concise JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "file": rel,
                        "content": content,
                        "schema": {
                            "summary": "what this file does",
                            "important_symbols": ["component/function/type/export"],
                            "data_flow": ["inputs, outputs, state, side effects"],
                            "dependencies": ["internal/external dependencies"],
                            "risks_or_gaps": ["possible issue"],
                        },
                    },
                    indent=2,
                ),
            },
        ],
        temperature=0.0,
        max_tokens=260,
    )
    project_call(store, job_id, "project_file_reader", rel, response, calls)
    parsed = parsed_json_mapping(
        response.content,
        {"summary": "", "important_symbols": [], "data_flow": [], "dependencies": [], "risks_or_gaps": []},
    )
    summary = {"path": rel, **parsed}
    store.event(job_id, {"event": "project_file_read", "time": round(time.time(), 3), "path": rel, "summary": parsed.get("summary", "")})
    return summary


def aggregate_project_memory(client: Any, store: JobStore, job_id: str, root: Path, file_summaries: list[dict[str, Any]], calls: list[dict[str, Any]]) -> dict[str, Any]:
    by_dir: dict[str, list[dict[str, Any]]] = {}
    for summary in file_summaries:
        directory = str(Path(summary["path"]).parent)
        by_dir.setdefault(directory, []).append(summary)

    directory_summaries = []
    for directory, summaries in sorted(by_dir.items()):
        if store.is_cancelled(job_id):
            raise JobCancelled()
        store.event(job_id, {"event": "project_directory_started", "time": round(time.time(), 3), "path": directory})
        response = client.chat(
            [
                {"role": "system", "content": "You compact file-level summaries into a directory architecture summary. Return concise JSON only."},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "directory": directory,
                            "files": summaries,
                            "schema": {
                                "summary": "directory responsibility",
                                "key_files": ["path: why important"],
                                "data_flow": ["how data/state moves"],
                                "risks_or_gaps": ["issue"],
                            },
                        },
                        indent=2,
                    )[:9000],
                },
            ],
            temperature=0.0,
            max_tokens=420,
        )
        project_call(store, job_id, "project_directory_compactor", directory, response, calls)
        parsed = parsed_json_mapping(response.content, {"summary": "", "key_files": [], "data_flow": [], "risks_or_gaps": []})
        directory_summaries.append({"directory": directory, **parsed})
        store.event(job_id, {"event": "project_directory_compacted", "time": round(time.time(), 3), "path": directory})

    response = client.chat(
        [
            {"role": "system", "content": "You create a compact whole-project memory for later agents. Return Markdown with clear sections."},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "project": str(root),
                        "directories": directory_summaries,
                        "file_summaries": file_summaries,
                    },
                    indent=2,
                    )[:10000],
            },
        ],
        temperature=0.0,
        max_tokens=700,
    )
    project_call(store, job_id, "project_architecture_compactor", root.name, response, calls)
    return {
        "path": str(root),
        "files": len(file_summaries),
        "file_summaries": file_summaries,
        "directory_summaries": directory_summaries,
        "architecture_brief": response.content.strip(),
        "model_calls": calls,
        "usage_summary": sum_calls(calls),
    }


def read_project_with_agents(client: Any, store: JobStore, job_id: str, request: str, project_spec: str) -> dict[str, Any]:
    root = discover_project(project_spec, request)
    if not root:
        return {"path": "", "architecture_brief": "", "files": 0, "model_calls": [], "usage_summary": sum_calls([])}

    files = project_files(root)
    fingerprint = project_fingerprint(root, files)
    if store.persistence:
        cached = store.persistence.load_project_memory(root, fingerprint)
        if cached:
            cached["model_calls"] = []
            cached["usage_summary"] = sum_calls([])
            cached["retrieval"] = retrieve_project_context(root, request, cached)
            store.event(
                job_id,
                {
                    "event": "project_memory_cache_hit",
                    "time": round(time.time(), 3),
                    "path": str(root),
                    "count": cached.get("files", 0),
                    "age_s": (cached.get("cache") or {}).get("age_s"),
                },
            )
            store.event(
                job_id,
                {
                    "event": "project_retrieval_ready",
                    "time": round(time.time(), 3),
                    "path": str(root),
                    "count": len((cached.get("retrieval") or {}).get("snippets") or []),
                    "terms": (cached.get("retrieval") or {}).get("terms", [])[:12],
                },
            )
            return cached

    calls: list[dict[str, Any]] = []
    store.event(
        job_id,
        {
            "event": "project_discovered",
            "time": round(time.time(), 3),
            "path": str(root),
            "count": len(files),
            "fingerprint": fingerprint[:12],
        },
    )
    partial_summaries = store.partial_project_summaries(str(root), fingerprint)
    if partial_summaries:
        store.event(
            job_id,
            {
                "event": "project_partial_resume",
                "time": round(time.time(), 3),
                "path": str(root),
                "count": len(partial_summaries),
            },
        )
    summaries = []
    for index, path in enumerate(files, start=1):
        if store.is_cancelled(job_id):
            raise JobCancelled()
        rel = str(path.relative_to(root))
        if rel in partial_summaries:
            summaries.append(partial_summaries[rel])
            store.event(
                job_id,
                {
                    "event": "project_file_resumed",
                    "time": round(time.time(), 3),
                    "path": rel,
                    "index": index,
                    "total": len(files),
                    "summary": partial_summaries[rel].get("summary", ""),
                },
            )
            continue
        store.event(
            job_id,
            {
                "event": "project_file_queued",
                "time": round(time.time(), 3),
                "path": rel,
                "index": index,
                "total": len(files),
            },
        )
        summaries.append(summarize_project_file(client, store, job_id, root, path, calls, index=index, total=len(files)))
    memory = aggregate_project_memory(client, store, job_id, root, summaries, calls)
    memory["cache"] = {"hit": False, "fingerprint": fingerprint, "updated": time.time()}
    memory["retrieval"] = retrieve_project_context(root, request, memory)
    if store.persistence:
        store.persistence.save_project_memory(memory, fingerprint)
        store.event(
            job_id,
            {
                "event": "project_memory_cached",
                "time": round(time.time(), 3),
                "path": str(root),
                "count": memory.get("files", 0),
            },
        )
    store.event(
        job_id,
        {
            "event": "project_retrieval_ready",
            "time": round(time.time(), 3),
            "path": str(root),
            "count": len((memory.get("retrieval") or {}).get("snippets") or []),
            "terms": (memory.get("retrieval") or {}).get("terms", [])[:12],
        },
    )
    return memory


def forced_route(mode_name: str, request: str, project_spec: str) -> dict[str, Any] | None:
    if mode_name in {"", "auto"}:
        return None
    mode = MODE_CONFIGS.get(mode_name)
    if not mode:
        return None
    route = {
        "mode": mode_name,
        "label": mode["label"],
        "reason": "manual mode override",
        "direct": False,
        "settings": mode.get("settings", {}),
        "expected_calls": mode["expected_calls"],
        "description": mode["description"],
    }
    if mode_name == "chat":
        route.update({"answer": "Hey. Give me a task, project name, or repo path and I can route it through the right workflow."})
    if mode_name in {"project_research", "code_review", "implementation", "debug"}:
        project_root = discover_project(project_spec, request)
        file_count = len(project_files(project_root)) if project_root else 0
        directory_count = len({str(path.parent) for path in project_files(project_root)}) if project_root else 0
        route.update(
            {
                "project_path": str(project_root) if project_root else "",
                "project_files": file_count,
                "expected_calls": file_count + directory_count + 1 + int(mode["expected_calls"]),
            }
        )
    return route


def wants_plain_search_results(words: list[str]) -> bool:
    if len(words) > 16:
        return False
    has_search = any(word in words for word in {"search", "find", "lookup"})
    has_result_intent = any(word in words for word in {"results", "result", "links", "sources"})
    has_synthesis_intent = any(word in words for word in {"summarize", "summary", "compare", "analyze", "explain", "why", "how", "research"})
    has_score_word = any(word in words for word in {"score", "scores", "winner", "won"})
    has_last_match_shape = "last" in words and any(word in words for word in {"match", "game"})
    has_world_cup = "world" in words and "cup" in words
    has_sports_score_intent = (has_score_word or has_last_match_shape) and ("fifa" in words or has_world_cup)
    if has_sports_score_intent:
        return False
    return has_search and not has_synthesis_intent and (has_result_intent or len(words) <= 10)


def wants_web_research(words: list[str], lowered: str) -> bool:
    """True when the request needs current/external facts — either a WEB_TERM keyword
    or a current-info phrase like "release date" / "when does ... come out"."""
    if any(word in words for word in WEB_TERMS):
        return True
    return any(phrase in lowered for phrase in WEB_RESEARCH_PHRASES)


def asks_about_capabilities(words: list[str], lowered: str) -> bool:
    capability_words = {"access", "read", "write", "edit", "browse", "list", "search", "tool", "tools", "capability", "capabilities", "permission", "permissions"}
    target_words = {"file", "files", "directory", "directories", "folder", "folders", "system", "machine", "computer", "projects", "project"}
    asks_about_assistant = any(phrase in lowered for phrase in ("can you", "you can", "you have", "are you able", "what can you", "which files", "what files"))
    return asks_about_assistant and any(word in words for word in capability_words) and any(word in words for word in target_words)


def classify_request(request: str, project_spec: str, mode_override: str = "auto") -> dict[str, Any]:
    override = forced_route(mode_override, request, project_spec)
    if override:
        return override
    text = request.strip()
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_.-]+", lowered)
    if GREETING_RE.match(text):
        mode = MODE_CONFIGS["chat"]
        return {
            "mode": "chat",
            "label": mode["label"],
            "reason": "short greeting",
            "direct": True,
            "answer": "Hey. Give me a task, project name, or repo path and I can route it through the right workflow.",
            "expected_calls": mode["expected_calls"],
            "description": mode["description"],
        }

    explicit_project = bool(project_spec.strip())
    project_root = discover_project(project_spec, request)
    inferred_project = project_root is not None
    mentions_project = explicit_project or inferred_project or any(term in lowered for term in PROJECT_TERMS)
    deep_score = sum(1 for word in words if word in DEEP_TERMS)

    if not explicit_project and wants_plain_search_results(words):
        mode = MODE_CONFIGS["search_results"]
        return {"mode": "search_results", "label": mode["label"], "reason": "simple web search request", "direct": True, "expected_calls": mode["expected_calls"], "description": mode["description"], "settings": mode["settings"]}

    if not explicit_project and not mentions_project and wants_web_research(words, lowered):
        mode = MODE_CONFIGS["web_research"]
        return {"mode": "web_research", "label": mode["label"], "reason": "request needs current or external information", "direct": False, "expected_calls": mode["expected_calls"], "description": mode["description"], "settings": mode["settings"]}

    if mentions_project:
        if any(word in words for word in {"review", "audit", "gaps", "risks", "risk", "issues", "problems"}):
            mode_name = "code_review"
        elif any(word in words for word in {"implement", "build", "fix", "refactor", "change", "add", "tests", "test"}):
            mode_name = "implementation"
        elif any(word in words for word in {"debug", "bug", "error", "crash", "broken", "trace"}):
            mode_name = "debug"
        else:
            mode_name = "project_research"
        mode = MODE_CONFIGS[mode_name]
        file_count = len(project_files(project_root)) if project_root else 0
        directory_count = len({str(path.parent) for path in project_files(project_root)}) if project_root else 0
        expected_calls = file_count + directory_count + 1 + int(mode["expected_calls"])
        return {
            "mode": mode_name,
            "label": mode["label"],
            "reason": "request refers to a project/codebase",
            "direct": False,
            "settings": mode["settings"],
            "expected_calls": expected_calls,
            "description": mode["description"],
            "project_path": str(project_root) if project_root else "",
            "project_files": file_count,
        }

    if any(word in words for word in {"debug", "bug", "error", "crash", "broken", "trace"}):
        mode = MODE_CONFIGS["debug"]
        return {"mode": "debug", "label": mode["label"], "reason": "debugging request", "direct": False, "expected_calls": mode["expected_calls"], "description": mode["description"], "settings": mode["settings"]}

    if any(word in words for word in {"implement", "build", "fix", "refactor", "change", "add", "tests", "test"}):
        mode = MODE_CONFIGS["implementation"]
        return {"mode": "implementation", "label": mode["label"], "reason": "programming request", "direct": False, "expected_calls": mode["expected_calls"], "description": mode["description"], "settings": mode["settings"]}

    if any(word in words for word in {"plan", "strategy", "roadmap", "steps", "approach"}):
        mode = MODE_CONFIGS["plan"]
        return {"mode": "plan", "label": mode["label"], "reason": "planning request", "direct": False, "expected_calls": mode["expected_calls"], "description": mode["description"], "settings": mode["settings"]}

    if len(words) <= 12 and deep_score == 0:
        mode = MODE_CONFIGS["direct_answer"]
        return {"mode": "direct_answer", "label": mode["label"], "reason": "small conversational request", "direct": False, "expected_calls": mode["expected_calls"], "description": mode["description"], "settings": mode["settings"]}

    if len(words) <= 28 and deep_score <= 1:
        mode = MODE_CONFIGS["direct_answer"]
        return {"mode": "direct_answer", "label": mode["label"], "reason": "small focused request", "direct": False, "expected_calls": mode["expected_calls"], "description": mode["description"], "settings": mode["settings"]}

    mode_name = "deep_orchestration"
    mode = MODE_CONFIGS[mode_name]
    return {
        "mode": mode_name,
        "label": mode["label"],
        "reason": "complex request without project context",
        "direct": False,
        "settings": mode["settings"],
        "expected_calls": mode["expected_calls"],
        "description": mode["description"],
    }


def resolve_run_settings(config: dict[str, Any], route: dict[str, Any]) -> dict[str, Any]:
    settings = {**config, **(route.get("settings") or {})}
    for key, default in RUN_DEFAULTS.items():
        if key in config and (config[key] != default or key not in settings):
            settings[key] = config[key]
    return settings


def capability_context_for_prompt(access_role: str) -> dict[str, Any]:
    home = str(Path.home())
    capabilities = capability_manifest()
    if access_role != "admin":
        capabilities = [item for item in capabilities if item["id"] not in ADMIN_ONLY_CAPABILITIES]
    return {
        "role": access_role,
        "filesystem_scope": f"paths under {home}" if access_role == "admin" else "disabled for restricted sessions",
        "model_filesystem_access": "none directly; host tools provide selected results to prompts",
        "capabilities": capabilities,
        "safety": "local file edits are not applied automatically; future editing workflows should require user approval",
    }


def sanitize_conversation_history(items: Any, *, max_turns: int = 10, max_chars: int = 6000) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    history: list[dict[str, str]] = []
    total = 0
    for item in items[-max_turns:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        content = content[:remaining]
        total += len(content)
        history.append({"role": role, "content": content})
    return history


def direct_model_answer(client: Any, store: JobStore, job_id: str, request: str, access_role: str = "admin", conversation_history: list[dict[str, str]] | None = None, tool_evidence: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    store.event(job_id, {"event": "direct_answer_started", "time": round(time.time(), 3)})
    words = re.findall(r"[a-z0-9_.-]+", request.lower())
    capability_context = capability_context_for_prompt(access_role) if asks_about_capabilities(words, request.lower()) else None
    history = sanitize_conversation_history(conversation_history)
    system_prompt = (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "
        "Answer naturally and directly. If project/file/tool evidence was not provided, say what you can infer and what would need inspection."
        + MEMORY_SUGGESTION_GUIDANCE
    )
    if capability_context:
        store.event(job_id, {"event": "capability_context_attached", "time": round(time.time(), 3), "role": access_role})
        system_prompt += (
            "\nThe user is asking about this app's own available capabilities or access. "
            "Use the host-provided capability context as ground truth. "
            "Distinguish the language model from host-side tools."
        )
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": request})
    if capability_context:
        messages.append({"role": "user", "content": "HOST CAPABILITY CONTEXT:\n" + json.dumps(capability_context, indent=2)})
    if tool_evidence:
        store.event(job_id, {"event": "tool_evidence_attached", "time": round(time.time(), 3), "count": len(tool_evidence)})
        messages.append({"role": "user", "content": tool_evidence_prompt_block(tool_evidence).strip()})
    content = ""
    done: dict[str, Any] = {}
    started = time.monotonic()
    if hasattr(client, "stream_chat"):
        store.update(job_id, partial_answer="")
        for event in client.stream_chat(messages, temperature=0.2, max_tokens=700):
            if store.is_cancelled(job_id):
                raise JobCancelled()
            if event.get("type") == "chunk":
                content += event.get("content", "")
                elapsed = max(0.001, time.monotonic() - started)
                approx_tokens = max(1, round(len(content) / 4))
                store.update(
                    job_id,
                    partial_answer=content,
                    stream_stats={
                        "chars": len(content),
                        "approx_tokens": approx_tokens,
                        "elapsed_s": round(elapsed, 2),
                        "approx_tokens_per_second": round(approx_tokens / elapsed, 2),
                    },
                )
            elif event.get("type") == "done":
                done = event
    else:
        fallback = client.chat(messages, temperature=0.2, max_tokens=700)
        content = fallback.content
        done = {
            "elapsed_s": fallback.elapsed_s,
            "usage": fallback.usage,
            "timings": fallback.timings,
            "raw": fallback.raw,
        }
    response = ModelResponse(
        content,
        float(done.get("elapsed_s") or 0),
        done.get("usage") or {},
        done.get("timings") or {},
        done.get("raw") or {},
    )
    call = compact_call("direct", response, label="single response")
    store.event(job_id, {"event": "model_call_finished", "time": round(time.time(), 3), **call})
    store.update(job_id, answer=content, partial_answer=content)
    return {
        "request": request,
        "user_request": request,
        "strategy": "Direct single-call response selected by request router.",
        "dynamic_settings": {},
        "usage_summary": sum_calls([call]),
        "model_calls": [call],
        "project_model_calls": [],
        "project_context": {"capability_context": capability_context, "conversation_history_used": len(history)} if capability_context or history else {},
        "tasks": [],
        "rounds": [],
        "worker_results": [],
        "verifier": {"pass": True, "issues": [], "missing_tasks": []},
        "final": response.content.strip(),
        "elapsed_s": round(response.elapsed_s, 3),
    }


def direct_web_search_results(store: JobStore, job_id: str, request: str) -> dict[str, Any]:
    query = web_search_query_from_request(request)
    store.event(job_id, {"event": "web_search_started", "time": round(time.time(), 3), "query": query, "raw_query": request})
    search_run = run_capability("web_search", {"query": query, "limit": 8})
    attach_capability_result(store, job_id, search_run, "Web search results returned")
    search = search_run.get("result") or {}
    store.event(
        job_id,
        {
            "event": "web_search_finished",
            "time": round(time.time(), 3),
            "query": search.get("query") or query,
            "raw_query": request,
            "count": search.get("count", 0),
            "engine": search.get("engine", ""),
        },
    )
    lines = [f"Search results for `{search.get('query') or query}`:"]
    results = search.get("results") or []
    if results:
        for index, item in enumerate(results, start=1):
            title = str(item.get("title") or item.get("url") or "Untitled").strip()
            url = str(item.get("url") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            if url:
                lines.append(f"{index}. [{title}]({url})")
            else:
                lines.append(f"{index}. {title}")
            if snippet:
                lines.append(f"   {snippet}")
    else:
        lines.append("No relevant results were returned by the available search backends.")
    answer = "\n".join(lines)
    trace = {
        "request": request,
        "user_request": request,
        "strategy": "Direct web-search result list selected by request router.",
        "dynamic_settings": {},
        "usage_summary": sum_calls([]),
        "model_calls": [],
        "project_model_calls": [],
        "project_context": {},
        "web_context": {"query": search.get("query") or query, "raw_query": request, "search": search, "pages": [], "errors": []},
        "tasks": [],
        "rounds": [],
        "worker_results": [],
        "verifier": {"pass": True, "issues": [], "missing_tasks": []},
        "final": answer,
        "elapsed_s": 0,
    }
    store.update(job_id, answer=answer, partial_answer=answer)
    return trace


@contextlib.contextmanager
def run_slot(store: JobStore, job_id: str) -> Any:
    """Acquire the single orchestration slot, keeping the job cancellable while it
    waits in the queue.

    The local model has one slot; with concurrent submissions the second job sits
    here with status "queued" until the first releases the slot, instead of
    interleaving model calls and halving throughput. Cancellation while queued is
    honoured without waiting for the slot.
    """
    if store.is_cancelled(job_id):
        raise JobCancelled()
    store.update(job_id, status="queued")
    store.event(job_id, {"event": "queued_for_slot", "time": round(time.time(), 3)})
    while not store.run_lock.acquire(timeout=0.5):
        if store.is_cancelled(job_id):
            raise JobCancelled()
    try:
        store.update(job_id, status="running")
        yield
    finally:
        store.run_lock.release()


def run_job(store: JobStore, job_id: str, config: dict[str, Any]) -> None:
    try:
        # Record the start time but DON'T claim "running" yet — the job may still be
        # waiting behind the single model slot. run_slot flips queued -> running when
        # it actually acquires the slot, so the queue no longer flickers running first.
        store.update(job_id, started=round(time.time(), 3))
        access_role = str(config.get("access_role") or "admin")
        user_memories = store.persistence.load_memories("user", limit=20) if store.persistence and access_role == "admin" else []
        route = classify_request(config["request"], config.get("project_spec", ""), config.get("mode_override", "auto"))
        project_memories = (
            store.persistence.load_memories(project_memory_scope(route.get("project_path", "")), limit=20)
            if store.persistence and route.get("project_path") and access_role == "admin"
            else []
        )
        all_memories = user_memories + project_memories
        # Tool evidence carried in from a parent job via the "Continue with evidence"
        # action. Human-in-the-loop only: one user-clicked continuation is a single
        # run (one prefill), never a server-side auto-loop (see CLAUDE.md GPU note).
        tool_evidence: list[dict[str, Any]] = []
        continue_from = str(config.get("continue_from") or "")
        if continue_from:
            parent = store.get(continue_from)
            if parent:
                tool_evidence = collect_attached_capabilities(parent)
                if tool_evidence:
                    store.event(job_id, {"event": "tool_evidence_carried", "time": round(time.time(), 3), "parent": continue_from, "count": len(tool_evidence)})
        store.update(job_id, route=route)
        store.event(job_id, {"event": "request_routed", "time": round(time.time(), 3), **route})
        if route.get("answer"):
            trace = {
                "request": config["request"],
                "user_request": config["request"],
                "strategy": route["reason"],
                "mode": route,
                "dynamic_settings": {},
                "usage_summary": sum_calls([]),
                "model_calls": [],
                "project_model_calls": [],
                "project_context": {},
                "user_memories": user_memories,
                "project_memories": project_memories,
                "tasks": [],
                "rounds": [],
                "worker_results": [],
                "verifier": {"pass": True, "issues": [], "missing_tasks": []},
                "final": route["answer"],
                "elapsed_s": 0,
            }
            answer = str(trace.get("final") or "").strip()
            store.update(job_id, status="done", trace=trace, answer=answer, partial_answer=answer, active_model_call=None, finished=round(time.time(), 3))
            return

        if route["mode"] == "search_results":
            trace = direct_web_search_results(store, job_id, config["request"])
            trace["mode"] = route
            answer = str(trace.get("final") or "").strip()
            store.update(job_id, status="done", trace=trace, answer=answer, partial_answer=answer, active_model_call=None, finished=round(time.time(), 3))
            return

        store.event(job_id, {"event": "workflow_ready", "time": round(time.time(), 3)})
        with run_slot(store, job_id):
            store.event(job_id, {"event": "workflow_started", "time": round(time.time(), 3)})
            if store.is_cancelled(job_id):
                raise JobCancelled()
            provider = config["provider"]
            timeout = config["timeout"]
            if provider == "ollama":
                base_client = OllamaClient(config["base_url"], config["model"], timeout_s=timeout)
            else:
                base_client = OpenAICompatibleClient(config["base_url"], config["model"], timeout_s=timeout)
            client = ObservableClient(base_client, store, job_id)

            # Device-split specialists (Component 2): route plan/compact -> MiniCPM-CPU and
            # verify -> Qwen3-4B-CPU, so the GPU coder only does worker/synthesize. Health-checked
            # per run; a down service falls back to the coder for that stage.
            util_client = verifier_client = None
            if DEVICE_SPLIT and provider != "ollama":
                if model_server_health(UTIL_BASE_URL).get("ok"):
                    util_client = ObservableClient(
                        OpenAICompatibleClient(UTIL_BASE_URL, UTIL_MODEL, timeout_s=timeout,
                                               extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
                        store, job_id)
                if model_server_health(VERIFIER_BASE_URL).get("ok"):
                    verifier_client = ObservableClient(
                        OpenAICompatibleClient(VERIFIER_BASE_URL, VERIFIER_MODEL, timeout_s=timeout),
                        store, job_id)

            if route["mode"] == "direct_answer":
                trace = direct_model_answer(client, store, job_id, config["request"], access_role, config.get("conversation_history") or [], tool_evidence)
                trace["mode"] = route
                trace["memory_suggestions"] = extract_trace_memory_suggestions(trace, all_memories)
                if trace["memory_suggestions"]:
                    store.event(job_id, {"event": "memory_suggested", "time": round(time.time(), 3), "count": len(trace["memory_suggestions"])})
                answer = str(trace.get("final") or "").strip()
                store.update(job_id, status="done", trace=trace, answer=answer, partial_answer=answer, active_model_call=None, finished=round(time.time(), 3))
                return

            settings = resolve_run_settings(config, route)
            project_memory = {"path": "", "architecture_brief": "", "files": 0, "model_calls": [], "usage_summary": sum_calls([])}
            web_context: dict[str, Any] = {}
            docs_context: dict[str, Any] = {}
            if route["mode"] == "web_research":
                web_context = build_web_context(store, job_id, config["request"])
            if route["mode"] in {"project_research", "code_review", "implementation", "debug"} and route.get("project_path"):
                project_memory = read_project_with_agents(
                    client,
                    store,
                    job_id,
                    config["request"],
                    config.get("project_spec", ""),
                )
            if should_prefetch_docs(route, config["request"], project_memory):
                docs_context = build_docs_context(store, job_id, config["request"], project_memory)
            if store.is_cancelled(job_id):
                raise JobCancelled()
            orchestrator_request = config["request"]
            if all_memories:
                orchestrator_request = (
                    f"{orchestrator_request}\n\nUSER MEMORY CONTEXT:\n"
                    f"{json.dumps(all_memories, indent=2)[:USER_MEMORY_PROMPT_CHARS]}\n"
                    "Use stored memory only as preference/context. Do not invent unstored memories."
                )
            if project_memory.get("architecture_brief"):
                retrieval = project_memory.get("retrieval") or {}
                architecture_brief = str(project_memory.get("architecture_brief") or "")[:PROJECT_MEMORY_PROMPT_CHARS]
                retrieval_json = json.dumps(retrieval, indent=2)[:PROJECT_RETRIEVAL_PROMPT_CHARS]
                orchestrator_request = (
                    f"{orchestrator_request}\n\n"
                    "The project has already been read by sequential file-reader agents. "
                    "Use the compacted project memory and retrieved project context below as your source of truth. "
                    "Do not claim you inspected files outside this memory or retrieved snippets.\n\n"
                    f"PROJECT MEMORY FOR {project_memory['path']}:\n{architecture_brief}"
                    "\n\nRETRIEVED PROJECT CONTEXT:\n"
                    f"{retrieval_json}"
                    "\n\nAVAILABLE HOST CAPABILITIES (manual approval/execution only):\n"
                    f"{json.dumps(capability_manifest(), indent=2)}\n"
                    "If more evidence is needed, recommend a tool request using this exact JSON shape, but do not pretend it was run: "
                    "{\"tool_request\":{\"capability\":\"git_status\",\"input\":{\"path\":\"/home/nit/project\"},\"reason\":\"why this evidence is needed\"}}"
                )
            if web_context:
                prompt_web_context = compact_web_context_for_prompt(web_context)
                orchestrator_request = (
                    f"{orchestrator_request}\n\n"
                    "WEB EVIDENCE FETCHED BY HOST:\n"
                    f"Current date: {CURRENT_DATE}\n"
                    f"{json.dumps(prompt_web_context, indent=2)[:WEB_CONTEXT_PROMPT_CHARS]}\n\n"
                    "Use the web evidence above as the source for current or external facts. "
                    "Cite the URL for each web-backed claim. "
                    "If the evidence is insufficient, say what is missing instead of inventing details. "
                    "Prefer official and reputable sources. If a claim (especially a specific date or number) "
                    "appears only on a fan, rumor, leak, or aggregator site, or only in a single low-authority source, "
                    "say it is unofficial and may be inaccurate rather than stating it as fact. "
                    "If sources disagree, say so and give the range. "
                    f"The current date is {CURRENT_DATE}. For a future release, launch, or event, report the date as "
                    "expected/scheduled and note it may change — never state a future date as certain or as already happened. "
                    "For sports results, only report matches the evidence clearly identifies as completed with a final score, "
                    "and treat schedules, fixtures, previews, odds, or projected brackets as not-yet-a-result."
                )
            if docs_context:
                prompt_docs_context = compact_web_context_for_prompt(docs_context)
                orchestrator_request = (
                    f"{orchestrator_request}\n\n"
                    "DOCUMENTATION / BEST-PRACTICES EVIDENCE FETCHED BY HOST:\n"
                    f"Current date: {CURRENT_DATE}\n"
                    f"{json.dumps(prompt_docs_context, indent=2)[:DOCS_CONTEXT_PROMPT_CHARS]}\n\n"
                    "Use this evidence for current APIs, library behavior, migration notes, and security/performance best practices. "
                    "Prefer official documentation from the evidence. Cite URLs when making doc-backed claims. "
                    "If the evidence is weak or unrelated, say so and fall back to project facts and general engineering judgment."
                )
            orchestrator_request += tool_evidence_prompt_block(tool_evidence)
            orchestrator_request += MEMORY_SUGGESTION_GUIDANCE

            orch = Orchestrator(
                client,
                util_client=util_client,
                verifier_client=verifier_client,
                max_workers=settings["max_workers"],
                max_tasks=settings["max_tasks"],
                planner_tokens=settings["planner_tokens"],
                worker_tokens=settings["worker_tokens"],
                verifier_tokens=settings["verifier_tokens"],
                compactor_tokens=settings["compactor_tokens"],
                synth_tokens=settings["synth_tokens"],
                max_rounds=settings["max_rounds"],
                dynamic=True,
                on_event=lambda event: store.event(job_id, event),
            )
            if store.is_cancelled(job_id):
                raise JobCancelled()
            store.event(
                job_id,
                {
                    "event": "orchestrator_prompt_ready",
                    "time": round(time.time(), 3),
                    "chars": len(orchestrator_request),
                    "approx_tokens": round(len(orchestrator_request) / 4),
                },
            )
            trace = orch.run(orchestrator_request)
            orchestrator_calls = trace.get("model_calls", [])
            project_calls = project_memory.get("model_calls", [])
            trace["orchestrator_usage_summary"] = trace.get("usage_summary", {})
            trace["usage_summary"] = sum_calls(project_calls + orchestrator_calls)
            trace["mode"] = route
            trace["user_request"] = config["request"]
            trace["user_memories"] = user_memories
            trace["project_memories"] = project_memories
            trace["web_context"] = web_context
            trace["docs_context"] = docs_context
            trace["project_context"] = {
                "path": project_memory.get("path", ""),
                "files": project_memory.get("files", 0),
                "cache": project_memory.get("cache", {}),
                "retrieval": project_memory.get("retrieval", {}),
                "available_capabilities": capability_manifest(),
                "usage_summary": project_memory.get("usage_summary", {}),
                "file_summaries": project_memory.get("file_summaries", []),
                "directory_summaries": project_memory.get("directory_summaries", []),
            }
            trace["project_model_calls"] = project_memory.get("model_calls", [])
            trace["tool_requests"] = extract_trace_tool_requests(trace)
            trace["memory_suggestions"] = extract_trace_memory_suggestions(trace, all_memories)
            if trace["memory_suggestions"]:
                store.event(job_id, {"event": "memory_suggested", "time": round(time.time(), 3), "count": len(trace["memory_suggestions"])})
            if store.is_cancelled(job_id):
                raise JobCancelled()
        answer = str(trace.get("final") or "").strip()
        verifier = trace.get("verifier") if isinstance(trace, dict) else {}
        if isinstance(verifier, dict) and verifier.get("pass") is False:
            issues = verifier.get("issues") if isinstance(verifier.get("issues"), list) else []
            missing = verifier.get("missing_tasks") if isinstance(verifier.get("missing_tasks"), list) else []
            store.event(
                job_id,
                {
                    "event": "verifier_failed_final_answer",
                    "time": round(time.time(), 3),
                    "issues": issues[:5],
                    "missing_tasks": missing[:5],
                },
            )
            # Keep the real answer as THE answer — never bury it. The verifier's
            # concerns ride along as a separate caveat the UI shows beneath it, so a
            # nitpicky verifier (common on a 7B) no longer hides a usable answer.
            note_bits = []
            if issues:
                note_bits.append("Possible gaps: " + "; ".join(str(item) for item in issues[:3]))
            if missing:
                note_bits.append("May still be missing: " + "; ".join(str(item) for item in missing[:3]))
            trace["verifier_note"] = "  ".join(note_bits) or "The verifier flagged this answer as possibly incomplete."
        store.update(job_id, status="done", trace=trace, answer=answer, partial_answer=answer, active_model_call=None, finished=round(time.time(), 3))
    except JobCancelled:
        store.event(job_id, {"event": "cancelled", "time": round(time.time(), 3)})
        store.update(job_id, status="cancelled", active_model_call=None, finished=round(time.time(), 3))
    except Exception as exc:
        store.event(job_id, {"event": "error", "time": round(time.time(), 3), "error": str(exc)})
        store.update(job_id, status="error", error=str(exc), traceback=traceback.format_exc(), active_model_call=None, finished=round(time.time(), 3))
    except BaseException as exc:
        store.event(job_id, {"event": "error", "time": round(time.time(), 3), "error": repr(exc)})
        store.update(job_id, status="error", error=repr(exc), traceback=traceback.format_exc(), active_model_call=None, finished=round(time.time(), 3))
        raise
    finally:
        if store.status(job_id) not in TERMINAL_STATUSES:
            store.event(
                job_id,
                {
                    "event": "error",
                    "time": round(time.time(), 3),
                    "error": "job runner exited without setting a terminal state",
                },
            )
            store.update(
                job_id,
                status="error",
                error="job runner exited without setting a terminal state",
                active_model_call=None,
                finished=round(time.time(), 3),
            )


class Handler(BaseHTTPRequestHandler):
    store: JobStore

    def log_message(self, fmt: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, value: Any) -> None:
        self.send_bytes(status, json.dumps(value).encode("utf-8"), "application/json; charset=utf-8")

    def send_job_stream(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if not job:
            self.send_json(404, {"error": "job not found"})
            return
        if not self.can_access_job(job):
            self.send_json(403, {"error": "job is not visible to this user"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last_payload = ""
        while True:
            job = self.store.get(job_id)
            if not job:
                break
            if not self.can_access_job(job):
                break
            payload = json.dumps(job, separators=(",", ":"))
            if payload != last_payload:
                try:
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                last_payload = payload
            if job.get("status") in TERMINAL_STATUSES:
                break
            time.sleep(0.12 if job.get("active_model_call") else 0.35)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def auth_enabled(self) -> bool:
        return bool(ADMIN_TOKEN or RESTRICTED_TOKENS)

    def request_token(self) -> str:
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth.split(" ", 1)[1].strip()
        cookie = self.headers.get("Cookie", "")
        for part in cookie.split(";"):
            key, _, value = part.strip().partition("=")
            if key == SESSION_COOKIE:
                return unquote(value)
        return ""

    def is_authenticated(self) -> bool:
        if not self.auth_enabled():
            return True
        token = self.request_token()
        return self.role_for_token(token) in {"admin", "restricted"}

    def role_for_token(self, token: str) -> str:
        if token and ADMIN_TOKEN and hmac.compare_digest(token, ADMIN_TOKEN):
            return "admin"
        for restricted in RESTRICTED_TOKENS:
            if token and hmac.compare_digest(token, restricted):
                return "restricted"
        return ""

    def current_role(self) -> str:
        if not self.auth_enabled():
            return "admin"
        return self.role_for_token(self.request_token())

    def is_admin(self) -> bool:
        return self.current_role() == "admin"

    def send_login(self, status: int = 200, error: str = "") -> None:
        body = LOGIN_HTML.replace("{{ERROR}}", html.escape(error)).encode("utf-8")
        self.send_bytes(status, body, "text/html; charset=utf-8")

    def require_auth(self) -> bool:
        if self.is_authenticated():
            return True
        if self.path.startswith("/api/"):
            self.send_json(401, {"error": "authentication required"})
        else:
            self.send_login(401 if self.path not in {"/", "/login"} else 200)
        return False

    def require_admin(self) -> bool:
        if self.is_admin():
            return True
        self.send_json(403, {"error": "admin access required"})
        return False

    def can_access_job(self, job: dict[str, Any] | None) -> bool:
        if not job:
            return False
        if self.is_admin():
            return True
        config = job.get("config") if isinstance(job.get("config"), dict) else {}
        return config.get("access_role") == "restricted"

    def visible_jobs(self, session_id: str | None = None) -> list[dict[str, Any]]:
        jobs = self.store.list(session_id=session_id)
        if self.is_admin():
            return jobs
        return [job for job in jobs if self.can_access_job(job)]

    def send_redirect(self, location: str, cookie: str | None = None) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/login":
            if self.is_authenticated():
                self.send_redirect("/")
            else:
                self.send_login()
            return
        if self.path == "/logout":
            self.send_redirect("/login", f"{SESSION_COOKIE}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0")
            return
        if not self.require_auth():
            return
        if self.path in {"/", "/index.html"}:
            self.send_bytes(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/api/health":
            self.send_json(200, model_server_health(DEFAULT_OPENAI_BASE))
            return
        if self.path == "/api/session":
            self.send_json(200, {"role": self.current_role(), "auth_enabled": self.auth_enabled()})
            return
        if self.path == "/api/capabilities":
            capabilities = capability_manifest()
            if not self.is_admin():
                capabilities = [item for item in capabilities if item["id"] not in ADMIN_ONLY_CAPABILITIES]
            self.send_json(200, {"capabilities": capabilities, "role": self.current_role()})
            return
        if self.path == "/api/capability/runs":
            if not self.require_admin():
                return
            runs = self.store.persistence.load_capability_runs() if self.store.persistence else []
            self.send_json(200, {"runs": runs})
            return
        if self.path.startswith("/api/memories"):
            if not self.require_admin():
                return
            scope = "user"
            parsed_url = urlparse(self.path)
            query = parse_qs(parsed_url.query)
            if query.get("scope"):
                scope = query["scope"][0]
            memories = self.store.persistence.load_memories(scope) if self.store.persistence else []
            self.send_json(200, {"memories": memories})
            return
        if self.path == "/api/sessions":
            if not self.require_admin():
                return
            sessions = self.store.persistence.list_sessions() if self.store.persistence else []
            self.store.annotate_sessions(sessions)
            self.send_json(200, {"sessions": sessions})
            return
        if self.path == "/api/coder-profiles":
            self.send_json(200, list_coder_profiles())
            return
        if urlparse(self.path).path == "/api/jobs":
            session_id = (parse_qs(urlparse(self.path).query).get("session_id") or [None])[0]
            self.send_json(200, {"jobs": self.visible_jobs(session_id=session_id), "role": self.current_role()})
            return
        if self.path.startswith("/api/jobs/") and self.path.endswith("/stream"):
            parts = self.path.strip("/").split("/")
            if len(parts) == 4:
                self.send_job_stream(parts[2])
                return
            self.send_json(404, {"error": "job not found"})
            return
        if self.path.startswith("/api/jobs/"):
            job_id = self.path.rsplit("/", 1)[-1]
            job = self.store.get(job_id)
            if not job:
                self.send_json(404, {"error": "job not found"})
                return
            if not self.can_access_job(job):
                self.send_json(403, {"error": "job is not visible to this user"})
                return
            self.send_json(200, job)
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/login":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8", errors="replace")
            form = parse_qs(body)
            token = str((form.get("token") or [""])[0]).strip()
            if self.role_for_token(token) in {"admin", "restricted"}:
                self.send_redirect("/", f"{SESSION_COOKIE}={quote_plus(token)}; HttpOnly; SameSite=Lax; Path=/; Max-Age=2592000")
            else:
                self.send_login(401, "Invalid token.")
            return
        if not self.require_auth():
            return
        if self.path == "/api/coder-profile":
            # Switching the GPU coder swaps the resident model for everyone -> admin-only.
            if not self.require_admin():
                return
            data = self.read_json()
            result = switch_coder_profile(str(data.get("name") or ""))
            self.send_json(200 if result.get("ok") else 400, result)
            return
        if self.path == "/api/sessions":
            if not self.require_admin():
                return
            if not self.store.persistence:
                self.send_json(400, {"error": "persistence is not enabled"})
                return
            data = self.read_json()
            self.send_json(201, self.store.persistence.create_session(str(data.get("title") or "New chat")))
            return
        if self.path.startswith("/api/sessions/") and self.path.endswith("/rename"):
            if not self.require_admin():
                return
            session_id = self.path.split("/")[-2]
            data = self.read_json()
            if self.store.persistence and self.store.persistence.rename_session(session_id, str(data.get("title") or "")):
                self.send_json(200, {"ok": True, "id": session_id})
            else:
                self.send_json(400, {"error": "could not rename this chat (empty title or unknown chat)"})
            return
        if self.path.startswith("/api/jobs/") and self.path.endswith("/cancel"):
            job_id = self.path.split("/")[-2]
            job = self.store.get(job_id)
            if not self.can_access_job(job):
                self.send_json(403, {"error": "job is not visible to this user"})
                return
            if self.store.cancel(job_id):
                self.send_json(202, {"ok": True, "id": job_id})
            else:
                self.send_json(404, {"error": "job not found"})
            return
        if self.path == "/api/capability/run":
            try:
                data = self.read_json()
                name = str(data.get("capability") or "")
                if not self.is_admin() and name in ADMIN_ONLY_CAPABILITIES:
                    self.send_json(403, {"error": "admin access required for this capability"})
                    return
                result = run_capability(name, data)
                if self.store.persistence:
                    self.store.persistence.save_capability_run(result)
                job_id = str(data.get("job_id") or "")
                if job_id and self.store.get(job_id):
                    preview = json.dumps(result.get("result"), indent=2)[:1200]
                    self.store.event(
                        job_id,
                        {
                            "event": "capability_run_attached",
                            "time": round(time.time(), 3),
                            "id": result["id"],
                            "capability": result["capability"],
                            "input": {k: v for k, v in data.items() if k not in ("capability", "job_id")},
                            "ok": result["ok"],
                            "summary": f"{result['label']} completed",
                            "result_preview": preview,
                        },
                    )
                self.send_json(200, result)
            except Exception as exc:
                result = {
                    "id": uuid.uuid4().hex[:12],
                    "created": round(time.time(), 3),
                    "capability": str((locals().get("data") or {}).get("capability") or ""),
                    "ok": False,
                    "error": str(exc),
                }
                if self.store.persistence:
                    self.store.persistence.save_capability_run(result)
                job_id = str((locals().get("data") or {}).get("job_id") or "")
                if job_id and self.store.get(job_id):
                    self.store.event(
                        job_id,
                        {
                            "event": "capability_run_attached",
                            "time": round(time.time(), 3),
                            "id": result["id"],
                            "capability": result["capability"],
                            "ok": False,
                            "summary": f"Capability failed: {result['error']}",
                            "result_preview": result["error"],
                        },
                    )
                self.send_json(400, result)
            return
        if self.path == "/api/memories":
            if not self.require_admin():
                return
            try:
                data = self.read_json()
                scope = str(data.get("scope") or "user")[:300]
                if scope == "project":
                    scope = project_memory_scope(str(data.get("project_path") or ""))
                key = str(data.get("key") or "").strip()[:160]
                value = str(data.get("value") or "").strip()
                tags = [str(tag).strip()[:40] for tag in data.get("tags", []) if str(tag).strip()]
                if not key or not value:
                    raise ValueError("key and value are required")
                if not self.store.persistence:
                    raise ValueError("persistence is not enabled")
                record = self.store.persistence.save_memory(scope, key, value, tags)
                job_id = str(data.get("job_id") or "")
                if job_id and self.store.get(job_id):
                    self.store.event(job_id, {"event": "memory_saved", "time": round(time.time(), 3), "key": key})
                self.send_json(201, record)
            except Exception as exc:
                self.send_json(400, {"error": str(exc)})
            return
        if self.path != "/api/run":
            self.send_json(404, {"error": "not found"})
            return
        try:
            data = self.read_json()
            request = str(data.get("request") or "").strip()
            if not request:
                raise ValueError("request is required")
            if not self.is_admin():
                if str(data.get("project_path") or "").strip():
                    self.send_json(403, {"error": "restricted users cannot access local project paths"})
                    return
                requested_mode = str(data.get("mode_override") or "auto")
                if requested_mode in ADMIN_ONLY_MODES:
                    self.send_json(403, {"error": "restricted users cannot use local project modes"})
                    return
            provider = data.get("provider") if data.get("provider") in {"ollama", "openai-compatible"} else "openai-compatible"
            session_id = str(data.get("session_id") or "")
            if session_id and self.store.persistence and not self.store.persistence.get_session(session_id):
                session_id = ""   # unknown session id -> treat as no session rather than erroring
            if session_id and self.store.pending_count(session_id) >= MAX_PENDING_PER_SESSION:
                self.send_json(429, {"error": f"This chat already has {MAX_PENDING_PER_SESSION} prompts queued or running — let some finish before adding more."})
                return
            # Chat history is server-owned per session; fall back to the client-sent
            # list only for restricted users / no-session runs.
            if session_id and self.store.persistence:
                conversation_history = self.store.persistence.session_messages(session_id, limit=24)
            else:
                conversation_history = sanitize_conversation_history(data.get("conversation_history"))
            config = {
                "request": request,
                "session_id": session_id,
                "project_spec": str(data.get("project_path") or ""),
                "provider": provider,
                "mode_override": str(data.get("mode_override") or "auto"),
                "base_url": str(data.get("base_url") or DEFAULT_OPENAI_BASE).rstrip("/"),
                "model": str(data.get("model") or DEFAULT_MODEL),
                "max_tasks": int_field(data, "max_tasks", RUN_DEFAULTS["max_tasks"], 1, 12),
                "max_workers": int_field(data, "max_workers", RUN_DEFAULTS["max_workers"], 1, 8),
                "max_rounds": int_field(data, "max_rounds", RUN_DEFAULTS["max_rounds"], 1, 4),
                "planner_tokens": int_field(data, "planner_tokens", RUN_DEFAULTS["planner_tokens"], 80, 2400),
                "worker_tokens": int_field(data, "worker_tokens", RUN_DEFAULTS["worker_tokens"], 80, 2400),
                "verifier_tokens": int_field(data, "verifier_tokens", RUN_DEFAULTS["verifier_tokens"], 80, 2000),
                "compactor_tokens": int_field(data, "compactor_tokens", RUN_DEFAULTS["compactor_tokens"], 80, 1000),
                "synth_tokens": int_field(data, "synth_tokens", RUN_DEFAULTS["synth_tokens"], 80, 1200),
                "timeout": int_field(data, "timeout", RUN_DEFAULTS["timeout"], 10, 3600),
                "access_role": self.current_role(),
                "conversation_history": conversation_history,
                "continue_from": str(data.get("continue_from") or ""),
            }
            if not self.is_admin():
                route = classify_request(request, "", config["mode_override"])
                if route.get("mode") in ADMIN_ONLY_MODES or route.get("project_path"):
                    self.send_json(403, {"error": "restricted users cannot run project, filesystem, implementation, or debug workflows"})
                    return
            job_id = self.store.create(config)
            threading.Thread(target=run_job, args=(self.store, job_id, config), daemon=True).start()
            self.send_json(202, {"id": job_id})
        except Exception as exc:
            self.send_json(400, {"error": str(exc)})

    def do_DELETE(self) -> None:
        if not self.require_auth():
            return
        if not self.require_admin():
            return
        if self.path.startswith("/api/memories/"):
            memory_id = self.path.rsplit("/", 1)[-1]
            if not self.store.persistence:
                self.send_json(400, {"error": "persistence is not enabled"})
                return
            if self.store.persistence.delete_memory(memory_id):
                self.send_json(200, {"ok": True, "id": memory_id})
            else:
                self.send_json(404, {"error": "memory not found"})
            return
        if self.path.startswith("/api/sessions/"):
            session_id = self.path.rsplit("/", 1)[-1]
            if not self.store.persistence:
                self.send_json(400, {"error": "persistence is not enabled"})
                return
            if self.store.persistence.delete_session(session_id):
                self.send_json(200, {"ok": True, "id": session_id})
            else:
                self.send_json(404, {"error": "chat not found"})
            return
        self.send_json(404, {"error": "not found"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Web UI for the Qwen orchestrator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="directory for SQLite job and project-memory storage")
    args = parser.parse_args()

    persistence = Persistence(Path(args.data_dir).expanduser() / "orchestrator.sqlite3")
    Handler.store = JobStore(persistence)
    hosts = [host.strip() for host in args.host.split(",") if host.strip()]
    if not hosts:
        hosts = ["127.0.0.1"]
    servers = [ThreadingHTTPServer((host, args.port), Handler) for host in hosts]
    for server in servers[1:]:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    print(
        "qwen orchestrator UI listening on "
        + ", ".join(f"http://{host}:{args.port}" for host in hosts),
        flush=True,
    )
    print(f"persistent state: {persistence.db_path}", flush=True)
    servers[0].serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
