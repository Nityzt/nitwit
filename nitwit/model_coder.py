"""ModelCoder: the real Coder — one bounded GPU-model call per iteration that proposes
full-file edits as fenced `file:` blocks. The engine applies edits and runs tests; the
coder only proposes."""
from __future__ import annotations

import re

from nitwit.coder import CoderResponse, MissionContext
from nitwit.workspace import FileEdit

_FILE_BLOCK = re.compile(r"```file:(?P<path>[^\n`]+)\n(?P<body>.*?)```", re.DOTALL)

CODER_SYSTEM = (
    "You are an autonomous software engineer working toward a goal in a code repository. "
    "You are given the goal, hard constraints, the current repository files, and the latest "
    "test output. Produce the file edits that make progress toward the goal and make the tests "
    "pass. For EVERY file you create or change, output a fenced block exactly in this form:\n"
    "```file:relative/path.ext\n<the COMPLETE new content of that file>\n```\n"
    "Output ONLY these file blocks — no explanation, no diff syntax, no partial files. "
    "Give the whole file content each time. If no edit is needed, output nothing."
)


def parse_file_edits(text: str) -> list[FileEdit]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    edits: list[FileEdit] = []
    for m in _FILE_BLOCK.finditer(text):
        path = m.group("path").strip()
        body = m.group("body")
        if path:
            edits.append(FileEdit(path=path, content=body))
    return edits


def _repo_snapshot_text(repo_files: dict[str, str], max_snippet: int) -> str:
    if not repo_files:
        return "(empty repository)"
    parts = []
    for path, content in repo_files.items():
        clipped = content[:max_snippet]
        parts.append(f"--- {path} ---\n{clipped}")
    return "\n\n".join(parts)


def build_coder_messages(ctx: MissionContext, max_snippet: int = 12000) -> list[dict]:
    constraints = "\n".join(f"- {c}" for c in ctx.constraints) or "(none)"
    user = (
        f"GOAL:\n{ctx.goal}\n\n"
        f"HARD CONSTRAINTS:\n{constraints}\n\n"
        f"LATEST TEST OUTPUT:\n{ctx.last_test_output or '(no tests run yet)'}\n\n"
        f"NOTES SO FAR:\n{ctx.notes or '(none)'}\n\n"
        f"CURRENT REPOSITORY FILES:\n{_repo_snapshot_text(ctx.repo_files, max_snippet)}"
    )
    return [
        {"role": "system", "content": CODER_SYSTEM},
        {"role": "user", "content": user},
    ]


class ModelCoder:
    def __init__(self, client, max_tokens: int = 1600) -> None:
        self.client = client
        self.max_tokens = max_tokens

    def propose(self, ctx: MissionContext) -> CoderResponse:
        messages = build_coder_messages(ctx)
        response = self.client.chat(messages, temperature=0.0, max_tokens=self.max_tokens)
        edits = parse_file_edits(response.content)
        note = "proposed edits: " + ", ".join(e.path for e in edits) if edits else "no edits proposed"
        return CoderResponse(edits=edits, note=note)
