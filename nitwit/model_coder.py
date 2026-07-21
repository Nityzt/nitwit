"""ModelCoder: the real Coder — one bounded GPU-model call per iteration that proposes
full-file edits as fenced `file:` blocks. The engine applies edits and runs tests; the
coder only proposes."""
from __future__ import annotations

import re

from nitwit.coder import CoderResponse, MissionContext
from nitwit.workspace import FileEdit

_FILE_OPENER = re.compile(r"```file:([^\n`]+)\n")
_STANDALONE_FENCE = re.compile(r"(?m)^```[ \t]*$")

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
    # Drop closed <think>...</think> blocks. If a <think> remains unterminated
    # (e.g. the model was cut off by max_tokens mid-reasoning), everything from
    # that point on is incomplete draft content, never final output — discard it
    # rather than let a fence the model was merely drafting become an edit.
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    if "<think>" in text:
        text = text[: text.index("<think>")]

    matches = list(_FILE_OPENER.finditer(text))
    edits: list[FileEdit] = []
    for i, m in enumerate(matches):
        path = m.group(1).strip()
        start = m.end()
        seg_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[start:seg_end]
        # The body runs up to the LAST stand-alone closing fence in this segment
        # (a line that is exactly ```), not the first ``` substring — otherwise
        # a fence embedded in legitimate file content (READMEs, docstrings)
        # would silently truncate the edit.
        closes = list(_STANDALONE_FENCE.finditer(segment))
        body = segment[: closes[-1].start()] if closes else segment
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
