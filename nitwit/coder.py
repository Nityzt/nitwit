"""Coder/Verifier interfaces the engine depends on, plus deterministic fakes for testing.

The real model-backed Coder (wrapping the GPU coder + tool loop) is Phase 2; the engine only
ever sees these interfaces, so the whole loop is testable offline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from nitwit.workspace import FileEdit


@dataclass
class MissionContext:
    goal: str
    constraints: list[str]
    notes: str
    last_test_output: str
    repo_files: dict[str, str]  # path -> content snapshot fed to the coder


@dataclass
class CoderResponse:
    edits: list[FileEdit] = field(default_factory=list)
    note: str = ""
    question: str = ""  # non-empty => the coder needs the user (Phase 4 wires needs_input)


class Coder(Protocol):
    def propose(self, ctx: MissionContext) -> CoderResponse: ...


class Verifier(Protocol):
    def judge(self, description: str, ctx: MissionContext) -> bool: ...


class FakeCoder:
    """Returns scripted responses in order; empty responses once exhausted."""

    def __init__(self, scripted: list[CoderResponse]) -> None:
        self._scripted = list(scripted)
        self.calls = 0

    def propose(self, ctx: MissionContext) -> CoderResponse:
        self.calls += 1
        if self._scripted:
            return self._scripted.pop(0)
        return CoderResponse()


class FakeVerifier:
    def __init__(self, verdict: bool = True) -> None:
        self.verdict = verdict
        self.calls = 0

    def judge(self, description: str, ctx: MissionContext) -> bool:
        self.calls += 1
        return self.verdict
