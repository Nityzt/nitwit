"""Workspace: git branch + file edits + sandboxed test runs for one repo. Never push/merge."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


class DirtyRepo(Exception):
    pass


class UnsafeEditPath(Exception):
    pass


@dataclass
class FileEdit:
    path: str      # repo-relative
    content: str   # full new file content (write_file semantics)


@dataclass
class TestResult:
    passed: bool
    output: str


def git(repo_path: str, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", repo_path, *args],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


class Workspace:
    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path

    def is_clean(self) -> bool:
        return git(self.repo_path, "status", "--porcelain") == ""

    def ensure_branch(self, branch: str) -> None:
        if not self.is_clean():
            raise DirtyRepo(f"{self.repo_path} has uncommitted changes; refusing to start")
        existing = git(self.repo_path, "branch", "--list", branch)
        if existing:
            git(self.repo_path, "checkout", "-q", branch)
        else:
            git(self.repo_path, "checkout", "-q", "-b", branch)

    def reset_hard(self) -> None:
        """Discard all uncommitted changes (tracked + untracked). Only safe to call on a
        branch whose dirt is known to be the mission's own crashed iteration, never on a
        tree that might hold a user's own uncommitted work."""
        git(self.repo_path, "reset", "--hard", "HEAD")
        git(self.repo_path, "clean", "-fd")

    def apply_edits(self, edits: list[FileEdit]) -> None:
        repo_root = os.path.realpath(self.repo_path)
        for edit in edits:
            full = os.path.realpath(os.path.join(self.repo_path, edit.path))
            if full != repo_root and not full.startswith(repo_root + os.sep):
                raise UnsafeEditPath(f"edit path escapes repo: {edit.path!r}")
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh:
                fh.write(edit.content)

    def commit(self, message: str) -> str:
        git(self.repo_path, "add", "-A")
        if git(self.repo_path, "status", "--porcelain") == "":
            return ""
        git(self.repo_path, "commit", "-q", "-m", message)
        return git(self.repo_path, "rev-parse", "--short", "HEAD")

    def run_tests(self, cmd: str, timeout: int = 120) -> TestResult:
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=self.repo_path,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return TestResult(False, "TIMEOUT")
        output = (proc.stdout + proc.stderr).strip()
        return TestResult(proc.returncode == 0, output)
