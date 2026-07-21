"""MissionEngine: the durable iteration loop. Pure orchestration over injected Coder/Verifier."""
from __future__ import annotations

import os
import threading
import time

from nitwit.coder import Coder, CoderResponse, MissionContext, Verifier
from nitwit.missions import Mission, MissionStore
from nitwit.workspace import DirtyRepo, Workspace, git

# Files bigger than this are skipped in the context snapshot (keep prompt bounded).
_MAX_SNAPSHOT_BYTES = 20000


class MissionEngine:
    def __init__(self, store: MissionStore, coder: Coder, verifier: Verifier,
                 workspace_factory=Workspace, max_iterations: int = 20, cooldown_s: float = 0.0, on_event=None) -> None:
        self.store = store
        self.coder = coder
        self.verifier = verifier
        self.workspace_factory = workspace_factory
        self.max_iterations = max_iterations
        self.cooldown_s = cooldown_s
        self.on_event = on_event
        self._paused = threading.Event()

    def _emit(self, event_type: str, mission_id: str, **data) -> None:
        if self.on_event:
            self.on_event({"event": event_type, "mission_id": mission_id,
                           "time": round(time.time(), 3), **data})

    def _snapshot(self, repo_path: str) -> dict[str, str]:
        files: dict[str, str] = {}
        for root, dirs, names in os.walk(repo_path):
            if ".git" in dirs:
                dirs.remove(".git")
            for name in names:
                full = os.path.join(root, name)
                try:
                    if os.path.getsize(full) > _MAX_SNAPSHOT_BYTES:
                        continue
                    with open(full, "r", errors="replace") as fh:
                        files[os.path.relpath(full, repo_path)] = fh.read()
                except OSError:
                    continue
        return files

    def build_context(self, mission: Mission, workspaces: dict[str, Workspace], last_test_output: str) -> MissionContext:
        primary = mission.repos[0]["path"] if mission.repos else ""
        repo_files = self._snapshot(primary) if primary else {}
        return MissionContext(
            goal=mission.goal, constraints=mission.constraints, notes=mission.notes,
            last_test_output=last_test_output, repo_files=repo_files,
        )

    def evaluate_criteria(self, mission: Mission, workspaces: dict[str, Workspace]) -> tuple[bool, str]:
        if not mission.success_criteria:
            return False, "no success criteria defined"
        summaries = []
        all_passed = True
        ctx = self.build_context(mission, workspaces, "")
        for crit in mission.success_criteria:
            kind = crit.get("type")
            if kind == "tests":
                ws = workspaces[crit["repo"]]
                result = ws.run_tests(crit["cmd"])
                ok = result.passed
                summaries.append(f"tests({crit['cmd']}): {'pass' if ok else 'fail'}")
            elif kind == "verifier":
                ok = self.verifier.judge(crit.get("description", ""), ctx)
                summaries.append(f"verifier: {'pass' if ok else 'fail'}")
            else:
                ok = False
                summaries.append(f"{kind}: unsupported (phase 1)")
            all_passed = all_passed and ok
        return all_passed, "; ".join(summaries)

    def run_iteration(self, mission: Mission, workspaces: dict[str, Workspace]) -> tuple[Mission, bool]:
        self._emit("iteration_started", mission.id, iteration=mission.iteration + 1)
        ctx = self.build_context(mission, workspaces, "")
        response: CoderResponse = self.coder.propose(ctx)
        primary_path = mission.repos[0]["path"]
        ws = workspaces[primary_path]
        if response.edits:
            ws.apply_edits(response.edits)
            self._emit("edits_applied", mission.id, paths=[e.path for e in response.edits])
            ws.commit(f"iteration {mission.iteration + 1}: {response.note or 'edits'}")
        mission = self.store.bump_iteration(mission.id)
        if response.note:
            mission = self.store.append_note(mission.id, response.note)
        done, summary = self.evaluate_criteria(mission, workspaces)
        self._emit("criteria_evaluated", mission.id, passed=done, summary=summary)
        mission = self.store.append_note(mission.id, f"criteria -> {summary}")
        return mission, done

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def reconcile(self) -> int:
        rewound = 0
        for m in self.store.list(state="running"):
            self.store.set_state(m.id, "queued")
            rewound += 1
        return rewound

    def _prepare_workspaces(self, mission: Mission) -> dict[str, Workspace]:
        workspaces = {}
        for repo in mission.repos:
            ws = self.workspace_factory(repo["path"])
            # A mission is RESUMING this repo iff its agent branch already exists. But the
            # branch existing does NOT prove the current dirty tree is this mission's own
            # crashed iteration -- between the crash and this resume, the user may have
            # checked the repo out to their own branch and done ordinary work there. The
            # real safety property is whether the repo is CURRENTLY parked on the agent
            # branch: only then is any dirt provably this mission's own debris, safe to
            # blow away with reset_hard. On any other branch the tree may hold the user's
            # work, so we refuse (via ws.is_clean()'s check) rather than silently carrying
            # their untracked files/edits onto the agent branch and then destroying them.
            existing = git(repo["path"], "branch", "--list", repo["branch"])
            if existing:
                current = git(repo["path"], "rev-parse", "--abbrev-ref", "HEAD")
                if current == repo["branch"]:
                    ws.reset_hard()
                else:
                    if not ws.is_clean():
                        raise DirtyRepo(
                            f"{repo['path']} has uncommitted changes on '{current}'; "
                            f"commit or stash before resuming this mission"
                        )
                    git(repo["path"], "checkout", "-q", repo["branch"])
                    ws.reset_hard()  # safe: tree was verified clean before switching
            else:
                ws.ensure_branch(repo["branch"])
            workspaces[repo["path"]] = ws
        return workspaces

    def run_mission(self, mission_id: str) -> Mission:
        mission = self.store.get(mission_id)
        if self._paused.is_set():
            if mission.state != "paused":
                mission = self.store.set_state(mission.id, "paused")
                self._emit("mission_finished", mission.id, state="paused")
            return mission
        mission = self.store.set_state(mission.id, "running")
        self._emit("mission_started", mission.id)
        workspaces = self._prepare_workspaces(mission)
        while True:
            if self._paused.is_set():
                mission = self.store.set_state(mission.id, "paused")
                self._emit("mission_finished", mission.id, state="paused")
                return mission
            mission, done = self.run_iteration(mission, workspaces)
            if done:
                mission = self.store.set_state(mission.id, "done")
                self._emit("mission_finished", mission.id, state="done")
                return mission
            if mission.iteration >= self.max_iterations:
                self.store.append_note(mission.id, "hit max_iterations; awaiting input")
                mission = self.store.set_state(mission.id, "needs_input")
                self._emit("mission_finished", mission.id, state="needs_input")
                return mission
            if self.cooldown_s:
                time.sleep(self.cooldown_s)
