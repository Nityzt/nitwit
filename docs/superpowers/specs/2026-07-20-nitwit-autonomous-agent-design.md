# Nitwit — a general local agent with durable autonomous missions

**Date:** 2026-07-20
**Status:** Design approved; pending spec review → implementation plan.
**Renames:** project `qwen-orchestrator` → **Nitwit**; CLI binary → **`wit`**; service → `nitwit.service`.

## Context

nitbox already runs a local orchestration service (chat, routing, web research, a plan→
worker→verify→synthesize pipeline) over a device-split model tier: MiniCPM-1B (plan/compact)
and Qwen3-4B (verify) resident on CPU, a swappable GPU coder slot (Qwen2.5-Coder-7B @ 64k /
14B @ 8k) with a VRAM guard, and an async verify pipeline (CPU verify overlaps GPU synth,
measured 260s→147s). The model layer was benchmarked on the user's real stack (TypeScript,
FastAPI, agentic repair) with executable tests; the 30B was rejected (17 GB model > 15 GB RAM
→ disk-thrashing, not viable).

**The need.** Reface this as a user-friendly, general-purpose local agent — a "coworker"
that handles quick chats and lookups *and* can take a feature/plan and **loop, self-correct,
and verify to objective completion, however long it takes (possibly days)**. It must run
24/7 headless when the user is away, be toggleable on/off when the user needs the GPU
(gaming/video editing), and survive reboots. This is an evolution of the existing pipeline,
not a rewrite: the pipeline's bounded plan→verify loop becomes the *inner step* of a durable,
unbounded outer loop.

**Why the GPU shapes everything.** The RX 580 faults under *sustained* GPU agentic loops
(proven in the crash investigation) but is safe for *bounded* GPU bursts separated by CPU
work. The whole durability design leans on this: a days-long mission is many short GPU bursts
(coder calls) with CPU work (running tests, CPU verify, git) between them — the crash-safe
shape by construction.

## Requirements (locked with the user)

1. **Done-oracle:** a task is complete when the target repo's **executable tests pass** AND
   the model verifier approves ("meaningful", not just green). Objective ground truth.
2. **Workspace:** the agent works in a **branch of a git repo the user points it at**
   (`agent/<slug>`), runs that repo's test command, and **commits as it goes**. It never
   pushes or merges. The branch is the deliverable *and* the durable checkpoint.
3. **Toggle + resume:** toggling off finishes the current GPU call, commits WIP, **stops the
   coder container (frees VRAM)**, and parks. Toggling on **resumes from the branch + mission
   state**. Reboot-proof (durable state, not RAM).
4. **General scope, not just missions:** the router is the front door. chat / lookup /
   quick-code answer inline and fast; only long-horizon work becomes a background mission.
5. **Any-lane escalation:** every message starts synchronous; it escalates to a mission when
   the router is confident up front OR when an inline attempt discovers depth (tests fail,
   multi-file fix). The inline work already done becomes the mission's iteration 0.
6. **Interfaces:** a persistent **daemon** exposes an **HTTP + SSE API**; the **`wit` CLI is an
   interactive REPL agent** (Claude Code / codex / agy style) — the primary client, built first
   (fastest, fully testable, and the headless control surface). Closing the REPL leaves missions
   running in the daemon. **UI second** (refit the existing chat UI as a second client for
   MacBook/iPhone). All clients are thin, over one API.
7. **24/7 headless**, single GPU slot (missions run one at a time; chats interleave).

## Architecture: decoupled mission engine

A background **systemd user service** owns the loop, fully decoupled from any client. All
resumable state lives in **SQLite** (mission rows, iteration log, control flag) and the **git
branch** (the work). Clients (CLI now, UI later) observe and steer over the API. Chosen over
(A) extending the in-request job loop — too fragile for days-long headless runs — and (C) a
full tool-calling agent rewrite — revives the unbounded GPU loop that faulted the card and
discards the working pipeline.

### Components (each one clear purpose, independently testable)

- **`missions.py` — mission store.** Durable CRUD + state machine over SQLite.
  Mission = `(id, title, task_prompt, repo_path, test_cmd, branch, status, iteration,
  checkpoint_commit, created, updated)`. Status:
  `queued → running → (paused | needs_input) → done | failed`. No model logic — data +
  transitions only. Extends the existing SQLite persistence.
- **`workspace.py` — workspace manager.** Owns the target repo: create/checkout
  `agent/<slug>`, apply file edits to the working tree, run `test_cmd` in a **sandboxed
  subprocess** (timeout, captured output), `git add/commit` WIP. Never pushes/merges. The
  single source of "did the tests pass" and "commit the checkpoint".
- **`engine.py` — mission engine.** One background worker thread (respects the single GPU
  slot). Owns the `RUNNING | PAUSED` control flag, the outer loop, the stop condition, the
  per-iteration GPU-call cap, and the inter-iteration cooldown. Reconciles orphaned
  `running` missions on startup. Emits SSE progress events.
- **Tool surface** — host-executed tools the coder requests (text-form calls parsed via the
  existing `tool_loop.py` logic, fed back as a `user` turn — proven safe): `read_file`,
  `list_dir`, `search`, `write_file`/`apply_patch`, `run_tests` (the oracle), `web_search`
  (docs/StackOverflow; the 7B's search discipline was clean — 0 over-search). The coder never
  executes anything itself; the host runs tools and logs every call to the iteration record.
- **`wit` CLI** — interactive REPL agent + non-interactive subcommands; the primary client
  over the daemon API and the reference client that proves the SSE contract the UI reuses
  (detailed under "CLI surface").

### The loop

**One iteration = one bounded GPU burst + CPU work** (the crash-safe unit):

```
a. build context: task + relevant repo files + last test output + prior notes → fit ~64k
b. bounded coder call(s) on GPU: request tools / propose edits      ← the only GPU work
c. apply edits to the working tree                                   (host)
d. run_tests() in sandbox → pass/fail + output                       (CPU) ← the oracle
e. CPU verifier reviews for meaningfulness                            (CPU)
f. git commit WIP + persist iteration record                         (checkpoint)
g. emit progress → SSE stream (CLI/UI)
h. STOP if tests all green AND verifier approves → status=done
   else check control flag → if PAUSED, park; else cooldown, loop
```

Unbounded in **iterations**, bounded in **GPU work per iteration**. A safety cap (max
iterations / max wall-time) flips a stuck mission to `needs_input` rather than spinning
forever.

### Durability, toggle, resume

- **Toggle** (`wit off`): finish current GPU call → commit WIP → stop coder container (VRAM
  freed) → set `PAUSED` → engine idles. CPU utility models stay up (no VRAM). `wit on`: start
  coder profile → pick oldest non-terminal mission → **reconstruct context from branch +
  iteration log** → continue.
- **Resume is reconstruction, not memory:** `git checkout agent/<slug>`, read last iteration's
  test output + verifier notes + compacted history from SQLite, refit to ~64k. Objective
  oracle means a resumed mission can't lose its place.
- **Crash/reboot:** `systemd Restart=on-failure`; on start, any mission stuck `running` is
  rewound to its last commit and set `queued`. At most one iteration's speculative GPU work is
  lost.
- **GPU safety across days:** per-iteration GPU-call cap + inter-iteration cooldown; never a
  sustained GPU spiral. Reuses the device split (verify/plan/compact on CPU) so GPU time is
  minimized.

### Lane coexistence (general-purpose scope)

Router front door classifies each message:
- **chat / lookup** → synchronous, inline, immediate; a pure chat can answer on MiniCPM
  without waking the GPU.
- **research** → existing web pipeline, returns a result.
- **mission** → escalated (up front or mid-run) into the background loop; streams progress.

Single GPU slot ⇒ missions run one at a time (global FIFO, reusing the existing single-slot
queue). A synchronous request slips in at a mission's **iteration boundary**, so quick
questions stay responsive while a mission runs.

## CLI surface (`wit`) — interactive REPL agent (Claude Code / codex / agy style)

Two distinct things: the **daemon** (`nitwit.service`) runs missions 24/7; the **`wit` client**
is how you talk to it. Primary shape is an **interactive REPL**, not a subcommand dispatcher.

**Interactive session** (`wit`, no args) — like `claude` / `codex` / `agy`:
- Natural conversation; streams the agent's thinking + **tool calls live** (read_file,
  run_tests, edits) as they happen.
- Quick things answer inline; a long task **escalates into a background mission** you watch
  stream — or detach from with the session still open.
- **Key synergy with 24/7:** the REPL is a *client* to the persistent daemon. **Close the REPL
  and missions keep running**; reopen `wit` later (even post-reboot) and `/ls` shows them still
  going or done. Claude-Code feel + durable background work.
- **Slash commands** for control without leaving the session:
  `/new <task> --repo … --test …`, `/ls`, `/tail <id>`, `/pause <id>`, `/resume <id>`,
  `/cancel <id>`, `/diff <id>`, `/approve <id>`, `/on`, `/off`, `/status`, `/help`, `/clear`.

**Non-interactive modes** (scripting + headless + tests):
```
wit -p "prompt"                    # one-shot chat/lookup: run, stream to stdout, exit (pipeable)
wit new "task" --repo PATH --test "CMD" [--detach]   # start a mission non-interactively; prints its id
wit ls | tail <id> | pause <id> | resume <id> | cancel <id> | diff <id> | approve <id>
wit on | off | status              # direct daemon control (systemd/scripts, no REPL)
```
These are the same operations as the REPL slash commands, exposed as subcommands so scripts and
the integration tests can drive missions without a REPL.

The interactive REPL, the one-shot mode, and the future UI all consume the **same SSE event
stream** from the daemon — the CLI is the reference client that proves the contract.

## Error handling

- **Test command missing/misconfigured** → mission → `needs_input` with the error; never loops
  blindly.
- **Coder returns no usable edit / malformed tool call** → retry once with a corrective note;
  repeated failure → `needs_input`.
- **Verifier returns empty (thinking overflow)** → already handled: treat as pass-with-caveat,
  don't crash the loop (fixed this session).
- **Safety cap hit** (max iterations/time) → `needs_input`, not `failed` — the user decides.
- **GPU fault** (should not occur with bounded bursts) → engine catches the coder-call error,
  commits WIP, parks the mission `paused`, surfaces the incident; host stays reachable.
- **Dirty target repo** (uncommitted user changes) → refuse to start; require a clean tree.

## Testing strategy

- **Unit:** `missions.py` (state machine transitions), `workspace.py` (branch/edit/test/commit
  against a temp repo), engine stop-condition + reconcile logic.
- **Integration (through the CLI):** a throwaway git repo with a known **failing test** →
  assert `wit new` branches, iterates, reaches green, commits, stops `done`. Kill the engine
  mid-mission → assert it resumes from the branch and completes. Assert toggle-off frees the
  coder container and toggle-on resumes.
- **Regression:** existing `python3 -m unittest` stays green; `node --check` on any embedded
  UI JS (UI phase); each engine change parse-checked.

## Build order (phased; each phase testable before the next)

1. **Engine core** — `missions.py` + `workspace.py` + `engine.py` + the outer loop, driven by
   a temp-repo integration test. No UI, no service yet.
2. **Daemon API + `wit` REPL** — HTTP + SSE endpoints; `wit` interactive REPL (streaming
   tool calls, slash commands) + `wit -p` one-shot as the reference clients. Toggle/resume via
   `wit on/off` and `/on /off`.
3. **systemd service** — `nitwit.service` (the daemon: enabled, `Restart=on-failure`), headless
   24/7, reconcile-on-start. `wit` attaches to it; closing `wit` leaves missions running.
4. **Escalation + lane routing** — wire chat/lookup synchronous lanes + mid-run escalation
   into a mission.
5. **UI refit** — the existing chat UI becomes a second client over the same API (MacBook/
   iPhone), including a toggle switch and mission views.
6. **Rename** — `qwen-orchestrator` → Nitwit throughout (README, services, paths), and add the
   git remote the user provides.

## Not doing

- No auto-push/auto-merge — the branch is handed to the user for review.
- No sustained/unbounded GPU tool-loop — bounded bursts only.
- No 30B (doesn't fit RAM). 14B Q4 remains the capability ceiling that fits.
- No raising GPU clocks/layers.
