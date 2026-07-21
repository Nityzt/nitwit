# Nitwit Capability Upgrade — Design

**Date:** 2026-07-21
**Status:** Approved; building phase by phase.

## Context
Nitwit is a working local agent: an interactive `wit` CLI (conversational, with memory), a
durable mission engine (loop until tests pass + verifier approves, on a git branch), a daemon +
HTTP/SSE API, and a device-split model tier that was benchmarked but only partly wired. Four
gaps surfaced in real use:
1. Tasks require a git repo — should work anywhere.
2. No tool calling (e.g. web search) — the agent can't fetch current info.
3. Everything runs on the CPU-bound GPU coder — chat is slow; no per-task model choice.
4. Memory is per-session only — nothing persists across restarts.

## Decisions (locked)
- **Tasks anywhere → isolated scratch workspace.** No git repo in CWD ⇒ the mission runs in a
  fresh git repo under `~/.local/share/nitwit/workspaces/<id>/`; `wit export <id> [dest]` copies
  results out. In a git repo, unchanged (works on `agent/<slug>`). CWD is never touched for the
  scratch case.
- **Auto device-split routing.** A router picks model+endpoint per stage: chat/lookup/plan/
  compact → CPU (MiniCPM-1B :8081 utility, Qwen3-4B :8086 substantive chat); coding/synthesis →
  GPU Qwen-7B :8080; verify → Qwen3-4B :8086. Falls back to the GPU coder if a CPU service is
  down. Result: instant chat, GPU reserved for coding, best model per task.
- **Tool calling: web search auto; shell gated.** `web_search` (searxng capability) auto-invoked
  read-only in chat + missions; missions read files + run the declared test command; a general
  `run_command` needs approval (interactive: y/n; headless mission: only its test_cmd, no
  arbitrary shell). Bounded loops (GPU-safe; tool loops run on CPU-hosted models anyway).
- **Persistent memory: auto-propose + approve, auto-recall.** A durable SQLite `memories` store.
  The agent proposes durable facts (stack, preferences, project conventions); you approve; they
  persist and are auto-recalled into chat + mission context. Reuses the orchestrator's
  memory-extraction logic.

## Phases (each independently shippable, highest value first)
1. **Router / device-split** (+ fix the chat identity hallucination): a `nitwit/router.py` that
   maps a stage → (base_url, model, extra_body); wire chat (`stream_answer`) and the mission
   engine's stages to it; give the chat a correct system identity ("Nitwit, a local self-hosted
   assistant running open models on the user's machine; not made by OpenAI/Anthropic").
2. **Tasks anywhere**: scratch-workspace creation for repo-less tasks + `wit export`.
3. **Tool calling**: `web_search` in chat + missions (bounded loop, results fed back); gated
   `run_command`.
4. **Persistent memory**: store + propose/approve UI in the session + auto-recall into context.

## Constraints (unchanged)
- Stdlib + existing project modules; loopback-only services; bounded GPU bursts (no sustained GPU
  tool-loops); never push/merge; the coder never executes anything the host didn't run.

## Verification
Each phase: TDD (offline unit tests with fakes), per-task review, then a live check (e.g. Phase 1:
`wit` chat answers correctly + fast on the CPU model; "who made you" is answered correctly; a
mission still codes on the GPU). Full suite + legacy stay green.
