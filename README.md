# Nitwit

A local, self-hosted AI coding and research agent that runs entirely on your own
machine using open models through `llama.cpp` — no cloud, no API keys, nothing
leaves the box. You talk to it through an interactive CLI (`wit`), it answers
questions, researches the web with real sources, and takes on longer coding work
as durable, self-correcting **missions** that run in the background.

Nitwit is the successor to the earlier `qwen-orchestrator` experiment (still in
this repo as `orchestrator.py` / `webui.py`); the `nitwit/` package is the current
system.

## What it does

- **Conversational CLI** — open `wit` in any directory and talk naturally, the way
  you would with Claude Code / codex / the agy CLI. Questions are answered inline
  and streamed; tasks turn into missions.
- **Grounded web research** — when a question needs current or external facts, Nitwit
  searches (SearXNG), fetches the top result pages, and answers from them with source
  URLs, running a self-correcting loop that verifies every specific against the fetched
  sources before replying — so it doesn't invent versions, dates, or numbers.
- **Durable missions** — hand it a feature or a fix and it works on an `agent/<slug>`
  branch, looping and correcting itself until the goal is met, committing as it goes.
  Missions survive detaching, restarts, and reboots (they run in a background daemon).
- **Persistent memory** — tell it your preferences ("I use pnpm", "call me Wit") and it
  proposes to remember them; approved facts persist across sessions and are recalled
  into context automatically.
- **Automatic model routing** — a device-split router picks the right local model for
  each job and balances CPU/GPU so chat stays fast while heavier work uses the GPU.
- **Local-only and private** — every service binds to `127.0.0.1`; reach it remotely
  over SSH/Tailscale, never by exposing a port.

## Quickstart

Install the launcher once:

```bash
bash deploy/install-wit.sh      # -> ~/.local/bin/wit
```

Then, from anywhere:

```bash
cd ~/my-project
wit
```

`wit` auto-starts the mission daemon, detects the current git repo and its test
command, and drops you into a conversational session.

## The `wit` session

- **Ask a question** (`what does this function do?`, `latest Next.js version?`) — answered
  inline and streamed. Current-info questions transparently trigger grounded web research.
- **Give it a task** (`add a /health endpoint`, `fix the failing test`) — auto-escalates
  into a **mission** on an `agent/<slug>` branch. Watch it work; **Ctrl-C detaches** and
  the mission keeps running in the daemon. Reopen `wit` and it's still there. Your main
  branch is never touched and nothing is pushed.

Slash commands inside the session:

| command | purpose |
|---------|---------|
| `/missions` | list missions and their state |
| `/mission <goal>` | force a task to run as a mission |
| `/diff <id>` | inspect a mission's work |
| `/status`, `/on`, `/off` | daemon status / toggle the worker on and off |
| `/remember <text>`, `/memories`, `/forget <id>` | manage persistent memory |
| `/export <id> [dest]` | copy a scratch-workspace mission's result out |
| `/clear`, `/help`, `/quit` | clear context, help, exit |

For scripting, subcommands work non-interactively too: `wit new "..." --repo P --test CMD`,
`wit ls`, `wit tail <id>`, `wit on|off|status`, `wit -p "quick question"`.

## Architecture

```
wit (CLI)  ──HTTP+SSE──►  mission daemon  ──►  engine + models
   │                          │                    │
   │  chat / web research     │  durable missions  │  device-split router
   ▼                          ▼                    ▼
 session.py                missions.py        router.py ─► local llama.cpp servers
 webanswer.py              engine.py                        (CPU 4B / GPU 7B / 1B)
 memory.py                 daemon.py / api.py
```

- **`nitwit/session.py`** — interactive chat: intent classification, streamed answers,
  memory recall, and delegation to the web-research loop when a search is needed.
- **`nitwit/webanswer.py`** — the grounded research loop: search → fetch pages →
  synthesize → verify grounding → self-correct, repeating until the answer is grounded
  or the iteration budget is spent. No hardcoded fact gates — a model judges the answer
  against the fetched sources.
- **`nitwit/router.py`** — maps each work stage to the best local endpoint (see below).
- **`nitwit/missions.py`, `engine.py`, `daemon.py`, `api.py`** — first-class Mission
  objects (goal, constraints, success criteria, repos, state) and the 24/7 worker that
  loops on them until their success criteria pass.
- **`nitwit/memory.py`** — durable SQLite memory with an approval-gated propose step.
- **`nitwit/tools.py`** — `web_search` (SearXNG), `fetch_url`, `gather_context`.

## Models and routing

Nitwit runs several small local models and routes work to the right one automatically.
Endpoints are OpenAI-compatible `llama.cpp` servers, all on loopback:

| stage | model | endpoint | device |
|-------|-------|----------|--------|
| chat / verify | Qwen3-4B | `127.0.0.1:8086` | CPU |
| code / web synthesis | Qwen2.5-Coder-7B | `127.0.0.1:8080` | GPU |
| utility | MiniCPM-1B | `127.0.0.1:8081` | CPU |

Keeping chat and verification on the CPU 4B leaves the GPU free for coding and
synthesis and keeps conversation responsive. Every endpoint and default is overridable
via `NITWIT_*_URL` / `NITWIT_*_MODEL` environment variables; a stage falls back to a
healthy endpoint if its own is down. Web search uses SearXNG on `127.0.0.1:8888`.

## Missions

A mission is a durable, structured goal the daemon works until it's actually done:

- Runs on a dedicated `agent/<slug>` branch (or an isolated scratch workspace for
  tasks with no repo), never your main branch.
- Loops: plan → edit → run the declared tests / verifier → correct → repeat, committing
  progress as it goes.
- Survives detaching and restarts — the daemon owns the work, `wit` is just a client.
- Toggle the worker with `/off` when you need the machine (gaming, video editing) and
  `/on` to resume.

Review a mission with `wit diff <id>` or `git -C <repo> diff main..agent/<slug>`, then
merge it yourself.

## Development

Run the full test suite:

```bash
cd /home/nit/qwen-orchestrator
python3 -m unittest discover -s . -p 'test_*.py'
```

The `nitwit/` package is covered by `test_nitwit_*.py`; the legacy orchestrator is
covered by `test_orchestrator.py`.

## Machine notes

This instance runs on a Fedora workstation with an RX 580 (8 GB, Vulkan). That GPU is
fragile under sustained load, so the model profile (context size, GPU layers, clock
lock) is tuned conservatively and the GPU is reserved for bounded, single-shot work.
Operational details, tuning, and incident history live in `~/infra/` and are outside
the scope of this repo.

## Legacy: qwen-orchestrator

The original multi-call orchestration experiment and its web UI still ship here:

```bash
python3 orchestrator.py "Design a small Python library for validating JSON config files"
python3 webui.py         # prompt console + workflow trace on 127.0.0.1:8091
```

It decomposes a prompt into planner → workers → verifier → synthesizer stages. Nitwit
reuses its model client and structured-output machinery; new work targets `nitwit/`.
