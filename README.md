# qwen-orchestrator

Small proof of concept for making a local Qwen2.5 Coder 7B model more useful by
calling it many times with narrow prompts instead of once with a large prompt.

This is not a traditional chatbot and not a full agent framework. It is a
long-running orchestration experiment:

1. Planner chooses a workflow, worker roles, dependencies, and token budgets.
2. Workers run focused subproblems with only relevant prior results.
3. Verifier checks coverage, contradictions, and missing work.
4. Follow-up workers run when the verifier finds gaps.
5. Synthesizer produces one final answer from all worker results.

## Current machine notes

On this Fedora machine, Qwen is running through a rootless Podman `llama`
container, not through a visible Ollama daemon. It is exposed as an
OpenAI-compatible llama.cpp server on `127.0.0.1:8080` with model alias
`qwen2.5-coder-7b`.

The RX 580/Vulkan path is the default model server path. A sustained UI
orchestration run completed at `GPU_LAYERS=20`, `BATCH_SIZE=128`,
`UBATCH_SIZE=32`, `--parallel 1` with 20/20 model calls and about 30k total
tokens. `GPU_LAYERS=28` crashed the GPU under a real orchestration workload.
Kernel logs showed `amdgpu` ring timeouts, `device lost from bus`, and
`GPU Recovery Failed` with `llama-server` as the active process. Treat 20 layers
as the current sustained-test ceiling.

Start the model server:

```bash
cd /home/nit/qwen-orchestrator
./start_llama_gpu.sh
```

This starts llama.cpp with:

```text
/app/llama-server --model /models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 --port 8080 --n-gpu-layers 20 --ctx-size 4096 \
  --parallel 1 --batch-size 128 --ubatch-size 32 --threads 4 \
  --temp 0.30 --top-p 0.90 --alias qwen2.5-coder-7b
```

`ollama` is not currently on `PATH`, no `ollama.service` is registered, and
nothing is listening on `127.0.0.1:11434`. The orchestrator still supports
Ollama for later experiments, but the defaults target the live llama.cpp server.

For now, keep orchestration sequential. The server is single-slot
(`--parallel 1`). The web UI also defaults to a project scan cap of 80 files and
a 3 second cooldown between model calls:

```bash
QWEN_PROJECT_MAX_FILES=80 QWEN_MODEL_CALL_COOLDOWN_S=3 ./start_webui_safe.sh
```

Raise those only after a full project run completes without system instability.

The systemd user service `qwen-llama.service` starts this GPU launcher on boot
with the same conservative 20-layer profile. Keep `--parallel 1` and check
`journalctl -k` for `amdgpu` ring timeout/reset messages during long jobs. Do
not use 28 layers for sustained orchestration on this machine.

The orchestrator can talk to either:

- Ollama native API: `--provider ollama --base-url http://127.0.0.1:11434`
- OpenAI-compatible local API: `--provider openai-compatible --base-url http://127.0.0.1:8080`

## Run

With the current live llama.cpp endpoint:

```bash
cd /home/nit/qwen-orchestrator
python3 orchestrator.py \
  "Design a tiny Python library for validating JSON config files"
```

Deeper run with full trace:

```bash
python3 orchestrator.py --json \
  "Design a robust Python CLI architecture for a local file sync tool. Include module boundaries, failure modes, config/logging, test strategy, and incremental implementation steps."
```

Explicit llama.cpp settings:

```bash
cd /home/nit/qwen-orchestrator
python3 orchestrator.py --provider openai-compatible --base-url http://127.0.0.1:8080 \
  --model qwen2.5-coder-7b \
  "Design a tiny Python library for validating JSON config files"
```

With Ollama, if you later start an Ollama daemon:

```bash
python3 orchestrator.py --provider ollama --base-url http://127.0.0.1:11434 \
  --model qwen2.5-coder:7b \
  "Design a tiny Python library for validating JSON config files"
```

Print the full trace:

```bash
python3 orchestrator.py --json "Explain how to refactor a messy CLI into modules"
```

## Web UI

```bash
cd /home/nit/qwen-orchestrator
./start_webui_safe.sh
```

Open:

```text
http://127.0.0.1:8091
```

The UI acts as a prompt console. Submit a request, then watch the workflow pane
as the planner, workers, verifier, follow-up rounds, and synthesizer run. The
advanced controls are collapsed by default; the normal path is to let Qwen choose
the orchestration shape while the host enforces the single-worker GPU constraint.

Persistent state is stored in SQLite at:

```text
/home/nit/qwen-orchestrator/data/orchestrator.sqlite3
```

Override it when needed:

```bash
python3 webui.py --data-dir ~/.local/share/qwen-orchestrator
```

The database currently stores:

- recent job snapshots and event traces, so job history survives UI restarts
- compact project memories keyed by project path and a file fingerprint

When a project has not changed, project modes reuse the cached architecture
memory instead of rescanning every file. If included source files change, the
fingerprint changes and the project is scanned again.

If the UI or machine restarts during a project scan, the old job is marked
interrupted on startup. The exact Python worker cannot continue after reboot,
but a new project run can reuse completed per-file summaries from the interrupted
job when the project fingerprint still matches. Those files appear as resumed in
the workflow.

Project modes also run a lightweight local retrieval step over cached file and
directory summaries. It ranks relevant files from the request terms, reads a few
small source snippets, and includes only that focused context in the downstream
orchestration prompt. This is lexical retrieval for now, with no embedding
dependency; the retrieval interface can later be backed by semantic embeddings.

The UI also includes a manual **Local capabilities** panel. These are small,
read-only or restricted tools exposed through `/api/capabilities` and
`/api/capability/run`:

- `git_status`: read-only branch/status/recent commits/diff stat
- `list_dir`: directory listing under the home directory
- `file_preview`: bounded text preview for a file under the home directory
- `search_text`: ripgrep search under the home directory
- `web_search`: no-key web search. It cleans conversational prompts into search
  queries, tries local SearXNG first when available at `QWEN_SEARXNG_URL` or
  `http://127.0.0.1:8888`, then falls back to DuckDuckGo HTML/lite and Bing. It
  rejects obviously irrelevant result sets.
- `webpage_summary`: fetch a webpage and extract bounded readable text for
  grounding
- `python_eval`: restricted expression-only Python calculation

Most capabilities are manual by default. Simple requests like "search X and give
me the results" route to `search_results`, which runs `web_search` and returns
links without spending model tokens. Deeper current-info requests route to
`web_research`: the host runs `web_search`, fetches a couple of result pages with
`webpage_summary`, logs those capability runs, and injects bounded web evidence
into the orchestration prompt. Manual capability runs are persisted in SQLite so
tool behavior can be audited and reused while tuning prompts.

Project-mode orchestration now includes the capability manifest in the prompt as
manual-only context. Agents may recommend a capability and exact input when more
evidence is needed, but the host does not execute tools automatically yet.

### Local SearXNG

The web search path works best with a local SearXNG container because it exposes
a JSON API and mixes multiple search engines. Start it with:

```bash
cd /home/nit/qwen-orchestrator
./start_searxng.sh
```

It listens on `http://127.0.0.1:8888` and uses
`searxng/settings.yml`, which enables `html` and `json` formats. The
orchestrator falls back to direct HTML scraping if SearXNG is down.

Suggested tool requests are extracted from worker/final answers when they use
this JSON shape:

```json
{
  "tool_request": {
    "capability": "git_status",
    "input": {"path": "/home/nit/project"},
    "reason": "why this evidence is needed"
  }
}
```

The UI displays these as manual requested tools. You can run one directly from
the workflow card, or run the matching capability from the Local capabilities
panel.

When a capability is run while a job is selected, the result is attached back to
that job as context memory. This is the first step toward an approval-based tool
loop: the model can request evidence, the user can run the tool, and the result
is preserved with the workflow trace.

The UI also has a manual **User memory** panel. Memories are stored as
scope/key/value/tags records in SQLite and loaded into future jobs as preference
or environment context. User-scoped memories load into all jobs. Project-scoped
memories use `project:/absolute/path` and load only when that project is routed.
Memories can be deleted from the UI. The model cannot write memories on its own.

## Incremental roadmap

The implementation order is intentionally conservative:

1. **Persistence and project-memory cache**: done. This gives the orchestrator
   durable job history and reusable project knowledge without new dependencies.
2. **Retrieval for large projects**: initial lexical retrieval is done. The next
   improvement is a persistent snippet index and optional semantic embeddings.
3. **Safe local tools**: initial manual capabilities, persisted tool traces,
   prompt-level capability awareness, tool-request extraction, manual workflow
   execution buttons, and attached tool evidence are done. Next, add an
   orchestrator loop that can pause for user approval when a worker requests
   evidence.
4. **Manual memory**: basic durable user and project-scoped memory is done.
   Next, add memory suggestions that require user approval.
5. **Specialized workers and plugins**: web-search grounding is started through
   `search_results` and `web_research`; next route architecture, testing,
   security, and performance workers through a small capability registry.
6. **Patch workflow**: generate diffs, show them in the UI, and apply only after
   explicit approval.

## Verify control flow

```bash
cd /home/nit/qwen-orchestrator
python3 -m unittest -v
```

## Borrowed ideas worth keeping

- Task decomposition should optimize for solvability, completeness, and
  non-overlap. That matches recent agent-oriented planning work.
- DAG-shaped subtasks are useful, but this machine should usually run them
  sequentially because the live llama.cpp service has one slot.
- Verification should be a first-class step. For a small local model, this is
  often cheaper than making every worker prompt huge.
- Prefer structured JSON between stages. Small models will still sometimes wrap
  JSON in prose, so the parser tolerates fenced output and surrounding text.

## Next increments

- Add a benchmark harness that compares single-call Qwen vs orchestrated Qwen on
  the same tasks.
- Add a local context retriever for files, but only pass each worker the smallest
  relevant snippets.
- Improve replanning beyond the current verifier `missing_tasks` follow-up loop.
- Persist traces as JSONL so failures can be inspected and prompts can be tuned.

## `wit` — the interactive agent CLI

Install the launcher once:

```bash
bash deploy/install-wit.sh      # -> ~/.local/bin/wit
```

Then, from inside any git repo, just run:

```bash
cd ~/my-project
wit
```

`wit` opens a conversational session (like Claude Code / codex / agy). It auto-starts the
mission daemon, detects the current repo and its test command, and you talk to it naturally:

- **A question** (`what does parse() do?`) is answered inline, streamed.
- **A task** (`add a /health endpoint`, `fix the failing test`) auto-escalates into a durable
  **mission**: it works on an `agent/<slug>` branch, streams its progress, and commits as it goes.
  Press **Ctrl-C to detach** — the mission keeps running in the daemon; reopen `wit` and it's still
  there. Nothing ever touches your main branch or gets pushed.

Slash commands inside the session: `/missions`, `/diff <id>`, `/status`, `/on`, `/off`,
`/mission <goal>` (force a mission), `/quit`. Review a mission's work with `wit diff <id>` or
`git -C <repo> diff main..agent/<slug>`, then merge it yourself.

For scripting, the subcommands still work non-interactively: `wit new "..." --repo P --test CMD`,
`wit ls`, `wit tail <id>`, `wit on|off|status`, `wit -p "quick question"`.
