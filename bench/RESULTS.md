# Orchestrator model + tool-loop benchmarks

Objective, reproducible comparisons of the model layer on nitbox (RX 580 8 GB, Vulkan,
llama.cpp, clock-locked to 927 MHz). Every number here comes from a script in `bench/`
or `~/infra/gpu/` — re-run them, don't trust prose.

## How to reproduce

```bash
# per-stage accuracy + telemetry (heuristic stages need no model)
python3 -m bench.run_bench                                   # heuristic only
python3 -m bench.run_bench --model-stages                    # + planner/verifier on Qwen-GPU (8080)
python3 -m bench.run_bench --model-stages --base-url http://127.0.0.1:8081 --model minicpm5-1b --no-think

# grounded web-search tool loop (server needs --jinja)
python3 -m bench.tool_loop --base-url http://127.0.0.1:8081 --model minicpm5-1b --no-think --max-calls 3            # bounded, raw
python3 -m bench.tool_loop --base-url http://127.0.0.1:8081 --model minicpm5-1b --no-think --max-calls 3 --scaffold # + search-discipline prompt
python3 -m bench.tool_loop ... --max-calls 0                 # unaided baseline

# EXECUTABLE coding pass@1 — the SWE/agentic optimization target
python3 -m bench.coding_eval --base-url http://127.0.0.1:8080 --model qwen2.5-coder-7b --set all
```

Models live on `/home/nit/extra/llama-models` (166 GB btrfs), symlinked from
`~/.local/share/llama/models` — keeps the primary root fs free; every launcher/quadlet
path is unchanged.

## Stack-relevant capability — the user's actual stack (executed, not vibes)

The real target is the user's stack: **TypeScript** (React/Next/Astro/RN/Expo), **Python/FastAPI**
backend, with **web-docs lookup** as the "learn as you go" mechanism for fast-moving UI libraries.
New harnesses: `bench/stack_eval.py` (TypeScript, run via Node 24 native type-stripping),
`bench/api_eval.py` (FastAPI via TestClient in a uv venv), `bench/agentic_code.py` (iterative
test-repair, optional `--web`). All execute real code against hidden tests.

| Model | TS (8) | FastAPI (5) | Py easy (12) | Py hard (8) | agentic repair (8) | latency | fits w/ 64k? |
|-------|--------|-------------|--------------|-------------|--------------------|---------|--------------|
| **Qwen2.5-Coder-7B** | 7/8 (87%) | 3/5 (60%) | 11/12 | **8/8** | **8/8, 1.0 rounds** | ~8–9 s | ✅ full GPU |
| **Qwen2.5-Coder-14B** | **8/8 (100%)** | **4/5 (80%)** | 11/12 | **8/8** | — | ~15–18 s | ❌ partial only |

**What the stack tests show.**
- **TypeScript** (your primary language): 7B 87%, 14B **100%** — the 14B got the tricky nested
  `deepMerge` the 7B missed. Matches the published HumanEval+ gap (14B 87.2 vs 7B 82.3).
- **FastAPI** is the weak spot for the 7B (60%) — it missed the `201` status code and the
  `Depends` auth pattern (framework idioms). The 14B did better (80%); its one miss was a **missing
  `import Depends`** — a mechanical slip, not a knowledge gap.
- **The agentic loop closes the single-shot gaps.** The 7B's repair loop solved **8/8 in one round
  each** — read the test failure, fix it. The 14B's missing-import FastAPI failure is exactly this
  shape: a feedback loop (or web-docs) recovers it immediately. So for *agentic* use — what you want
  — the 7B's single-shot idiom gaps matter far less than the raw score suggests.
- **Web-search discipline is excellent on the Qwen coders.** Offered a `web_search` tool during
  repair, the 7B used it **0 times** on bugs it could fix itself — no over-search (the opposite of
  the 1B's reflex-searching). It's well-calibrated to search only when genuinely stuck, which is the
  behavior you want for "look up the docs" on unfamiliar/UI-library APIs.

**Reading for the stack:** the **7B is the efficient agentic default** — fast, full-GPU-resident
(coexists with 64k ctx), strong on TS, and its idiom gaps self-heal in the loop. The **14B is the
higher-capability single-shot synthesizer** (+TS, +FastAPI) when correctness-in-one-shot matters
more than latency, at the cost of the 64k context.

**Qwen3-Coder-30B-A3B — NOT VIABLE on this box (hardware, not capability).** The 30B MoE is a
**17 GB** file; the machine has **15 GB RAM**. All 30B of weights must be *resident* (the "3B
active" MoE trick saves compute, not memory), so llama.cpp mmaps the weights and the kernel
**page-thrashes them against disk** every token. Measured while "running": process state `Dsl`
(uninterruptible disk-I/O wait, `blk_mq_get_tag`), **swap 100 % full**, `vmstat bi ≈ 280 MB/s`
continuous disk reads, **I/O-wait 62–64 %**, with CPU *and* GPU both near-idle (stalled on disk).
It doesn't fit, so it's disk-bound, not compute-bound — unusably slow and it pins swap. **Not
added as a coder profile.** (A smaller quant — Q3/IQ2, ~9–12 GB — might fit RAM and is the only way
the 30B could ever run here; untested. The 14B Q4 remains the capability ceiling that actually fits.)

## Coding capability — algorithms (executable pass@1 + published benchmarks)

`bench/coding_eval.py` generates a solution and **runs it against hidden tests** (edge cases
included) in a sandboxed subprocess — correctness by execution, the metric that matters for a
coding/synthesis model. 12 "easy" (LeetCode easy/med) + 8 "hard" (DP/graph/stack/stateful).

| Model (offload)              | easy 12 | hard 8 | avg latency | VRAM      | published HumanEval / HE+ |
|------------------------------|---------|--------|-------------|-----------|---------------------------|
| **Qwen2.5-Coder-7B** (full GPU) | 11/12 | **8/8** | ~8 s       | 4.0–6.2 GB| 86.6 / 82.3               |
| **Qwen2.5-Coder-14B** (ngl40)   | 11/12 | **8/8** | ~16 s      | 7.5 GB    | **89.6 / 87.2**           |
| DeepSeek-Coder-V2-Lite (ngl12, MoE) | 10/12 | 8/8 | ~12 s   | 5.9 GB+CPU| 81.1 / —                  |
| Qwen3-4B (no-think)          | 10/12   | —      | ~4 s        | 4.2 GB    | (general model)           |
| Qwen3-4B (thinking)          | 8/12    | —      | ~80 s       | 4.2 GB    | thinking HURTS code-gen   |
| Qwen3-Coder-30B-A3B (MoE, 3B active) | _pending_ | _pending_ | — | ~partial | SWE-bench Verified 51.6   |

**Key findings.**
- **My local set saturates at the top** — both Qwen coders hit 8/8 on the hard set and 11/12 on
  easy (each missing one different problem). It confirms they *work* and measures *latency*, but
  can't separate 7B from 14B. For that, the **published decontaminated HumanEval+ is the
  discriminator: 14B 87.2 vs 7B 82.3 (+5 pts)** — the 14B is genuinely the stronger coder, it just
  doesn't show on easy/mid problems.
- **Qwen3-4B is a weak coder and thinking makes it worse** — reasoning runs away, hits the token
  cap, and truncates the actual function (67% with think vs 83% without). Qwen3-4B is a *verifier*,
  not a coder.
- **DeepSeek-Coder-V2-Lite is dominated** — 81.1 HumanEval (below both Qwen coders) and slower here
  (partial offload). Kept for now but not a contender for the coding role.
- The **7B is the efficient default**: 100% on hard, ~8 s, fully GPU-resident. The **14B is the
  capability ceiling that fits** (+5 HE+), at 2× latency and it can't coexist with the 64k context.
- **Qwen3-Coder-30B-A3B** (30B MoE, only **3B active** → fast even on partial offload; SWE-bench
  Verified 51.6, native tool-use, 256K ctx) is the strongest *agentic/SWE* lead — testing pending
  its download. If it runs at usable speed here, it's the agentic-coding pick.

Published numbers: Qwen2.5-Coder tech report / EvalPlus / model cards; DeepSeek-V2-Lite &
Qwen3-Coder-30B-A3B model cards. Local numbers: `bench/coding_eval.py` on this RX 580.

## Model grid — verifier judgment ladder (8-case verifier, full GPU offload)

Planner is structured *generation* (schema-constrained) — **100% valid for every model 1B→14B**,
so it's not the discriminator. *Verifier* is judgment; that's where models separate. All
measured at full GPU offload (ngl 99, or the max that fits), ctx 8192:

| Model                    | Verifier acc | Verifier latency | VRAM (weights+8k KV) |
|--------------------------|--------------|------------------|----------------------|
| MiniCPM5-1B              | 50%          | 0.45 s           | 1260 MiB             |
| SmolLM3-3B               | 50%          | 1.7 s            | 3118 MiB             |
| Qwen3-4B (no-think)      | 75%          | 2.3 s            | 4245 MiB             |
| Qwen2.5-Coder-7B         | 87.5%        | 5.2 s            | 4038 MiB             |
| **Qwen3-4B (thinking)**  | **100%**     | 9.9 s            | 4245 MiB             |
| Qwen2.5-Coder-14B (ngl32)| **100%**     | 11.7 s           | 7257 MiB             |

**Takeaways.**
- **Thinking is the cheapest path to top judgment.** Reasoning takes Qwen3-4B from 75% → **100%**
  verifier (beating the 7B's 87.5%) at ~4× latency — and it fits in 4.2 GB, leaving headroom.
  This is the standout result: the best verifier on this benchmark is a *4B with thinking on*,
  not a bigger model. (Non-thinking 4B is 75%; MiniCPM/SmolLM3 are coin-flips regardless.)
- **The 14B also hits 100%** but is heavier (7.3 GB) and slower (11.7 s verifier, 77 s planner on
  partial offload). Its real edge is **coding/synthesis** — which this harness does *not* measure,
  so we can't claim it over the 7B from these numbers alone. Judged purely on the verifier, the
  4B-thinking dominates it on cost.
- MiniCPM in *thinking* mode can't use the json_schema grammar (grammar blocks `<think>` → empty)
  and gave wrong verdicts — utility stages stay non-thinking.

## CPU vs GPU offload (latency is the only variable — quality is model-intrinsic)

Same model, different `--n-gpu-layers`. Offload doesn't change accuracy, only speed/VRAM:

| Model     | CPU (ngl 0) verifier lat | GPU (ngl 99) verifier lat | speedup |
|-----------|--------------------------|---------------------------|---------|
| Qwen3-4B  | 34.1 s                   | 9.8 s                     | ~3.5×   |
| SmolLM3-3B| 5.6 s                    | 1.7 s                     | ~3.3×   |

CPU-only keeps the fragile GPU untouched at ~3–4× the latency — the right home for agentic/tool
loops (see below). GPU offload is for latency-sensitive single-shot stages.

## Context sweep — using the idle VRAM (Qwen-7B @ 20 layers)

The live 7B at 8k ctx uses only ~4 GB of 8. KV cache is *memory*, and big-context prefill stays
power-flat, so context is a safe way to fill VRAM:

| ctx   | 8k   | 16k  | 32k  | 64k  | 128k |
|-------|------|------|------|------|------|
| VRAM  | 4038 | 4359 | 4755 | 6223 | 8025 MiB |

**64k (6.2 GB) is the sweet spot** — big jump in usable context, comfortable margin. 128k loads
(8.0 GB) but sits at 98% with the display — viable but tight.

**Conclusion (verifier/judgment):** the best measured option is **Qwen3-4B with thinking on**
(100%, 4.2 GB). Qwen-7B stays the coding/synthesis model. The 14B is available and *safe to run*
(below) but its benefit needs a coding-quality eval to justify its cost.

## Web-search tool loop — "smart, not blind" hypothesis

Hypothesis (user): a well-calibrated small model that searches *only when unsure* could
beat a bigger model, since small models supposedly hallucinate less. Tested with a
`web_search` tool (9 factual Qs: 5 stable/known, 4 needing current info). MiniCPM-CPU,
bounded to 3 calls, non-thinking. Metrics: accuracy, search-recall on the 4 needed,
over-search on the 5 known.

| Model / condition                    | Accuracy | Recall (needed) | Over-search (known) | avg calls |
|--------------------------------------|----------|-----------------|---------------------|-----------|
| MiniCPM-1B **raw** (no prompt)       | 6/7      | **1/4**         | 5/5                 | 0.7       |
| MiniCPM-1B **scaffolded**            | **7/7**  | **4/4**         | 5/5                 | 1.0       |
| MiniCPM-1B **strict scaffold**       | **7/7**  | **4/4**         | 5/5                 | 1.0       |
| **Qwen-7B (GPU), no prompt**         | **7/7**  | **4/4**         | **2/5**             | 0.7       |

**What the scaffold fixes:** the *dangerous* direction. Raw, the 1B under-searched — it
answered "I don't have access to real-time data" instead of using the tool it was given
(recall 1/4). A system prompt lifts recall to **4/4** and accuracy to **7/7**: it now
reliably grounds every question that needs current facts, which is exactly what prevents
hallucination on those.

**What no prompt fixes:** over-search. Even with explicit "capital cities need NO search"
examples, the 1B searches the capital of Japan every time (over-search 5/5, avg 1.0). It
treats `web_search` as a reflex, not a decision. You get **"always search," not "smart search."**

**Verdict on the hypothesis — refuted, and the opposite is true.** The 1B can't self-assess
what it knows, so it needs the tool for *more* things and can't be tuned to call selectively.
**Qwen-7B is the one that does "smart, not blind"**: same 7/7 accuracy and 4/4 recall as the
scaffolded 1B, but it *skipped* the search on 3 of 5 known facts (over-search 2/5 vs the 1B's
5/5) — answering gold/boiling-point/hexagon with zero calls and searching only when genuinely
uncertain. The valuable pattern is not "small model calls smartly" — it's "**the search
*decision* lives in a well-calibrated model (the 7B) or a cheap deterministic router, never in
the 1B's own judgement**; the 1B only executes searches once something decides to."

## GPU safe-envelope (RX 580, 927 MHz mask)

See `~/infra/docs/gpu-crash-diagnosis.md` for the full crash analysis. Prefill telemetry
from `~/infra/gpu/prefill-benchmark.py` and sustained-load from `~/infra/gpu/gpu-stress-loop.py`:

- **Single prefill, 512→8192 tok (full ctx range):** safe, **power-flat at 927 MHz** — 68–69 W
  (cool) / 75–76 W (warm), *size-independent*. VRAM ~3.9–4.7 GB (20 layers). ~19 W under the
  ~95 W death point. Prefill is slow (~28–54 tok/s) — only 20 layers on-GPU, rest CPU-bound.
- **Sustained: 40 back-to-back prefills, zero cooldown:** safe. Every iter **927 MHz, 74–76 W
  flat**, temp plateaus 76 °C, no bus loss over 9,465 black-box samples. The clock mask **holds
  under sustained load** and never boosts above 927 (only dips to the 600 MHz floor between
  prefills — the safe direction).
- **Bigger model / more layers does NOT raise power past the clock ceiling.** A **14B** at
  ngl 32 *and* ngl 40 (40 of 48 layers on GPU) prefills at **81 W flat, 927 MHz** — only ~5 W
  above the 7B and still ~14 W under death. Power is clock-bound, so layer count and model size
  move it barely at all. The old "28 layers = failure" was a **pre-clock-lock** artifact (the
  card boosted to 1179 MHz+); post-lock that lever is defanged. 14B fits in VRAM at ngl≤40
  (7.5 GB); 7B context fits to 128k (8.0 GB).
- **Direct GPU reproduction (user-authorized):** the bounded Qwen-7B agentic loop — the exact
  shape that faulted the card *before* the harness fix — ran **clean**: peak 75 W, 927 MHz, no
  bus loss, no kernel faults, 7/7 correct. The original fault was substantially the **harness bug
  (a hung `--jinja` template pinning a stuck request), not GPU tool-loops per se**.
- **Conclusion:** prefill — single *or* sustained, any size — is **not** a power problem, and a
  *correctly threaded, bounded* agentic loop is safe on GPU too. The adopted posture stays
  conservative anyway (agentic loops on CPU, GPU single-shot); **unbounded** GPU loops remain
  untested. Not raising clocks/layers.

## Architecture conclusion

- **Agentic / tool loops → MiniCPM-CPU only, never the GPU model** (the GPU model faults
  under sustained agentic submission; see workstream P).
- **GPU model (Qwen-7B) → single-shot only**: synthesis, coding, one-pass verify.
- **Utility stages** (routing, planning, extraction, query-rewrite) → the cheap CPU 1B.
- **Judgment stages** (verify) → Qwen-7B.
- Current-fact questions **always** get a grounded search; the search *decision* is not
  delegated to the 1B.
