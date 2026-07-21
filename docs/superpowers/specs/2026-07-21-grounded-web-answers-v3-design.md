# Grounded Web Answers — self-correcting loop

**Problem (Image #6, reproduced):** chat web answers state fabricated specifics. "latest one piece
updates" → *"chapter 1187, releases July 5 2026"*; `1187` was in the fetched page but `July 5` was
invented. Root cause: the model fills gaps with plausible specifics, and a single synth pass has no
check.

**Goal (per the Nitwit vision):** a general, model-verified, *self-correcting* research loop that
returns an accurate, source-grounded answer for any topic — accuracy over latency. **No hardcoded
fact/date/number gates**: the verifier is a model judging the answer against the fetched sources,
and correction is another synthesis pass, so the same machinery generalizes.

## The loop (`nitwit/webanswer.py :: answer_web`)

```
context = gather_context(query)          # search (searxng) + fetch top result PAGES
warm the GPU (tiny prefill) if synth is on the GPU
answer = synthesize(query, context)      # grounded prompt: only state specifics present in CONTEXT
repeat up to max_iters (default 3):
    unsupported = verify_grounding(answer, context)   # model enumerates each specific, judges vs CONTEXT
    if not unsupported: break                          # grounded → done
    answer = synthesize(query, context, correction=unsupported)   # self-correct
clean(answer)                            # strip any leading "can't search" disclaimer, tidy
```

- **Synthesis + verification both run on the GPU 7B** (`route("synth")` → :8080). The 7B is a
  markedly more reliable judge than the 4B (bench: 87.5% vs 75%) and, on the GPU, fast enough to
  loop (~40–90 s/query end-to-end). If :8080 is down / `NITWIT_SYNTH_URL` points at CPU, the whole
  loop runs on the CPU model instead.
- **Verifier is enumerate-and-judge**, JSON-constrained: it lists every specific token (number,
  date, version, name, superlative) and marks each supported/unsupported against CONTEXT. Fails
  OPEN (`[]`) on any error so a flaky judge never blocks an answer.
- **Correction** feeds the unsupported list back: "remove these or replace only with values in
  CONTEXT." Bounded by `max_iters`; on budget exhaustion the best-effort answer is returned (no
  hardcoded stripping).

## GPU safety (RX 580) — the hard-won part

Incident 2026-07-21: an earlier GPU pipeline lost the card from the PCIe bus (`ring comp timeout` →
`device lost from bus` → `ret=-19`, **hard restart required**). Investigation with `gpu-blackbox`
telemetry (the instrumentation prior crashes lacked) established:

- The crash was a **cold-start** compute-ring timeout (first heavy prefill on an idle/cold SMU),
  on a **drifted llama config** (20-layer / 64k / ubatch-32, not the stable 99-layer / 4k /
  ubatch-256 quadlet). It was **not** an inherent property of GPU synthesis.
- **Warmed up, the whole loop is power-flat and safe:** across 5 diverse queries (each doing
  synth + verify + correction prefills), peak **77 W** at the 927 MHz lock — ~18 W below the ~95 W
  death point — device healthy, zero ring errors.

Mitigations, in code:
- **Warm-up guard:** a tiny prefill wakes the GPU before the first real synthesis (cold-start
  guard). `_warmup` never raises.
- **Cooldown before every GPU prefill** (`cooldown`, default 2 s): power drops to ~29 W idle
  between prefills, keeping each an *isolated single prefill*. The card faults on sustained
  back-to-back prefills, not on isolated ones.
- **Bounded** (`max_iters`): no unbounded GPU loop.
- CPU path (synth on the CPU model) skips warm-up and cooldown entirely.

## Components

- `nitwit/tools.py` — `web_search` (searxng), `fetch_url` (HTML→text, capped, never raises),
  `gather_context` (search + fetch top pages → CONTEXT + sources).
- `nitwit/router.py` — `synth` stage → GPU 7B (:8080), env-overridable, falls back to the CPU chat
  model (never the coder) if :8080 is down.
- `nitwit/webanswer.py` — `synthesize`, `verify_grounding`, `clean`, `_warmup`, `answer_web` (the
  loop). All return safe values on failure; `answer_web` never raises.
- `nitwit/session.py` — `stream_answer` delegates to `answer_web` whenever a search fires
  (proactive heuristic OR the model emitting `SEARCH:`); ordinary chat still streams live on CPU.

## Verified (live, this session)

- 5 diverse queries (python version, one piece chapter, CEO of X, newest iphone, react version):
  **4/5 fully grounded**, no fabricated specifics; 1 residual ungrounded date the 7B judge missed
  on one pass (recall limit, not systemic). GPU peak 77 W, no crash.
- Full unit suite green (`python3 -m unittest discover`).

## Known limits / follow-ons

- Verifier recall isn't perfect — an occasional ungrounded date slips a single pass. Reduce with
  verifier self-consistency (2 samples, union) or a couple more `max_iters`; both cost extra GPU
  prefills (safe but slower).
- Groundedness ≠ source truth: the loop makes the answer match the fetched sources, not fact-check
  the sources themselves.
- Re-search on gaps (refine the query when CONTEXT lacks the asked-for fact) is a natural extension
  of the loop, not yet implemented.
