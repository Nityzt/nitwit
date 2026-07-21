# Grounded Web Answers v3 — accuracy pipeline

**Problem (Image #6, reproduced):** chat web answers state fabricated specifics. For "latest one
piece manga updates" the model produced *"chapter 1187, releases July 5 2026"* — neither string
appears anywhere in the search results. Root cause is **fabrication over thin retrieval**: searxng
snippets are homepage blurbs that lack the actual fact, and the CPU 4B invents plausible specifics
to fill the gap. Prompt instructions ("only state what's in the results") do not reliably suppress
this on a 4B.

**Goal:** trade latency for accuracy. Put the *real* facts into context (deeper retrieval), use a
stronger synthesizer, and let a verifier catch ungrounded specifics before they reach the user.

## Pipeline (bounded — never a GPU tool-loop)

```
query
  → search        searxng, top ~6 (existing tools.web_search)
  → fetch         readable text from the top 2–3 result PAGES (new tools.fetch_url)
  → CONTEXT       search snippets + fetched page text + source URLs, capped for the 4096 ctx
  → synthesize    GPU Qwen2.5-Coder-7B (:8080), single-shot; strict "answer only from CONTEXT,
                  cite URLs, if a specific isn't present say so"   [CPU 4B fallback if :8080 down]
  → verify        CPU 4B, grammar-constrained JSON: {"unsupported": ["<claim>", ...]}
  → if unsupported and GPU synth was used → re-synthesize ONCE on GPU with the unsupported list
                  (cooldown between prefills); else hedge/strip on CPU
  → clean         dedup source URLs, collapse whitespace, drop leading cutoff disclaimer
  → emit
```

## GPU safety (RX 580, hard constraints)

- **At most two 7B prefills per web answer** (synthesis + at most one correction), never a loop.
- A **cooldown** (`NITWIT_GPU_COOLDOWN_S`, default 2.5s) between the two prefills. The card faults
  on *sustained back-to-back* prefills; single prefills at the 927MHz lock are power-flat (~68W).
  A cooldown keeps them as two isolated single prefills, within the proven-safe envelope.
- **CONTEXT capped** (~3 pages × ~1500 chars) so the prefill stays well under the 4096-token live
  profile (`--parallel 1`, `--ubatch-size 256`).
- If llama:8080 is not running (it does not auto-start), the health check fails and the **entire
  pipeline runs on the CPU 4B**; the correction step then hedges on CPU (no GPU prefill at all).
- `NITWIT_WEB_SYNTH=cpu` forces the CPU path (kill-switch if the card degrades further).

## Components / file structure

- `nitwit/tools.py` — add:
  - `fetch_url(url, *, timeout=6, max_chars=1500, _get=None) -> str` — GET, strip HTML to readable
    text via `html.parser`, cap, **never raises** (returns "" on any failure). `_get` injectable.
  - `gather_context(query, *, k_results=6, k_pages=3, _search=None, _fetch=None) -> dict` — returns
    `{"context": str, "sources": [url], "results": str}`; never raises.
- `nitwit/router.py` — add a `synth` stage → GPU 7B (:8080) with CPU-4B fallback; env overrides
  `NITWIT_SYNTH_URL` / `NITWIT_SYNTH_MODEL`. `route("synth")` health-gates to the CPU chat endpoint.
- `nitwit/webanswer.py` (new) — orchestrates the pipeline as pure-ish functions with injectable
  seams (no network in tests):
  - `synthesize(query, context, *, client, cite=True) -> str`
  - `verify_grounding(answer, context, *, client) -> list[str]` (structured JSON, never raises →
    `[]` on failure so verification failure never blocks an answer)
  - `clean(answer, sources) -> str` (dedup URLs, whitespace, strip lead disclaimer)
  - `answer_web(query, *, out, route, factory, search, fetch, cooldown) -> str` — the whole thing;
    honors the GPU-prefill cap + cooldown + CPU fallback; **never raises**.
- `nitwit/session.py` — when a search fires (proactive OR model-decided `SEARCH:`), delegate to
  `webanswer.answer_web` instead of the current inline single-synth. Non-search chat is unchanged.

## Verifier contract

Grammar-constrained (reuse the project's `response_format` json_schema path). Prompt: "Here is
CONTEXT and an ANSWER. List every specific factual claim in ANSWER (numbers, dates, names, versions)
that is NOT supported by CONTEXT. If all claims are supported, return an empty list." Output schema
`{"unsupported": [string]}`. Runs on CPU 4B (verify stage). On any error/malformed output → `[]`
(fail-open: never block a usable answer on a flaky verifier).

## Correction / hedge

- **GPU path chosen (this build):** if `unsupported` is non-empty, re-synthesize once on GPU with an
  added system line: "The following statements were NOT supported by CONTEXT — remove them or
  replace them only with facts actually present in CONTEXT: <list>." One extra prefill, after the
  cooldown. Then re-verify is **skipped** (bounded: no loop); remaining unsupported specifics are
  stripped by `clean`'s sentence filter as a floor.
- **CPU path (fallback / kill-switch):** skip re-synth; `clean` strips/hedges the unsupported
  sentences directly.

## Never-raises / non-regression

- Every new function returns a safe value on failure; `answer_web` degrades to
  `search → single synth → clean` and, worst case, to showing the raw results — chat never breaks.
- Existing `stream_answer` seams (`_endpoint`, `_client_factory`, `_search_fn`, `memories`,
  `allow_search`, history) preserved. Non-search chat path and its tests unchanged.
- Bounded: one search, one fetch batch (≤3 pages), ≤2 GPU prefills, one verify. No agentic loop.

## Tests (unittest, root `test_nitwit_*.py`, no network)

- `fetch_url`: extracts text from sample HTML, strips tags/scripts, caps length, returns "" on
  raising `_get`.
- `gather_context`: builds CONTEXT + sources from injected search/fetch; never raises on failures.
- `router`: `route("synth")` picks GPU 7B when healthy, CPU chat when :8080 down; env overrides.
- `webanswer`: synthesize uses injected client; verify_grounding parses JSON → list, fails open to
  `[]`; clean dedups URLs + strips disclaimer; `answer_web` — (a) clean answer passes through,
  (b) unsupported claim triggers exactly one GPU re-synth then stops, (c) CPU path hedges without a
  GPU call, (d) cooldown invoked between the two GPU prefills (inject a fake sleep, assert called
  once), (e) whole thing never raises when every stage fails.
- `session`: a triggered search delegates to `answer_web`; non-search chat unchanged.

## Not doing (later)

- Multi-hop / iterative retrieval (bounded single pass only — GPU safety).
- A general (non-coder) GPU 7B; reuse whatever is on :8080.
- Caching fetched pages across turns.
