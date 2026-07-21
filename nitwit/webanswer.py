"""Grounded web-answer loop: search → fetch pages → synthesize → verify grounding → self-correct,
repeating until the answer is grounded in the fetched sources or the iteration budget runs out.

Design goals (per the Nitwit vision): a general self-correcting research loop that returns an
accurate, source-grounded answer regardless of latency. No hardcoded fact/date/number gates — the
verifier is a model judging the answer against the fetched CONTEXT, and correction is another
synthesis pass, so the same machinery works for any topic.

GPU safety (RX 580): synthesis may run on the GPU 7B (route('synth')). The card hard-faults on a
cold compute submission and on sustained back-to-back prefills, so this module (a) warms the GPU
with a tiny prefill before the first real synthesis, and (b) sleeps a cooldown between successive
GPU prefills, keeping each as an isolated single prefill (proven power-flat at the 927MHz lock).
If the synth endpoint is the CPU model, warm-up and cooldown are skipped. Every function returns a
safe value on failure; `answer_web` NEVER raises.
"""
from __future__ import annotations

import os
import re
import subprocess
import time

_MAX_CONTEXT = 6000

_GROUNDING_SYSTEM = (
    "You are Nitwit, a local self-hosted assistant with live web access. Answer the user's question "
    "using ONLY the CONTEXT below, which was fetched from the web moments ago and is current. "
    "Ground every specific — each number, date, version, name, quantity — in CONTEXT: state a "
    "specific ONLY if that exact value appears in CONTEXT. If CONTEXT does not contain a specific "
    "the user asked for, say plainly that the sources don't state it; never guess, infer, estimate, "
    "or invent one (especially dates). Cite the source URLs inline. Do NOT add disclaimers about "
    "knowledge cutoffs, training dates, or being unable to search in real time; treat CONTEXT as "
    "present fact."
)

# The verifier enumerates each specific claim and judges it against CONTEXT, quoting the supporting
# text. Enumerate-and-quote is far steadier than "list the unsupported ones" on a small model.
_VERIFY_SYSTEM = (
    "You verify an ANSWER against CONTEXT. Enumerate EVERY specific factual token in ANSWER — each "
    "number, date, version, chapter/episode number, name, quantity, or superlative ('the latest', "
    "'the newest'). For each, search CONTEXT for the exact value: put the supporting CONTEXT "
    "substring in \"quote\" and set supported=true ONLY if such a substring truly exists; otherwise "
    "set supported=false and quote=\"\". A date/number is supported only if that exact value appears "
    "in CONTEXT. Judge only against CONTEXT, never prior knowledge. Return JSON only."
)

_VERIFY_FORMAT = {"type": "json_schema", "json_schema": {"name": "grounding", "schema": {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {
        "type": "object",
        "properties": {"claim": {"type": "string"}, "quote": {"type": "string"},
                       "supported": {"type": "boolean"}},
        "required": ["claim", "supported"],
    }}},
    "required": ["claims"],
}}}


def _content(resp) -> str:
    return (getattr(resp, "content", "") or "").strip()


def synthesize(query: str, context: str, *, client, correction=None) -> str:
    """One grounded synthesis (or correction) pass. Returns "" on failure (never raises)."""
    messages = [
        {"role": "system", "content": _GROUNDING_SYSTEM},
        {"role": "system", "content": "CONTEXT:\n" + (context or "")[:_MAX_CONTEXT]},
    ]
    if correction:
        messages.append({"role": "system",
                         "content": "Your previous answer stated these specifics that are NOT "
                                    "supported by CONTEXT. Remove each one, or replace it only with a "
                                    "value that actually appears in CONTEXT. If CONTEXT has no such "
                                    "value, say the sources don't state it. Unsupported:\n"
                                    + "\n".join(f"- {c}" for c in correction)})
    messages.append({"role": "user", "content": query})
    try:
        return _content(client.chat(messages, temperature=0.1 if correction else 0.2, max_tokens=700))
    except Exception:
        return ""


def verify_grounding(answer: str, context: str, *, client) -> list[str]:
    """Return the specific claims in `answer` NOT supported by `context`, via an enumerate-and-quote
    model check. Fails OPEN: any error/malformed output yields [] so a flaky verifier never blocks
    a usable answer (the loop simply stops correcting)."""
    if not (answer or "").strip():
        return []
    messages = [
        {"role": "system", "content": _VERIFY_SYSTEM},
        {"role": "system", "content": "CONTEXT:\n" + (context or "")[:_MAX_CONTEXT]},
        {"role": "user", "content": "ANSWER:\n" + answer},
    ]
    try:
        from orchestrator import extract_json
        resp = client.chat(messages, temperature=0.0, max_tokens=600, response_format=_VERIFY_FORMAT)
        data = extract_json(_content(resp))
        claims = data.get("claims") if isinstance(data, dict) else None
        if isinstance(claims, list):
            return [str(c.get("claim", "")).strip() for c in claims
                    if isinstance(c, dict) and c.get("supported") is False and str(c.get("claim", "")).strip()]
    except Exception:
        pass
    return []


def clean(answer: str) -> str:
    """Drop any leading cutoff/real-time disclaimer, collapse whitespace. Returns tidy text."""
    from nitwit.session import _strip_lead_disclaimer
    text = _strip_lead_disclaimer(answer or "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _warmup(client) -> bool:
    """Wake the GPU with a tiny prefill before the first real synthesis. Returns True if the
    endpoint responded. Never raises."""
    try:
        client.chat([{"role": "user", "content": "Reply with: ok"}], temperature=0.0, max_tokens=2)
        return True
    except Exception:
        return False


def _gpu_undervolt_active(_probe=None) -> bool:
    """The RX 580 needs CoreCtrl's undervolt applied before any 927 MHz GPU compute, or it faults
    and drops off the PCIe bus (incident 2026-07-21). CoreCtrl applies the undervolt at login, so
    pre-login / headless there is none — and the 24/7 daemon can run then. Gate GPU synthesis on
    CoreCtrl being active; without it, synthesis falls back to the CPU model. Override with
    NITWIT_GPU_UNPROTECTED_OK=1 when the undervolt is applied some other way. Never raises."""
    if os.environ.get("NITWIT_GPU_UNPROTECTED_OK") == "1":
        return True
    probe = _probe or (lambda: subprocess.run(["pgrep", "-x", "corectrl"],
                                              capture_output=True).returncode == 0)
    try:
        return bool(probe())
    except Exception:
        return False


def answer_web(query, *, out, route, factory, search=None, fetch=None,
               max_iters=3, cooldown=2.0, sleep=None, warmup=None, gpu_ok=None):
    """Search, fetch pages, then synthesize and self-correct in a loop until the answer is grounded
    in CONTEXT or `max_iters` verification passes are spent. Streams a '[searching…]' note then the
    final answer to `out`; returns the answer text. NEVER raises.

    GPU: if route('synth') resolves to the GPU, warms it once and sleeps `cooldown` between
    successive GPU synthesis prefills. On the CPU model both are skipped.
    """
    sleep = sleep or time.sleep
    warmup = warmup or _warmup
    gpu_ok = gpu_ok or _gpu_undervolt_active
    from nitwit import tools
    out("[searching the web…]\n")
    ctx = tools.gather_context(query, _search=search, _fetch=fetch)
    context = ctx["context"]

    synth_ep = route("synth")
    on_gpu = "8080" in synth_ep.base_url
    if on_gpu and not gpu_ok():
        # No CoreCtrl undervolt applied → GPU compute would fault the card. Use the CPU model.
        synth_ep = route("chat")
        on_gpu = "8080" in synth_ep.base_url
    client = factory(synth_ep.base_url, synth_ep.model, extra_body=synth_ep.extra_body)
    if on_gpu:
        warmup(client)                                   # wake the GPU before the first real prefill

    # Synthesis AND verification run on the same model: the 7B is a markedly more reliable judge
    # than the 4B (bench: 87.5% vs 75%) and, on the GPU, fast enough to loop. A cooldown before
    # every GPU prefill keeps each an isolated single prefill (the card faults on sustained
    # back-to-back prefills, not on isolated ones).
    def gpu_gap():
        if on_gpu:
            sleep(cooldown)

    answer = synthesize(query, context, client=client)   # prefill #1
    checks = 0
    while checks < max_iters:
        gpu_gap()
        unsupported = verify_grounding(answer, context, client=client)   # prefill
        checks += 1
        if not unsupported:
            break                                        # grounded → done
        if checks >= max_iters:
            break                                        # correction budget spent; keep best effort
        gpu_gap()
        corrected = synthesize(query, context, client=client, correction=unsupported)  # prefill
        if not corrected.strip():
            break                                        # correction failed; keep prior answer
        answer = corrected

    final = clean(answer)
    if not final:
        final = ctx["results"] or "(no results)"         # worst case: show the raw results
    out(final)
    return final
