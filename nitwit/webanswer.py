"""Grounded web-answer pipeline: search → fetch pages → synthesize (GPU 7B) → verify grounding
(CPU 4B) → correct once (GPU) or hedge (CPU) → clean. Bounded — never a GPU tool-loop:

  * at most TWO 7B prefills per answer (synthesis + at most one correction), never a loop;
  * a cooldown between the two prefills so they stay isolated single prefills (the RX 580 faults on
    sustained back-to-back prefills, not on single ones);
  * the whole thing runs on the CPU 4B if llama:8080 is down, and NITWIT_WEB_SYNTH=cpu forces it.

Every function returns a safe value on failure; `answer_web` NEVER raises and always emits something.
"""
from __future__ import annotations

import os
import re
import time

_GROUNDING_SYSTEM = (
    "You are Nitwit, a local self-hosted assistant with live web access. Answer the user's question "
    "ONLY using the CONTEXT below, which was fetched from the web just now and is current. Cite the "
    "source URLs inline. If CONTEXT does not contain a specific fact the user asked for — an exact "
    "number, date, version, chapter/episode number, or name — say plainly that the sources don't "
    "state it rather than guessing or inventing one. Do NOT add disclaimers about knowledge cutoffs, "
    "training dates, or being unable to search in real time; treat CONTEXT as present fact."
)

_VERIFY_FORMAT = {"type": "json_schema", "json_schema": {"name": "grounding", "schema": {
    "type": "object",
    "properties": {"unsupported": {"type": "array", "items": {"type": "string"}}},
    "required": ["unsupported"],
}}}

_MAX_CONTEXT = 6000


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
                         "content": "Your previous answer contained statements NOT supported by "
                                    "CONTEXT. Remove them, or replace them only with facts actually "
                                    "present in CONTEXT. Unsupported statements:\n"
                                    + "\n".join(f"- {c}" for c in correction)})
    messages.append({"role": "user", "content": query})
    try:
        return _content(client.chat(messages, temperature=0.1 if correction else 0.2, max_tokens=700))
    except Exception:
        return ""


def verify_grounding(answer: str, context: str, *, client) -> list[str]:
    """Return the specific claims in `answer` NOT supported by `context`. Fails OPEN: any error or
    malformed output yields [] so a flaky verifier never blocks a usable answer."""
    if not (answer or "").strip():
        return []
    messages = [
        {"role": "system",
         "content": "You check whether an ANSWER is grounded in CONTEXT. List every specific factual "
                    "claim in ANSWER — exact numbers, dates, versions, chapter/episode numbers, names "
                    "— that is NOT explicitly supported by CONTEXT. If every specific claim is "
                    "supported, return an empty list. Return JSON only."},
        {"role": "system", "content": "CONTEXT:\n" + (context or "")[:_MAX_CONTEXT]},
        {"role": "user", "content": "ANSWER:\n" + answer},
    ]
    try:
        from orchestrator import extract_json
        resp = client.chat(messages, temperature=0.0, max_tokens=300, response_format=_VERIFY_FORMAT)
        data = extract_json(_content(resp))
        items = data.get("unsupported") if isinstance(data, dict) else None
        if isinstance(items, list):
            return [str(x).strip() for x in items if str(x).strip()]
    except Exception:
        pass
    return []


def _hedge(answer: str, unsupported: list[str]) -> str:
    """CPU fallback: drop sentences carrying an unsupported specific (a number or 'Month DD')."""
    bad = set()
    for u in unsupported:
        for tok in re.findall(r"\b\d[\d.,/:-]*\b|\b[A-Z][a-z]+ \d{1,2}\b", u):
            bad.add(tok.lower())
    if not bad:
        return answer
    kept = [s for s in re.split(r"(?<=[.!?])\s+|\n", answer) if not any(t in s.lower() for t in bad)]
    out = " ".join(s for s in kept if s.strip()).strip()
    return out or ("The sources I found don't state those specifics — you may want to open the cited "
                   "pages directly.")


def clean(answer: str, sources=None) -> str:
    """Drop any leading cutoff/real-time disclaimer, collapse whitespace. Returns tidy text."""
    from nitwit.session import _strip_lead_disclaimer
    text = _strip_lead_disclaimer(answer or "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def answer_web(query, *, out, route, factory, search=None, fetch=None,
               cooldown=2.5, sleep=None, allow_gpu_correct=True):
    """Run the full grounded pipeline for `query`, streaming a '[searching…]' note then the final
    answer to `out`. Returns the answer text. NEVER raises; degrades to the raw results worst-case."""
    sleep = sleep or time.sleep
    from nitwit import tools
    out("[searching the web…]\n")
    ctx = tools.gather_context(query, _search=search, _fetch=fetch)
    context, sources = ctx["context"], ctx["sources"]

    force_cpu = os.environ.get("NITWIT_WEB_SYNTH", "").lower() == "cpu"
    try:
        synth_ep = route("chat" if force_cpu else "synth")
    except Exception:
        synth_ep = route("chat")
    on_gpu = (not force_cpu) and "8080" in synth_ep.base_url
    client = factory(synth_ep.base_url, synth_ep.model, extra_body=synth_ep.extra_body)

    answer = synthesize(query, context, client=client)

    try:
        ver_ep = route("verify")
        vclient = factory(ver_ep.base_url, ver_ep.model, extra_body=ver_ep.extra_body)
        unsupported = verify_grounding(answer, context, client=vclient)
    except Exception:
        unsupported = []

    if unsupported:
        if on_gpu and allow_gpu_correct:
            sleep(cooldown)                       # keep the 2 GPU prefills as isolated singles
            corrected = synthesize(query, context, client=client, correction=unsupported)
            if corrected.strip():
                answer = corrected                # bounded: no re-verify, no loop
            else:
                answer = _hedge(answer, unsupported)
        else:
            answer = _hedge(answer, unsupported)  # CPU path: no GPU prefill

    final = clean(answer, sources)
    if not final:
        final = (ctx["results"] or "(no results)")   # worst case: show the raw results
    out(final)
    return final
