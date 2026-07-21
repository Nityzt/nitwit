"""Device-split router: map a work 'stage' to the best-fit local model endpoint, balancing
CPU/GPU. Health-checked with fallback to the GPU coder so a down CPU service never breaks chat."""
from __future__ import annotations

import copy
import os
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Endpoint:
    base_url: str
    model: str
    extra_body: dict = field(default_factory=dict)


def _env(url_key, url_default, model_key, model_default):
    return (os.environ.get(url_key, url_default), os.environ.get(model_key, model_default))


def _build_defaults() -> dict[str, Endpoint]:
    chat_url, chat_model = _env("NITWIT_CHAT_URL", "http://127.0.0.1:8086", "NITWIT_CHAT_MODEL", "qwen3-4b")
    util_url, util_model = _env("NITWIT_UTIL_URL", "http://127.0.0.1:8081", "NITWIT_UTIL_MODEL", "minicpm5-1b")
    code_url, code_model = _env("NITWIT_CODE_URL", "http://127.0.0.1:8080", "NITWIT_CODE_MODEL", "qwen2.5-coder-7b")
    ver_url, ver_model = _env("NITWIT_VERIFY_URL", "http://127.0.0.1:8086", "NITWIT_VERIFY_MODEL", "qwen3-4b")
    synth_url, synth_model = _env("NITWIT_SYNTH_URL", "http://127.0.0.1:8080", "NITWIT_SYNTH_MODEL", "qwen2.5-coder-7b")
    nothink = {"chat_template_kwargs": {"enable_thinking": False}}
    return {
        "chat": Endpoint(chat_url, chat_model, copy.deepcopy(nothink)),
        "utility": Endpoint(util_url, util_model, copy.deepcopy(nothink)),
        "code": Endpoint(code_url, code_model, {}),
        "verify": Endpoint(ver_url, ver_model, {}),
        # web-answer synthesis: the GPU 7B, but falls back to the CPU chat model (not the coder)
        # so a stopped llama:8080 degrades to a usable answer instead of a dead endpoint.
        "synth": Endpoint(synth_url, synth_model, {}),
    }


# Which stage each stage falls back to when its own endpoint is down. Unlisted stages fall back to
# the GPU coder (the historical default); synth falls back to the CPU chat model instead.
_STAGE_FALLBACK = {"synth": "chat"}


STAGE_DEFAULTS = _build_defaults()


def _default_health(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def route(stage: str, *, health=_default_health) -> Endpoint:
    ep = STAGE_DEFAULTS.get(stage) or STAGE_DEFAULTS["code"]
    if health(ep.base_url):
        return ep
    fb = STAGE_DEFAULTS.get(_STAGE_FALLBACK.get(stage, "code"))
    if not fb or fb.base_url == ep.base_url or not health(fb.base_url):
        return ep  # nothing better to fall back to; return the original (caller handles failure)
    return Endpoint(fb.base_url, fb.model, copy.deepcopy(fb.extra_body))
