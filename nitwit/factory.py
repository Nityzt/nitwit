"""Factory: wire a MissionEngine to the live local models (GPU coder + CPU verifier)."""
from __future__ import annotations

import urllib.request

from orchestrator import OpenAICompatibleClient
from nitwit.engine import MissionEngine
from nitwit.missions import MissionStore
from nitwit.model_coder import ModelCoder
from nitwit.model_verifier import ModelVerifier


def endpoint_healthy(base_url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=timeout) as res:
            return res.status == 200
    except Exception:
        return False


def build_model_engine(store: MissionStore, *,
                       coder_url: str = "http://127.0.0.1:8080",
                       coder_model: str = "qwen2.5-coder-7b",
                       verifier_url: str = "http://127.0.0.1:8086",
                       verifier_model: str = "qwen3-4b",
                       max_iterations: int = 12,
                       cooldown_s: float = 0.0) -> MissionEngine:
    coder = ModelCoder(OpenAICompatibleClient(coder_url, coder_model))
    # The verifier is a thinking model; give it headroom so <think> + JSON both fit.
    verifier = ModelVerifier(OpenAICompatibleClient(verifier_url, verifier_model), max_tokens=1000)
    return MissionEngine(store, coder, verifier, max_iterations=max_iterations, cooldown_s=cooldown_s)
