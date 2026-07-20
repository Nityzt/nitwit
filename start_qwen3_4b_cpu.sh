#!/usr/bin/env bash
# Qwen3-4B on CPU — the orchestrator's VERIFY specialist. Benchmarks put it at 100% on the
# 8-case verifier with reasoning on (beating the 7B and 14B), fitting entirely in system RAM
# (~2.4 GB), so the fragile RX 580 stays free for the coder slot. Loopback only, on-demand.
# Thinking is a per-REQUEST flag (chat_template_kwargs.enable_thinking), not a server flag,
# so the orchestrator decides think-vs-fast per call; this just serves the model.
#   ./start_qwen3_4b_cpu.sh                 # 8k ctx verify server on :8086
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/nit/.local/share/llama/models}"
MODEL_FILE="${MODEL_FILE:-Qwen3-4B-Q4_K_M.gguf}"
IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-vulkan}"
CTX_SIZE="${CTX_SIZE:-8192}"
THREADS="${THREADS:-6}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PORT="${PORT:-8086}"

podman rm -f qwen3-verifier >/dev/null 2>&1 || true

# No --device /dev/dri and -ngl 0 -> pure CPU; the GPU is untouched.
exec podman run -d --rm \
  --name qwen3-verifier \
  -p 127.0.0.1:${PORT}:${PORT} \
  -v "${MODEL_DIR}:/models:ro" \
  "${IMAGE}" \
  --model "/models/${MODEL_FILE}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --n-gpu-layers 0 \
  --ctx-size "${CTX_SIZE}" \
  --parallel 1 \
  --batch-size "${BATCH_SIZE}" \
  --threads "${THREADS}" \
  --jinja \
  --alias qwen3-4b
