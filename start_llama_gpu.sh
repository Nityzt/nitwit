#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/nit/.local/share/llama/models}"
MODEL_FILE="${MODEL_FILE:-qwen2.5-coder-7b-instruct-q4_k_m.gguf}"
IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-vulkan}"
GPU_LAYERS="${GPU_LAYERS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
UBATCH_SIZE="${UBATCH_SIZE:-32}"
CTX_SIZE="${CTX_SIZE:-4096}"

podman rm -f llama >/dev/null 2>&1 || true

exec podman run -d --rm \
  --name llama \
  -p 127.0.0.1:8080:8080 \
  -v "${MODEL_DIR}:/models:ro" \
  --device /dev/dri \
  --security-opt label=disable \
  "${IMAGE}" \
  --model "/models/${MODEL_FILE}" \
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers "${GPU_LAYERS}" \
  --ctx-size "${CTX_SIZE}" \
  --parallel 1 \
  --batch-size "${BATCH_SIZE}" \
  --ubatch-size "${UBATCH_SIZE}" \
  --threads 4 \
  --temp 0.30 \
  --top-p 0.90 \
  --alias qwen2.5-coder-7b
