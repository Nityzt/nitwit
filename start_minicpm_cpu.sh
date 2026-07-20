#!/usr/bin/env bash
# MiniCPM5-1B on CPU — a utility co-model with a 128K context, kept OFF the fragile
# RX 580 entirely (no --device, -ngl 0). Binds loopback only. On-demand, not enabled.
#   ./start_minicpm_cpu.sh                    # 8k ctx utility server
#   CTX_SIZE=131072 ./start_minicpm_cpu.sh    # long-context sweep (big KV in system RAM)
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/nit/.local/share/llama/models}"
MODEL_FILE="${MODEL_FILE:-MiniCPM5-1B-Q4_K_M.gguf}"
IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-vulkan}"
CTX_SIZE="${CTX_SIZE:-8192}"
THREADS="${THREADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"

podman rm -f minicpm >/dev/null 2>&1 || true

# No --device /dev/dri and --n-gpu-layers 0 -> pure CPU; the fragile GPU is untouched.
exec podman run -d --rm \
  --name minicpm \
  -p 127.0.0.1:8081:8081 \
  -v "${MODEL_DIR}:/models:ro" \
  "${IMAGE}" \
  --model "/models/${MODEL_FILE}" \
  --host 0.0.0.0 \
  --port 8081 \
  --n-gpu-layers 0 \
  --ctx-size "${CTX_SIZE}" \
  --parallel 1 \
  --batch-size "${BATCH_SIZE}" \
  --threads "${THREADS}" \
  --jinja \
  --alias minicpm5-1b
