#!/usr/bin/env bash
# MiniCPM5-1B fully offloaded to the RX 580 (all layers). Only safe because the
# 927 MHz clock lock makes prefill power-flat (~69 W) regardless of size — verified
# by ~/infra/gpu/prefill-benchmark.py. Do NOT run this alongside a large-ctx Qwen:
# it competes for the 8 GB. Loopback only, on-demand, port 8082.
#   ./start_minicpm_gpu.sh                    # 8k ctx
#   CTX_SIZE=32768 ./start_minicpm_gpu.sh     # bigger (watch VRAM if Qwen is also resident)
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/nit/.local/share/llama/models}"
MODEL_FILE="${MODEL_FILE:-MiniCPM5-1B-Q4_K_M.gguf}"
IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-vulkan}"
CTX_SIZE="${CTX_SIZE:-8192}"
GPU_LAYERS="${GPU_LAYERS:-99}"     # 1B is tiny; offload everything
BATCH_SIZE="${BATCH_SIZE:-128}"
UBATCH_SIZE="${UBATCH_SIZE:-32}"

podman rm -f minicpm-gpu >/dev/null 2>&1 || true

exec podman run -d --rm \
  --name minicpm-gpu \
  -p 127.0.0.1:8082:8082 \
  -v "${MODEL_DIR}:/models:ro" \
  --device /dev/dri \
  --security-opt label=disable \
  "${IMAGE}" \
  --model "/models/${MODEL_FILE}" \
  --host 0.0.0.0 \
  --port 8082 \
  --n-gpu-layers "${GPU_LAYERS}" \
  --ctx-size "${CTX_SIZE}" \
  --parallel 1 \
  --batch-size "${BATCH_SIZE}" \
  --ubatch-size "${UBATCH_SIZE}" \
  --threads 4 \
  --alias minicpm5-1b
