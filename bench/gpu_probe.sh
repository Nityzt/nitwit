#!/usr/bin/env bash
# Launch a throwaway GPU llama.cpp server for one (model, layers, ctx) config,
# wait for it to load, report VRAM used + load time, then tear it down. Used by the
# model/context/offload sweeps so each config is measured on identical footing.
#
# Run ONE at a time — two GPU servers won't fit in 8 GB. The live `llama` service
# must be stopped first (systemctl --user stop qwen-llama).
#
#   bench/gpu_probe.sh <model_file> <ngl> <ctx> [port]
# prints:  VRAM_USED_MIB=<n>  LOAD_S=<n>  OK|FAIL
set -uo pipefail

MODEL_FILE="$1"; NGL="$2"; CTX="$3"; PORT="${4:-8083}"
MODEL_DIR="/home/nit/.local/share/llama/models"
IMAGE="ghcr.io/ggml-org/llama.cpp:server-vulkan"
NAME="gpuprobe"
CARD=/sys/class/drm/card0/device

vram_mib() { echo $(( $(cat "$CARD/mem_info_vram_used") / 1048576 )); }

podman rm -f "$NAME" >/dev/null 2>&1 || true
base_vram=$(vram_mib)

podman run -d --rm --name "$NAME" \
  -p 127.0.0.1:${PORT}:${PORT} \
  -v "${MODEL_DIR}:/models:ro" \
  --device /dev/dri --security-opt label=disable \
  "$IMAGE" \
  --model "/models/${MODEL_FILE}" --host 0.0.0.0 --port "${PORT}" \
  --n-gpu-layers "${NGL}" --ctx-size "${CTX}" --parallel 1 \
  --batch-size 128 --ubatch-size 32 --threads 4 --alias probe >/dev/null 2>&1

started=$(date +%s)
ok=FAIL
for _ in $(seq 1 90); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:${PORT}/health 2>/dev/null)" = "200" ]; then ok=OK; break; fi
  # bail early if the container died (e.g. OOM)
  if ! podman ps --format '{{.Names}}' | grep -q "$NAME"; then ok=FAIL; break; fi
  sleep 1
done
load_s=$(( $(date +%s) - started ))

peak_vram=0
if [ "$ok" = OK ]; then
  # sample VRAM a few times to catch full KV allocation
  for _ in $(seq 1 5); do v=$(vram_mib); [ "$v" -gt "$peak_vram" ] && peak_vram=$v; sleep 0.3; done
fi
model_vram=$(( peak_vram - base_vram ))
[ "$model_vram" -lt 0 ] && model_vram=$peak_vram

echo "VRAM_USED_MIB=${peak_vram} MODEL_VRAM_MIB=${model_vram} LOAD_S=${load_s} ${ok}"
# leave it running only if caller wants to probe further; default tear down
if [ "${KEEP:-0}" != "1" ]; then podman rm -f "$NAME" >/dev/null 2>&1 || true; fi
