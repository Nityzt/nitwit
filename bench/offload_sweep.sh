#!/usr/bin/env bash
# Sweep a model across CPU/GPU offload levels, measuring VRAM + per-stage latency
# and accuracy (via bench.run_bench). Quality is model-intrinsic (offload doesn't
# change it) — the point is the latency/VRAM cost of each split. One server at a time.
#
#   bench/offload_sweep.sh <model_file> <label> "<ngl_list>" [ctx]
#     e.g. bench/offload_sweep.sh Qwen3-4B-Q4_K_M.gguf qwen3-4b "0 20 99" 8192
set -uo pipefail

MODEL_FILE="$1"; LABEL="$2"; NGLS="$3"; CTX="${4:-8192}"
MODEL_DIR="/home/nit/.local/share/llama/models"
IMAGE="ghcr.io/ggml-org/llama.cpp:server-vulkan"
PORT=8083; NAME=gpuprobe
CARD=/sys/class/drm/card0/device
vram_mib(){ echo $(( $(cat "$CARD/mem_info_vram_used") / 1048576 )); }
cd /home/nit/qwen-orchestrator

for ngl in $NGLS; do
  podman rm -f "$NAME" >/dev/null 2>&1 || true
  base=$(vram_mib)
  dev=(); [ "$ngl" != "0" ] && dev=(--device /dev/dri --security-opt label=disable)
  podman run -d --rm --name "$NAME" -p 127.0.0.1:${PORT}:${PORT} \
    -v "${MODEL_DIR}:/models:ro" "${dev[@]}" "$IMAGE" \
    --model "/models/${MODEL_FILE}" --host 0.0.0.0 --port $PORT \
    --n-gpu-layers "$ngl" --ctx-size "$CTX" --parallel 1 \
    --batch-size 128 --ubatch-size 32 --threads 8 --alias probe >/dev/null 2>&1
  ok=FAIL
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:${PORT}/health 2>/dev/null)" = "200" ] && { ok=OK; break; }
    podman ps --format '{{.Names}}' | grep -q "$NAME" || break
    sleep 1
  done
  if [ "$ok" != OK ]; then echo "${LABEL} ngl=${ngl}: FAILED to load"; continue; fi
  peak=0; for _ in $(seq 1 4); do v=$(vram_mib); [ "$v" -gt "$peak" ] && peak=$v; sleep 0.3; done
  # run the model stages; capture verifier + planner rows
  res=$(python3 -m bench.run_bench --model-stages --base-url http://127.0.0.1:${PORT} --model probe 2>/dev/null)
  vrow=$(echo "$res" | grep 'verifier')
  prow=$(echo "$res" | grep 'planner')
  vacc=$(echo "$vrow" | awk '{print $4}'); vlat=$(echo "$vrow" | awk '{print $5}')
  pacc=$(echo "$prow" | awk '{print $4}'); plat=$(echo "$prow" | awk '{print $5}')
  printf "%-12s ngl=%-3s ctx=%-6s VRAM=%5sMiB | verifier acc=%5s lat=%7sms | planner acc=%5s lat=%7sms\n" \
    "$LABEL" "$ngl" "$CTX" "$peak" "${vacc:-?}" "${vlat:-?}" "${pacc:-?}" "${plat:-?}"
  podman rm -f "$NAME" >/dev/null 2>&1 || true
done
