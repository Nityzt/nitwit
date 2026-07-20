#!/usr/bin/env bash
# Switch the single GPU coder slot between profiles. A profile is (model, ctx, gpu_layers,
# vram) and maps 1:1 to a launch config on :8080. Only ONE coder is resident at a time (8 GB
# card), so this stops the current container and starts the new one — session-level, never
# per-request (cold load is seconds, tens for the 30B). Utility/verify models live on CPU and
# are untouched. A hard VRAM guard refuses any profile over the cap so a switch can't
# oversubscribe the card and fault the RX 580.
#
#   switch_coder.sh list                 # show profiles + which is active
#   switch_coder.sh coder-14b            # switch the GPU slot to the 14B profile
#   switch_coder.sh coder-7b             # back to the fast default (7B @ 64k)
set -uo pipefail

VRAM_CAP_MIB="${VRAM_CAP_MIB:-7500}"          # usable ceiling = card 8192 - ~700 display headroom
STATE_FILE="${STATE_FILE:-$HOME/.local/state/qwen-orchestrator/coder-profile}"
LAUNCHER="/home/nit/qwen-orchestrator/start_llama_gpu.sh"
CARD=/sys/class/drm/card0/device

# profile: MODEL_FILE | GPU_LAYERS | CTX_SIZE | VRAM_MIB | LABEL
declare -A PROFILES=(
  [coder-7b]="qwen2.5-coder-7b-instruct-q4_k_m.gguf|20|65536|6300|Qwen2.5-Coder-7B @ 64k (fast default, full GPU)"
  [coder-14b]="Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf|32|8192|7420|Qwen2.5-Coder-14B @ 8k (max capability, partial offload)"
  # coder-30b is added by the installer only AFTER it clears the sustained-loop crash test.
)
# Optional 30B profile, enabled once tested (fill GPU_LAYERS/CTX/VRAM from the probe):
if [ -f "$HOME/.local/state/qwen-orchestrator/coder-30b.profile" ]; then
  PROFILES[coder-30b]="$(cat "$HOME/.local/state/qwen-orchestrator/coder-30b.profile")"
fi

active_profile() { [ -f "$STATE_FILE" ] && cat "$STATE_FILE" || echo "coder-7b(assumed)"; }
vram_used_mib() { echo $(( $(cat "$CARD/mem_info_vram_used" 2>/dev/null || echo 0) / 1048576 )); }

cmd="${1:-list}"

if [ "$cmd" = "list" ]; then
  printf "%-11s %-6s %-7s %-8s  %s\n" PROFILE NGL CTX VRAM DESCRIPTION
  for name in coder-7b coder-14b coder-30b; do
    [ -n "${PROFILES[$name]:-}" ] || continue
    IFS='|' read -r mf ngl ctx vram label <<<"${PROFILES[$name]}"
    mark=" "; [ "$name" = "$(active_profile)" ] && mark="*"
    printf "%s%-10s %-6s %-7s %-8s  %s\n" "$mark" "$name" "$ngl" "$ctx" "${vram}MiB" "$label"
  done
  echo "active: $(active_profile)   VRAM cap: ${VRAM_CAP_MIB} MiB   in use now: $(vram_used_mib) MiB"
  exit 0
fi

spec="${PROFILES[$cmd]:-}"
if [ -z "$spec" ]; then echo "error: unknown profile '$cmd' (try: list)" >&2; exit 1; fi
IFS='|' read -r MODEL_FILE GPU_LAYERS CTX_SIZE VRAM_MIB LABEL <<<"$spec"

# --- VRAM guard: refuse a profile that would oversubscribe the card ---
if [ "$VRAM_MIB" -gt "$VRAM_CAP_MIB" ]; then
  echo "REFUSED: $cmd needs ~${VRAM_MIB} MiB > cap ${VRAM_CAP_MIB} MiB. Would oversubscribe the 8 GB card." >&2
  exit 2
fi

echo "switching GPU coder -> $cmd ($LABEL)"
# Stop the systemd-managed default so it doesn't fight the manual container.
systemctl --user stop qwen-llama >/dev/null 2>&1 || true
podman rm -f llama >/dev/null 2>&1 || true

MODEL_FILE="$MODEL_FILE" GPU_LAYERS="$GPU_LAYERS" CTX_SIZE="$CTX_SIZE" "$LAUNCHER" >/dev/null 2>&1

# Wait for health (the 30B is a big file -> allow generous load time).
for i in $(seq 1 90); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null)" = "200" ] && break
  podman ps --format '{{.Names}}' | grep -q '^llama$' || { echo "ERROR: coder container died on load (VRAM?)"; exit 3; }
  sleep 1
done
if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null)" != "200" ]; then
  echo "ERROR: coder did not become healthy" >&2; exit 3
fi

mkdir -p "$(dirname "$STATE_FILE")"; echo "$cmd" > "$STATE_FILE"
echo "active coder: $cmd  (n_ctx=$(curl -s http://127.0.0.1:8080/props 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["default_generation_settings"]["n_ctx"])' 2>/dev/null))  VRAM in use: $(vram_used_mib) MiB"
