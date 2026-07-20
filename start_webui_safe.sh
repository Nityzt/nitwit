#!/usr/bin/env bash
set -euo pipefail

export QWEN_PROJECT_MAX_FILES="${QWEN_PROJECT_MAX_FILES:-80}"
export QWEN_MODEL_CALL_COOLDOWN_S="${QWEN_MODEL_CALL_COOLDOWN_S:-3}"

exec python3 webui.py --host 127.0.0.1 --port 8091
