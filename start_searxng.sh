#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

podman rm -f qwen-searxng >/dev/null 2>&1 || true
podman run -d \
  --name qwen-searxng \
  --replace \
  -p 127.0.0.1:8888:8080 \
  -v "$PWD/searxng/settings.yml:/etc/searxng/settings.yml:ro,Z" \
  docker.io/searxng/searxng:latest

echo "SearXNG starting at http://127.0.0.1:8888"
