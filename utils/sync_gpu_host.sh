#!/usr/bin/env bash
# Sync this repository with the GPU inference host over rsync (no git remote needed).
#
# Usage:
#   utils/sync_gpu_host.sh push              # local working tree -> HOST:DEST (code, docs, .git, uv.lock)
#   utils/sync_gpu_host.sh pull [RUN_DIR]    # HOST:DEST/_testbed/outputs[/RUN_DIR] -> local _testbed/outputs/
#
# Environment overrides: SYNC_HOST (default: gpu-host), SYNC_DEST (default: dev/det-depth-distance, relative to $HOME).
# `push` never touches the remote .venv/, _testbed/ or weights/ (excluded, so --delete leaves them alone).
set -euo pipefail

HOST="${SYNC_HOST:-gpu-host}"
DEST="${SYNC_DEST:-dev/det-depth-distance}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${1:-}" in
  push)
    rsync -az --delete \
      --exclude '.venv/' --exclude '_testbed/' --exclude 'weights/' \
      --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.ruff_cache/' \
      "$ROOT/" "$HOST:$DEST/"
    echo "pushed $ROOT -> $HOST:$DEST"
    ;;
  pull)
    mkdir -p "$ROOT/_testbed/outputs"
    rsync -az --info=progress2 "$HOST:$DEST/_testbed/outputs/${2:-}" "$ROOT/_testbed/outputs/"
    ;;
  *)
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
