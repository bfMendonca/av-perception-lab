#!/usr/bin/env bash
# Start the web interface from a fresh clone with one command.
#
# Usage:
#   utils/run_web.sh [--gpu] [--rebuild] [--dry-run] [SERVER OPTIONS...]
#
#   --gpu        use the GPU stack (extra `infer`); installs it on the first run
#                (about 9 GB of packages). Without it the script keeps the GPU
#                stack when it is already installed and otherwise starts the
#                interface without it (models show "Not ready").
#   --rebuild    rebuild the front-end even if web/dist exists.
#   --dry-run    print what would run and run nothing.
#   -h, --help   this text.
# Everything else goes to det-depth-distance-web (for example --port 8010;
# `uv run --extra web det-depth-distance-web --help` lists the options).
#
# The front-end is built only when web/dist/index.html is missing (needs Node.js
# 22.12 or newer). The extras are never fewer than the ones already installed:
# `uv run` syncs exactly, so dropping `infer` would remove PyTorch.
set -euo pipefail

usage() {
  sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
}

DRY=0
GPU=0
REBUILD=0
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --gpu) GPU=1 ;;
    --rebuild) REBUILD=1 ;;
    --dry-run) DRY=1 ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."

fail() {
  echo "error: $1" >&2
  exit 1
}

command -v uv > /dev/null 2>&1 ||
  fail "uv not found. Install it from https://docs.astral.sh/uv/getting-started/installation/ and run this again."

NEED_BUILD=0
if [ "$REBUILD" = 1 ] || [ ! -f web/dist/index.html ]; then
  NEED_BUILD=1
fi

if [ "$NEED_BUILD" = 1 ]; then
  command -v node > /dev/null 2>&1 && command -v npm > /dev/null 2>&1 ||
    fail "building the web interface needs Node.js 22.12 or newer with npm (https://nodejs.org/). Install it and run this again."
  NODE_VERSION="$(node --version)"
  VERSION="${NODE_VERSION#v}"
  MAJOR="${VERSION%%.*}"
  REST="${VERSION#*.}"
  MINOR="${REST%%.*}"
  if [ "$MAJOR" -lt 22 ] || { [ "$MAJOR" -eq 22 ] && [ "$MINOR" -lt 12 ]; }; then
    fail "Node.js 22.12 or newer is required to build the web interface (found $NODE_VERSION)."
  fi
fi

ENV_DIR="${UV_PROJECT_ENVIRONMENT:-.venv}"
HAS_TORCH=0
if compgen -G "$ENV_DIR/lib/python*/site-packages/torch" > /dev/null; then
  HAS_TORCH=1
fi

if [ "$GPU" = 1 ] || [ "$HAS_TORCH" = 1 ]; then
  EXTRAS=(--extra infer --extra web)
  if [ "$HAS_TORCH" = 1 ]; then
    echo "GPU stack found in $ENV_DIR: keeping it."
  else
    echo "Installing the GPU stack (about 9 GB of packages) on this first run."
  fi
else
  EXTRAS=(--extra web)
  echo "No GPU stack in $ENV_DIR: starting without it (models show \"Not ready\"). Use --gpu to install it."
fi

if [ "$NEED_BUILD" = 1 ]; then
  echo "Building the web interface."
  if [ "$DRY" = 1 ]; then
    echo "+ (cd web && npm ci)"
    echo "+ (cd web && npm run build)"
  else
    (cd web && npm ci && npm run build)
  fi
fi

HOST=127.0.0.1
PORT=8000
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --host) HOST="${ARGS[$((i + 1))]:-$HOST}" ;;
    --port) PORT="${ARGS[$((i + 1))]:-$PORT}" ;;
  esac
done

echo "Starting the web interface at http://$HOST:$PORT (Ctrl-C to stop)."
if [ "$DRY" = 1 ]; then
  echo "+ uv run ${EXTRAS[*]} det-depth-distance-web${ARGS[*]:+ ${ARGS[*]}}"
else
  exec uv run "${EXTRAS[@]}" det-depth-distance-web ${ARGS[@]+"${ARGS[@]}"}
fi
