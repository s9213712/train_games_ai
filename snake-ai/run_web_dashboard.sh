#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/main"
exec python3 main/web_dashboard.py
