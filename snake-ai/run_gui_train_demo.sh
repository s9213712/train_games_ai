#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/main"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
exec python3 gui_train_demo.py "$@"
