#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PORT="${PORT:-7873}"
echo "Chess AI Trainer: http://localhost:${PORT}/"
python3 app.py --host 127.0.0.1 --port "${PORT}"
