#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PORT="${PORT:-7872}"
echo "Soccer AI Trainer: http://localhost:${PORT}/"
python3 server.py --host 127.0.0.1 --port "${PORT}"
