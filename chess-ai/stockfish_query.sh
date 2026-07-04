#!/usr/bin/env bash
set -euo pipefail

stockfish_path="${STOCKFISH_PATH:-/home/s92137/reference_repos/Stockfish/src/stockfish}"
fen="${1:?fen required}"
depth="${2:-8}"
timeout_seconds="${STOCKFISH_QUERY_TIMEOUT:-45}"

cd "$(dirname "$stockfish_path")"

coproc SF { "$stockfish_path"; }

cleanup() {
  {
    printf 'quit\n' >&"${SF[1]}"
  } 2>/dev/null || true
  wait "${SF_PID}" 2>/dev/null || true
}
trap cleanup EXIT

send() {
  printf '%s\n' "$1" >&"${SF[1]}"
}

read_until() {
  local needle="$1"
  local deadline=$((SECONDS + timeout_seconds))
  local line
  while (( SECONDS < deadline )); do
    if IFS= read -r -t 1 line <&"${SF[0]}"; then
      printf '%s\n' "$line"
      [[ "$line" == "$needle"* ]] && return 0
    fi
  done
  printf 'error timeout waiting for %s\n' "$needle"
  return 1
}

send "uci"
read_until "uciok"
send "setoption name Threads value 1"
send "setoption name Hash value 32"
send "setoption name MultiPV value 5"
send "isready"
read_until "readyok"
send "position fen $fen"
send "go depth $depth"
read_until "bestmove"
