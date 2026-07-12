from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import select
import shutil
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path

import chess
from flask import Flask, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
RUNTIME_DIR = ROOT / "runtime"
CHECKPOINT_PATH = RUNTIME_DIR / "chess_policy.json"
BEST_CHECKPOINT_PATH = RUNTIME_DIR / "chess_policy.best.json"
CHECKPOINT_VERSION = 2
TRAINING_PROTOCOL = "teacher_ranker_guard_holdout_audit_v2"

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

DEFAULT_WEIGHTS = {
    "material": 1.0,
    "mobility": 0.12,
    "center": 0.34,
    "king_safety": 0.22,
    # Learned coefficient for a deterministic one-opponent-reply lookahead.
    # The verified baseline starts at zero so training must earn any change.
    "reply_safety": 0.0,
}
FEATURE_KEYS = tuple(DEFAULT_WEIGHTS.keys())
CORE_FEATURE_KEYS = tuple(key for key in FEATURE_KEYS if key != "reply_safety")
CORE_LEARNING_SCALE = 0.05
CORE_WEIGHT_RADIUS = 0.75
TEACHER_BUFFER_LIMIT = 800
# The student always plays White.  Guard and holdout positions therefore use
# White-to-move positions and remain disjoint from live self-play games.
GUARD_FENS = (
    "rnb1k2r/3p1p1p/1p3q1b/p3pP2/2pPn2p/2P1Q3/PPN1PBPR/R3KBN1 w Qkq - 0 15",
    "r3kbnr/2p1p1p1/n3b3/pp2p2p/P4P2/1PNK1P1N/2PP2PP/R1Q2B1R w k - 0 15",
    "r1bq1bn1/2pkpppr/n7/pp1pP2p/7P/2NP1P2/PPP3P1/R1BQKBNR w KQ - 1 8",
    "r1bqkb1r/2pp1pp1/Bp2p2n/p2n4/PP2P2p/2N2P2/2PP2PP/R1BQK1NR w q - 5 13",
    "r1bqk1n1/p1ppp2r/5pp1/1p5P/Q2Pn1p1/P3B2B/1PP2P2/RN2K1NR w q - 0 14",
    "r1b1kbnr/pq1p2p1/2nPpp2/1pp4p/5B2/P4NP1/1PPKPP1P/RN1Q1B1R w kq - 0 10",
)
HOLDOUT_FENS = (
    "1q2kbnr/rb2p2p/2pp1pp1/pp1N4/PP5P/5P2/3PPKPR/R1BQ1BN1 w k - 2 14",
    "rnbk1b1r/p1p3pp/1n3p2/1p2p3/2qPP1P1/N7/PPQP1P1P/R1B1K1NR w KQ - 2 13",
    "rnbqkb1r/2pnpp2/p7/p2P2pp/2P5/N2P2PP/PP3P2/R1BQK1NR w KQkq - 0 9",
    "rnb1kbnr/pp2pppp/3p4/2p5/P7/1P4q1/R2PPP2/1NBQKB1R w kq - 0 13",
    "r1b1kbnr/p2pp1p1/n4p2/1pp4p/2P1P3/1Q4q1/PP1P1P1P/RNB1KBNR w KQkq - 2 11",
    "r1bqkbnr/p3p1p1/1p1p3p/n1N2p2/8/2BP1P1N/PPP1P1PP/R2QKB1R w KQkq - 0 9",
)
AUDIT_FENS = (
    "1rbqkbr1/ppp1npp1/2n4p/3pp1P1/3P1P2/4P3/PPPK3P/RNBQ1BNR w - - 1 8",
    "rnbqkbnr/pp2p2p/2p5/1N3pp1/PP1pP3/B2P1P1N/2P3PP/2RQKB1R w Kkq - 0 11",
    "1n2kbnr/3qp2p/1pp1b3/3p2pP/3PRp2/2P3P1/rP1KPPB1/RNBQ2N1 w k - 0 14",
    "3rkbnr/p1p1pp2/nq5p/1p1p3p/PQ6/1P1P1N2/1BP1PPb1/R2NKB2 w Qk - 4 15",
    "r2qkb1r/p2bp1n1/np6/2pP2p1/7p/P2P1p1N/RPPBNPPP/4KB1R w Kkq - 2 15",
    "rnbq1b1r/pp2nk1p/2p2pp1/2Ppp1P1/8/P3P3/1P1P1PBP/RNBQK1NR w KQ - 0 8",
)
# This split is deliberately not consulted by training, acceptance, checkpoint
# promotion, or startup selection.  It exists for independent offline audits.
INDEPENDENT_AUDIT_FENS = (
    "r1b1kb1r/ppp1pppp/8/3q4/1n1P2n1/1P5P/P1P1KPP1/RNBQ1BNR w kq - 1 7",
    "rnb1kbnr/pq1pp1p1/2p4p/1p3p2/4P3/1PPP1P2/P5PP/RNBQKBNR w KQkq - 1 8",
    "r2qkbnr/p1pbp1pp/6n1/1p1p1p2/8/3P1PPP/PPP1PK2/RNB1QBNR w kq - 0 9",
    "rnb2bnr/pp2pk2/1qp2p2/1B1p2pp/NP1P4/4PP1N/P1P3PP/1RBQK2R w K - 1 10",
    "r1bqk1nr/pp1p1p1p/2pb4/8/1nPP4/N1N2pp1/PP2P1PP/R1BQKB1R w KQkq - 0 11",
    "rnb1kbnr/1p2p1pp/1qp2p2/p2p1P2/Q7/P1P5/1P1PP1PP/RNB1KBNR w KQkq - 2 7",
    "1rbqkbnr/p3p1pp/n1pp4/1p2p3/1PP4P/3P4/P4PPR/RNBQKBN1 w Qk - 0 8",
    "1r1qkbnr/p2bpp2/2n4p/1Bpp2p1/4P2P/7N/PPPP1PPR/RNBQK3 w Qk - 4 9",
    "r1bqkb1r/1p1npppp/8/p1pp1n2/Q2P4/2P1PPPN/PP1B3P/RN2KB1R w KQkq - 0 10",
    "rnbqkbn1/p1pp1p2/1p6/4p1Pr/QP2P2p/2PB3P/P2PN1P1/RNB1K2R w KQq - 1 11",
    "r1b1k1nr/1ppp1ppp/n4q2/p1b1p3/2BP4/1PP1P2N/P4PPP/RNBQK2R w KQkq - 5 7",
    "rnb1k1nr/3p1ppp/1q1b4/ppp1p3/1PB1PPP1/3P4/P1P4P/RNBQ1KNR w kq - 1 8",
)
BENCHMARK_FENS = GUARD_FENS + HOLDOUT_FENS
PROMOTION_FENS = BENCHMARK_FENS + AUDIT_FENS


def clamp_student_weight(key: str, value: float) -> float:
    if key == "reply_safety":
        return max(-2.0, min(3.0, float(value)))
    anchor = float(DEFAULT_WEIGHTS[key])
    return max(anchor - CORE_WEIGHT_RADIUS, min(anchor + CORE_WEIGHT_RADIUS, float(value)))


def policy_fingerprint(weights: dict[str, float]) -> str:
    payload = {
        key: round(float(weights[key]), 12)
        for key in FEATURE_KEYS
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def resolve_stockfish_path() -> str:
    for key in ("STOCKFISH_PATH", "HTML_LEARNING_CHESS_STOCKFISH_PATH"):
        value = os.environ.get(key, "").strip()
        if value:
            path = Path(value).expanduser()
            if path.exists() and os.access(path, os.X_OK):
                return str(path.resolve())
    found = shutil.which("stockfish")
    if found:
        return str(Path(found).resolve())
    for candidate in (Path("/usr/games/stockfish"), Path("/usr/local/bin/stockfish")):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return ""


def material_score(board: chess.Board) -> int:
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score


def center_score(board: chess.Board) -> int:
    center = [chess.D4, chess.E4, chess.D5, chess.E5]
    extended = [chess.C3, chess.D3, chess.E3, chess.F3, chess.C4, chess.F4, chess.C5, chess.F5, chess.C6, chess.D6, chess.E6, chess.F6]
    score = 0
    for square in center:
      piece = board.piece_at(square)
      if piece:
          score += 30 if piece.color == chess.WHITE else -30
    for square in extended:
      piece = board.piece_at(square)
      if piece:
          score += 10 if piece.color == chess.WHITE else -10
    return score


def king_safety_score(board: chess.Board) -> int:
    score = 0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        king = board.king(color)
        if king is None:
            continue
        attackers = len(board.attackers(not color, king))
        rank = chess.square_rank(king)
        back_rank_bonus = 12 if (color == chess.WHITE and rank <= 1) or (color == chess.BLACK and rank >= 6) else 0
        score += sign * (back_rank_bonus - attackers * 35)
    return score


def mobility_score(board: chess.Board) -> int:
    turn = board.turn
    board.turn = chess.WHITE
    white = board.legal_moves.count()
    board.turn = chess.BLACK
    black = board.legal_moves.count()
    board.turn = turn
    return (white - black) * 8


def evaluate_board(board: chess.Board, weights: dict[str, float]) -> float:
    if board.is_checkmate():
        return -100000 if board.turn == chess.WHITE else 100000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    return (
        weights["material"] * material_score(board)
        + weights["mobility"] * mobility_score(board)
        + weights["center"] * center_score(board)
        + weights["king_safety"] * king_safety_score(board)
    )


def component_scores(board: chess.Board) -> dict[str, float]:
    return {
        "material": material_score(board) / 100.0,
        "mobility": mobility_score(board) / 100.0,
        "center": center_score(board) / 100.0,
        "king_safety": king_safety_score(board) / 100.0,
    }


def captured_piece_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        square = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        piece = board.piece_at(square)
    else:
        piece = board.piece_at(move.to_square)
    return PIECE_VALUES.get(piece.piece_type, 0) if piece else 0


def move_features(board: chess.Board, move: chess.Move) -> dict[str, float]:
    side = board.turn
    before = component_scores(board)
    capture_bonus = captured_piece_value(board, move) / 700.0
    promotion_bonus = PIECE_VALUES.get(move.promotion, 0) / 700.0 if move.promotion else 0.0
    check_bonus = 0.18 if board.gives_check(move) else 0.0
    castle_bonus = 0.12 if board.is_castling(move) else 0.0
    board.push(move)
    after = component_scores(board)
    board.pop()

    sign = 1.0 if side == chess.WHITE else -1.0
    features = {key: sign * (after[key] - before[key]) for key in before}
    features["material"] += capture_bonus + promotion_bonus
    features["king_safety"] += check_bonus + castle_bonus
    features["reply_safety"] = reply_safety_feature(board.fen(), move.uci())
    return {key: float(features.get(key, 0.0)) for key in FEATURE_KEYS}


@lru_cache(maxsize=20_000)
def reply_safety_feature(fen: str, move_uci: str) -> float:
    """Score the position after the opponent's best deterministic reply."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return 0.0
    mover = board.turn
    board.push(move)
    if board.is_game_over(claim_draw=True):
        score = evaluate_board(board, DEFAULT_WEIGHTS)
    else:
        reply_scores = []
        for reply in board.legal_moves:
            board.push(reply)
            reply_scores.append(evaluate_board(board, DEFAULT_WEIGHTS))
            board.pop()
        if not reply_scores:
            score = evaluate_board(board, DEFAULT_WEIGHTS)
        else:
            score = min(reply_scores) if mover == chess.WHITE else max(reply_scores)
    sign = 1.0 if mover == chess.WHITE else -1.0
    return sign * float(score) / 100.0


def score_move(board: chess.Board, move: chess.Move, weights: dict[str, float]) -> float:
    features = move_features(board, move)
    return sum(float(weights.get(key, 0.0)) * features[key] for key in FEATURE_KEYS)


def material_for_color(board: chess.Board, color: chess.Color) -> int:
    score = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES.get(piece.piece_type, 0)
        score += value if piece.color == color else -value
    return score


def tactical_safety_report(board: chess.Board, move: chess.Move | None) -> dict:
    if move is None:
        return {"safe": False, "reason": "missing_move"}
    if move not in board.legal_moves:
        return {"safe": False, "reason": "illegal_move", "move": move.uci()}

    mover = board.turn
    before = material_for_color(board, mover)
    after = board.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        return {"safe": True, "reason": "immediate_checkmate", "move": move.uci()}

    moved_piece = after.piece_at(move.to_square)
    if moved_piece is None or moved_piece.color != mover or moved_piece.piece_type == chess.KING:
        return {"safe": True, "reason": "no_hanging_moved_piece", "move": move.uci()}

    worst_loss = 0
    worst_reply = ""
    for reply in after.legal_moves:
        if not after.is_capture(reply):
            continue
        capture_square = reply.to_square
        if after.is_en_passant(reply):
            capture_square = chess.square(chess.square_file(reply.to_square), chess.square_rank(reply.from_square))
        if capture_square != move.to_square:
            continue
        reply_board = after.copy(stack=False)
        reply_board.push(reply)
        loss = before - material_for_color(reply_board, mover)
        if loss > worst_loss:
            worst_loss = loss
            worst_reply = reply.uci()

    if worst_loss <= 80:
        return {"safe": True, "reason": "direct_loss_within_window", "move": move.uci(), "worst_loss_cp": worst_loss}
    return {
        "safe": False,
        "reason": "direct_hanging_piece_without_compensation",
        "move": move.uci(),
        "worst_loss_cp": worst_loss,
        "worst_reply": worst_reply,
    }


def choose_tactically_safe_move(
    board: chess.Board,
    proposed_move: chess.Move | None,
    score_candidate,
) -> tuple[chess.Move | None, dict]:
    report = tactical_safety_report(board, proposed_move)
    if report.get("safe"):
        report["fallback_applied"] = False
        return proposed_move, report
    candidates = []
    for candidate in board.legal_moves:
        candidate_report = tactical_safety_report(board, candidate)
        if not candidate_report.get("safe"):
            continue
        candidates.append((float(score_candidate(candidate)), candidate.uci(), candidate, candidate_report))
    if not candidates:
        report["fallback_applied"] = False
        return proposed_move, report
    _score, _uci, fallback, fallback_report = max(candidates, key=lambda item: (item[0], item[1]))
    return fallback, {
        "safe": True,
        "reason": "fallback_selected",
        "fallback_applied": True,
        "blocked_move": proposed_move.uci() if proposed_move else "",
        "blocked_report": report,
        "fallback_move": fallback.uci(),
        "fallback_report": fallback_report,
    }


def fallback_ranked_moves(board: chess.Board, depth: int = 2) -> list[dict]:
    def search(position: chess.Board, ply: int, alpha: float, beta: float) -> float:
        if ply == 0 or position.is_game_over():
            return evaluate_board(position, DEFAULT_WEIGHTS)
        if position.turn == chess.WHITE:
            best = -math.inf
            for move in ordered_moves(position):
                position.push(move)
                best = max(best, search(position, ply - 1, alpha, beta))
                position.pop()
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best
        best = math.inf
        for move in ordered_moves(position):
            position.push(move)
            best = min(best, search(position, ply - 1, alpha, beta))
            position.pop()
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best

    rows = []
    for move in ordered_moves(board):
        board.push(move)
        score = search(board, max(0, depth - 1), -math.inf, math.inf)
        board.pop()
        rows.append({"move": move.uci(), "score_cp": int(score), "pv": [move.uci()]})
    rows.sort(key=lambda row: row["score_cp"], reverse=board.turn == chess.WHITE)
    return rows


def fallback_teacher(board: chess.Board, depth: int = 2) -> dict:
    rows = fallback_ranked_moves(board, depth=depth)
    best = rows[0] if rows else {"move": "", "score_cp": 0, "pv": []}
    return {
        "available": False,
        "source": "fallback",
        "best_move": best["move"],
        "eval_cp": best["score_cp"],
        "lines": rows[:5],
    }


@lru_cache(maxsize=64)
def benchmark_ranked_moves(fen: str) -> tuple[tuple[str, float], ...]:
    """Frozen deterministic depth-2 teacher rows used by guard/holdout checks."""
    rows = fallback_ranked_moves(chess.Board(fen), depth=2)
    return tuple((str(row["move"]), float(row["score_cp"])) for row in rows)


def ordered_moves(board: chess.Board) -> list[chess.Move]:
    def key(move: chess.Move) -> tuple[int, str]:
        capture = board.is_capture(move)
        promo = move.promotion is not None
        check = board.gives_check(move)
        return (int(capture) * 3 + int(promo) * 2 + int(check), move.uci())
    return sorted(board.legal_moves, key=key, reverse=True)


def _send_uci(proc: subprocess.Popen, text: str) -> None:
    if proc.stdin is None:
        raise RuntimeError("stockfish stdin is closed")
    proc.stdin.write(text.rstrip("\n") + "\n")
    proc.stdin.flush()


def _readline_timeout(proc: subprocess.Popen, deadline: float) -> str:
    if proc.stdout is None:
        raise RuntimeError("stockfish stdout is closed")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("timed out waiting for stockfish")
    readable, _, _ = select.select([proc.stdout], [], [], remaining)
    if not readable:
        raise TimeoutError("timed out waiting for stockfish")
    line = proc.stdout.readline()
    if line == "":
        raise RuntimeError("stockfish exited unexpectedly")
    return line.rstrip("\n")


def _read_until(proc: subprocess.Popen, deadline: float, predicate) -> list[str]:
    lines = []
    while True:
        line = _readline_timeout(proc, deadline)
        lines.append(line)
        if predicate(line):
            return lines


def _parse_uci_info(lines: list[str], board: chess.Board) -> list[dict]:
    latest: dict[int, dict] = {}
    for line in lines:
        if not line.startswith("info "):
            continue
        tokens = line.split()
        multipv = 1
        if "multipv" in tokens:
            try:
                multipv = int(tokens[tokens.index("multipv") + 1])
            except Exception:
                multipv = 1
        row = dict(latest.get(multipv) or {"rank": multipv})
        if "depth" in tokens:
            try:
                row["depth"] = int(tokens[tokens.index("depth") + 1])
            except Exception:
                pass
        if "score" in tokens:
            idx = tokens.index("score")
            if idx + 2 < len(tokens):
                kind = tokens[idx + 1]
                try:
                    raw = int(tokens[idx + 2])
                except Exception:
                    raw = 0
                if kind == "mate":
                    row["score_cp"] = 100000 - abs(raw) if raw > 0 else -100000 + abs(raw)
                    row["mate"] = raw
                else:
                    row["score_cp"] = raw
        if "pv" in tokens:
            pv = tokens[tokens.index("pv") + 1 :]
            if pv:
                row["move"] = pv[0]
                row["pv"] = pv[:8]
        latest[multipv] = row

    rows = []
    for _rank, row in sorted(latest.items()):
        uci = str(row.get("move") or "")
        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            continue
        if move not in board.legal_moves:
            continue
        row.setdefault("score_cp", 0)
        row.setdefault("depth", 0)
        row.setdefault("pv", [move.uci()])
        row["move"] = move.uci()
        rows.append(row)
    return rows


def stockfish_teacher(path: str, board: chess.Board, depth: int) -> dict:
    if not path:
        raise FileNotFoundError("Stockfish binary path is empty")
    helper = ROOT / "stockfish_query.sh"
    timeout_seconds = max(10, min(90, int(depth) * 6))
    env = os.environ.copy()
    env["STOCKFISH_PATH"] = path
    env["STOCKFISH_QUERY_TIMEOUT"] = str(timeout_seconds)
    proc = subprocess.run(
        [str(helper), board.fen(), str(max(1, int(depth)))],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds + 5,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or f"stockfish helper exited {proc.returncode}").strip())
    lines = proc.stdout.splitlines()
    rows = _parse_uci_info(lines, board)
    if not rows:
        best = ""
        for line in reversed(lines):
            if line.startswith("bestmove "):
                parts = line.split()
                best = parts[1] if len(parts) > 1 else ""
                break
        if best:
            rows = [{"move": best, "score_cp": 0, "depth": depth, "pv": [best]}]
    if not rows:
        raise RuntimeError("Stockfish returned no legal principal variation")
    best = rows[0]
    return {
        "available": True,
        "source": "stockfish",
        "path": path,
        "best_move": best["move"],
        "eval_cp": best["score_cp"],
        "lines": rows[:5],
    }


class Trainer:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.step_lock = threading.RLock()
        self.checkpoint_path = CHECKPOINT_PATH
        self.best_checkpoint_path = BEST_CHECKPOINT_PATH
        self.running = False
        self.thread: threading.Thread | None = None
        self.stop_requested = False
        self.stockfish_path = resolve_stockfish_path()
        self.engine_error = ""
        self.engine_verified = False
        self.teacher_depth = 4
        self.student_depth = 1
        self.chunk_moves = 80
        self.exploration = 0.08
        self.mutation = 0.12
        self.learning_rate = 0.025
        self.teacher_learning_rate = 0.05
        self.guard_enabled = True
        self.guard_min_gap_delta = 0.0
        self.guard_holdout_tolerance = 0.0
        self.discount = 0.94
        self.guard_in_progress = False
        self.guard_public_state: dict | None = None
        self.closed = False
        self.reset()

    def reset(self, *, load_checkpoint: bool = True) -> None:
        with self.step_lock, self.lock:
            self.board = chess.Board()
            self.weights = dict(DEFAULT_WEIGHTS)
            self.best_weights = dict(DEFAULT_WEIGHTS)
            self.game = 0
            self.ply = 0
            self.reward = 0.0
            self.best_reward = -math.inf
            self.student_matches = 0
            self.teacher_samples = 0
            self.teacher_updates = 0
            self.rl_samples = 0
            self.policy_updates = 0
            self.accepted_chunks = 0
            self.rejected_chunks = 0
            self.td_error_ema = 0.0
            self.last_td_error = 0.0
            self.teacher_buffer = []
            self.episode_trace = []
            self.completed_games = 0
            self.white_wins = 0
            self.black_wins = 0
            self.draws = 0
            self.history = []
            self.last_guard = {}
            self.accepted_guard = {}
            self.moves = []
            self.teacher = {"available": False, "source": "none", "best_move": "", "eval_cp": 0, "lines": []}
            self.last_event = "reset"
            self.last_error = ""
            self.loaded_checkpoint: dict = {}
            self.guard_in_progress = False
            self.guard_public_state = None
            if load_checkpoint:
                self._load_checkpoint_locked()

    def _sanitize_weights(self, payload: dict | None) -> dict[str, float] | None:
        if not isinstance(payload, dict):
            return None
        weights = {}
        for key in FEATURE_KEYS:
            if key not in payload:
                return None
            weights[key] = clamp_student_weight(key, float(payload[key]))
        return weights

    def _checkpoint_payload_locked(self, *, teacher: str = "dashboard", extra: dict | None = None) -> dict:
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "training_protocol": TRAINING_PROTOCOL,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "teacher": teacher,
            "stockfish_path": self.stockfish_path,
            "weights": dict(self.weights),
            "policy_fingerprint": policy_fingerprint(self.weights),
            "best_weights": dict(self.best_weights),
            "best_reward": None if not math.isfinite(self.best_reward) else self.best_reward,
            "learning": {
                "game": self.game,
                "ply": self.ply,
                "teacher_samples": self.teacher_samples,
                "teacher_updates": self.teacher_updates,
                "student_matches": self.student_matches,
                "rl_samples": self.rl_samples,
                "policy_updates": self.policy_updates,
                "accepted_chunks": self.accepted_chunks,
                "rejected_chunks": self.rejected_chunks,
                "completed_games": self.completed_games,
                "wins": self.white_wins,
                "losses": self.black_wins,
                "draws": self.draws,
            },
            "config": {
                "teacher_depth": self.teacher_depth,
                "student_depth": self.student_depth,
                "chunk_moves": self.chunk_moves,
                "exploration": self.exploration,
                "mutation": self.mutation,
                "learning_rate": self.learning_rate,
                "teacher_learning_rate": self.teacher_learning_rate,
                "guard_enabled": self.guard_enabled,
                "guard_min_gap_delta": self.guard_min_gap_delta,
                "guard_holdout_tolerance": self.guard_holdout_tolerance,
                "discount": self.discount,
            },
            "guard": dict(self.last_guard),
            "accepted_guard": dict(self.accepted_guard),
        }
        if extra:
            payload.update(extra)
        return payload

    def _save_checkpoint_locked(self) -> None:
        # A guarded chunk mutates the live in-memory policy while it is still a
        # candidate.  Never make that unaccepted state crash-durable.
        if self.guard_in_progress:
            return
        payload = self._checkpoint_payload_locked()
        self._atomic_write_json(self.checkpoint_path, payload)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _quality_key(metrics: dict) -> tuple[float, float, float]:
        return (
            float(metrics.get("avg_gap", math.inf)),
            -float(metrics.get("match_rate", 0.0)),
            -float(metrics.get("top5_rate", 0.0)),
        )

    def validate_policy(self, weights: dict[str, float]) -> dict[str, dict]:
        """Evaluate every frozen split instead of hiding regressions in one mean."""
        return {
            "guard": self.evaluate_teacher_gap(weights, fens=GUARD_FENS),
            "holdout": self.evaluate_teacher_gap(weights, fens=HOLDOUT_FENS),
            "audit": self.evaluate_teacher_gap(weights, fens=AUDIT_FENS),
            "promotion": self.evaluate_teacher_gap(weights, fens=PROMOTION_FENS),
        }

    def _validation_not_worse(self, candidate: dict, baseline: dict) -> bool:
        return all(
            self._quality_key(candidate[name]) <= self._quality_key(baseline[name])
            for name in ("guard", "holdout", "audit")
        )

    @staticmethod
    def _checkpoint_provenance_valid(payload: dict, weights: dict[str, float]) -> bool:
        try:
            checkpoint_version = int(payload.get("checkpoint_version", 0))
            accepted_chunks = int((payload.get("learning") or {}).get("accepted_chunks", 0))
            policy_updates = int((payload.get("learning") or {}).get("policy_updates", 0))
            teacher_updates = int((payload.get("learning") or {}).get("teacher_updates", 0))
            rl_samples = int((payload.get("learning") or {}).get("rl_samples", 0))
        except (TypeError, ValueError, AttributeError):
            return False
        if checkpoint_version != CHECKPOINT_VERSION:
            return False
        if payload.get("training_protocol") != TRAINING_PROTOCOL:
            return False
        if (payload.get("config") or {}).get("guard_enabled") is False:
            return False
        fingerprint = policy_fingerprint(weights)
        if payload.get("policy_fingerprint") != fingerprint:
            return False
        learning = payload.get("learning")
        accepted_guard = payload.get("accepted_guard")
        if not isinstance(learning, dict) or not isinstance(accepted_guard, dict):
            return False
        return bool(
            accepted_chunks > 0
            and policy_updates > 0
            and (teacher_updates > 0 or rl_samples > 0)
            and accepted_guard.get("accepted") is True
            and accepted_guard.get("behavior_changed") is True
            and accepted_guard.get("candidate_fingerprint") == fingerprint
            and isinstance(accepted_guard.get("baseline"), dict)
            and isinstance(accepted_guard.get("candidate"), dict)
            and isinstance(accepted_guard.get("holdout_baseline"), dict)
            and isinstance(accepted_guard.get("holdout_candidate"), dict)
        )

    def _save_fallback_best_checkpoint_locked(self, guard: dict) -> None:
        candidate = dict(guard.get("candidate") or {})
        if not candidate:
            return
        candidate_validation = self.validate_policy(self.weights)
        candidate_quality = candidate_validation["promotion"]
        default_validation = self.validate_policy(DEFAULT_WEIGHTS)
        if self.best_checkpoint_path.exists():
            try:
                current = json.loads(self.best_checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = {}
            current_weights = self._sanitize_weights(current.get("weights"))
            current_validation = (
                self.validate_policy(current_weights)
                if current_weights is not None
                and self._checkpoint_provenance_valid(current, current_weights)
                else {}
            )
            current_safe = bool(current_validation) and self._validation_not_worse(
                current_validation,
                default_validation,
            )
            current_quality = current_validation.get("promotion", {"avg_gap": math.inf})
            if current_safe and self._quality_key(candidate_quality) >= self._quality_key(current_quality):
                return
        payload = self._checkpoint_payload_locked(
            teacher="verified_guard",
            extra={
                "guard": dict(guard),
                "baseline": dict(guard.get("baseline") or {}),
                "final": candidate,
                "validation": candidate_validation,
            },
        )
        self._atomic_write_json(self.best_checkpoint_path, payload)

    def _load_checkpoint_locked(self) -> None:
        candidates = []
        errors = []
        for path in (self.checkpoint_path, self.best_checkpoint_path):
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                weights = self._sanitize_weights(payload.get("weights"))
                if weights is None:
                    errors.append(f"{path.name}: invalid weights")
                    continue
                if not self._checkpoint_provenance_valid(payload, weights):
                    errors.append(
                        f"{path.name}: missing current schema or policy-bound acceptance evidence"
                    )
                    continue
                validation = self.validate_policy(weights)
                candidates.append({"path": path, "payload": payload, "weights": weights, "validation": validation})
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

        default_validation = self.validate_policy(DEFAULT_WEIGHTS)
        safe_candidates = [
            row
            for row in candidates
            if self._validation_not_worse(row["validation"], default_validation)
            and self._quality_key(row["validation"]["promotion"])
            < self._quality_key(default_validation["promotion"])
        ]
        safe_candidates.append(
            {
                "path": None,
                "payload": {},
                "weights": dict(DEFAULT_WEIGHTS),
                "validation": default_validation,
            }
        )
        selected = min(
            safe_candidates,
            key=lambda row: self._quality_key(row["validation"]["promotion"]),
        )
        rejected = [
            {"path": str(row["path"]), "validation": row["validation"]}
            for row in candidates
            if row["path"] is not None and row is not selected
        ]
        if selected["path"] is None:
            self.loaded_checkpoint = {
                "path": "",
                "validation": default_validation,
                "rejected": rejected,
                "reason": "saved checkpoints did not beat the verified default policy",
            }
            self.last_event = "using verified default policy"
            self.last_error = "; ".join(errors)
            return

        path = selected["path"]
        payload = selected["payload"]
        weights = selected["weights"]
        self.weights = dict(weights)
        self.best_weights = dict(weights)
        if payload.get("best_reward") is not None:
            self.best_reward = float(payload["best_reward"])
        learning = dict(payload.get("learning") or {})
        self.game = int(learning.get("game", self.game))
        self.teacher_samples = int(learning.get("teacher_samples", self.teacher_samples))
        self.teacher_updates = int(learning.get("teacher_updates", self.teacher_updates))
        self.student_matches = int(learning.get("student_matches", self.student_matches))
        self.rl_samples = int(learning.get("rl_samples", self.rl_samples))
        self.policy_updates = int(learning.get("policy_updates", self.policy_updates))
        self.accepted_chunks = int(learning.get("accepted_chunks", self.accepted_chunks))
        self.rejected_chunks = int(learning.get("rejected_chunks", self.rejected_chunks))
        self.completed_games = int(learning.get("completed_games", self.completed_games))
        self.white_wins = int(learning.get("wins", self.white_wins))
        self.black_wins = int(learning.get("losses", self.black_wins))
        self.draws = int(learning.get("draws", self.draws))
        self.last_guard = copy.deepcopy(payload.get("guard") or {})
        self.accepted_guard = copy.deepcopy(payload.get("accepted_guard") or {})
        config = dict(payload.get("config") or {})
        self.teacher_depth = max(1, min(18, int(config.get("teacher_depth", self.teacher_depth))))
        self.student_depth = max(1, min(3, int(config.get("student_depth", self.student_depth))))
        self.chunk_moves = max(1, min(120, int(config.get("chunk_moves", self.chunk_moves))))
        self.exploration = max(0.0, min(0.8, float(config.get("exploration", self.exploration))))
        self.mutation = max(0.0, min(1.0, float(config.get("mutation", self.mutation))))
        self.learning_rate = max(0.0, min(0.5, float(config.get("learning_rate", self.learning_rate))))
        self.teacher_learning_rate = max(0.0, min(0.5, float(config.get("teacher_learning_rate", self.teacher_learning_rate))))
        # Serving unverified in-memory candidates is intentionally unsupported.
        # Keep this true even when a legacy checkpoint asked to disable it.
        self.guard_enabled = True
        self.guard_min_gap_delta = max(0.0, min(1000.0, float(config.get("guard_min_gap_delta", self.guard_min_gap_delta))))
        self.guard_holdout_tolerance = max(0.0, min(1000.0, float(config.get("guard_holdout_tolerance", self.guard_holdout_tolerance))))
        self.discount = max(0.0, min(1.0, float(config.get("discount", self.discount))))
        self.loaded_checkpoint = {
            "path": str(path),
            "teacher": payload.get("teacher", ""),
            "created_at": payload.get("created_at", ""),
            "baseline": payload.get("baseline", {}),
            "final": payload.get("final", {}),
            "validation": selected["validation"],
            "rejected": rejected,
        }
        self.last_event = f"loaded verified {path.name}"
        self.last_error = "; ".join(errors)

    def close(self) -> None:
        with self.lock:
            self.stop_requested = True
            self.running = False
            self.closed = True
            thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=30)
        # Also wait for a manually requested guarded step, if any.  Its final
        # commit/rollback happens before close returns.
        with self.step_lock:
            pass

    def teacher_analysis(self, board: chess.Board) -> dict:
        if not self.stockfish_path:
            return fallback_teacher(board)
        try:
            teacher = stockfish_teacher(self.stockfish_path, board, self.teacher_depth)
            self.engine_error = ""
            self.engine_verified = True
            self.last_error = ""
            return teacher
        except Exception as exc:
            self.engine_error = f"{type(exc).__name__}: {exc}"
            self.engine_verified = False
            fallback = fallback_teacher(board)
            fallback["engine_error"] = self.engine_error
            self.last_error = f"Stockfish analysis failed: {self.engine_error}"
            return fallback

    def student_move(self, board: chess.Board, teacher: dict) -> chess.Move | None:
        moves = list(board.legal_moves)
        if not moves:
            return None
        if random.random() < self.exploration:
            return random.choice(moves)
        rows = [(score_move(board, move, self.weights), move) for move in moves]
        rows.sort(key=lambda item: item[0], reverse=True)
        return rows[0][1]

    def safe_student_move(self, board: chess.Board, teacher: dict) -> tuple[chess.Move | None, dict]:
        proposed = self.student_move(board, teacher)
        return choose_tactically_safe_move(
            board,
            proposed,
            lambda candidate: score_move(board, candidate, self.weights),
        )

    def opponent_move(self, board: chess.Board, teacher: dict) -> chess.Move | None:
        best = str(teacher.get("best_move") or "")
        if best:
            try:
                move = chess.Move.from_uci(best)
                if move in board.legal_moves:
                    return move
            except Exception:
                pass
        return self.student_move(board, teacher)

    def record_teacher_diagnostic(self, board: chess.Board, chosen: chess.Move | None, teacher: dict) -> dict:
        best = str(teacher.get("best_move") or "")
        if chosen is None or not best:
            return {"sampled": False, "match": False}
        try:
            teacher_move = chess.Move.from_uci(best)
        except Exception:
            return {"sampled": False, "match": False}
        if teacher_move not in board.legal_moves or chosen not in board.legal_moves:
            return {"sampled": False, "match": False}
        match = chosen == teacher_move

        self.teacher_samples += 1
        self.student_matches += int(match)
        self.teacher_buffer.append({
            "fen": board.fen(),
            "teacher": teacher_move.uci(),
            "student": chosen.uci(),
            "match": match,
            "source": teacher.get("source", ""),
        })
        if len(self.teacher_buffer) > TEACHER_BUFFER_LIMIT:
            self.teacher_buffer = self.teacher_buffer[-TEACHER_BUFFER_LIMIT:]
        return {"sampled": True, "match": match, "teacher_move": teacher_move.uci()}

    def apply_rl_update(self, features: dict[str, float], reward_signal: float) -> None:
        clipped = max(-3.0, min(3.0, float(reward_signal)))
        if abs(clipped) < 0.0001:
            return
        self.rl_samples += 1
        self.policy_updates += 1
        self.last_td_error = clipped
        self.td_error_ema = clipped if self.rl_samples == 1 else self.td_error_ema * 0.92 + clipped * 0.08
        rate = float(self.learning_rate)
        for key in FEATURE_KEYS:
            feature_rate = rate if key == "reply_safety" else rate * CORE_LEARNING_SCALE
            self.weights[key] = clamp_student_weight(
                key,
                self.weights[key] + feature_rate * clipped * float(features.get(key, 0.0)),
            )

    def apply_teacher_update(self, board: chess.Board, chosen: chess.Move | None, teacher_move_uci: str) -> None:
        if chosen is None or not teacher_move_uci:
            return
        try:
            teacher_move = chess.Move.from_uci(teacher_move_uci)
        except Exception:
            return
        if teacher_move not in board.legal_moves or chosen not in board.legal_moves or teacher_move == chosen:
            return
        teacher_features = move_features(board, teacher_move)
        chosen_features = move_features(board, chosen)
        rate = float(self.teacher_learning_rate)
        for key in FEATURE_KEYS:
            delta = teacher_features.get(key, 0.0) - chosen_features.get(key, 0.0)
            feature_rate = rate if key == "reply_safety" else rate * CORE_LEARNING_SCALE
            self.weights[key] = clamp_student_weight(key, self.weights[key] + feature_rate * delta)
        self.teacher_updates += 1
        self.policy_updates += 1

    @staticmethod
    def deterministic_student_move(board: chess.Board, weights: dict[str, float]) -> chess.Move | None:
        moves = list(board.legal_moves)
        if not moves:
            return None
        proposed = max(moves, key=lambda move: (score_move(board, move, weights), move.uci()))
        selected, _report = choose_tactically_safe_move(
            board,
            proposed,
            lambda candidate: score_move(board, candidate, weights),
        )
        return selected

    def evaluate_teacher_gap(
        self,
        weights: dict[str, float] | None = None,
        *,
        fens: tuple[str, ...] = GUARD_FENS,
    ) -> dict:
        eval_weights = weights or self.weights
        gaps = []
        matches = 0
        top5 = 0
        choices = []
        for fen in fens:
            board = chess.Board(fen)
            ranked = benchmark_ranked_moves(fen)
            if not ranked:
                continue
            chosen = self.deterministic_student_move(board, eval_weights)
            if chosen is None:
                continue
            best = ranked[0][0]
            score_by_move = dict(ranked)
            rank_by_move = {move: index + 1 for index, (move, _score) in enumerate(ranked)}
            chosen_uci = chosen.uci()
            chosen_rank = rank_by_move[chosen_uci]
            matches += int(chosen_uci == best)
            top5 += int(chosen_rank <= 5)
            best_score = ranked[0][1]
            chosen_score = score_by_move[chosen_uci]
            gap = max(0.0, best_score - chosen_score)
            gaps.append(gap)
            choices.append({"fen": fen, "best": best, "chosen": chosen_uci, "rank": chosen_rank, "gap": gap})
        count = max(1, len(gaps))
        return {
            "positions": count,
            "avg_gap": round(sum(gaps) / count, 4),
            "match_rate": round(matches / count, 4),
            "top5_rate": round(top5 / count, 4),
            "choices": choices,
        }

    def _capture_state_locked(self) -> dict:
        return {
            "board": self.board.copy(stack=True),
            "weights": dict(self.weights),
            "best_weights": dict(self.best_weights),
            "game": self.game,
            "ply": self.ply,
            "reward": self.reward,
            "best_reward": self.best_reward,
            "student_matches": self.student_matches,
            "teacher_samples": self.teacher_samples,
            "teacher_updates": self.teacher_updates,
            "rl_samples": self.rl_samples,
            "policy_updates": self.policy_updates,
            "accepted_chunks": self.accepted_chunks,
            "rejected_chunks": self.rejected_chunks,
            "td_error_ema": self.td_error_ema,
            "last_td_error": self.last_td_error,
            "teacher_buffer": copy.deepcopy(self.teacher_buffer),
            "episode_trace": copy.deepcopy(self.episode_trace),
            "completed_games": self.completed_games,
            "white_wins": self.white_wins,
            "black_wins": self.black_wins,
            "draws": self.draws,
            "history": copy.deepcopy(self.history),
            "moves": copy.deepcopy(self.moves),
            "teacher": dict(self.teacher),
            "last_guard": copy.deepcopy(self.last_guard),
            "accepted_guard": copy.deepcopy(self.accepted_guard),
            "last_event": self.last_event,
            "last_error": self.last_error,
        }

    def _restore_state_locked(self, state: dict) -> None:
        self.board = state["board"].copy(stack=True)
        self.weights = dict(state["weights"])
        self.best_weights = dict(state["best_weights"])
        self.game = state["game"]
        self.ply = state["ply"]
        self.reward = state["reward"]
        self.best_reward = state["best_reward"]
        self.student_matches = state["student_matches"]
        self.teacher_samples = state["teacher_samples"]
        self.teacher_updates = state["teacher_updates"]
        self.rl_samples = state["rl_samples"]
        self.policy_updates = state["policy_updates"]
        self.accepted_chunks = state["accepted_chunks"]
        self.rejected_chunks = state["rejected_chunks"]
        self.td_error_ema = state["td_error_ema"]
        self.last_td_error = state["last_td_error"]
        self.teacher_buffer = copy.deepcopy(state["teacher_buffer"])
        self.episode_trace = copy.deepcopy(state["episode_trace"])
        self.completed_games = state["completed_games"]
        self.white_wins = state["white_wins"]
        self.black_wins = state["black_wins"]
        self.draws = state["draws"]
        self.history = copy.deepcopy(state["history"])
        self.moves = copy.deepcopy(state["moves"])
        self.teacher = dict(state["teacher"])
        self.last_guard = copy.deepcopy(state["last_guard"])
        self.accepted_guard = copy.deepcopy(state["accepted_guard"])
        self.last_event = state["last_event"]
        self.last_error = state["last_error"]

    def apply_episode_result_update(self, final_signal: float) -> None:
        if not self.episode_trace or abs(final_signal) < 0.0001:
            return
        total = len(self.episode_trace)
        for index, row in enumerate(self.episode_trace):
            credit = final_signal * (self.discount ** (total - index - 1))
            self.apply_rl_update(row["features"], credit)

    def mutate_weights(self) -> None:
        scale = float(self.mutation)
        for key in self.weights:
            feature_scale = scale if key == "reply_safety" else scale * CORE_LEARNING_SCALE
            self.weights[key] = clamp_student_weight(
                key,
                self.best_weights[key] + random.uniform(-feature_scale, feature_scale),
            )

    def finish_game(self) -> None:
        result = self.board.result(claim_draw=True)
        final = self.reward
        final_signal = 0.0
        if result == "1-0":
            final += 500
            self.white_wins += 1
            final_signal = 2.5
        elif result == "0-1":
            final -= 500
            self.black_wins += 1
            final_signal = -2.5
        else:
            self.draws += 1
        self.apply_episode_result_update(final_signal)
        self.completed_games += 1
        match_rate = self.student_matches / self.teacher_samples if self.teacher_samples else 0.0
        win_rate = self.white_wins / self.completed_games if self.completed_games else 0.0
        self.history.append({
            "game": self.game,
            "reward": round(final, 1),
            "result": result,
            "ply": self.ply,
            "match_rate": round(match_rate, 3),
            "win_rate": round(win_rate, 3),
        })
        if final >= self.best_reward:
            self.best_reward = final
            self.best_weights = dict(self.weights)
            self.last_event = "new best"
        else:
            self.weights = dict(self.best_weights)
        self.mutate_weights()
        self.game += 1
        self.board = chess.Board()
        self.ply = 0
        self.reward = 0.0
        self.moves = []
        self.episode_trace = []
        self._save_checkpoint_locked()

    def step_once(self) -> dict:
        """Public one-step entry point; it may never bypass acceptance."""
        return self.step_guarded(1)

    def step_guarded(self, count: int) -> dict:
        count = max(1, min(500, int(count)))
        with self.step_lock:
            with self.lock:
                before_state = self._capture_state_locked()
                self.guard_in_progress = True
                self.guard_public_state = copy.deepcopy(before_state)
            try:
                baseline = self.evaluate_teacher_gap(before_state["weights"])
                holdout_baseline = self.evaluate_teacher_gap(before_state["weights"], fens=HOLDOUT_FENS)
                audit_baseline = self.evaluate_teacher_gap(before_state["weights"], fens=AUDIT_FENS)
                for _ in range(count):
                    self._step_once()
                with self.lock:
                    candidate_weights = dict(self.weights)
                candidate = self.evaluate_teacher_gap(candidate_weights)
                holdout_candidate = self.evaluate_teacher_gap(candidate_weights, fens=HOLDOUT_FENS)
                audit_candidate = self.evaluate_teacher_gap(candidate_weights, fens=AUDIT_FENS)
            except Exception as exc:
                with self.lock:
                    self._restore_state_locked(before_state)
                    self.rejected_chunks += 1
                    self.guard_in_progress = False
                    self.guard_public_state = None
                    self.last_guard = {
                        "enabled": True,
                        "accepted": False,
                        "steps": count,
                        "reason": f"candidate evaluation failed: {type(exc).__name__}: {exc}",
                    }
                    self.last_event = "candidate rolled back after evaluation error"
                    self._save_checkpoint_locked()
                raise
            baseline_moves = [row["chosen"] for row in baseline.get("choices", [])]
            candidate_moves = [row["chosen"] for row in candidate.get("choices", [])]
            behavior_changed = candidate_moves != baseline_moves
            guard_improved = (
                self._quality_key(candidate) < self._quality_key(baseline)
                and candidate["avg_gap"] <= baseline["avg_gap"] - self.guard_min_gap_delta
            )
            holdout_improved = (
                self._quality_key(holdout_candidate) < self._quality_key(holdout_baseline)
                and holdout_candidate["avg_gap"] <= holdout_baseline["avg_gap"] + self.guard_holdout_tolerance
                and holdout_candidate["match_rate"] >= holdout_baseline["match_rate"]
                and holdout_candidate["top5_rate"] >= holdout_baseline["top5_rate"]
            )
            audit_ok = self._quality_key(audit_candidate) <= self._quality_key(audit_baseline)
            accepted = behavior_changed and guard_improved and holdout_improved and audit_ok
            if not behavior_changed:
                reason = "no verified policy behavior change"
            elif not guard_improved:
                reason = "guard benchmark did not improve"
            elif not holdout_improved:
                reason = "fixed holdout validation did not improve"
            elif not audit_ok:
                reason = "promotion audit regressed"
            else:
                reason = "guard and holdout improved without promotion-audit regression"
            guard = {
                "enabled": True,
                "accepted": accepted,
                "steps": count,
                "min_gap_delta": self.guard_min_gap_delta,
                "holdout_tolerance": self.guard_holdout_tolerance,
                "behavior_changed": behavior_changed,
                "reason": reason,
                "baseline": baseline,
                "candidate": candidate,
                "holdout_baseline": holdout_baseline,
                "holdout_candidate": holdout_candidate,
                "audit_baseline": audit_baseline,
                "audit_candidate": audit_candidate,
                "baseline_fingerprint": policy_fingerprint(before_state["weights"]),
                "candidate_fingerprint": policy_fingerprint(candidate_weights),
            }
            with self.lock:
                if not accepted:
                    self._restore_state_locked(before_state)
                    self.rejected_chunks += 1
                else:
                    self.accepted_chunks += 1
                    self.accepted_guard = copy.deepcopy(guard)
                self.guard_in_progress = False
                self.guard_public_state = None
                if accepted:
                    self._save_fallback_best_checkpoint_locked(guard)
                self.last_guard = guard
                self.last_event = (
                    f"verified {baseline['avg_gap']:.1f}->{candidate['avg_gap']:.1f}; holdout {holdout_candidate['avg_gap']:.1f}"
                    if accepted
                    else f"candidate rejected: {reason}"
                )
                self._save_checkpoint_locked()
            return guard

    def _step_once(self) -> None:
        with self.lock:
            if self.board.is_game_over(claim_draw=True) or self.ply >= 160:
                self.finish_game()
                return
            board_copy = self.board.copy(stack=False)

        teacher = self.teacher_analysis(board_copy)
        with self.lock:
            self.teacher = teacher
            diagnostic = {"sampled": False, "match": False}
            safety = {"safe": True, "reason": "teacher_or_opponent"}
            action_features = None
            if self.board.turn == chess.WHITE:
                pre_move_board = self.board.copy(stack=False)
                move, safety = self.safe_student_move(self.board, teacher)
                diagnostic = self.record_teacher_diagnostic(pre_move_board, move, teacher)
                if move is not None:
                    action_features = move_features(pre_move_board, move)
                    self.apply_teacher_update(pre_move_board, move, str(teacher.get("best_move") or ""))
            else:
                move = self.opponent_move(self.board, teacher)
            if move is None:
                self.finish_game()
                return
            moving_side = self.board.turn
            before = evaluate_board(self.board, DEFAULT_WEIGHTS)
            san = self.board.san(move)
            self.board.push(move)
            after = evaluate_board(self.board, DEFAULT_WEIGHTS)
            delta = after - before
            if moving_side == chess.WHITE:
                self.reward += delta / 25
                if action_features is not None:
                    self.episode_trace.append({"ply": self.ply, "features": action_features})
                    self.apply_rl_update(action_features, delta / 140.0)
            else:
                self.reward -= delta / 25
                if self.episode_trace:
                    self.apply_rl_update(self.episode_trace[-1]["features"], delta / 220.0)
            self.ply += 1
            self.moves.append({
                "ply": self.ply,
                "san": san,
                "uci": move.uci(),
                "teacher": teacher.get("best_move", ""),
                "eval_cp": teacher.get("eval_cp", 0),
                "match": bool(diagnostic.get("match")),
                "loss": round(float(self.last_td_error), 3),
                "safety": safety.get("reason", ""),
                "fallback": bool(safety.get("fallback_applied")),
            })
            self.last_event = f"{'white' if not self.board.turn else 'black'} {san}"
            if self.board.is_game_over(claim_draw=True) or self.ply >= 160:
                self.finish_game()

    def loop(self) -> None:
        while True:
            with self.lock:
                if self.stop_requested:
                    return
                active = self.running
                chunk = int(self.chunk_moves)
            if not active:
                time.sleep(0.08)
                continue
            try:
                # Pair the final run-state check with the same transaction lock
                # used by pause/reset. This prevents a chunk from starting just
                # after Pause has already returned.
                with self.step_lock:
                    with self.lock:
                        if self.stop_requested:
                            return
                        if not self.running:
                            continue
                    self.step_guarded(max(1, chunk))
            except Exception as exc:
                with self.lock:
                    self.last_error = str(exc)
                    self.last_event = "error"
                    self.running = False
            time.sleep(0.02)

    def start(self) -> None:
        with self.lock:
            if self.closed:
                raise RuntimeError("trainer is closed")
            self.running = True
            self.stop_requested = False
            self.last_event = "training candidates; only verified improvements are retained"
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self.loop, daemon=True)
                self.thread.start()

    def pause(self) -> None:
        with self.lock:
            self.running = False
        with self.step_lock:
            pass
        with self.lock:
            self.last_event = "paused"

    def update_config(self, updates: dict) -> None:
        with self.step_lock, self.lock:
            weights = updates.get("weights")
            if isinstance(weights, dict):
                candidate_weights = dict(self.weights)
                for key in candidate_weights:
                    if key in weights:
                        candidate_weights[key] = clamp_student_weight(key, float(weights[key]))
                if candidate_weights != self.weights:
                    raise ValueError(
                        "manual policy-weight changes are disabled; use guarded training"
                    )
            self.teacher_depth = max(1, min(18, int(updates.get("teacher_depth", self.teacher_depth))))
            self.student_depth = max(1, min(3, int(updates.get("student_depth", self.student_depth))))
            self.chunk_moves = max(1, min(120, int(updates.get("chunk_moves", self.chunk_moves))))
            self.exploration = max(0.0, min(0.8, float(updates.get("exploration", self.exploration))))
            self.mutation = max(0.0, min(1.0, float(updates.get("mutation", self.mutation))))
            self.learning_rate = max(0.0, min(0.5, float(updates.get("learning_rate", self.learning_rate))))
            self.teacher_learning_rate = max(0.0, min(0.5, float(updates.get("teacher_learning_rate", self.teacher_learning_rate))))
            self.guard_enabled = True
            self.guard_min_gap_delta = max(0.0, min(1000.0, float(updates.get("guard_min_gap_delta", self.guard_min_gap_delta))))
            self.guard_holdout_tolerance = max(
                0.0,
                min(1000.0, float(updates.get("guard_holdout_tolerance", self.guard_holdout_tolerance))),
            )
            self.last_event = "settings updated"
            self._save_checkpoint_locked()

    def snapshot(self) -> dict:
        with self.lock:
            public = self.guard_public_state if self.guard_in_progress else None

            def state_value(name):
                return public[name] if public is not None else getattr(self, name)

            board = state_value("board").copy(stack=False)
            student_matches = int(state_value("student_matches"))
            teacher_samples = int(state_value("teacher_samples"))
            completed_games = int(state_value("completed_games"))
            white_wins = int(state_value("white_wins"))
            black_wins = int(state_value("black_wins"))
            draws = int(state_value("draws"))
            match_rate = student_matches / teacher_samples if teacher_samples else 0.0
            win_rate = white_wins / completed_games if completed_games else 0.0
            best_reward = float(state_value("best_reward"))
            return {
                "running": self.running,
                "guard_in_progress": self.guard_in_progress,
                "fen": board.fen(),
                "turn": "white" if board.turn == chess.WHITE else "black",
                "legal_moves": [move.uci() for move in board.legal_moves],
                "game": int(state_value("game")),
                "ply": int(state_value("ply")),
                "result": board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*",
                "reward": round(float(state_value("reward")), 2),
                "best_reward": 0 if not math.isfinite(best_reward) else round(best_reward, 2),
                "student_matches": student_matches,
                "learning": {
                    "mode": "reward-shaped teacher ranker",
                    "samples": int(state_value("rl_samples")),
                    "teacher_samples": teacher_samples,
                    "matches": student_matches,
                    "match_rate": round(match_rate, 4),
                    "teacher_fit_used": True,
                    "teacher_updates": int(state_value("teacher_updates")),
                    "update_signal_ema": round(float(state_value("td_error_ema")), 4),
                    "last_update_signal": round(float(state_value("last_td_error")), 4),
                    "updates": int(state_value("policy_updates")),
                    "buffer": len(state_value("teacher_buffer")),
                    "learning_rate": self.learning_rate,
                    "teacher_learning_rate": self.teacher_learning_rate,
                    "discount": self.discount,
                    "accepted_chunks": int(state_value("accepted_chunks")),
                    "rejected_chunks": int(state_value("rejected_chunks")),
                    "completed_games": completed_games,
                    "wins": white_wins,
                    "losses": black_wins,
                    "draws": draws,
                    "win_rate": round(win_rate, 4),
                },
                "weights": dict(state_value("weights")),
                "guard": copy.deepcopy(state_value("last_guard")),
                "history": copy.deepcopy(state_value("history")[-80:]),
                "moves": copy.deepcopy(state_value("moves")[-20:]),
                "teacher": dict(state_value("teacher")),
                "stockfish": {
                    "available": bool(self.stockfish_path),
                    "path": self.stockfish_path,
                    "connected": self.engine_verified and not bool(self.engine_error) and bool(self.stockfish_path),
                    "error": self.engine_error,
                },
                "config": {
                    "teacher_depth": self.teacher_depth,
                    "student_depth": self.student_depth,
                    "chunk_moves": self.chunk_moves,
                    "exploration": self.exploration,
                    "mutation": self.mutation,
                    "learning_rate": self.learning_rate,
                    "teacher_learning_rate": self.teacher_learning_rate,
                    "guard_enabled": self.guard_enabled,
                    "guard_min_gap_delta": self.guard_min_gap_delta,
                    "guard_holdout_tolerance": self.guard_holdout_tolerance,
                },
                "checkpoint": {
                    "current": str(self.checkpoint_path),
                    "best": str(self.best_checkpoint_path),
                    "loaded": dict(self.loaded_checkpoint),
                },
                "last_event": (
                    "guard evaluating private candidate; serving last accepted state"
                    if public is not None
                    else self.last_event
                ),
                "last_error": str(state_value("last_error")),
            }


trainer = Trainer()
app = Flask(__name__, static_folder=str(WEB_DIR))


def json_error(message, status=400):
    return jsonify({"ok": False, "error": str(message)}), status


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:path>")
def static_file(path):
    return send_from_directory(WEB_DIR, path)


@app.get("/api/state")
def api_state():
    return jsonify(trainer.snapshot())


@app.post("/api/start")
def api_start():
    try:
        trainer.start()
    except RuntimeError as exc:
        return json_error(exc, 409)
    return jsonify(trainer.snapshot())


@app.post("/api/pause")
def api_pause():
    trainer.pause()
    return jsonify(trainer.snapshot())


@app.post("/api/reset")
def api_reset():
    trainer.pause()
    trainer.reset(load_checkpoint=False)
    with trainer.lock:
        trainer._save_checkpoint_locked()
    return jsonify(trainer.snapshot())


@app.post("/api/step")
def api_step():
    try:
        count = int((request.get_json(silent=True) or {}).get("count", 1))
    except (TypeError, ValueError) as exc:
        return json_error(exc, 400)
    trainer.step_guarded(count)
    return jsonify(trainer.snapshot())


@app.post("/api/config")
def api_config():
    try:
        trainer.update_config(request.get_json(silent=True) or {})
    except (TypeError, ValueError) as exc:
        return json_error(exc, 400)
    return jsonify(trainer.snapshot())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7873)
    args = parser.parse_args()
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
