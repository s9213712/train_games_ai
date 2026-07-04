from __future__ import annotations

import argparse
import math
import os
import random
import select
import shutil
import subprocess
import threading
import time
from pathlib import Path

import chess
from flask import Flask, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"

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
}
FEATURE_KEYS = tuple(DEFAULT_WEIGHTS.keys())
TEACHER_BUFFER_LIMIT = 800


def resolve_stockfish_path() -> str:
    for key in ("STOCKFISH_PATH", "HTML_LEARNING_CHESS_STOCKFISH_PATH"):
        value = os.environ.get(key, "").strip()
        if value:
            path = Path(value).expanduser()
            if path.exists() and os.access(path, os.X_OK):
                return str(path.resolve())
    found = shutil.which("stockfish")
    return str(Path(found).resolve()) if found else ""


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
    return {key: float(features.get(key, 0.0)) for key in FEATURE_KEYS}


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


def fallback_teacher(board: chess.Board, depth: int = 2) -> dict:
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
    best = rows[0] if rows else {"move": "", "score_cp": 0, "pv": []}
    return {
        "available": False,
        "source": "fallback",
        "best_move": best["move"],
        "eval_cp": best["score_cp"],
        "lines": rows[:5],
    }


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
        self.running = False
        self.thread: threading.Thread | None = None
        self.stop_requested = False
        self.stockfish_path = resolve_stockfish_path()
        self.engine_error = ""
        self.teacher_depth = 8
        self.student_depth = 1
        self.chunk_moves = 1
        self.exploration = 0.08
        self.mutation = 0.12
        self.learning_rate = 0.025
        self.discount = 0.94
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.board = chess.Board()
            self.weights = dict(DEFAULT_WEIGHTS)
            self.best_weights = dict(DEFAULT_WEIGHTS)
            self.game = 0
            self.ply = 0
            self.reward = 0.0
            self.best_reward = -math.inf
            self.student_matches = 0
            self.teacher_samples = 0
            self.rl_samples = 0
            self.policy_updates = 0
            self.td_error_ema = 0.0
            self.last_td_error = 0.0
            self.teacher_buffer = []
            self.episode_trace = []
            self.completed_games = 0
            self.white_wins = 0
            self.black_wins = 0
            self.draws = 0
            self.history = []
            self.moves = []
            self.teacher = {"available": False, "source": "none", "best_move": "", "eval_cp": 0, "lines": []}
            self.last_event = "reset"
            self.last_error = ""

    def close(self) -> None:
        with self.lock:
            self.stop_requested = True
            self.running = False

    def teacher_analysis(self, board: chess.Board) -> dict:
        if not self.stockfish_path:
            return fallback_teacher(board)
        try:
            teacher = stockfish_teacher(self.stockfish_path, board, self.teacher_depth)
            self.engine_error = ""
            self.last_error = ""
            return teacher
        except Exception as exc:
            self.engine_error = f"{type(exc).__name__}: {exc}"
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
            self.weights[key] = max(-2.0, min(3.0, self.weights[key] + rate * clipped * float(features.get(key, 0.0))))

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
            self.weights[key] = max(-2.0, min(3.0, self.best_weights[key] + random.uniform(-scale, scale)))

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

    def step_once(self) -> None:
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
                for _ in range(max(1, chunk)):
                    self.step_once()
            except Exception as exc:
                with self.lock:
                    self.last_error = str(exc)
                    self.last_event = "error"
                    self.running = False
            time.sleep(0.02)

    def start(self) -> None:
        with self.lock:
            self.running = True
            self.stop_requested = False
            self.last_event = "training"
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self.loop, daemon=True)
                self.thread.start()

    def pause(self) -> None:
        with self.lock:
            self.running = False
            self.last_event = "paused"

    def update_config(self, updates: dict) -> None:
        with self.lock:
            self.teacher_depth = max(1, min(18, int(updates.get("teacher_depth", self.teacher_depth))))
            self.student_depth = max(1, min(3, int(updates.get("student_depth", self.student_depth))))
            self.chunk_moves = max(1, min(30, int(updates.get("chunk_moves", self.chunk_moves))))
            self.exploration = max(0.0, min(0.8, float(updates.get("exploration", self.exploration))))
            self.mutation = max(0.0, min(1.0, float(updates.get("mutation", self.mutation))))
            self.learning_rate = max(0.0, min(0.5, float(updates.get("learning_rate", self.learning_rate))))
            weights = updates.get("weights")
            if isinstance(weights, dict):
                for key in self.weights:
                    if key in weights:
                        self.weights[key] = max(-2.0, min(3.0, float(weights[key])))
            self.last_event = "settings updated"

    def snapshot(self) -> dict:
        with self.lock:
            board = self.board.copy(stack=False)
            match_rate = self.student_matches / self.teacher_samples if self.teacher_samples else 0.0
            win_rate = self.white_wins / self.completed_games if self.completed_games else 0.0
            strength = int(800 + win_rate * 500 + min(400, self.policy_updates * 0.8) + max(-200, min(300, self.best_reward if math.isfinite(self.best_reward) else 0)) / 3)
            return {
                "running": self.running,
                "fen": board.fen(),
                "turn": "white" if board.turn == chess.WHITE else "black",
                "legal_moves": [move.uci() for move in board.legal_moves],
                "game": self.game,
                "ply": self.ply,
                "result": board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*",
                "reward": round(self.reward, 2),
                "best_reward": 0 if not math.isfinite(self.best_reward) else round(self.best_reward, 2),
                "student_matches": self.student_matches,
                "learning": {
                    "mode": "reinforcement",
                    "samples": self.rl_samples,
                    "teacher_samples": self.teacher_samples,
                    "matches": self.student_matches,
                    "match_rate": round(match_rate, 4),
                    "teacher_fit_used": False,
                    "loss": round(abs(self.td_error_ema), 4),
                    "td_error": round(self.td_error_ema, 4),
                    "last_td_error": round(self.last_td_error, 4),
                    "updates": self.policy_updates,
                    "buffer": len(self.teacher_buffer),
                    "learning_rate": self.learning_rate,
                    "discount": self.discount,
                    "strength": strength,
                    "completed_games": self.completed_games,
                    "wins": self.white_wins,
                    "losses": self.black_wins,
                    "draws": self.draws,
                    "win_rate": round(win_rate, 4),
                },
                "weights": dict(self.weights),
                "history": self.history[-80:],
                "moves": self.moves[-20:],
                "teacher": dict(self.teacher),
                "stockfish": {
                    "available": bool(self.stockfish_path),
                    "path": self.stockfish_path,
                    "connected": not bool(self.engine_error) and bool(self.stockfish_path),
                    "error": self.engine_error,
                },
                "config": {
                    "teacher_depth": self.teacher_depth,
                    "student_depth": self.student_depth,
                    "chunk_moves": self.chunk_moves,
                    "exploration": self.exploration,
                    "mutation": self.mutation,
                    "learning_rate": self.learning_rate,
                },
                "last_event": self.last_event,
                "last_error": self.last_error,
            }


trainer = Trainer()
app = Flask(__name__, static_folder=str(WEB_DIR))


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
    trainer.start()
    return jsonify(trainer.snapshot())


@app.post("/api/pause")
def api_pause():
    trainer.pause()
    return jsonify(trainer.snapshot())


@app.post("/api/reset")
def api_reset():
    trainer.pause()
    trainer.reset()
    return jsonify(trainer.snapshot())


@app.post("/api/step")
def api_step():
    count = int((request.get_json(silent=True) or {}).get("count", 1))
    for _ in range(max(1, min(500, count))):
        trainer.step_once()
    return jsonify(trainer.snapshot())


@app.post("/api/config")
def api_config():
    trainer.update_config(request.get_json(silent=True) or {})
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
