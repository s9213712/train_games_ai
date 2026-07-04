from __future__ import annotations

import random
from dataclasses import dataclass, field


COLS = 10
ROWS = 20
PIECES = (
    (((0, 0), (0, 1), (0, 2), (0, 3)),),
    (((0, 0), (1, 0), (1, 1), (1, 2)),),
    (((0, 2), (1, 0), (1, 1), (1, 2)),),
    (((0, 0), (0, 1), (1, 0), (1, 1)),),
    (((0, 1), (0, 2), (1, 0), (1, 1)),),
    (((0, 1), (1, 0), (1, 1), (1, 2)),),
    (((0, 0), (0, 1), (1, 1), (1, 2)),),
)


def normalize(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def rotations(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    out = []
    current = normalize(shape)
    for _ in range(4):
        if current not in out:
            out.append(current)
        current = normalize(tuple((c, -r) for r, c in current))
    return tuple(out)


ROTATIONS = tuple(rotations(piece[0]) for piece in PIECES)
UNKNOWN_NEXT_PIECE = len(PIECES) // 2


@dataclass
class TetrisStats:
    score: int = 0
    lines: int = 0
    pieces: int = 0
    tetrises: int = 0
    holds: int = 0
    max_height: int = 0
    holes: int = 0
    bumpiness: int = 0
    wells: int = 0


@dataclass
class TetrisState:
    board: list[list[int]] = field(default_factory=lambda: [[0 for _ in range(COLS)] for _ in range(ROWS)])
    current: int = 0
    next_piece: int = 0
    hold_piece: int | None = None
    bag: list[int] = field(default_factory=list)
    stats: TetrisStats = field(default_factory=TetrisStats)
    done: bool = False
    last_event: str = "ready"


class TetrisEnv:
    """Fast Tetris afterstate environment for backend training."""

    def __init__(self, *, seed: int | None = None, max_pieces: int = 900) -> None:
        self.random = random.Random(seed)
        self.max_pieces = int(max_pieces)
        self.state = TetrisState()
        self.replay: list[dict] = []
        self.last_reward_terms: dict[str, float] = {}

    def reset(self) -> list[float]:
        self.state = TetrisState()
        self.state.current = self._draw_piece()
        self.state.next_piece = self._draw_piece()
        self.state.last_event = "spawn"
        self.replay = []
        self.last_reward_terms = {}
        self._record_frame()
        return self.observation()

    def observation(self) -> list[float]:
        feats = self.features(self.state.board, 0)
        return self.feature_vector(feats, self.state.current, self.state.next_piece)

    def legal_moves(self) -> list[dict]:
        moves = self.moves_for(self.state.board, self.state.current, self.state.next_piece)
        draws, bag, random_state = self._simulate_draws(1)
        for move in moves:
            move.update(
                {
                    "hold": False,
                    "piece": self.state.current,
                    "next_current": self.state.next_piece,
                    "next_piece": draws[0],
                    "next_hold_piece": self.state.hold_piece,
                    "bag_after": bag,
                    "random_state_after": random_state,
                }
            )
        hold_moves = self.hold_moves()
        return moves + hold_moves

    def hold_moves(self) -> list[dict]:
        if self.state.hold_piece is None:
            draws, bag, random_state = self._simulate_draws(2)
            active_piece = self.state.next_piece
            visible_next = draws[0]
            next_current = draws[0]
            next_piece = draws[1]
            next_hold_piece = self.state.current
        else:
            draws, bag, random_state = self._simulate_draws(1)
            active_piece = self.state.hold_piece
            visible_next = self.state.next_piece
            next_current = self.state.next_piece
            next_piece = draws[0]
            next_hold_piece = self.state.current
        moves = self.moves_for(self.state.board, active_piece, visible_next)
        for move in moves:
            move.update(
                {
                    "hold": True,
                    "piece": active_piece,
                    "next_current": next_current,
                    "next_piece": next_piece,
                    "next_hold_piece": next_hold_piece,
                    "bag_after": bag,
                    "random_state_after": random_state,
                }
            )
            move["immediate_reward"] = round(float(move["immediate_reward"]) - 0.03, 4)
            move["immediate_terms"] = dict(move["immediate_terms"], hold=-0.03)
        return moves

    def future_legal_moves_for(self, move: dict, *, include_hold: bool = False) -> list[dict]:
        current = int(move.get("next_current", self.state.next_piece))
        next_piece = int(move.get("next_piece", UNKNOWN_NEXT_PIECE))
        hold_piece = move.get("next_hold_piece")
        board = move["board"]
        moves = self.moves_for(board, current, next_piece)
        for row in moves:
            row.update({"hold": False, "piece": current})
        if not include_hold:
            return moves
        if hold_piece is None:
            active_piece = next_piece
            visible_next = UNKNOWN_NEXT_PIECE
            next_current = UNKNOWN_NEXT_PIECE
            next_after = UNKNOWN_NEXT_PIECE
            next_hold_piece = current
        else:
            active_piece = int(hold_piece)
            visible_next = next_piece
            next_current = next_piece
            next_after = UNKNOWN_NEXT_PIECE
            next_hold_piece = current
        hold_moves = self.moves_for(board, active_piece, visible_next)
        for row in hold_moves:
            row.update(
                {
                    "hold": True,
                    "piece": active_piece,
                    "next_current": next_current,
                    "next_piece": next_after,
                    "next_hold_piece": next_hold_piece,
                }
            )
            row["immediate_reward"] = round(float(row["immediate_reward"]) - 0.03, 4)
            row["immediate_terms"] = dict(row["immediate_terms"], hold=-0.03)
        return moves + hold_moves

    def moves_for(self, board: list[list[int]], piece_id: int, next_piece: int) -> list[dict]:
        moves = []
        before = self.features(board, 0)
        for rot_index, shape in enumerate(ROTATIONS[piece_id]):
            width = max(c for _, c in shape) + 1
            for col in range(COLS - width + 1):
                row = self.drop_row(board, shape, col)
                placed = self.place(board, shape, row, col, piece_id + 1)
                if placed is None:
                    continue
                feats = self.features(placed["board"], placed["cleared"])
                feats["eroded_cells"] = placed["eroded_cells"]
                immediate_terms = self.reward_terms(before, feats, placed["cleared"], False)
                moves.append(
                    {
                        "rotation": rot_index,
                        "col": col,
                        "row": row,
                        "hold": False,
                        "piece": piece_id,
                        "board": placed["board"],
                        "cleared": placed["cleared"],
                        "eroded_cells": placed["eroded_cells"],
                        "features": feats,
                        "immediate_reward": round(sum(immediate_terms.values()), 4),
                        "immediate_terms": immediate_terms,
                        "vector": self.feature_vector(feats, piece_id, next_piece),
                    }
                )
        return moves

    def step(self, move: dict, *, check_terminal: bool = True) -> tuple[list[float], float, bool, dict]:
        if self.state.done:
            return self.observation(), 0.0, True, self.info()
        before = self.features(self.state.board, 0)
        if not move:
            self.state.done = True
            self.state.last_event = "top out"
            reward_terms = {"survival": -5.0, "lines": 0.0, "shape": -4.0, "risk": -3.0}
            reward = sum(reward_terms.values())
            self.last_reward_terms = reward_terms
            self._record_frame(reward=reward, reward_terms=reward_terms)
            return self.observation(), reward, True, self.info()

        self.state.board = [row[:] for row in move["board"]]
        cleared = int(move["cleared"])
        self.state.stats.pieces += 1
        if move.get("hold"):
            self.state.stats.holds += 1
        self.state.stats.lines += cleared
        self.state.stats.score += (0, 100, 300, 500, 800)[cleared] + 1
        if cleared == 4:
            self.state.stats.tetrises += 1
        after = self.features(self.state.board, cleared)
        after["eroded_cells"] = int(move.get("eroded_cells", 0))
        self.state.stats.max_height = max(self.state.stats.max_height, after["max_height"])
        self.state.stats.holes = after["holes"]
        self.state.stats.bumpiness = after["bumpiness"]
        self.state.stats.wells = after["wells"]
        action = "hold" if move.get("hold") else "place"
        self.state.last_event = f"{action} p{move.get('piece', self.state.current)} r{move['rotation']} c{move['col']} clear {cleared}"
        if "next_current" in move and "next_piece" in move:
            self.state.current = int(move["next_current"])
            self.state.next_piece = int(move["next_piece"])
            self.state.hold_piece = move.get("next_hold_piece")
            self.state.bag = list(move.get("bag_after", self.state.bag))
            if "random_state_after" in move:
                self.random.setstate(move["random_state_after"])
        else:
            self.state.current = self.state.next_piece
            self.state.next_piece = self._draw_piece()
        if self.state.stats.pieces >= self.max_pieces or (check_terminal and not self.legal_moves()):
            self.state.done = True
            if self.state.stats.pieces < self.max_pieces:
                self.state.last_event = "top out"
        terms = self.reward_terms(before, after, cleared, self.state.done)
        if move.get("hold"):
            terms["hold"] = -0.03
        reward = sum(terms.values())
        self.last_reward_terms = terms
        self._record_frame(move=move, reward=reward, reward_terms=terms)
        return self.observation(), reward, self.state.done, self.info()

    def info(self) -> dict:
        feats = self.features(self.state.board, 0)
        return {
            "score": self.state.stats.score,
            "lines": self.state.stats.lines,
            "pieces": self.state.stats.pieces,
            "tetrises": self.state.stats.tetrises,
            "holds": self.state.stats.holds,
            "done": self.state.done,
            "current": self.state.current,
            "next_piece": self.state.next_piece,
            "hold_piece": self.state.hold_piece,
            "last_event": self.state.last_event,
            "features": feats,
            "reward_terms": dict(self.last_reward_terms),
        }

    def reward_terms(self, before: dict, after: dict, cleared: int, done: bool) -> dict[str, float]:
        tetris_ready_delta = after["tetris_ready"] - before["tetris_ready"]
        well_delta = after["right_well_depth"] - before["right_well_depth"]
        covered_delta = before["covered_holes"] - after["covered_holes"]
        safe_setup = after["holes"] <= 8 and after["max_height"] <= 16
        if safe_setup:
            well_reward = well_delta * 0.45 + tetris_ready_delta * 1.0 + after["tetris_ready"] * 0.14 + after["right_well_depth"] * 0.04
        else:
            well_reward = min(0.18, max(-0.4, well_delta * 0.06)) - max(0, tetris_ready_delta) * 0.3
        return {
            "survival": -8.0 if done and self.state.stats.pieces < self.max_pieces else 0.02,
            "lines": (0.0, 0.25, 1.1, 2.4, 12.0)[cleared],
            "shape": round((before["holes"] - after["holes"]) * 1.1 + (before["bumpiness"] - after["bumpiness"]) * 0.06 + covered_delta * 0.35, 4),
            "risk": round((before["height"] - after["height"]) * 0.012 + (before["max_height"] - after["max_height"]) * 0.1, 4),
            "well": round(well_reward, 4),
            "eroded": round(after["eroded_cells"] * 0.35, 4),
            "tetris": 7.5 if cleared == 4 else 0.0,
        }

    def _draw_piece(self) -> int:
        if not self.state.bag:
            self.state.bag = list(range(len(PIECES)))
            self.random.shuffle(self.state.bag)
        return self.state.bag.pop()

    def _simulate_draws(self, count: int) -> tuple[list[int], list[int], object]:
        rng = random.Random()
        rng.setstate(self.random.getstate())
        bag = list(self.state.bag)
        draws = []
        for _ in range(count):
            if not bag:
                bag = list(range(len(PIECES)))
                rng.shuffle(bag)
            draws.append(bag.pop())
        return draws, bag, rng.getstate()

    def _record_frame(self, **extra) -> None:
        frame = self.info()
        frame["board"] = [row[:] for row in self.state.board]
        frame.update(extra)
        if "move" in frame:
            move = frame["move"]
            frame["move"] = {
                "rotation": move["rotation"],
                "col": move["col"],
                "row": move["row"],
                "cleared": move["cleared"],
                "hold": bool(move.get("hold")),
                "piece": move.get("piece"),
            }
        self.replay.append(frame)
        replay_limit = self.max_pieces + 1
        if len(self.replay) > replay_limit:
            self.replay = self.replay[-replay_limit:]

    @staticmethod
    def collide(board: list[list[int]], shape: tuple[tuple[int, int], ...], row: int, col: int) -> bool:
        for r, c in shape:
            rr = row + r
            cc = col + c
            if cc < 0 or cc >= COLS or rr >= ROWS:
                return True
            if rr >= 0 and board[rr][cc]:
                return True
        return False

    @classmethod
    def drop_row(cls, board: list[list[int]], shape: tuple[tuple[int, int], ...], col: int) -> int:
        row = -4
        while not cls.collide(board, shape, row + 1, col):
            row += 1
        return row

    @classmethod
    def place(cls, board: list[list[int]], shape: tuple[tuple[int, int], ...], row: int, col: int, value: int) -> dict | None:
        out = [line[:] for line in board]
        placed_cells = set()
        for r, c in shape:
            rr = row + r
            if rr < 0:
                return None
            out[rr][col + c] = value
            placed_cells.add((rr, col + c))
        cleared_rows = {index for index, line in enumerate(out) if all(line)}
        eroded_cells = sum(1 for cell in placed_cells if cell[0] in cleared_rows)
        kept = [line for line in out if any(cell == 0 for cell in line)]
        cleared = ROWS - len(kept)
        while len(kept) < ROWS:
            kept.insert(0, [0 for _ in range(COLS)])
        return {"board": kept, "cleared": cleared, "eroded_cells": eroded_cells * cleared}

    @staticmethod
    def features(board: list[list[int]], cleared: int) -> dict:
        heights = []
        holes = 0
        covered_holes = 0
        row_transitions = 0
        col_transitions = 0
        for c in range(COLS):
            seen = False
            blockers = 0
            height = 0
            prev = 1
            for r in range(ROWS):
                filled = 1 if board[r][c] else 0
                if filled and not seen:
                    height = ROWS - r
                    seen = True
                    blockers += 1
                elif filled and seen:
                    blockers += 1
                elif not filled and seen:
                    holes += 1
                    covered_holes += blockers
                if filled != prev:
                    col_transitions += 1
                prev = filled
            if prev == 0:
                col_transitions += 1
            heights.append(height)
        for row in board:
            prev = 1
            for cell in row:
                filled = 1 if cell else 0
                if filled != prev:
                    row_transitions += 1
                prev = filled
            if prev == 0:
                row_transitions += 1
        wells = 0
        right_well_depth = max(0, heights[COLS - 2] - heights[COLS - 1]) if COLS >= 2 else 0
        right_well_depth = min(8, right_well_depth)
        for c in range(COLS):
            left = ROWS if c == 0 else heights[c - 1]
            right = ROWS if c == COLS - 1 else heights[c + 1]
            if heights[c] < left and heights[c] < right:
                depth = min(left, right) - heights[c]
                wells += depth
        bumpiness = sum(abs(heights[c] - heights[c + 1]) for c in range(COLS - 1))
        height = sum(heights)
        eroded_cells = cleared * sum(1 for cell in board[-1] if cell)
        tetris_ready = 0
        for row in board[max(0, ROWS - 8) :]:
            if row[-1] == 0 and sum(1 for cell in row[:-1] if cell) >= 9:
                tetris_ready += 1
        if holes > 10 or max(heights) > 17:
            tetris_ready = min(tetris_ready, 1)
        return {
            "lines": int(cleared),
            "height": int(height),
            "max_height": int(max(heights) if heights else 0),
            "holes": int(holes),
            "covered_holes": int(covered_holes),
            "bumpiness": int(bumpiness),
            "wells": int(wells),
            "right_well_depth": int(right_well_depth),
            "eroded_cells": int(eroded_cells),
            "tetris_ready": int(tetris_ready),
            "row_transitions": int(row_transitions),
            "col_transitions": int(col_transitions),
        }

    @staticmethod
    def feature_vector(feats: dict, current: int, next_piece: int) -> list[float]:
        return [
            1.0,
            feats["lines"] / 4.0,
            feats["height"] / 200.0,
            feats["max_height"] / 20.0,
            feats["holes"] / 80.0,
            feats["covered_holes"] / 300.0,
            feats["bumpiness"] / 80.0,
            feats["wells"] / 40.0,
            feats["right_well_depth"] / 20.0,
            feats["eroded_cells"] / 40.0,
            feats["tetris_ready"] / 4.0,
            feats["row_transitions"] / 200.0,
            feats["col_transitions"] / 200.0,
            current / 6.0,
            next_piece / 6.0,
        ]
