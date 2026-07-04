from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rl_trainer import RLTrainer
from tetris_env import TetrisEnv


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start_text, end_text = value.split(":", 1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise argparse.ArgumentTypeError("seed range end must be >= start")
        return list(range(start, end + 1))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return float(ordered[index])


def evaluate(root: Path, seeds: list[int], max_pieces: int, future_hold: bool) -> dict:
    trainer = RLTrainer(root)
    rows = []
    for seed in seeds:
        env = TetrisEnv(seed=seed, max_pieces=max_pieces)
        env.reset()
        done = False
        total_reward = 0.0
        while not done:
            move = trainer.policy.choose(
                env.legal_moves(),
                epsilon=0.0,
                temperature=0.0,
                next_moves_provider=lambda row: env.future_legal_moves_for(row, include_hold=future_hold),
                lookahead_weight=trainer.lookahead_weight,
                lookahead_candidates=trainer.lookahead_candidates,
                lookahead_include_hold=future_hold,
            )
            _obs, reward, done, _info = env.step(move or {}, check_terminal=False)
            total_reward += reward
        info = env.info()
        rows.append(
            {
                "seed": seed,
                "score": info["score"],
                "lines": info["lines"],
                "pieces": info["pieces"],
                "tetrises": info["tetrises"],
                "holds": info.get("holds", 0),
                "reward": round(total_reward, 3),
                "piece_cap_hit": info["pieces"] >= max_pieces,
                "topout": info["pieces"] < max_pieces,
            }
        )
    scores = [row["score"] for row in rows]
    tetrises = [row["tetrises"] for row in rows]
    return {
        "episodes": len(rows),
        "seeds": seeds,
        "max_pieces": max_pieces,
        "future_hold": future_hold,
        "avg_score": round(sum(scores) / max(1, len(scores)), 2),
        "median_score": round(statistics.median(scores), 2) if scores else 0.0,
        "p10_score": round(percentile(scores, 0.10), 2),
        "p90_score": round(percentile(scores, 0.90), 2),
        "avg_tetrises": round(sum(tetrises) / max(1, len(tetrises)), 3),
        "topout_rate": round(sum(row["topout"] for row in rows) / max(1, len(rows)), 3),
        "piece_cap_hit_rate": round(sum(row["piece_cap_hit"] for row in rows) / max(1, len(rows)), 3),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict Tetris policy evaluation")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("1000:1099"))
    parser.add_argument("--max-pieces", type=int, default=900)
    parser.add_argument("--future-hold", action="store_true")
    args = parser.parse_args()
    seeds = args.seeds[: max(1, args.episodes)]
    print(json.dumps(evaluate(args.root, seeds, args.max_pieces, args.future_hold), indent=2))


if __name__ == "__main__":
    main()
