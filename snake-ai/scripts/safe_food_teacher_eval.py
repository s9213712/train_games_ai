import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main"
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from safe_food_teacher import safe_food_action  # noqa: E402
from snake_game import SnakeGame  # noqa: E402


def run_episode(board_size, seed, max_steps):
    game = SnakeGame(seed=seed, board_size=board_size, silent_mode=True)
    food_steps = []
    last_food_step = 0
    done = False
    steps = 0

    while not done and steps < max_steps and len(game.snake) < game.grid_size:
        action = safe_food_action(game)
        done, info = game.step(action)
        steps += 1
        if info["food_obtained"]:
            food_steps.append(steps - last_food_step)
            last_food_step = steps

    return {
        "seed": seed,
        "score": game.score,
        "length": len(game.snake),
        "steps": steps,
        "filled": len(game.snake) == game.grid_size,
        "done": done,
        "food_count": len(food_steps),
        "avg_steps_per_food": sum(food_steps) / len(food_steps) if food_steps else 0.0,
        "max_steps_per_food": max(food_steps) if food_steps else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the safe-food teacher headlessly.")
    parser.add_argument("--board-size", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--max-steps", type=int, default=12000)
    args = parser.parse_args()

    results = [
        run_episode(args.board_size, args.seed + episode, args.max_steps)
        for episode in range(args.episodes)
    ]
    summary = {
        "board_size": args.board_size,
        "episodes": args.episodes,
        "filled": sum(item["filled"] for item in results),
        "score_avg": sum(item["score"] for item in results) / len(results),
        "score_max": max(item["score"] for item in results),
        "length_avg": sum(item["length"] for item in results) / len(results),
        "length_max": max(item["length"] for item in results),
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
