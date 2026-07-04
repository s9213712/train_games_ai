import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main"
sys.path.insert(0, str(MAIN))

from snake_game import SnakeGame  # noqa: E402


ACTIONS = {
    (-1, 0): 0,
    (0, -1): 1,
    (0, 1): 2,
    (1, 0): 3,
}


def generate_even_board_cycle(board_size):
    if board_size % 2 != 0:
        raise ValueError("A Hamiltonian cycle on a square grid requires an even board size.")

    path = [(0, c) for c in range(board_size)]
    for row in range(1, board_size):
        cols = range(board_size - 1, 0, -1) if row % 2 else range(1, board_size)
        for col in cols:
            path.append((row, col))
    for row in range(board_size - 1, 0, -1):
        path.append((row, 0))

    if len(path) != board_size * board_size or len(set(path)) != len(path):
        raise AssertionError("Generated path does not visit every cell exactly once.")
    for current, nxt in zip(path, path[1:] + path[:1]):
        if abs(current[0] - nxt[0]) + abs(current[1] - nxt[1]) != 1:
            raise AssertionError(f"Non-adjacent cycle edge: {current} -> {nxt}")
    return path


def action_between(current, nxt):
    delta = (nxt[0] - current[0], nxt[1] - current[1])
    if delta not in ACTIONS:
        raise ValueError(f"Cells are not adjacent: {current} -> {nxt}")
    return ACTIONS[delta]


def run_even_cycle_episode(board_size, seed, max_steps):
    game = SnakeGame(seed=seed, board_size=board_size, silent_mode=True)
    cycle = generate_even_board_cycle(board_size)
    cycle_index = {cell: index for index, cell in enumerate(cycle)}

    food_steps = []
    last_food_step = 0
    steps = 0
    done = False

    while len(game.snake) < game.grid_size and not done and steps < max_steps:
        head = game.snake[0]
        index = cycle_index[head]
        nxt = cycle[(index + 1) % len(cycle)]
        action = action_between(head, nxt)
        done, info = game.step(action)
        steps += 1
        if info["food_obtained"]:
            food_steps.append(steps - last_food_step)
            last_food_step = steps

    return {
        "seed": seed,
        "board_size": board_size,
        "target_length": board_size * board_size,
        "length": len(game.snake),
        "score": game.score,
        "steps": steps,
        "done": done,
        "filled_board": len(game.snake) == game.grid_size,
        "food_count": len(food_steps),
        "avg_steps_per_food": sum(food_steps) / len(food_steps) if food_steps else 0.0,
        "max_steps_per_food": max(food_steps) if food_steps else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Headless oracle evaluation for Snake full-board targets.")
    parser.add_argument("--board-size", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    if args.board_size % 2 != 0:
        result = {
            "board_size": args.board_size,
            "target_length": args.board_size * args.board_size,
            "hamiltonian_cycle_available": False,
            "reason": "Odd-by-odd grid graphs have an odd number of cells, so no Hamiltonian cycle exists.",
            "next_oracle": "Use a Hamiltonian path or tail-aware planner for 21x21 curriculum/evaluation.",
        }
        print(json.dumps(result, indent=2))
        return

    max_steps = args.max_steps or args.board_size * args.board_size * args.board_size * 2
    results = [
        run_even_cycle_episode(args.board_size, args.seed + episode, max_steps)
        for episode in range(args.episodes)
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
