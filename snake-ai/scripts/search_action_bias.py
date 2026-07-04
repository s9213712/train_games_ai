import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch

sys.modules.setdefault("gym", gym)
sys.modules.setdefault("gym.spaces", gym.spaces)

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main"
sys.path.insert(0, str(MAIN))

from sb3_contrib import MaskablePPO  # noqa: E402

from snake_env import SnakeCnnEnv  # noqa: E402
from train_cnn_oracle_bc import evaluate  # noqa: E402


def parse_delta(text):
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("delta must contain four comma-separated numbers")
    return parts


def load_model(path, env, device):
    return MaskablePPO.load(
        path,
        env=env,
        device=device,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
    )


def apply_bias_delta(model, base_state, delta):
    state = {key: value.clone() for key, value in base_state.items()}
    base_bias = base_state["action_net.bias"]
    delta_tensor = torch.as_tensor(delta, dtype=base_bias.dtype)
    delta_tensor = delta_tensor - delta_tensor.mean()
    state["action_net.bias"] = base_bias + delta_tensor
    model.policy.load_state_dict(state)


def main():
    parser = argparse.ArgumentParser(description="Search small action-head bias edits for Snake CNN policy.")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--board-size", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--screen-episodes", type=int, default=20)
    parser.add_argument("--confirm-episodes", type=int, default=40)
    parser.add_argument("--screen-threshold", type=float, default=777.0)
    parser.add_argument("--best-threshold", type=float, default=784.25)
    parser.add_argument("--candidate", action="append", type=parse_delta, required=True)
    parser.add_argument("--candidate-name", action="append", required=True)
    parser.add_argument("--force-exit", action="store_true")
    args = parser.parse_args()

    if len(args.candidate_name) != len(args.candidate):
        raise SystemExit("--candidate-name count must match --candidate count")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = SnakeCnnEnv(
        seed=1,
        board_size=args.board_size,
        silent_mode=True,
        channel_first=True,
    )
    print(f"loading={args.source_model}", flush=True)
    model = load_model(args.source_model, env, args.device)
    base_state = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}

    best_score = args.best_threshold
    best_name = None
    try:
        for name, delta in zip(args.candidate_name, args.candidate):
            apply_bias_delta(model, base_state, delta)
            screen_metrics = evaluate(
                model,
                args.board_size,
                args.seed,
                args.screen_episodes,
                args.max_steps,
            )
            print(f"{name} screen={screen_metrics}", flush=True)
            if screen_metrics["score_avg"] < args.screen_threshold:
                continue

            confirm_metrics = evaluate(
                model,
                args.board_size,
                args.seed,
                args.confirm_episodes,
                args.max_steps,
            )
            print(f"{name} confirm={confirm_metrics}", flush=True)
            if confirm_metrics["score_avg"] > best_score:
                best_score = confirm_metrics["score_avg"]
                best_name = name
                output = args.output_dir / f"ppo_snake_cnn_107007936_bias_{name}.zip"
                model.save(output)
                print(f"best_saved={output} best_score={best_score}", flush=True)
    finally:
        env.close()

    print(f"done best_name={best_name} best_score={best_score}", flush=True)
    if args.force_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
