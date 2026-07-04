import argparse
import os
import random
import time
from pathlib import Path

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_env import SnakeCnnEnv, SnakeMlpEnv
from train import linear_schedule, select_device


def make_train_env(env_cls, seed, board_size):
    def _init():
        env = env_cls(seed=seed, board_size=board_size, silent_mode=True, limit_step=True)
        env = Monitor(env)
        env = ActionMasker(env, lambda wrapped_env: wrapped_env.unwrapped.get_action_mask())
        env.reset(seed=seed)
        return env

    return _init


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train a tiny Snake agent and periodically show it in a Pygame window."
    )
    parser.add_argument("--agent", choices=("mlp", "cnn"), default="mlp")
    parser.add_argument("--board-size", type=int, default=12)
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--chunk-timesteps", type=int, default=512)
    parser.add_argument("--preview-steps", type=int, default=180)
    parser.add_argument("--frame-delay", type=float, default=0.04)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.94)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--final-learning-rate", type=float, default=2.5e-6)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--final-clip-range", type=float, default=0.025)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-dir", default="gui_train_demo_models")
    return parser


def preview_policy(model, env_cls, args, trained_steps):
    import pygame

    env = env_cls(
        seed=args.seed + trained_steps,
        board_size=args.board_size,
        silent_mode=False,
        limit_step=False,
        render_mode="human",
    )
    pygame.display.set_caption(f"Snake AI GUI training demo - {args.agent.upper()}")

    obs, _ = env.reset(seed=args.seed + trained_steps)
    for step in range(args.preview_steps):
        env.game.status_text = f"trained: {trained_steps}/{args.total_timesteps}"
        action, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=args.seed + trained_steps + step + 1)
        time.sleep(args.frame_delay)

    env.close()


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    env_cls = SnakeMlpEnv if args.agent == "mlp" else SnakeCnnEnv
    policy = "MlpPolicy" if args.agent == "mlp" else "CnnPolicy"
    device = select_device(args.device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_env = DummyVecEnv([make_train_env(env_cls, args.seed, args.board_size)])
    model = MaskablePPO(
        policy,
        train_env,
        device=device,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        learning_rate=linear_schedule(args.learning_rate, args.final_learning_rate),
        clip_range=linear_schedule(args.clip_range, args.final_clip_range),
    )

    trained_steps = 0
    print(
        "Starting GUI training demo. Close the Pygame window or press Ctrl+C in the terminal to stop."
    )
    try:
        while trained_steps < args.total_timesteps:
            next_chunk = min(args.chunk_timesteps, args.total_timesteps - trained_steps)
            print(f"Training chunk: {trained_steps} -> {trained_steps + next_chunk}")
            model.learn(
                total_timesteps=next_chunk,
                reset_num_timesteps=(trained_steps == 0),
                progress_bar=False,
            )
            trained_steps += next_chunk
            print(f"Previewing policy after {trained_steps} timesteps")
            preview_policy(model, env_cls, args, trained_steps)
    finally:
        train_env.close()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_dir / f"ppo_snake_{args.agent}_gui_demo.zip")
    print(f"Saved demo model to {save_dir}")


if __name__ == "__main__":
    main()
