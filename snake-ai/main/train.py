import argparse
import random
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_env import SnakeCnnEnv, SnakeMlpEnv


def linear_schedule(initial_value, final_value=0.0):
    initial_value = float(initial_value)
    final_value = float(final_value)

    def scheduler(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)

    return scheduler


def select_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_env(
    env_cls,
    seed,
    board_size,
    limit_step,
    food_time_penalty,
    food_step_limit_multiplier,
    food_reward_bonus,
    distance_reward_scale,
    reachable_space_penalty,
    reachable_space_min_ratio,
    loop_penalty,
    loop_window,
    oscillation_penalty,
    oscillation_window,
    cnn_channel_first,
):
    def _init():
        env = env_cls(
            seed=seed,
            board_size=board_size,
            silent_mode=True,
            limit_step=limit_step,
            food_time_penalty=food_time_penalty,
            food_step_limit_multiplier=food_step_limit_multiplier,
            food_reward_bonus=food_reward_bonus,
            distance_reward_scale=distance_reward_scale,
            reachable_space_penalty=reachable_space_penalty,
            reachable_space_min_ratio=reachable_space_min_ratio,
            loop_penalty=loop_penalty,
            loop_window=loop_window,
            oscillation_penalty=oscillation_penalty,
            oscillation_window=oscillation_window,
            channel_first=cnn_channel_first if env_cls is SnakeCnnEnv else False,
        )
        env = Monitor(env)
        env = ActionMasker(env, lambda wrapped_env: wrapped_env.unwrapped.get_action_mask())
        env.reset(seed=seed)
        return env

    return _init


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train a MaskablePPO Snake agent.")
    parser.add_argument("--agent", choices=("cnn", "mlp"), default="cnn")
    parser.add_argument("--board-size", type=int, default=12)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--total-timesteps", type=int, default=100_000_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.94)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--final-learning-rate", type=float, default=2.5e-6)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--final-clip-range", type=float, default=0.025)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--food-time-penalty", type=float, default=0.0)
    parser.add_argument("--food-step-limit-multiplier", type=float, default=4.0)
    parser.add_argument("--food-reward-bonus", type=float, default=0.0)
    parser.add_argument("--distance-reward-scale", type=float, default=0.1)
    parser.add_argument("--reachable-space-penalty", type=float, default=0.0)
    parser.add_argument("--reachable-space-min-ratio", type=float, default=0.35)
    parser.add_argument("--loop-penalty", type=float, default=0.0)
    parser.add_argument("--loop-window", type=int, default=16)
    parser.add_argument("--oscillation-penalty", type=float, default=0.0)
    parser.add_argument("--oscillation-window", type=int, default=12)
    parser.add_argument("--checkpoint-interval", type=int, default=15_625)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--cnn-channel-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--load-model", default=None)
    parser.add_argument("--no-stdout-log", action="store_true")
    parser.add_argument("--no-step-limit", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    env_cls = SnakeCnnEnv if args.agent == "cnn" else SnakeMlpEnv
    policy = "CnnPolicy" if args.agent == "cnn" else "MlpPolicy"
    device = select_device(args.device)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir or f"trained_models_{args.agent}")
    log_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    seeds = random.sample(range(1_000_000_000), args.num_envs)
    env = SubprocVecEnv(
        [
            make_env(
                env_cls,
                seed=s,
                board_size=args.board_size,
                limit_step=not args.no_step_limit,
                food_time_penalty=args.food_time_penalty,
                food_step_limit_multiplier=args.food_step_limit_multiplier,
                food_reward_bonus=args.food_reward_bonus,
                distance_reward_scale=args.distance_reward_scale,
                reachable_space_penalty=args.reachable_space_penalty,
                reachable_space_min_ratio=args.reachable_space_min_ratio,
                loop_penalty=args.loop_penalty,
                loop_window=args.loop_window,
                oscillation_penalty=args.oscillation_penalty,
                oscillation_window=args.oscillation_window,
                cnn_channel_first=args.cnn_channel_first,
            )
            for s in seeds
        ]
    )

    if args.load_model:
        learning_rate = linear_schedule(args.learning_rate, args.final_learning_rate)
        clip_range = linear_schedule(args.clip_range, args.final_clip_range)
        model = MaskablePPO.load(
            args.load_model,
            env=env,
            device=device,
            custom_objects={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "gamma": args.gamma,
                "ent_coef": args.ent_coef,
                "learning_rate": learning_rate,
                "clip_range": clip_range,
                "tensorboard_log": str(log_dir),
            },
        )
        model.verbose = 1
    else:
        model = MaskablePPO(
            policy,
            env,
            device=device,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            learning_rate=linear_schedule(args.learning_rate, args.final_learning_rate),
            clip_range=linear_schedule(args.clip_range, args.final_clip_range),
            tensorboard_log=str(log_dir),
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_interval,
        save_path=str(save_dir),
        name_prefix=f"ppo_snake_{args.agent}",
    )

    log_path = save_dir / "training_log.txt"
    context = nullcontext()
    log_file = None
    if not args.no_stdout_log:
        log_file = log_path.open("w")
        context = redirect_stdout(log_file)

    try:
        with context:
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=[checkpoint_callback],
                reset_num_timesteps=args.load_model is None,
            )
    finally:
        env.close()
        if log_file is not None:
            log_file.close()

    model.save(save_dir / "ppo_snake_final.zip")


if __name__ == "__main__":
    main()
