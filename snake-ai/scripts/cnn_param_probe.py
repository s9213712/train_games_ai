import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main"
sys.path.insert(0, str(MAIN))
sys.modules.setdefault("gym", gym)
sys.modules.setdefault("gym.spaces", gym.spaces)

from snake_env import SnakeCnnEnv  # noqa: E402
from train import select_device  # noqa: E402


ORIGINAL_CNN = MAIN / "original_models" / "trained_models_cnn" / "ppo_snake_final.zip"
PROTECTED_CNN = MAIN / "protected_models" / "ppo_snake_cnn_original.zip"


def make_env(seed, params, rank=0):
    def _init():
        env = SnakeCnnEnv(
            seed=seed + rank * 1009,
            board_size=params["board_size"],
            silent_mode=True,
            limit_step=True,
            channel_first=True,
            food_time_penalty=params["food_time_penalty"],
            food_step_limit_multiplier=params["food_step_limit_multiplier"],
            food_reward_bonus=params["food_reward_bonus"],
            distance_reward_scale=params["distance_reward_scale"],
            loop_penalty=params["loop_penalty"],
            loop_window=params["loop_window"],
            oscillation_penalty=params["oscillation_penalty"],
            oscillation_window=params["oscillation_window"],
        )
        env = Monitor(env)
        env = ActionMasker(env, lambda wrapped_env: wrapped_env.unwrapped.get_action_mask())
        env.reset(seed=seed + rank * 1009)
        return env

    return _init


def load_model(path, env, device, params):
    model = MaskablePPO.load(
        path,
        env=env,
        device=device,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
    )
    lr = params["learning_rate"]
    clip = params["clip_range"]
    model.learning_rate = lr
    model.lr_schedule = lambda _: lr
    model.clip_range = lambda _: clip
    model.gamma = params["gamma"]
    model.ent_coef = params["ent_coef"]
    model.n_epochs = params["n_epochs"]
    for group in model.policy.optimizer.param_groups:
        group["lr"] = lr
    return model


def evaluate(model, params, episodes, seed, max_steps):
    scores = []
    steps = []
    food_counts = []
    loops = []
    oscillations = []
    steps_per_food = []

    env = SnakeCnnEnv(
        seed=seed,
        board_size=params["board_size"],
        silent_mode=True,
        limit_step=True,
        channel_first=True,
        food_time_penalty=params["food_time_penalty"],
        food_step_limit_multiplier=params["food_step_limit_multiplier"],
        food_reward_bonus=params["food_reward_bonus"],
        distance_reward_scale=params["distance_reward_scale"],
        loop_penalty=params["loop_penalty"],
        loop_window=params["loop_window"],
        oscillation_penalty=params["oscillation_penalty"],
        oscillation_window=params["oscillation_window"],
    )

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        step = 0
        last_food_step = 0
        food_step_counts = []
        episode_loops = 0
        episode_oscillations = 0
        while not done and step < max_steps:
            action, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            step += 1
            done = terminated or truncated
            if info.get("food_obtained"):
                food_step_counts.append(step - last_food_step)
                last_food_step = step
            if info.get("loop_revisit"):
                episode_loops += 1
            if info.get("oscillation"):
                episode_oscillations += 1

        score = env.game.score
        scores.append(score)
        steps.append(step)
        food_counts.append(score // 10)
        loops.append(episode_loops)
        oscillations.append(episode_oscillations)
        if food_step_counts:
            steps_per_food.append(sum(food_step_counts) / len(food_step_counts))

    env.close()
    return {
        "score_avg": float(np.mean(scores)),
        "score_min": int(np.min(scores)),
        "score_max": int(np.max(scores)),
        "food_avg": float(np.mean(food_counts)),
        "steps_avg": float(np.mean(steps)),
        "steps_per_food_avg": float(np.mean(steps_per_food)) if steps_per_food else 0.0,
        "loop_avg": float(np.mean(loops)),
        "oscillation_avg": float(np.mean(oscillations)),
        "scores": scores,
    }


def variant_params():
    base = {
        "board_size": 12,
        "num_envs": 8,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 4,
        "gamma": 0.94,
        "learning_rate": 2.5e-4,
        "clip_range": 0.15,
        "ent_coef": 0.0,
        "food_time_penalty": 0.0,
        "food_step_limit_multiplier": 4.0,
        "food_reward_bonus": 0.0,
        "distance_reward_scale": 0.1,
        "loop_penalty": 0.0,
        "loop_window": 16,
        "oscillation_penalty": 0.0,
        "oscillation_window": 12,
    }
    variants = {
        "protected_baseline": {**base, "learning_rate": 0.0, "clip_range": 0.0},
        "gentle_continue": base,
        "mild_loop_penalty": {
            **base,
            "loop_penalty": 0.015,
            "oscillation_penalty": 0.002,
        },
        "lower_shaping": {
            **base,
            "distance_reward_scale": 0.025,
            "food_reward_bonus": 0.65,
            "food_time_penalty": 0.0015,
        },
        "shorter_food_limit": {
            **base,
            "food_step_limit_multiplier": 3.0,
            "food_time_penalty": 0.0015,
        },
    }
    return variants


def run_variant(name, params, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_env = DummyVecEnv([make_env(args.seed, params, rank) for rank in range(params["num_envs"])])
    try:
        model_path = ORIGINAL_CNN if ORIGINAL_CNN.exists() else PROTECTED_CNN
        model = load_model(model_path, train_env, args.device, params)
        if args.finetune_steps > 0 and params["learning_rate"] > 0:
            model.learn(total_timesteps=args.finetune_steps, reset_num_timesteps=False, progress_bar=False)
        metrics = evaluate(model, params, args.eval_episodes, args.seed + 50_000, args.max_steps)
    finally:
        train_env.close()

    return {
        "name": name,
        "finetune_steps": args.finetune_steps if params["learning_rate"] > 0 else 0,
        "params": params,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Probe CNN Snake hyperparameters without mutating protected models.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--variants", default="protected_baseline,gentle_continue,mild_loop_penalty,lower_shaping,shorter_food_limit")
    parser.add_argument("--finetune-steps", type=int, default=32_768)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--output", default="runtime/cnn_param_probe.json")
    args = parser.parse_args()
    args.device = select_device(args.device)

    if not ORIGINAL_CNN.exists() and not PROTECTED_CNN.exists():
        raise FileNotFoundError(f"Original CNN model not found: {ORIGINAL_CNN}")

    selected = [name.strip() for name in args.variants.split(",") if name.strip()]
    variants = variant_params()
    unknown = sorted(set(selected) - set(variants))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    results = [run_variant(name, variants[name], args) for name in selected]
    results.sort(key=lambda item: item["metrics"]["score_avg"], reverse=True)

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
