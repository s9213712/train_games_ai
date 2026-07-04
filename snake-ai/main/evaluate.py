import argparse
import random
import sys
import time
from pathlib import Path

import gymnasium as gym
from sb3_contrib import MaskablePPO

from snake_env import SnakeCnnEnv, SnakeMlpEnv
from train import select_device

sys.modules.setdefault("gym", gym)
sys.modules.setdefault("gym.spaces", gym.spaces)


MAIN_DIR = Path(__file__).resolve().parent


def default_model_path(agent, device):
    candidates = []
    if agent == "cnn" and device == "mps":
        candidates.append(MAIN_DIR / "original_models" / "trained_models_cnn_mps" / "ppo_snake_final.zip")
        candidates.append(MAIN_DIR / "trained_models_cnn_mps" / "ppo_snake_final.zip")
    else:
        candidates.append(MAIN_DIR / "original_models" / f"trained_models_{agent}" / "ppo_snake_final.zip")
        candidates.append(MAIN_DIR / f"trained_models_{agent}" / "ppo_snake_final.zip")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake agent.")
    parser.add_argument("--agent", choices=("cnn", "mlp"), default="cnn")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--board-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--cnn-channel-first", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frame-delay", type=float, default=0.05)
    parser.add_argument("--round-delay", type=float, default=5.0)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    env_cls = SnakeCnnEnv if args.agent == "cnn" else SnakeMlpEnv
    device = select_device(args.device)

    seed = args.seed if args.seed is not None else random.randint(0, 1_000_000_000)
    print(f"Using seed = {seed} for testing.")
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = args.board_size * args.board_size * 8

    env = env_cls(
        seed=seed,
        board_size=args.board_size,
        limit_step=False,
        silent_mode=not args.render,
        render_mode="human" if args.render else None,
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
        channel_first=args.cnn_channel_first if env_cls is SnakeCnnEnv else False,
    )

    model_path = Path(args.model_path) if args.model_path else default_model_path(args.agent, device)
    model = MaskablePPO.load(
        model_path,
        env=env,
        device=device,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
    )

    total_reward = 0.0
    total_score = 0
    min_score = float("inf")
    max_score = 0
    total_food = 0
    total_food_steps = 0
    total_loop_revisits = 0
    total_oscillations = 0

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        done = False
        num_step = 0
        info = None
        sum_step_reward = 0.0
        last_food_step = 0
        episode_food_steps = []
        episode_loop_revisits = 0
        episode_oscillations = 0

        print(f"=================== Episode {episode + 1} ==================")
        hit_eval_cap = False
        while not done:
            action, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
            num_step += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if max_steps > 0 and num_step >= max_steps and not done:
                hit_eval_cap = True
                done = True

            if done:
                last_action = ["UP", "LEFT", "RIGHT", "DOWN"][int(action)]
                if hit_eval_cap:
                    print(f"Evaluation step cap reached at {num_step} steps. Last action: {last_action}")
                elif info["snake_size"] == env.game.grid_size:
                    print(f"Victory reward: {reward:.4f}.")
                else:
                    print(f"Terminal reward: {reward:.4f}. Last action: {last_action}")
            elif info["food_obtained"]:
                steps_to_food = num_step - last_food_step
                episode_food_steps.append(steps_to_food)
                last_food_step = num_step
                print(
                    f"Food obtained at step {num_step:04d}. "
                    f"Steps To Food: {steps_to_food}. "
                    f"Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}"
                )
                sum_step_reward = 0.0
            else:
                sum_step_reward += reward
                if info.get("loop_revisit"):
                    episode_loop_revisits += 1
                if info.get("oscillation"):
                    episode_oscillations += 1

            episode_reward += reward
            if args.render:
                time.sleep(args.frame_delay)

        episode_score = env.game.score
        min_score = min(min_score, episode_score)
        max_score = max(max_score, episode_score)
        snake_size = info["snake_size"] if info else 0
        avg_food_steps = (
            sum(episode_food_steps) / len(episode_food_steps)
            if episode_food_steps
            else 0
        )
        print(
            f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, "
            f"Score: {episode_score}, Total Steps: {num_step}, Snake Size: {snake_size}, "
            f"Avg Steps/Food: {avg_food_steps:.2f}, Loop Hits: {episode_loop_revisits}, "
            f"Oscillation Hits: {episode_oscillations}"
        )
        total_reward += episode_reward
        total_score += episode_score
        total_food += len(episode_food_steps)
        total_food_steps += sum(episode_food_steps)
        total_loop_revisits += episode_loop_revisits
        total_oscillations += episode_oscillations
        if args.render:
            time.sleep(args.round_delay)

    env.close()
    avg_steps_per_food = total_food_steps / total_food if total_food else 0
    print("=================== Summary ==================")
    print(
        f"Average Score: {total_score / args.episodes}, Min Score: {min_score}, "
        f"Max Score: {max_score}, Average reward: {total_reward / args.episodes}, "
        f"Avg Steps/Food: {avg_steps_per_food:.2f}, Loop Hits: {total_loop_revisits}, "
        f"Oscillation Hits: {total_oscillations}"
    )


if __name__ == "__main__":
    main()
