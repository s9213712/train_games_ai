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
from train import (
    linear_schedule,
    save_model_atomic,
    select_device,
    write_json_atomic,
)


GUI_DEMO_REPORT_FORMAT = "snake-gui-demo-unverified-v1"
GUI_DEMO_UNVERIFIED_REASON = "gui_demo_has_no_promotion_evaluation"
GUI_DEMO_WARNING = (
    "UNVERIFIED CANDIDATE: the GUI demo previews PPO updates but does not run "
    "the development/holdout promotion guard. Use train.py for verified training."
)


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
        description=(
            "Preview an unverified Snake PPO candidate while it trains. "
            "This demo never produces an official/final model."
        ),
        epilog="Use train.py when you need a guarded, promotion-eligible model.",
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
    parser.add_argument(
        "--save-dir",
        default="gui_train_demo_models",
        help="directory for the explicitly unverified candidate bundle and report",
    )
    return parser


def unverified_candidate_paths(save_dir, agent):
    """Return the only artifact paths that this unguarded demo may write."""

    save_dir = Path(save_dir)
    stem = f"ppo_snake_{agent}_gui_demo_candidate_unverified"
    return save_dir / f"{stem}.zip", save_dir / f"{stem}.guard.json"


def save_unverified_candidate(
    model,
    args,
    trained_steps,
    *,
    termination_reason="completed_requested_budget",
):
    """Persist demo weights with durable evidence that they are not verified.

    The GUI loop intentionally prioritizes interactive previews and does not run
    the paired development/fixed-holdout transaction used by ``train.py``.
    Consequently its output is never eligible for a final or protected path.
    """

    candidate_path, report_path = unverified_candidate_paths(args.save_dir, args.agent)
    report = {
        "format": GUI_DEMO_REPORT_FORMAT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "gui_train_demo",
        "artifact": str(candidate_path),
        "artifact_status": "unverified_candidate",
        "agent": args.agent,
        "board_size": int(args.board_size),
        "requested_timesteps": int(args.total_timesteps),
        "attempted_timesteps": int(trained_steps),
        "termination_reason": str(termination_reason),
        "decision": {
            "accepted": False,
            "verified": False,
            "reason": GUI_DEMO_UNVERIFIED_REASON,
        },
        "promotion_evaluation": None,
        "warning": GUI_DEMO_WARNING,
    }
    save_model_atomic(model, candidate_path, embedded_guard_report=report)
    write_json_atomic(report_path, report)
    return candidate_path, report_path, report


def preview_policy(model, env_cls, args, trained_steps):
    import pygame

    env = env_cls(
        seed=args.seed + trained_steps,
        board_size=args.board_size,
        silent_mode=False,
        limit_step=False,
        render_mode="human",
    )
    try:
        pygame.display.set_caption(
            f"UNVERIFIED CANDIDATE - Snake AI GUI demo - {args.agent.upper()}"
        )

        obs, _ = env.reset(seed=args.seed + trained_steps)
        for step in range(args.preview_steps):
            env.game.status_text = (
                "UNVERIFIED candidate preview | "
                f"trained: {trained_steps}/{args.total_timesteps}"
            )
            action, _ = model.predict(
                obs,
                action_masks=env.get_action_mask(),
                deterministic=True,
            )
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=args.seed + trained_steps + step + 1)
            time.sleep(args.frame_delay)
    finally:
        env.close()


def train_preview_loop(model, env_cls, args):
    """Run preview chunks and return the actual PPO transition delta.

    SB3 may round a requested chunk up to a full rollout, so requested chunk
    sizes are not evidence of how many transitions were really collected.
    ``model.num_timesteps`` is the authoritative counter.  Ctrl+C is a normal
    interactive stop and returns the partial count; every other exception is
    deliberately allowed to propagate.
    """

    initial_timesteps = int(model.num_timesteps)
    trained_steps = 0
    termination_reason = "completed_requested_budget"

    try:
        while trained_steps < args.total_timesteps:
            next_chunk = min(
                args.chunk_timesteps,
                args.total_timesteps - trained_steps,
            )
            print(
                "Training UNVERIFIED candidate chunk: "
                f"{trained_steps} -> requested {trained_steps + next_chunk}"
            )
            before_learn = int(model.num_timesteps)
            model.learn(
                total_timesteps=next_chunk,
                # The model is newly initialized, and keeping the counter
                # monotonic makes the measured delta unambiguous across chunks.
                reset_num_timesteps=False,
                progress_bar=False,
            )
            trained_steps = max(0, int(model.num_timesteps) - initial_timesteps)
            if int(model.num_timesteps) <= before_learn:
                raise RuntimeError("PPO learn returned without collecting transitions")
            print(
                "Previewing UNVERIFIED candidate after "
                f"{trained_steps} actual timesteps"
            )
            preview_policy(model, env_cls, args, trained_steps)
    except KeyboardInterrupt:
        trained_steps = max(0, int(model.num_timesteps) - initial_timesteps)
        termination_reason = "keyboard_interrupt"
        print(
            "Ctrl+C received; preserving an explicitly UNVERIFIED candidate "
            f"after {trained_steps} actual timesteps."
        )

    return trained_steps, termination_reason


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

    print(GUI_DEMO_WARNING)
    print(
        "Starting GUI candidate demo. Close the Pygame window or press Ctrl+C "
        "in the terminal to stop."
    )
    try:
        trained_steps, termination_reason = train_preview_loop(
            model,
            env_cls,
            args,
        )
    finally:
        train_env.close()

    candidate_path, report_path, _report = save_unverified_candidate(
        model,
        args,
        trained_steps,
        termination_reason=termination_reason,
    )
    print(f"Saved UNVERIFIED candidate only: {candidate_path}")
    print(f"Unverified-candidate evidence: {report_path}")
    print("No final or protected model was written. Use train.py for guarded promotion.")


if __name__ == "__main__":
    main()
