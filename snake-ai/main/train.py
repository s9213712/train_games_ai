import argparse
import json
import os
import random
import sys
import tempfile
import time
import zipfile
from contextlib import nullcontext, redirect_stdout
from pathlib import Path


def build_arg_parser():
    """Build the complete CLI parser without importing the training runtime."""

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
    parser.add_argument(
        "--checkpoint-interval-timesteps",
        "--checkpoint-interval",
        dest="checkpoint_interval_timesteps",
        type=int,
        default=500_000,
        help=(
            "Checkpoint interval in total environment transitions (not vector-env callback "
            "calls). The legacy --checkpoint-interval spelling is retained as an alias."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps")
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help=(
            "CPU intra-op threads used by PyTorch. Snake's small networks normally run "
            "faster and more predictably with one thread; increase explicitly for a "
            "dedicated host after benchmarking."
        ),
    )
    parser.add_argument(
        "--cnn-channel-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CHW CNN observations (default); --no-cnn-channel-first selects HWC.",
    )
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--load-model", default=None)
    parser.add_argument("--no-stdout-log", action="store_true")
    parser.add_argument("--no-step-limit", action="store_true")
    parser.add_argument("--guard-eval-episodes", type=int, default=8)
    parser.add_argument("--guard-holdout-episodes", type=int, default=8)
    parser.add_argument("--guard-max-steps", type=int, default=None)
    parser.add_argument("--guard-min-delta", type=float, default=0.001)
    parser.add_argument(
        "--guard-min-training-timesteps",
        type=int,
        default=4096,
        help=(
            "Minimum candidate transitions before final-model verification. Shorter smoke "
            "runs exit successfully but save only an explicitly unverified candidate."
        ),
    )
    return parser

# Apply conservative native-library defaults before importing PyTorch.  The
# small Snake networks are latency-bound, and unconstrained BLAS/OpenMP pools
# can make even an eight-step smoke run stall for minutes on a shared host.
# SNAKE_NATIVE_THREADS is the deliberate opt-out; inherited generic BLAS values
# must not silently defeat this safety default.  --torch-threads below controls
# PyTorch itself after argument parsing.
if __name__ == "__main__":
    # Help is a metadata-only operation.  Exit before validating runtime-only
    # environment settings or importing PyTorch/SB3.
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        build_arg_parser().parse_args()

    _native_threads = os.environ.get("SNAKE_NATIVE_THREADS", "1")
    try:
        if int(_native_threads) < 1:
            raise ValueError
    except ValueError as exc:
        raise SystemExit("SNAKE_NATIVE_THREADS must be a positive integer") from exc
    for _thread_env_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_thread_env_name] = _native_threads

    # Reject an impossible CNN board before importing PyTorch/SB3.  Besides
    # producing a prompt, useful CLI error, this guarantees invalid input
    # cannot initialize native training runtimes or create output artifacts.
    _cnn_preflight = argparse.ArgumentParser(add_help=False)
    _cnn_preflight.add_argument("--agent", choices=("cnn", "mlp"), default="cnn")
    _cnn_preflight.add_argument("--board-size", type=int, default=12)
    _cnn_preflight_args, _unknown = _cnn_preflight.parse_known_args()
    if (
        _cnn_preflight_args.agent == "cnn"
        and "-h" not in sys.argv[1:]
        and "--help" not in sys.argv[1:]
    ):
        _compatible_board_sizes = [
            size for size in range(3, 85) if 84 % size == 0
        ]
        if _cnn_preflight_args.board_size not in _compatible_board_sizes:
            _cnn_preflight.error(
                f"CNN board_size={_cnn_preflight_args.board_size} is incompatible "
                "with image_size=84; board_size must divide 84 exactly. "
                "Compatible values: "
                + ", ".join(str(size) for size in _compatible_board_sizes)
            )

import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_env import (
    CNN_DEFAULT_IMAGE_SIZE,
    SnakeCnnEnv,
    SnakeMlpEnv,
    validate_cnn_board_size,
    validate_cnn_channel_mode,
)


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
        env_kwargs = dict(
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
        )
        if env_cls is SnakeCnnEnv:
            env_kwargs["channel_first"] = cnn_channel_first
        env = env_cls(**env_kwargs)
        env = Monitor(env)
        env = ActionMasker(env, lambda wrapped_env: wrapped_env.unwrapped.get_action_mask())
        env.reset(seed=seed)
        return env

    return _init


def checkpoint_callback_frequency(interval_timesteps, num_envs):
    """Convert a transition interval to SB3 callback calls.

    Stable-Baselines3 invokes callbacks once per vectorized ``env.step`` call, so
    each callback call represents ``num_envs`` transitions.  The returned
    cadence is rounded down to the nearest representable transition interval.
    """

    interval_timesteps = max(1, int(interval_timesteps))
    num_envs = max(1, int(num_envs))
    return max(interval_timesteps // num_envs, 1)


def open_training_log(log_path, args):
    """Open the durable CLI audit log and append a session boundary."""

    log_file = Path(log_path).open("a", encoding="utf-8")
    log_file.write(
        "\n=== training session "
        f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
        f"agent={args.agent} load_model={args.load_model or 'none'} "
        f"requested_timesteps={args.total_timesteps} num_envs={args.num_envs} "
        "===\n"
    )
    log_file.flush()
    return log_file


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_model_atomic(model, path, *, embedded_guard_report=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".zip", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        model.save(temporary)
        if embedded_guard_report is not None:
            with zipfile.ZipFile(temporary, mode="a", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr(
                    "training_guard.json",
                    json.dumps(embedded_guard_report, ensure_ascii=False, indent=2),
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def policy_objective(metrics):
    return (
        float(metrics.get("avg_score", 0.0))
        + float(metrics.get("avg_food", 0.0)) * 0.5
        + float(metrics.get("avg_reward", 0.0)) * 0.05
    )


def evaluate_policy_fixed_seeds(
    model,
    env_cls,
    *,
    seeds,
    max_steps,
    env_kwargs,
):
    """Deterministically evaluate one policy without touching its train VecEnv."""

    rows = []
    for seed in seeds:
        kwargs = dict(env_kwargs, seed=int(seed), silent_mode=True, limit_step=True)
        env = env_cls(**kwargs)
        try:
            obs, _info = env.reset(seed=int(seed))
            total_reward = 0.0
            food = 0
            steps = 0
            done = False
            while not done and steps < int(max_steps):
                action, _state = model.predict(
                    obs,
                    deterministic=True,
                    action_masks=env.get_action_mask(),
                )
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = bool(terminated or truncated)
                total_reward += float(reward)
                food += int(bool(info.get("food_obtained")))
                steps += 1
            rows.append(
                {
                    "seed": int(seed),
                    "score": len(env.game.snake) - env.init_snake_size,
                    "food": food,
                    "reward": round(total_reward, 6),
                    "steps": steps,
                }
            )
        finally:
            env.close()
    count = max(1, len(rows))
    metrics = {
        "episodes": len(rows),
        "seeds": [int(seed) for seed in seeds],
        "avg_score": round(sum(row["score"] for row in rows) / count, 5),
        "avg_food": round(sum(row["food"] for row in rows) / count, 5),
        "avg_reward": round(sum(row["reward"] for row in rows) / count, 6),
        "rows": rows,
        "max_steps": int(max_steps),
    }
    metrics["objective"] = round(policy_objective(metrics), 6)
    return metrics


def paired_evaluation_decision(
    baseline,
    candidate,
    holdout_baseline,
    holdout_candidate,
    *,
    min_delta,
    attempted_timesteps,
    min_training_timesteps,
    protected_holdout=None,
):
    """Gate a CLI candidate on paired development and fixed promotion validation."""

    required_delta = max(1e-5, float(min_delta))

    def comparison(reference, proposed):
        objective_delta = float(proposed["objective"]) - float(reference["objective"])
        food_delta = float(proposed.get("avg_food", 0.0)) - float(
            reference.get("avg_food", 0.0)
        )
        return {
            "objective_delta": round(objective_delta, 6),
            "food_delta": round(food_delta, 6),
            "passed": bool(
                objective_delta + 1e-12 >= required_delta and food_delta >= 0.0
            ),
        }

    development = comparison(baseline, candidate)
    holdout = comparison(holdout_baseline, holdout_candidate)
    protected = (
        comparison(protected_holdout, holdout_candidate)
        if protected_holdout is not None
        else None
    )
    enough_training = int(attempted_timesteps) >= int(min_training_timesteps)
    accepted = bool(
        enough_training
        and development["passed"]
        and holdout["passed"]
        and (protected is None or protected["passed"])
    )
    if not enough_training:
        reason = "insufficient_training_evidence"
    elif not development["passed"]:
        reason = "development_not_improved"
    elif not holdout["passed"]:
        reason = "fixed_holdout_not_improved"
    elif protected is not None and not protected["passed"]:
        reason = "protected_final_not_improved"
    else:
        reason = "paired_behavior_improved"
    return {
        "accepted": accepted,
        "verified": accepted,
        "reason": reason,
        "required_delta": required_delta,
        "attempted_timesteps": int(attempted_timesteps),
        "min_training_timesteps": int(min_training_timesteps),
        "development": development,
        "holdout": holdout,
        "protected": protected,
    }


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.agent == "cnn":
        try:
            validate_cnn_board_size(args.board_size, CNN_DEFAULT_IMAGE_SIZE)
            validate_cnn_channel_mode(args.cnn_channel_first)
        except ValueError as exc:
            parser.error(str(exc))

    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be at least 1")
    torch.set_num_threads(args.torch_threads)

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
    env_factories = [
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
    # A subprocess adds no parallelism for one environment and can fail in
    # constrained containers that intentionally disallow multiprocessing IPC.
    env = DummyVecEnv(env_factories) if args.num_envs == 1 else SubprocVecEnv(env_factories)

    if args.load_model:
        learning_rate = linear_schedule(args.learning_rate, args.final_learning_rate)
        clip_range = linear_schedule(args.clip_range, args.final_clip_range)
        # Do not override observation_space with the raw VecEnv space here.
        # SB3 legitimately wraps HWC image environments as CHW before checking
        # the checkpoint, so forcing the pre-wrap HWC space creates a false
        # mismatch and makes channel-last training impossible to resume.
        load_overrides = {
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "ent_coef": args.ent_coef,
            "learning_rate": learning_rate,
            "clip_range": clip_range,
            "tensorboard_log": str(log_dir),
        }
        if args.agent == "mlp":
            # Repository MLP checkpoints predate the Gymnasium migration. Their
            # Box spaces are semantically identical but fail class-based equality;
            # this narrow legacy override is safe for the fixed MLP layout. CNN,
            # especially HWC input, must keep SB3's real transpose-aware space.
            load_overrides.update(
                {
                    "observation_space": env.observation_space,
                    "action_space": env.action_space,
                }
            )
        model = MaskablePPO.load(
            args.load_model,
            env=env,
            device=device,
            custom_objects=load_overrides,
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
        save_freq=checkpoint_callback_frequency(
            args.checkpoint_interval_timesteps,
            args.num_envs,
        ),
        save_path=str(save_dir),
        name_prefix=f"ppo_snake_{args.agent}_candidate_unverified",
    )

    eval_env_kwargs = {
        "board_size": args.board_size,
        "food_time_penalty": args.food_time_penalty,
        "food_step_limit_multiplier": args.food_step_limit_multiplier,
        "food_reward_bonus": args.food_reward_bonus,
        "distance_reward_scale": args.distance_reward_scale,
        "reachable_space_penalty": args.reachable_space_penalty,
        "reachable_space_min_ratio": args.reachable_space_min_ratio,
        "loop_penalty": args.loop_penalty,
        "loop_window": args.loop_window,
        "oscillation_penalty": args.oscillation_penalty,
        "oscillation_window": args.oscillation_window,
    }
    if env_cls is SnakeCnnEnv:
        eval_env_kwargs["channel_first"] = args.cnn_channel_first
    guard_seed_root = 1_100_000_000 + (int(args.seed or 0) % 100_000_000)
    guard_eval_episodes = max(4, min(64, int(args.guard_eval_episodes)))
    guard_holdout_episodes = max(8, min(64, int(args.guard_holdout_episodes)))
    development_seeds = [
        guard_seed_root + index for index in range(guard_eval_episodes)
    ]
    holdout_seeds = [
        guard_seed_root + 1_000_000 + index
        for index in range(guard_holdout_episodes)
    ]
    guard_max_steps = max(
        10,
        int(args.guard_max_steps or args.board_size * args.board_size * 8),
    )
    guard_min_delta = max(1e-5, float(args.guard_min_delta))
    guard_min_training_timesteps = max(1, int(args.guard_min_training_timesteps))
    final_path = save_dir / "ppo_snake_final.zip"
    candidate_path = save_dir / "ppo_snake_candidate_unverified.zip"
    report_path = save_dir / "training_guard_report.json"
    protected_report_path = save_dir / "ppo_snake_final.guard.json"

    log_path = save_dir / "training_log.txt"
    context = nullcontext()
    log_file = None
    if not args.no_stdout_log:
        # Always append: a resumed run must not erase the evidence from the run
        # that produced the checkpoint it is continuing from.
        log_file = open_training_log(log_path, args)
        context = redirect_stdout(log_file)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_checkpoint = Path(tmpdir) / "baseline_before_training.zip"
            model.save(baseline_checkpoint)
            with context:
                baseline = evaluate_policy_fixed_seeds(
                    model,
                    env_cls,
                    seeds=development_seeds,
                    max_steps=guard_max_steps,
                    env_kwargs=eval_env_kwargs,
                )
                holdout_baseline = evaluate_policy_fixed_seeds(
                    model,
                    env_cls,
                    seeds=holdout_seeds,
                    max_steps=guard_max_steps,
                    env_kwargs=eval_env_kwargs,
                )
                protected_holdout = None
                protected_error = None
                if final_path.exists():
                    try:
                        protected_model = MaskablePPO.load(final_path, device=device)
                        protected_holdout = evaluate_policy_fixed_seeds(
                            protected_model,
                            env_cls,
                            seeds=holdout_seeds,
                            max_steps=guard_max_steps,
                            env_kwargs=eval_env_kwargs,
                        )
                    except Exception as exc:
                        protected_error = repr(exc)

                candidate_start_steps = int(model.num_timesteps)
                model.learn(
                    total_timesteps=args.total_timesteps,
                    callback=[checkpoint_callback],
                    reset_num_timesteps=args.load_model is None,
                )
                attempted_timesteps = max(
                    0, int(model.num_timesteps) - candidate_start_steps
                )
                candidate = evaluate_policy_fixed_seeds(
                    model,
                    env_cls,
                    seeds=development_seeds,
                    max_steps=guard_max_steps,
                    env_kwargs=eval_env_kwargs,
                )
                holdout_candidate = evaluate_policy_fixed_seeds(
                    model,
                    env_cls,
                    seeds=holdout_seeds,
                    max_steps=guard_max_steps,
                    env_kwargs=eval_env_kwargs,
                )
                decision = paired_evaluation_decision(
                    baseline,
                    candidate,
                    holdout_baseline,
                    holdout_candidate,
                    min_delta=guard_min_delta,
                    attempted_timesteps=attempted_timesteps,
                    min_training_timesteps=guard_min_training_timesteps,
                    protected_holdout=protected_holdout,
                )
                if protected_error is not None:
                    decision["accepted"] = False
                    decision["verified"] = False
                    decision["reason"] = "existing_protected_final_not_comparable"

                report = {
                    "format": "snake-cli-paired-evaluation-v1",
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "agent": args.agent,
                    "board_size": args.board_size,
                    "cnn_channel_first": (
                        bool(args.cnn_channel_first) if args.agent == "cnn" else None
                    ),
                    "load_model": args.load_model,
                    "decision": decision,
                    "baseline": baseline,
                    "candidate": candidate,
                    "holdout_baseline": holdout_baseline,
                    "holdout_candidate": holdout_candidate,
                    "protected_holdout": protected_holdout,
                    "protected_error": protected_error,
                    "protocol": {
                        "development_seeds": development_seeds,
                        "holdout_seeds": holdout_seeds,
                        "max_steps": guard_max_steps,
                        "min_delta": guard_min_delta,
                        "min_training_timesteps": guard_min_training_timesteps,
                    },
                }
                if decision["accepted"]:
                    report["artifact"] = str(final_path)
                    report["artifact_status"] = "verified_protected_final"
                    save_model_atomic(
                        model,
                        final_path,
                        embedded_guard_report=report,
                    )
                    write_json_atomic(protected_report_path, report)
                else:
                    report["artifact"] = str(candidate_path)
                    report["artifact_status"] = "unverified_candidate"
                    save_model_atomic(
                        model,
                        candidate_path,
                        embedded_guard_report=report,
                    )
                    # Restore serialized policy/optimizer parameters in-place.
                    # For HWC input, constructing a new algorithm with the raw
                    # env creates a false HWC/CHW space mismatch.  The CLI exits
                    # immediately after writing the rejection report, so its
                    # advanced counters/schedules are neither reused nor saved.
                    model.set_parameters(
                        baseline_checkpoint,
                        exact_match=True,
                        device=device,
                    )
                write_json_atomic(report_path, report)
                print(
                    f"CLI guard: {decision['reason']} -> {report['artifact_status']} "
                    f"({report['artifact']})"
                )
    finally:
        env.close()
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
