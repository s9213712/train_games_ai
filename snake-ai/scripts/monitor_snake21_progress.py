import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


SUMMARY_RE = re.compile(
    r"Average Score: (?P<avg>[-0-9.]+), Min Score: (?P<min>[-0-9.]+), "
    r"Max Score: (?P<max>[-0-9.]+), Average reward: (?P<reward>[-0-9.]+), "
    r"Avg Steps/Food: (?P<steps_food>[-0-9.]+), Loop Hits: (?P<loops>\d+), "
    r"Oscillation Hits: (?P<oscillations>\d+)"
)
STEP_RE = re.compile(r"ppo_snake_cnn_(\d+)_steps\.zip$")


def checkpoint_step(path):
    match = STEP_RE.search(path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(checkpoint_dir):
    checkpoints = list(Path(checkpoint_dir).glob("ppo_snake_cnn_*_steps.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=checkpoint_step)


def load_state(path, best_avg, best_max):
    if path.exists():
        return json.loads(path.read_text())
    return {
        "evaluated_steps": [],
        "best_avg": best_avg,
        "best_max": best_max,
        "best_avg_step": None,
        "best_max_step": None,
        "no_improve_count": 0,
        "stopped": False,
        "stop_reason": None,
    }


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def run_evaluation(args, model_path, episodes=None):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = args.pythonpath
    env["SDL_AUDIODRIVER"] = "dummy"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    cmd = [
        args.python,
        "-u",
        str(args.evaluate_script),
        "--agent",
        "cnn",
        "--board-size",
        "21",
        "--device",
        "cpu",
        "--episodes",
        str(episodes if episodes is not None else args.episodes),
        "--seed",
        str(args.seed),
        "--model-path",
        str(model_path),
        "--food-time-penalty",
        "0.001",
        "--food-step-limit-multiplier",
        "4.0",
        "--food-reward-bonus",
        "0.8",
        "--distance-reward-scale",
        "0.01",
        "--loop-penalty",
        "0.01",
        "--oscillation-penalty",
        "0.01",
        "--max-steps",
        str(args.max_steps),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(args.main_dir),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.eval_timeout,
    )
    match = SUMMARY_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(f"Could not parse evaluation summary for {model_path}\n{proc.stdout[-4000:]}")
    return {
        "avg_score": float(match.group("avg")),
        "min_score": int(float(match.group("min"))),
        "max_score": int(float(match.group("max"))),
        "avg_reward": float(match.group("reward")),
        "avg_steps_per_food": float(match.group("steps_food")),
        "loop_hits": int(match.group("loops")),
        "oscillation_hits": int(match.group("oscillations")),
        "raw_tail": proc.stdout[-6000:],
    }


def stop_training(session):
    if not session:
        return
    subprocess.run(["/usr/bin/tmux", "send-keys", "-t", session, "C-c"], check=False)


def append_jsonl(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(item, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor 21x21 Snake PPO checkpoints until plateau or regression.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--main-dir", type=Path, required=True)
    parser.add_argument("--evaluate-script", type=Path, required=True)
    parser.add_argument("--python", default="/home/s92137/.pyenv/versions/3.10.14/bin/python3")
    parser.add_argument("--pythonpath", default="/home/s92137/snake-ai/main")
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--jsonl-log", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--confirm-regression-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-timeout", type=int, default=240)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--regression-ratio", type=float, default=0.85)
    parser.add_argument("--initial-best-avg", type=float, default=733.3333333333334)
    parser.add_argument("--initial-best-max", type=float, default=980.0)
    parser.add_argument("--stop-session", default="snake21_ppo_cpu")
    args = parser.parse_args()

    state = load_state(args.state_file, args.initial_best_avg, args.initial_best_max)
    save_state(args.state_file, state)

    while True:
        if state.get("stopped"):
            return

        checkpoint = latest_checkpoint(args.checkpoint_dir)
        if checkpoint is None:
            time.sleep(args.interval)
            continue

        step = checkpoint_step(checkpoint)
        evaluated = set(state.get("evaluated_steps", []))
        if step in evaluated:
            time.sleep(args.interval)
            state = load_state(args.state_file, args.initial_best_avg, args.initial_best_max)
            continue

        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            metrics = run_evaluation(args, checkpoint)
            status = "ok"
            error = None
        except Exception as exc:
            metrics = {}
            status = "error"
            error = repr(exc)

        record = {
            "time": started_at,
            "checkpoint": str(checkpoint),
            "step": step,
            "status": status,
            "error": error,
            **metrics,
        }
        append_jsonl(args.jsonl_log, record)

        if status == "ok":
            evaluated = list(state.get("evaluated_steps", []))
            evaluated.append(step)
            state["evaluated_steps"] = evaluated[-1000:]

            improved = False
            if metrics["avg_score"] > float(state["best_avg"]):
                state["best_avg"] = metrics["avg_score"]
                state["best_avg_step"] = step
                improved = True
            if metrics["max_score"] > float(state["best_max"]):
                state["best_max"] = metrics["max_score"]
                state["best_max_step"] = step
                improved = True

            if improved:
                state["no_improve_count"] = 0
            else:
                state["no_improve_count"] = int(state.get("no_improve_count", 0)) + 1

            regression_threshold = float(state["best_avg"]) * args.regression_ratio
            if metrics["avg_score"] < regression_threshold:
                confirmed_metrics = None
                if args.confirm_regression_episodes > args.episodes:
                    try:
                        confirmed_metrics = run_evaluation(
                            args,
                            checkpoint,
                            episodes=args.confirm_regression_episodes,
                        )
                        append_jsonl(
                            args.jsonl_log,
                            {
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                "checkpoint": str(checkpoint),
                                "step": step,
                                "status": "confirm_regression",
                                "error": None,
                                **confirmed_metrics,
                            },
                        )
                    except Exception as exc:
                        append_jsonl(
                            args.jsonl_log,
                            {
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                "checkpoint": str(checkpoint),
                                "step": step,
                                "status": "confirm_regression_error",
                                "error": repr(exc),
                            },
                        )

                regression_avg = (
                    confirmed_metrics["avg_score"]
                    if confirmed_metrics is not None
                    else metrics["avg_score"]
                )
                if regression_avg < regression_threshold:
                    state["stopped"] = True
                    state["stop_reason"] = (
                        f"regression: avg={regression_avg:.2f}, "
                        f"best_avg={float(state['best_avg']):.2f}"
                    )
            elif int(state["no_improve_count"]) >= args.patience:
                state["stopped"] = True
                state["stop_reason"] = f"plateau: no improvement for {state['no_improve_count']} evaluations"

            if state["stopped"]:
                stop_training(args.stop_session)

        save_state(args.state_file, state)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
