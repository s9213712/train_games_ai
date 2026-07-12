from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path, import_dir: Path, *, clear: tuple[str, ...] = ()):
    for module_name in clear:
        sys.modules.pop(module_name, None)
    sys.path.insert(0, str(import_dir))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(import_dir))
        except ValueError:
            pass


def read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def capture_audit(name: str, callback) -> dict:
    try:
        return callback()
    except Exception as exc:  # A broken evaluator is itself an audit failure.
        return {
            "safe_to_serve": False,
            "training_verified": False,
            "error": f"{name}: {type(exc).__name__}: {exc}",
        }


def quality_key(metrics: dict) -> tuple[float, float, float]:
    return (
        float(metrics.get("avg_gap", math.inf)),
        -float(metrics.get("match_rate", 0.0)),
        -float(metrics.get("top5_rate", 0.0)),
    )


def chess_metric_summary(metrics: dict) -> dict:
    return {
        "positions": int(metrics.get("positions", 0)),
        "avg_gap": float(metrics.get("avg_gap", math.inf)),
        "match_rate": float(metrics.get("match_rate", 0.0)),
        "top5_rate": float(metrics.get("top5_rate", 0.0)),
    }


def snake_metric_summary(metrics: dict) -> dict:
    return {
        key: metrics.get(key)
        for key in ("episodes", "avg_score", "avg_food", "avg_reward", "objective")
    }


def soccer_metric_summary(trainer, evaluation: dict) -> dict:
    return {
        "episodes": int(evaluation.get("episodes", 0)),
        "record": dict(evaluation.get("record") or {}),
        "avg_reward": float(evaluation.get("avg_reward", 0.0)),
        "avg_goal_diff": float(evaluation.get("avg_goal_diff", 0.0)),
        "avg_xg_diff": float(evaluation.get("avg_xg_diff", 0.0)),
        "objective": round(float(trainer._guard_objective(evaluation)), 3),
    }


def tetris_metric_summary(metrics: dict) -> dict:
    return {
        key: metrics.get(key)
        for key in (
            "episodes",
            "avg_score",
            "avg_lines",
            "avg_pieces",
            "avg_tetrises",
            "best_score",
            "best_tetrises",
            "max_pieces",
        )
    }


def audit_chess() -> dict:
    chess_dir = ROOT / "chess-ai"
    app = load_module("usable_model_audit_chess", chess_dir / "app.py", chess_dir)
    trainer = app.Trainer()
    loaded = dict(trainer.snapshot().get("checkpoint", {}).get("loaded") or {})
    loaded_path = Path(str(loaded.get("path"))) if loaded.get("path") else None
    payload = read_json(loaded_path) if loaded_path else {}
    baseline = trainer.validate_policy(dict(app.DEFAULT_WEIGHTS))
    current = trainer.validate_policy(dict(trainer.weights))
    independent_baseline = trainer.evaluate_teacher_gap(
        dict(app.DEFAULT_WEIGHTS),
        fens=app.INDEPENDENT_AUDIT_FENS,
    )
    independent_candidate = trainer.evaluate_teacher_gap(
        dict(trainer.weights),
        fens=app.INDEPENDENT_AUDIT_FENS,
    )
    independent_non_regression = (
        quality_key(independent_candidate) <= quality_key(independent_baseline)
    )
    split_non_regression = {
        name: quality_key(current[name]) <= quality_key(baseline[name])
        for name in ("guard", "holdout", "audit")
    }
    independently_better = (
        all(split_non_regression.values())
        and independent_non_regression
        and quality_key(current["guard"]) < quality_key(baseline["guard"])
        and quality_key(current["holdout"]) < quality_key(baseline["holdout"])
        and quality_key(current["promotion"]) < quality_key(baseline["promotion"])
    )
    learning = dict(payload.get("learning") or {})
    accepted_guard = dict(payload.get("accepted_guard") or {})
    fingerprint = app.policy_fingerprint(trainer.weights)
    schema_current = (
        int(payload.get("checkpoint_version", 0)) == int(app.CHECKPOINT_VERSION)
        and payload.get("training_protocol") == app.TRAINING_PROTOCOL
    )
    accepted_evidence = (
        int(learning.get("accepted_chunks", 0)) > 0
        and int(learning.get("policy_updates", 0)) > 0
        and (
            int(learning.get("teacher_updates", 0)) > 0
            or int(learning.get("rl_samples", 0)) > 0
        )
        and payload.get("policy_fingerprint") == fingerprint
        and accepted_guard.get("accepted") is True
        and accepted_guard.get("behavior_changed") is True
        and accepted_guard.get("candidate_fingerprint") == fingerprint
        and isinstance(accepted_guard.get("baseline"), dict)
        and isinstance(accepted_guard.get("candidate"), dict)
        and isinstance(accepted_guard.get("holdout_baseline"), dict)
        and isinstance(accepted_guard.get("holdout_candidate"), dict)
    )
    training_verified = bool(
        loaded_path and schema_current and accepted_evidence and independently_better
    )
    return {
        "runtime_policy": str(loaded_path) if loaded_path else "verified built-in baseline",
        "artifact_exists": bool(loaded_path),
        "checkpoint_schema_current": schema_current,
        "accepted_training_evidence": accepted_evidence,
        "independent_comparison": {
            "baseline": {
                name: chess_metric_summary(metrics)
                for name, metrics in baseline.items()
            },
            "candidate": {
                name: chess_metric_summary(metrics)
                for name, metrics in current.items()
            },
            "split_non_regression": split_non_regression,
            "strictly_better": independently_better,
            "unseen_audit_baseline": chess_metric_summary(independent_baseline),
            "unseen_audit_candidate": chess_metric_summary(independent_candidate),
            "unseen_audit_non_regression": independent_non_regression,
        },
        "safe_to_serve": all(split_non_regression.values()) and independent_non_regression,
        "training_verified": training_verified,
        "note": (
            "loaded policy independently beats the built-in baseline"
            if training_verified
            else "no current, independently improved training artifact; dashboard serves its verified baseline"
        ),
    }


def _bundle_guard(metadata: dict) -> dict:
    guard = metadata.get("guard")
    if isinstance(guard, dict):
        return dict(guard)
    for row in reversed(list(metadata.get("history") or [])):
        if isinstance(row, dict) and isinstance(row.get("guard"), dict):
            return dict(row["guard"])
    return {}


def audit_snake(episodes: int) -> dict:
    snake_dir = ROOT / "snake-ai" / "main"
    dashboard_module = load_module(
        "usable_model_audit_snake",
        snake_dir / "web_dashboard.py",
        snake_dir,
        clear=("cnn_features", "snake_env", "snake_game", "train"),
    )
    checkpoint = ROOT / "snake-ai" / "runtime" / "snake_policy.best.snakeai.zip"
    if not checkpoint.exists():
        return {
            "checkpoint": str(checkpoint),
            "artifact_exists": False,
            "safe_to_serve": True,
            "training_verified": False,
            "note": "no protected bundle; dashboard creates or loads its explicitly configured baseline",
        }

    try:
        with zipfile.ZipFile(checkpoint) as bundle:
            metadata = json.loads(bundle.read("metadata.json").decode("utf-8"))
    except (OSError, KeyError, UnicodeDecodeError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        return {
            "checkpoint": str(checkpoint),
            "artifact_exists": True,
            "safe_to_serve": False,
            "training_verified": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    original_best_path = dashboard_module.BEST_MODEL_PATH
    with tempfile.TemporaryDirectory() as tmpdir:
        dashboard_module.BEST_MODEL_PATH = Path(tmpdir) / "no-implicit-best.snakeai.zip"
        dashboard = dashboard_module.TrainingDashboard()
        try:
            dashboard.config = dict(dashboard_module.DEFAULT_CONFIG)
            dashboard.resume_best_bundle = None
            dashboard.guard_benchmark = {}
            dashboard.holdout_protocol = {}
            dashboard.best_guard_objective = float("-inf")
            bundle_config = dict(metadata.get("config") or {})
            bundle_config.update({"model_profile": "new", "device": "cpu", "num_envs": 1})
            dashboard._merge_config(bundle_config)
            dashboard._ensure_model()
            eval_config = dict(dashboard.config)
            seed_base = int(eval_config.get("seed", 7)) + 7_000_000
            max_steps = min(1_200, max(120, int(eval_config.get("guard_eval_steps", 600))))
            baseline = dashboard._evaluate_model_score(
                dashboard.model,
                seed_base=seed_base,
                episodes=episodes,
                max_steps=max_steps,
                eval_config=eval_config,
            )
            candidate_model = dashboard._load_model_from_bundle(
                checkpoint,
                dashboard.train_env,
                "cpu",
            )
            candidate = dashboard._evaluate_model_score(
                candidate_model,
                seed_base=seed_base,
                episodes=episodes,
                max_steps=max_steps,
                eval_config=eval_config,
            )
            benchmark = dashboard._validated_guard_benchmark(metadata)
        finally:
            dashboard.close()
            dashboard_module.BEST_MODEL_PATH = original_best_path

    objective_delta = float(candidate["objective"]) - float(baseline["objective"])
    food_delta = float(candidate["avg_food"]) - float(baseline["avg_food"])
    independently_better = objective_delta > 1e-5 and food_delta >= 0.0
    safe_to_serve = objective_delta >= -1e-12 and food_delta >= 0.0
    guard = _bundle_guard(metadata)
    schema_current = int(metadata.get("format_version", 0)) >= 2 and bool(benchmark)
    accepted_evidence = (
        guard.get("accepted") is True
        and int(metadata.get("trained_steps", 0)) > 0
        and float(benchmark.get("objective", -math.inf)) > -math.inf
    )
    return {
        "checkpoint": str(checkpoint),
        "artifact_exists": True,
        "checkpoint_schema_current": schema_current,
        "accepted_training_evidence": accepted_evidence,
        "audit_protocol": {
            "seed_base": seed_base,
            "episodes": episodes,
            "max_steps": max_steps,
            "fresh_same_architecture_baseline": True,
        },
        "independent_comparison": {
            "baseline": snake_metric_summary(baseline),
            "candidate": snake_metric_summary(candidate),
            "objective_delta": round(objective_delta, 5),
            "food_delta": round(food_delta, 5),
            "strictly_better": independently_better,
        },
        "safe_to_serve": safe_to_serve,
        "training_verified": bool(schema_current and accepted_evidence and independently_better),
        "note": (
            "legacy bundle may be behaviorally useful, but it is not current protected-training evidence"
            if not schema_current
            else ""
        ),
    }


def audit_soccer(episodes: int) -> dict:
    soccer_dir = ROOT / "soccer-ai"
    module = load_module(
        "usable_model_audit_soccer",
        soccer_dir / "rl_trainer.py",
        soccer_dir,
        clear=("soccer_env",),
    )
    current_path = soccer_dir / "runtime" / "soccer_policy.json"
    payload = read_json(current_path)
    # Instantiate against the real runtime directory so this audit evaluates
    # exactly the policy selected by the production loader, not a preferred
    # artifact that the server would never serve.
    trainer = module.RLTrainer(soccer_dir)
    baseline_policy = module.SoftmaxPolicy(trainer.policy.obs_dim, trainer.policy.action_dim)
    candidate_policy = trainer.policy.clone(preserve_rng=True)
    served_fingerprint = trainer._policy_fingerprint(candidate_policy)
    payload_policy = module.SoftmaxPolicy(trainer.policy.obs_dim, trainer.policy.action_dim)
    payload_policy.load_json(dict(payload.get("policy") or {}))
    payload_fingerprint = trainer._policy_fingerprint(payload_policy)
    try:
        schema_current = bool(
            payload
            and int(payload.get("checkpoint_version", 0)) == int(module.CHECKPOINT_VERSION)
            and int(payload.get("guard_objective_version", 0)) == int(module.GUARD_OBJECTIVE_VERSION)
            and payload_fingerprint == served_fingerprint
            and (payload.get("config") or {}).get("guard_enabled", True) is not False
            and "checkpoint invalid" not in str(trainer.last_event)
        )
    except (TypeError, ValueError):
        schema_current = False

    red_policy = module.SoftmaxPolicy(trainer.policy.obs_dim, trainer.policy.action_dim, seed=20260712)
    pool: list[dict] = []
    seeds = [1_500_000_000 + index for index in range(episodes)]
    schedule = trainer._build_evaluation_schedule("mixed", episodes, pool)
    context = trainer._evaluation_context(red_policy, pool, schedule, seeds)
    context.update({"purpose": "independent_artifact_audit_v1"})
    baseline = trainer._evaluate_policy(
        baseline_policy,
        red_policy,
        pool,
        seeds,
        opponent="mixed",
        opponent_schedule=schedule,
        capture_observations=True,
        evaluation_context=context,
    )
    observations = baseline.pop("_observations", [])
    candidate = trainer._evaluate_policy(
        candidate_policy,
        red_policy,
        pool,
        seeds,
        opponent="mixed",
        opponent_schedule=schedule,
        evaluation_context=context,
    )
    behavior = trainer._holdout_behavior_change(
        baseline_policy,
        candidate_policy,
        observations,
    )
    baseline_objective = trainer._guard_objective(baseline)
    candidate_objective = trainer._guard_objective(candidate)
    baseline_points = trainer._match_points(baseline)
    candidate_points = trainer._match_points(candidate)
    expected_context_id = trainer._promotion_evaluation_setup(candidate_policy)[4][
        "context_id"
    ]

    objective_delta = candidate_objective - baseline_objective
    independently_better = (
        behavior["changed_actions"] > 0
        and objective_delta >= float(module.GUARD_MIN_EFFECT)
        and candidate_points + 1e-12 >= baseline_points
    )
    accepted_evidence = (
        schema_current
        and int(payload.get("episode", 0)) > 0
        and finite_number(payload.get("best_guard_objective"))
        and payload.get("best_guard_context") == expected_context_id
    )
    return {
        "checkpoint": str(current_path),
        "artifact_exists": current_path.exists(),
        "checkpoint_schema_current": schema_current,
        "accepted_training_evidence": accepted_evidence,
        "audit_protocol": {"seeds": [seeds[0], seeds[-1]], "episodes": episodes},
        "independent_comparison": {
            "baseline": soccer_metric_summary(trainer, baseline),
            "candidate": soccer_metric_summary(trainer, candidate),
            "objective_delta": round(objective_delta, 3),
            "baseline_match_points": round(baseline_points, 4),
            "candidate_match_points": round(candidate_points, 4),
            "behavior": behavior,
            "strictly_better": independently_better,
        },
        "safe_to_serve": (
            objective_delta >= -1e-12 and candidate_points + 1e-12 >= baseline_points
        ),
        "training_verified": bool(accepted_evidence and independently_better),
        "note": "runtime loader selected its baseline; legacy checkpoint is not training evidence" if not schema_current else "",
    }


def audit_tetris(episodes: int, max_pieces: int) -> dict:
    tetris_dir = ROOT / "tetris-ai"
    module = load_module(
        "usable_model_audit_tetris",
        tetris_dir / "rl_trainer.py",
        tetris_dir,
        clear=("tetris_env",),
    )
    # Use the production loader and its explicit selection record. This covers
    # protected-best, main fallback, and built-in fallback exactly as served.
    trainer = module.RLTrainer(tetris_dir)
    selection = trainer.checkpoint_selection()
    selected_path = Path(selection["path"]) if selection.get("path") else None
    payload = read_json(selected_path) if selected_path else {}
    schema_current = bool(
        selected_path
        and module.RLTrainer._checkpoint_is_compatible(
            payload,
            require_promotion=bool(selection.get("protected")),
        )
        and dict(selection.get("policy") or {}) == trainer.policy.to_json()
    )
    baseline_policy = module.AfterstateValue(trainer.policy.dim)
    candidate_policy = trainer.policy.clone(preserve_rng=True)
    seeds = [1_600_000_000 + index for index in range(episodes)]
    kwargs = {
        "episodes": episodes,
        "start": 0,
        "lookahead_weight": trainer.lookahead_weight,
        "lookahead_candidates": trainer.lookahead_candidates,
        "lookahead_include_hold": trainer.lookahead_include_hold,
        "max_pieces": max_pieces,
        "seeds": seeds,
    }
    baseline = trainer._evaluate_policy(baseline_policy, **kwargs)
    candidate = trainer._evaluate_policy(candidate_policy, **kwargs)
    baseline_objective = trainer._guard_objective(baseline)
    candidate_objective = trainer._guard_objective(candidate)
    behavior_changed = (
        trainer._evaluation_signature(candidate)
        != trainer._evaluation_signature(baseline)
    )

    objective_delta = candidate_objective - baseline_objective
    independently_better = behavior_changed and objective_delta >= 1.0
    promotion = dict(payload.get("promotion_benchmark") or {})
    accepted_evidence = (
        schema_current
        and int(payload.get("episode", 0)) > 0
        and finite_number(promotion.get("objective"))
        and len(list((promotion.get("protocol") or {}).get("seeds") or [])) >= 4
    )
    return {
        "checkpoint": str(selected_path) if selected_path else "built-in default",
        "artifact_exists": selected_path is not None,
        "loader_selection": {
            "source": selection.get("source"),
            "protected": bool(selection.get("protected")),
            "rejected": list(selection.get("rejected") or []),
        },
        "checkpoint_schema_current": schema_current,
        "accepted_training_evidence": accepted_evidence,
        "audit_protocol": {
            "seeds": [seeds[0], seeds[-1]],
            "episodes": episodes,
            "max_pieces": max_pieces,
        },
        "independent_comparison": {
            "baseline": tetris_metric_summary(baseline),
            "candidate": tetris_metric_summary(candidate),
            "baseline_objective": round(baseline_objective, 2),
            "candidate_objective": round(candidate_objective, 2),
            "objective_delta": round(objective_delta, 2),
            "behavior_changed": behavior_changed,
            "strictly_better": independently_better,
        },
        "safe_to_serve": objective_delta >= -1e-12,
        "training_verified": bool(accepted_evidence and independently_better),
        "note": (
            "runtime loader selected its built-in baseline; incompatible artifacts were quarantined"
            if not schema_current
            else ""
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Independently compare protected game-AI artifacts with fresh/built-in baselines. "
            "A weight change by itself never passes."
        )
    )
    parser.add_argument("--snake-episodes", type=int, default=12)
    parser.add_argument("--soccer-episodes", type=int, default=64)
    parser.add_argument("--tetris-episodes", type=int, default=8)
    parser.add_argument("--tetris-max-pieces", type=int, default=180)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runtime" / "usable_model_audit_latest.json",
    )
    args = parser.parse_args()

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "definition": (
            "training_verified requires current-schema acceptance evidence, a measurable policy "
            "behavior change, and independent same-seed improvement over a fresh/built-in baseline"
        ),
        "games": {
            "chess": capture_audit("chess", audit_chess),
            "snake": capture_audit(
                "snake",
                lambda: audit_snake(max(4, args.snake_episodes)),
            ),
            "soccer": capture_audit(
                "soccer",
                lambda: audit_soccer(max(32, args.soccer_episodes)),
            ),
            "tetris": capture_audit(
                "tetris",
                lambda: audit_tetris(
                    max(4, args.tetris_episodes),
                    max(40, min(1_200, args.tetris_max_pieces)),
                ),
            ),
        },
    }
    report["overall_safe_to_serve"] = all(
        row["safe_to_serve"] for row in report["games"].values()
    )
    report["overall_training_verified"] = all(
        row["training_verified"] for row in report["games"].values()
    )
    # Backwards-compatible field, now with the strict definition above.
    report["overall_pass"] = report["overall_training_verified"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
