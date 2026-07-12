from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
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


def _read_snake_bundle_metadata(path: Path) -> tuple[dict, str | None]:
    """Read only the small metadata member from a Snake dashboard bundle."""

    try:
        with zipfile.ZipFile(path) as bundle:
            info = bundle.getinfo("metadata.json")
            if info.file_size > 1024 * 1024:
                raise ValueError("metadata.json exceeds 1 MiB")
            metadata = json.loads(bundle.read("metadata.json").decode("utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("metadata.json is not an object")
    except (
        OSError,
        KeyError,
        UnicodeDecodeError,
        ValueError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    return metadata, None


def _stage_snake_audit_tree(source_root: Path, destination_root: Path) -> dict:
    """Create a minimal, private runtime tree for the production Snake loader.

    Importing ``web_dashboard`` constructs its singleton immediately.  The
    singleton is intentionally allowed to migrate or quarantine bundles, but
    only after all relevant code and runtime artifacts have been copied below a
    temporary root.  No production checkpoint is opened for writing, renamed,
    or unlinked by the audit.
    """

    source_main = source_root / "main"
    destination_main = destination_root / "main"
    destination_runtime = destination_root / "runtime"
    destination_protected = destination_runtime / "protected_best"
    destination_main.mkdir(parents=True, exist_ok=True)
    destination_protected.mkdir(parents=True, exist_ok=True)

    for source in source_main.glob("*.py"):
        shutil.copy2(source, destination_main / source.name)

    # The dashboard's production fallback chain is part of the runtime being
    # audited.  Copy the CPU originals and the full-board CNN into the private
    # tree as well; otherwise an audit with no protected checkpoint silently
    # exercises a newly initialized policy instead of what production serves.
    fallback_specs = {
        "mlp_repo_original": Path(
            "main/original_models/trained_models_mlp/ppo_snake_final.zip"
        ),
        "cnn_repo_original": Path(
            "main/original_models/trained_models_cnn/ppo_snake_final.zip"
        ),
        "cnn_fullboard_12x12": Path(
            "main/trained_models_cnn_oracle_bc/ppo_snake_bc_final_12x12.zip"
        ),
    }

    legacy_relative = Path("runtime/snake_policy.best.snakeai.zip")
    protected_relatives = [
        path.relative_to(source_root)
        for path in sorted(
            (source_root / "runtime" / "protected_best").glob(
                "snake_policy.*.best.snakeai.zip"
            )
        )
        if path.is_file()
    ]
    source_relatives = list(protected_relatives)
    if (source_root / legacy_relative).is_file():
        source_relatives.append(legacy_relative)

    for relative in source_relatives:
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)

    fallback_artifacts = []
    for name, relative in fallback_specs.items():
        source = source_root / relative
        row = {
            "name": name,
            "relative": relative,
            "source_exists": source.is_file(),
            "sha256": None,
            "staged_verified": False,
        }
        if source.is_file():
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            source_digest = _file_sha256(source)
            staged_digest = _file_sha256(destination)
            if source_digest != staged_digest:
                raise OSError(f"Snake fallback copy verification failed: {relative}")
            row.update({"sha256": source_digest, "staged_verified": True})
        fallback_artifacts.append(row)

    return {
        "source_relatives": source_relatives,
        "protected_relatives": protected_relatives,
        "legacy_relative": legacy_relative,
        "fallback_artifacts": fallback_artifacts,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_snake_baseline_model(dashboard_module, dashboard):
    """Construct the loader's fresh, same-architecture PPO initialization."""

    seed = int(dashboard.config["seed"])
    dashboard_module.random.seed(seed)
    dashboard_module.torch.manual_seed(seed)
    return dashboard_module.MaskablePPO(
        dashboard._policy(),
        dashboard.train_env,
        device="cpu",
        verbose=0,
        n_steps=dashboard.config["n_steps"],
        batch_size=dashboard.config["batch_size"],
        n_epochs=dashboard.config["n_epochs"],
        gamma=dashboard.config["gamma"],
        learning_rate=dashboard.config["learning_rate"],
        clip_range=dashboard.config["clip_range"],
        ent_coef=dashboard.config["ent_coef"],
        policy_kwargs=(
            dashboard._cnn_policy_kwargs()
            if dashboard.config["agent"] == "cnn"
            else None
        ),
    )


def _snake_original_checkpoint(
    local_path: Path,
    *,
    source_root: Path,
    isolated_root: Path,
    staged: dict,
) -> tuple[Path, bool]:
    """Map a private loader selection back to its production source path."""

    relative = local_path.relative_to(isolated_root)
    if relative in staged["protected_relatives"]:
        return source_root / relative, False
    # The only file the isolated loader can create at a new protected path is a
    # validated migration of the copied legacy-global bundle.
    return source_root / staged["legacy_relative"], True


def _audit_selected_snake_bundle(
    dashboard_module,
    dashboard,
    *,
    local_path: Path,
    source_path: Path,
    metadata: dict,
    episodes: int,
    migrated_from_legacy: bool,
) -> dict:
    model_sha256 = dashboard._read_bundle_model_sha256(local_path)
    provenance = dashboard._validated_bundle_provenance(
        metadata,
        model_sha256=model_sha256,
    )
    benchmark = dict(provenance.get("benchmark") or {})
    config = metadata.get("config") if isinstance(metadata, dict) else None
    base = {
        "checkpoint": str(source_path),
        "runtime_checkpoint": str(local_path),
        "runtime_kind": "protected",
        "migrated_from_legacy": migrated_from_legacy,
        "agent": (
            str(config.get("agent"))
            if isinstance(config, dict) and config.get("agent") in {"cnn", "mlp"}
            else "unknown"
        ),
        "protocol_id": (
            dashboard._protocol_id(benchmark["protocol"]) if benchmark else None
        ),
        "model_sha256": model_sha256,
    }
    schema_current = bool(dashboard._has_resumable_bundle_format(metadata))
    if not provenance or not benchmark or not isinstance(config, dict):
        return {
            **base,
            "runtime_selected": False,
            "safe_to_serve": True,
            "training_verified": False,
            "checkpoint_schema_current": schema_current,
            "accepted_training_evidence": False,
            "loader_disposition": (
                "production provenance validator rejected metadata/model identity "
                "before policy loading"
            ),
        }

    expected_path = dashboard._protected_best_path(benchmark["protocol"])
    if local_path != expected_path:
        return {
            **base,
            "runtime_selected": False,
            "safe_to_serve": True,
            "training_verified": False,
            "checkpoint_schema_current": False,
            "accepted_training_evidence": False,
            "loader_disposition": "rejected protected namespace mismatch",
        }

    runtime_config = dict(dashboard_module.DEFAULT_CONFIG)
    runtime_config.update(config)
    # Device and vector count do not participate in the fixed-validation
    # protocol.  Keeping this audit on one CPU environment makes the comparison
    # reproducible without changing which protected policy the real loader picks.
    runtime_config.update({"device": "cpu", "num_envs": 1})
    try:
        dashboard.reset(runtime_config)
    except (TypeError, ValueError, RuntimeError) as exc:
        return {
            **base,
            "runtime_selected": False,
            "safe_to_serve": True,
            "training_verified": False,
            "checkpoint_schema_current": False,
            "accepted_training_evidence": False,
            "loader_disposition": f"loader rejected configuration: {type(exc).__name__}: {exc}",
        }
    if (
        dashboard.resume_best_bundle != local_path
        or getattr(dashboard, "resume_best_model_sha256", None) != model_sha256
    ):
        return {
            **base,
            "runtime_selected": False,
            "safe_to_serve": True,
            "training_verified": False,
            "checkpoint_schema_current": False,
            "accepted_training_evidence": False,
            "loader_disposition": "loader selected baseline fallback",
        }

    # Restoring a bundle deliberately restores its complete training config.
    # Device, vector count, rollout length, and batch size are operational knobs
    # outside the policy architecture and fixed-validation namespace, so bound
    # them again only after the real loader has made its selection.
    audit_n_steps = min(2048, max(8, int(dashboard.config["n_steps"])))
    audit_batch_size = min(
        audit_n_steps,
        max(8, int(dashboard.config["batch_size"])),
    )
    dashboard.update_config(
        {
            "device": "cpu",
            "num_envs": 1,
            "n_steps": audit_n_steps,
            "batch_size": audit_batch_size,
        }
    )
    try:
        dashboard._ensure_model()
    except Exception as exc:
        # The production loader catches its supported archive/load failures and
        # falls back.  Anything escaping it is an evaluator/runtime failure.
        return {
            **base,
            "runtime_selected": False,
            "safe_to_serve": False,
            "training_verified": False,
            "checkpoint_schema_current": False,
            "accepted_training_evidence": False,
            "loader_disposition": f"policy load failed: {type(exc).__name__}: {exc}",
        }
    if (
        dashboard.resume_best_bundle != local_path
        or getattr(dashboard, "resume_best_model_sha256", None) != model_sha256
        or not local_path.exists()
    ):
        return {
            **base,
            "runtime_selected": False,
            "safe_to_serve": True,
            "training_verified": False,
            "checkpoint_schema_current": False,
            "accepted_training_evidence": False,
            "loader_disposition": "policy weights failed to load; baseline fallback selected",
        }

    eval_config = dict(dashboard.config)
    seed_base = int(eval_config.get("seed", 7)) + 7_000_000
    max_steps = min(
        1_200,
        max(120, int(eval_config.get("guard_eval_steps", 600))),
    )
    candidate = dashboard._evaluate_model_score(
        dashboard.model,
        seed_base=seed_base,
        episodes=episodes,
        max_steps=max_steps,
        eval_config=eval_config,
    )
    baseline_model = _new_snake_baseline_model(dashboard_module, dashboard)
    baseline = dashboard._evaluate_model_score(
        baseline_model,
        seed_base=seed_base,
        episodes=episodes,
        max_steps=max_steps,
        eval_config=eval_config,
    )
    objective_delta = float(candidate["objective"]) - float(baseline["objective"])
    food_delta = float(candidate["avg_food"]) - float(baseline["avg_food"])
    independently_better = objective_delta > 1e-5 and food_delta >= 0.0
    safe_to_serve = objective_delta >= -1e-12 and food_delta >= 0.0
    # Do not duplicate the dashboard's evidence rules here.  The production
    # validator binds the exact embedded model bytes to the accepted guard,
    # development comparison, fixed holdout, and positive attempted training.
    accepted_evidence = bool(provenance)
    return {
        **base,
        "runtime_selected": True,
        "loader_disposition": "served protected policy",
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
        "training_verified": bool(
            schema_current and accepted_evidence and independently_better
        ),
    }


def _audit_snake_runtime_fallback(
    dashboard_module,
    dashboard,
    *,
    source_root: Path,
    isolated_root: Path,
    agent: str,
    episodes: int,
) -> dict:
    """Exercise the production no-protected-checkpoint path for one agent."""

    profile = "repo_original" if agent == "mlp" else "fullboard_12x12"
    base = {
        "checkpoint": None,
        "runtime_checkpoint": None,
        "runtime_kind": "fallback",
        "migrated_from_legacy": False,
        "agent": agent,
        "protocol_id": None,
        "checkpoint_schema_current": False,
        "accepted_training_evidence": False,
        "training_verified": False,
    }
    runtime_config = dict(dashboard_module.DEFAULT_CONFIG)
    runtime_config.update(
        {
            "agent": agent,
            "model_profile": profile,
            "board_size": 12,
            "device": "cpu",
            "num_envs": 1,
            "n_steps": 64,
            "batch_size": 64,
        }
    )
    try:
        dashboard.reset(runtime_config)
    except Exception as exc:
        return {
            **base,
            "runtime_selected": False,
            "safe_to_serve": False,
            "loader_disposition": (
                f"{agent} fallback configuration failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }
    if dashboard.resume_best_bundle is not None:
        return {
            **base,
            "runtime_selected": False,
            "safe_to_serve": False,
            "loader_disposition": (
                f"{agent} fallback audit unexpectedly selected a protected bundle"
            ),
            "error": "protected policy selected while auditing fallback",
        }

    try:
        local_path = dashboard._initial_model_path("cpu")
    except Exception as exc:
        return {
            **base,
            "runtime_selected": False,
            "safe_to_serve": False,
            "loader_disposition": (
                f"{agent} fallback resolution failed: {type(exc).__name__}: {exc}"
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }
    if local_path is None or not Path(local_path).is_file():
        return {
            **base,
            "runtime_selected": False,
            "safe_to_serve": False,
            "loader_disposition": f"{agent} production fallback artifact is absent",
            "error": f"no {profile} fallback artifact",
        }

    local_path = Path(local_path)
    try:
        relative = local_path.relative_to(isolated_root)
        source_path = source_root / relative
    except ValueError:
        return {
            **base,
            "runtime_checkpoint": str(local_path),
            "runtime_selected": False,
            "safe_to_serve": False,
            "loader_disposition": f"{agent} fallback escaped the isolated audit tree",
            "error": "fallback path is outside isolated tree",
        }
    base.update(
        {
            "checkpoint": str(source_path),
            "runtime_checkpoint": str(local_path),
            "fallback_profile": profile,
            "artifact_sha256": _file_sha256(local_path),
        }
    )

    try:
        dashboard._ensure_model()
    except Exception as exc:
        return {
            **base,
            "runtime_selected": False,
            "safe_to_serve": False,
            "loader_disposition": (
                f"{agent} fallback failed to load: {type(exc).__name__}: {exc}"
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }
    loader_event = str(getattr(dashboard, "last_event", ""))
    runtime_selected = bool(
        dashboard.model is not None
        and dashboard.resume_best_bundle is None
        and "loaded original model from" in loader_event
    )
    if not runtime_selected:
        return {
            **base,
            "runtime_selected": False,
            "safe_to_serve": False,
            "loader_disposition": (
                f"{agent} loader did not confirm the staged fallback: {loader_event}"
            ),
            "error": "production loader did not select fallback artifact",
        }

    eval_config = dict(dashboard.config)
    seed_base = int(eval_config.get("seed", 7)) + (
        7_100_000 if agent == "mlp" else 7_200_000
    )
    max_steps = min(
        1_200,
        max(120, int(eval_config.get("guard_eval_steps", 600))),
    )
    try:
        candidate = dashboard._evaluate_model_score(
            dashboard.model,
            seed_base=seed_base,
            episodes=episodes,
            max_steps=max_steps,
            eval_config=eval_config,
        )
        baseline_model = _new_snake_baseline_model(dashboard_module, dashboard)
        baseline = dashboard._evaluate_model_score(
            baseline_model,
            seed_base=seed_base,
            episodes=episodes,
            max_steps=max_steps,
            eval_config=eval_config,
        )
    except Exception as exc:
        return {
            **base,
            "runtime_selected": True,
            "safe_to_serve": False,
            "loader_disposition": (
                f"served {agent} fallback but behavior evaluation failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }

    objective_delta = float(candidate["objective"]) - float(baseline["objective"])
    food_delta = float(candidate["avg_food"]) - float(baseline["avg_food"])
    independently_better = objective_delta > 1e-5 and food_delta >= 0.0
    safe_to_serve = objective_delta >= -1e-12 and food_delta >= 0.0
    return {
        **base,
        "runtime_selected": True,
        "loader_disposition": f"served production {agent} {profile} fallback",
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
    }


def _snake_agent_summary(runtime_rows: list[dict]) -> dict:
    """Require an actual, safe loader selection for both policy families."""

    summary = {}
    for agent in ("mlp", "cnn"):
        rows = [row for row in runtime_rows if row.get("agent") == agent]
        selected = [row for row in rows if row.get("runtime_selected") is True]
        runtime_selected = bool(selected)
        summary[agent] = {
            "audited_policies": len(rows),
            "selected_policies": len(selected),
            "runtime_selected": runtime_selected,
            "safe_to_serve": runtime_selected
            and all(row.get("safe_to_serve") is True for row in selected),
            "training_verified": runtime_selected
            and all(row.get("training_verified") is True for row in selected),
        }
    return summary


def audit_snake(episodes: int) -> dict:
    source_root = ROOT / "snake-ai"
    legacy_source = source_root / "runtime" / "snake_policy.best.snakeai.zip"
    with tempfile.TemporaryDirectory(prefix="snake-model-audit-") as tmpdir:
        isolated_root = Path(tmpdir) / "snake-ai"
        staged = _stage_snake_audit_tree(source_root, isolated_root)
        source_artifacts = []
        for relative in staged["source_relatives"]:
            source_path = source_root / relative
            metadata, error = _read_snake_bundle_metadata(source_path)
            source_artifacts.append(
                {
                    "checkpoint": str(source_path),
                    "layout": (
                        "legacy_global"
                        if relative == staged["legacy_relative"]
                        else "protected_namespace"
                    ),
                    "metadata_readable": not bool(error),
                    "metadata_error": error,
                    "agent": (metadata.get("config") or {}).get("agent"),
                    "runtime_disposition": "not selected",
                }
            )

        isolated_main = isolated_root / "main"
        dashboard_module = load_module(
            "usable_model_audit_snake",
            isolated_main / "web_dashboard.py",
            isolated_main,
            clear=("cnn_features", "snake_env", "snake_game", "train"),
        )
        dashboard = dashboard_module.dashboard
        startup_local = (
            Path(dashboard.resume_best_bundle)
            if dashboard.resume_best_bundle is not None
            else None
        )
        startup_source = None
        if startup_local is not None:
            startup_source, _migrated = _snake_original_checkpoint(
                startup_local,
                source_root=source_root,
                isolated_root=isolated_root,
                staged=staged,
            )

        # Startup may have migrated a validated legacy bundle or quarantined a
        # copied artifact.  Audit every remaining protected policy by selecting
        # its own exact config through TrainingDashboard.reset(), just as the
        # production API does when switching MLP/CNN/protocol.
        local_candidates = sorted(
            (isolated_root / "runtime" / "protected_best").glob(
                "snake_policy.*.best.snakeai.zip"
            )
        )
        policies = []
        try:
            for local_path in local_candidates:
                metadata, metadata_error = _read_snake_bundle_metadata(local_path)
                source_path, migrated = _snake_original_checkpoint(
                    local_path,
                    source_root=source_root,
                    isolated_root=isolated_root,
                    staged=staged,
                )
                if metadata_error:
                    policies.append(
                        {
                            "checkpoint": str(source_path),
                            "runtime_checkpoint": str(local_path),
                            "runtime_kind": "protected",
                            "migrated_from_legacy": migrated,
                            "agent": "unknown",
                            "protocol_id": None,
                            "runtime_selected": False,
                            "loader_disposition": (
                                "rejected metadata before policy loading: "
                                f"{metadata_error}"
                            ),
                            "checkpoint_schema_current": False,
                            "accepted_training_evidence": False,
                            "safe_to_serve": True,
                            "training_verified": False,
                        }
                    )
                    continue
                policies.append(
                    _audit_selected_snake_bundle(
                        dashboard_module,
                        dashboard,
                        local_path=local_path,
                        source_path=source_path,
                        metadata=metadata,
                        episodes=episodes,
                        migrated_from_legacy=migrated,
                    )
                )

            # A missing protected policy is not a successful audit by vacuity.
            # Start the real fallback path for that policy family, require the
            # loader to select the staged production artifact, and compare its
            # behavior with a fresh same-architecture initialization.
            protected_selected = {
                row.get("agent")
                for row in policies
                if row.get("runtime_selected") is True
                and row.get("runtime_kind") == "protected"
            }
            for agent in ("mlp", "cnn"):
                if agent not in protected_selected:
                    policies.append(
                        _audit_snake_runtime_fallback(
                            dashboard_module,
                            dashboard,
                            source_root=source_root,
                            isolated_root=isolated_root,
                            agent=agent,
                            episodes=episodes,
                        )
                    )
        finally:
            dashboard.close()

        for artifact in source_artifacts:
            matches = [
                row for row in policies if row["checkpoint"] == artifact["checkpoint"]
            ]
            if matches:
                artifact["runtime_disposition"] = matches[0]["loader_disposition"]
            elif artifact["layout"] == "legacy_global":
                quarantines = list(
                    (isolated_root / "runtime").glob(
                        "snake_policy.best.snakeai.quarantine-*.zip"
                    )
                )
                artifact["runtime_disposition"] = (
                    "quarantined by isolated production loader"
                    if quarantines
                    else "legacy fallback retained but not selected on this startup"
                )
            else:
                relative = Path(artifact["checkpoint"]).relative_to(source_root)
                local_path = isolated_root / relative
                artifact["runtime_disposition"] = (
                    "not selectable from bundle metadata"
                    if local_path.exists()
                    else "quarantined by isolated production loader"
                )

    selected = [row for row in policies if row.get("runtime_selected") is True]
    agent_summary = _snake_agent_summary(policies)

    startup_row = next(
        (
            row
            for row in selected
            if startup_source is not None
            and row["checkpoint"] == str(startup_source)
        ),
        None,
    )
    primary = startup_row or (selected[0] if selected else None)
    legacy_artifact = next(
        (row for row in source_artifacts if row["layout"] == "legacy_global"),
        None,
    )
    legacy_note = (
        "legacy bundle was evaluated only if the production loader validated and migrated it; "
        "otherwise the loader reported a baseline fallback"
        if legacy_artifact
        else "no legacy-global bundle present"
    )
    no_policy_note = (
        "missing protected policy families were exercised through their production fallback loader"
    )
    fallback_artifacts = [
        {
            **row,
            "relative": str(row["relative"]),
            "checkpoint": str(source_root / row["relative"]),
        }
        for row in staged["fallback_artifacts"]
    ]
    both_agents_safe = all(
        agent_summary[agent]["runtime_selected"]
        and agent_summary[agent]["safe_to_serve"]
        for agent in ("mlp", "cnn")
    )
    both_agents_verified = all(
        agent_summary[agent]["runtime_selected"]
        and agent_summary[agent]["training_verified"]
        for agent in ("mlp", "cnn")
    )
    return {
        "checkpoint": (
            primary["checkpoint"] if primary else str(legacy_source)
        ),
        "runtime_policy": (
            startup_row["checkpoint"]
            if startup_row
            else (primary["checkpoint"] if primary else "no runtime policy selected")
        ),
        "artifact_exists": bool(source_artifacts)
        or any(row["source_exists"] for row in fallback_artifacts),
        "checkpoint_schema_current": bool(primary)
        and bool(primary["checkpoint_schema_current"]),
        "accepted_training_evidence": bool(primary)
        and bool(primary["accepted_training_evidence"]),
        "audit_protocol": primary.get("audit_protocol") if primary else None,
        "independent_comparison": (
            primary.get("independent_comparison") if primary else None
        ),
        "policies": policies,
        "agent_summary": agent_summary,
        "artifacts": source_artifacts,
        "fallback_artifacts": fallback_artifacts,
        "legacy_fallback": {
            "checkpoint": str(legacy_source),
            "artifact_exists": legacy_artifact is not None,
            "runtime_disposition": (
                legacy_artifact["runtime_disposition"]
                if legacy_artifact
                else "absent"
            ),
        },
        "safe_to_serve": both_agents_safe,
        "training_verified": both_agents_verified,
        "note": (
            f"{legacy_note}; {no_policy_note}"
            if any(row.get("runtime_kind") == "fallback" for row in policies)
            else legacy_note
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
