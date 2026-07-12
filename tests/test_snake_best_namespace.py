import json
import hashlib
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAKE_MAIN = REPO_ROOT / "snake-ai" / "main"
sys.path.insert(0, str(SNAKE_MAIN))

import web_dashboard  # noqa: E402
from web_dashboard import FIXED_HOLDOUT_KIND, TrainingDashboard  # noqa: E402


def _isolate_paths(monkeypatch, tmp_path):
    legacy_path = tmp_path / "snake_policy.best.snakeai.zip"
    monkeypatch.setattr(web_dashboard, "BEST_MODEL_PATH", legacy_path)
    monkeypatch.setattr(
        web_dashboard, "_DEFAULT_LEGACY_BEST_MODEL_PATH", legacy_path
    )
    monkeypatch.setattr(web_dashboard, "PROTECTED_BEST_DIR", tmp_path / "protected")
    return legacy_path


def _benchmark(dashboard, objective):
    protocol = dashboard._expected_holdout_protocol()
    metrics = {
        "episodes": protocol["episodes"],
        "avg_score": 1.0,
        "avg_food": 1.0,
        "avg_reward": 0.0,
        "objective": float(objective),
        "episode_results": [],
    }
    return {
        "kind": FIXED_HOLDOUT_KIND,
        "protocol": protocol,
        "objective": float(objective),
        "metrics": metrics,
    }


def _accepted_guard(dashboard, benchmark):
    objective = float(benchmark["objective"])
    development_baseline = {
        "episodes": 4,
        "avg_score": 0.0,
        "avg_food": 0.0,
        "avg_reward": 0.0,
        "objective": objective - 1.0,
    }
    development_candidate = {
        **development_baseline,
        "avg_score": 1.0,
        "avg_food": 1.0,
        "objective": objective,
    }
    return {
        "accepted": True,
        "promoted_to_best": True,
        "attempted_timesteps": 8,
        "episodes": 4,
        "baseline": development_baseline,
        "candidate": development_candidate,
        "baseline_model_sha256": "0" * 64,
        "candidate_model_sha256": "f" * 64,
        "decision": {"accepted": True, "required_delta": 0.001},
        "promotion_basis": FIXED_HOLDOUT_KIND,
        "holdout_protocol": benchmark["protocol"],
        "holdout_seed_base": benchmark["protocol"]["seed_base"],
        "holdout_episodes": benchmark["protocol"]["episodes"],
        "holdout_max_steps": benchmark["protocol"]["max_steps"],
        "holdout_baseline": dict(benchmark["metrics"]),
        "holdout_candidate": dict(benchmark["metrics"]),
    }


def _write_bundle(path, metadata, model_payload=b"placeholder protected weights"):
    path.parent.mkdir(parents=True, exist_ok=True)
    model_sha256 = hashlib.sha256(model_payload).hexdigest()
    metadata["model_sha256"] = model_sha256
    metadata["guard"]["candidate_model_sha256"] = model_sha256
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("model.zip", model_payload)
        bundle.writestr("metadata.json", json.dumps(metadata))


def _write_current_bundle(dashboard, objective):
    benchmark = _benchmark(dashboard, objective)
    dashboard.guard_benchmark = benchmark
    dashboard.holdout_protocol = dict(benchmark["protocol"])
    dashboard.best_guard_objective = float(objective)
    dashboard.trained_steps = 8
    metadata = dashboard._model_bundle_metadata(
        extra={"guard": _accepted_guard(dashboard, benchmark)}
    )
    path = dashboard._protected_best_path(benchmark["protocol"])
    _write_bundle(path, metadata)
    return path


def test_mlp_and_cnn_promotions_have_distinct_protocol_namespaces(
    tmp_path, monkeypatch
):
    _isolate_paths(monkeypatch, tmp_path)
    dashboard = TrainingDashboard()
    dashboard.trained_steps = 8
    writes = []

    def write_bundle(destination, metadata):
        writes.append((destination, metadata))
        _write_bundle(destination, metadata)

    monkeypatch.setattr(dashboard, "_write_model_bundle", write_bundle)

    cnn_protocol = dashboard._holdout_protocol()
    cnn_metrics = {**_benchmark(dashboard, 1.0)["metrics"]}
    assert dashboard._promote_fixed_holdout_best(
        protocol=cnn_protocol,
        metrics=cnn_metrics,
        guard=_accepted_guard(dashboard, _benchmark(dashboard, 1.0)),
    )
    cnn_path = writes[-1][0]

    dashboard._merge_config({"agent": "mlp", "model_profile": "new"})
    dashboard._clear_verified_provenance()
    mlp_protocol = dashboard._holdout_protocol()
    mlp_metrics = {**_benchmark(dashboard, 1.0)["metrics"]}
    assert dashboard._promote_fixed_holdout_best(
        protocol=mlp_protocol,
        metrics=mlp_metrics,
        guard=_accepted_guard(dashboard, _benchmark(dashboard, 1.0)),
    )
    mlp_path = writes[-1][0]

    assert cnn_path != mlp_path
    assert ".cnn." in cnn_path.name
    assert ".mlp." in mlp_path.name
    assert dashboard._protocol_id(cnn_protocol) in cnn_path.name
    assert dashboard._protocol_id(mlp_protocol) in mlp_path.name
    dashboard.close()


def test_reset_switches_between_matching_cnn_and_mlp_protected_bests(
    tmp_path, monkeypatch
):
    _isolate_paths(monkeypatch, tmp_path)
    cnn_builder = TrainingDashboard()
    cnn_path = _write_current_bundle(cnn_builder, 2.0)
    cnn_builder.close()

    mlp_builder = TrainingDashboard()
    mlp_builder._merge_config({"agent": "mlp", "model_profile": "new"})
    mlp_path = _write_current_bundle(mlp_builder, 3.0)
    mlp_builder.close()

    dashboard = TrainingDashboard()
    assert dashboard.config["agent"] == "cnn"
    assert dashboard.resume_best_bundle == cnn_path
    assert dashboard.best_guard_objective == 2.0

    dashboard.reset({"agent": "mlp", "model_profile": "new"})
    assert dashboard.config["agent"] == "mlp"
    assert dashboard.resume_best_bundle == mlp_path
    assert dashboard.best_guard_objective == 3.0
    assert cnn_path.exists()
    assert mlp_path.exists()
    dashboard.close()


def test_valid_legacy_global_bundle_is_moved_into_its_agent_namespace(
    tmp_path, monkeypatch
):
    legacy_path = _isolate_paths(monkeypatch, tmp_path)
    builder = TrainingDashboard()
    builder._merge_config({"agent": "mlp", "model_profile": "new"})
    benchmark = _benchmark(builder, 4.0)
    builder.guard_benchmark = benchmark
    builder.holdout_protocol = dict(benchmark["protocol"])
    builder.best_guard_objective = 4.0
    builder.trained_steps = 8
    metadata = builder._model_bundle_metadata(
        extra={"guard": _accepted_guard(builder, benchmark)}
    )
    expected_path = builder._protected_best_path(benchmark["protocol"])
    builder.close()
    _write_bundle(legacy_path, metadata)

    dashboard = TrainingDashboard()

    assert not legacy_path.exists()
    assert expected_path.exists()
    assert dashboard.config["agent"] == "mlp"
    assert dashboard.resume_best_bundle == expected_path
    assert dashboard.best_guard_objective == 4.0
    assert "migrated legacy protected bundle" in dashboard.last_event
    dashboard.close()


def test_superseded_legacy_global_is_quarantined_without_clearing_namespace(
    tmp_path, monkeypatch
):
    legacy_path = _isolate_paths(monkeypatch, tmp_path)
    builder = TrainingDashboard()
    protected_path = _write_current_bundle(builder, 5.0)
    builder.close()
    legacy_path.write_bytes(b"ambiguous old global checkpoint")

    dashboard = TrainingDashboard()

    assert dashboard.resume_best_bundle == protected_path
    assert dashboard.best_guard_objective == 5.0
    assert not legacy_path.exists()
    quarantined = list(
        tmp_path.glob(
            "snake_policy.best.snakeai.quarantine-invalid-legacy-global-bundle-*.zip"
        )
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"ambiguous old global checkpoint"
    dashboard.close()


def test_invalid_expected_namespace_does_not_hide_valid_legacy_mlp(
    tmp_path, monkeypatch
):
    legacy_path = _isolate_paths(monkeypatch, tmp_path)
    default_builder = TrainingDashboard()
    invalid_namespace = default_builder._protected_best_path()
    default_builder.close()

    mlp_builder = TrainingDashboard()
    mlp_builder._merge_config({"agent": "mlp", "model_profile": "new"})
    benchmark = _benchmark(mlp_builder, 6.0)
    mlp_builder.guard_benchmark = benchmark
    mlp_builder.holdout_protocol = dict(benchmark["protocol"])
    mlp_builder.best_guard_objective = 6.0
    mlp_builder.trained_steps = 8
    metadata = mlp_builder._model_bundle_metadata(
        extra={"guard": _accepted_guard(mlp_builder, benchmark)}
    )
    expected_mlp_path = mlp_builder._protected_best_path(benchmark["protocol"])
    mlp_builder.close()
    invalid_namespace.parent.mkdir(parents=True, exist_ok=True)
    invalid_namespace.write_bytes(b"invalid namespaced candidate")
    _write_bundle(legacy_path, metadata)

    dashboard = TrainingDashboard()

    assert dashboard.config["agent"] == "mlp"
    assert dashboard.resume_best_bundle == expected_mlp_path
    assert expected_mlp_path.exists()
    assert not legacy_path.exists()
    assert not invalid_namespace.exists()
    assert list(invalid_namespace.parent.glob(f"{invalid_namespace.stem}.quarantine-*"))
    dashboard.close()


def test_valid_other_agent_legacy_is_migrated_without_replacing_current_cnn(
    tmp_path, monkeypatch
):
    legacy_path = _isolate_paths(monkeypatch, tmp_path)
    cnn_builder = TrainingDashboard()
    cnn_path = _write_current_bundle(cnn_builder, 5.0)
    cnn_builder.close()

    mlp_builder = TrainingDashboard()
    mlp_builder._merge_config({"agent": "mlp", "model_profile": "new"})
    benchmark = _benchmark(mlp_builder, 7.0)
    mlp_builder.guard_benchmark = benchmark
    mlp_builder.holdout_protocol = dict(benchmark["protocol"])
    mlp_builder.best_guard_objective = 7.0
    mlp_builder.trained_steps = 8
    metadata = mlp_builder._model_bundle_metadata(
        extra={"guard": _accepted_guard(mlp_builder, benchmark)}
    )
    mlp_path = mlp_builder._protected_best_path(benchmark["protocol"])
    mlp_builder.close()
    _write_bundle(legacy_path, metadata)

    dashboard = TrainingDashboard()

    assert dashboard.config["agent"] == "cnn"
    assert dashboard.resume_best_bundle == cnn_path
    assert dashboard.best_guard_objective == 5.0
    assert mlp_path.exists()
    assert not legacy_path.exists()
    dashboard.close()
