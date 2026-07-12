import os
import json
import hashlib
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAKE_MAIN = REPO_ROOT / "snake-ai" / "main"
sys.path.insert(0, str(SNAKE_MAIN))

from snake_env import SnakeMlpEnv  # noqa: E402
from train import (  # noqa: E402
    checkpoint_callback_frequency,
    make_env,
    open_training_log,
    paired_evaluation_decision,
)
import web_dashboard  # noqa: E402
from web_dashboard import FIXED_HOLDOUT_KIND, TrainingDashboard  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_protected_best(tmp_path, monkeypatch):
    monkeypatch.setattr(
        web_dashboard,
        "BEST_MODEL_PATH",
        tmp_path / "isolated-best.snakeai.zip",
    )


def _env_factory_kwargs():
    return {
        "env_cls": SnakeMlpEnv,
        "seed": 17,
        "board_size": 6,
        "limit_step": True,
        "food_time_penalty": 0.0,
        "food_step_limit_multiplier": 4.0,
        "food_reward_bonus": 0.0,
        "distance_reward_scale": 0.1,
        "reachable_space_penalty": 0.0,
        "reachable_space_min_ratio": 0.35,
        "loop_penalty": 0.0,
        "loop_window": 16,
        "oscillation_penalty": 0.0,
        "oscillation_window": 12,
        "cnn_channel_first": True,
    }


def _metric(objective, food=1.0):
    return {
        "objective": float(objective),
        "avg_food": float(food),
        "avg_score": float(food),
        "avg_reward": 0.0,
    }


def _fixed_benchmark(dashboard, objective=4.25):
    protocol = dashboard._holdout_protocol()
    metrics = {
        **_metric(objective),
        "episodes": protocol["episodes"],
        "episode_results": [],
    }
    return {
        "kind": FIXED_HOLDOUT_KIND,
        "protocol": protocol,
        "objective": float(objective),
        "metrics": metrics,
    }


def _write_dashboard_bundle(path, metadata, model_payload=b"placeholder model weights"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("model.zip", model_payload)
        bundle.writestr("metadata.json", json.dumps(metadata))


def _accepted_dashboard_guard(benchmark, *, development_episodes=4):
    objective = float(benchmark["objective"])
    baseline = {
        "episodes": development_episodes,
        "objective": objective - 1.0,
        "avg_score": 0.0,
        "avg_food": 0.0,
        "avg_reward": 0.0,
    }
    candidate = {
        **baseline,
        "objective": objective,
        "avg_score": 1.0,
        "avg_food": 1.0,
    }
    protocol = benchmark["protocol"]
    return {
        "accepted": True,
        "promoted_to_best": True,
        "attempted_timesteps": 8,
        "episodes": development_episodes,
        "baseline": baseline,
        "candidate": candidate,
        "baseline_model_sha256": "0" * 64,
        "candidate_model_sha256": "f" * 64,
        "decision": {"accepted": True, "required_delta": 0.001},
        "promotion_basis": FIXED_HOLDOUT_KIND,
        "holdout_protocol": protocol,
        "holdout_seed_base": protocol["seed_base"],
        "holdout_episodes": protocol["episodes"],
        "holdout_max_steps": protocol["max_steps"],
        "holdout_baseline": dict(benchmark["metrics"]),
        "holdout_candidate": dict(benchmark["metrics"]),
    }


def _write_fingerprinted_dashboard_bundle(path, metadata, model_payload=b"test model"):
    model_sha256 = hashlib.sha256(model_payload).hexdigest()
    metadata["model_sha256"] = model_sha256
    metadata["guard"]["candidate_model_sha256"] = model_sha256
    _write_dashboard_bundle(path, metadata, model_payload)


def test_mlp_factory_never_receives_cnn_only_channel_first():
    wrapped = make_env(**_env_factory_kwargs())()
    try:
        assert isinstance(wrapped.unwrapped, SnakeMlpEnv)
    finally:
        wrapped.close()


def test_checkpoint_interval_is_total_transitions_not_vecenv_calls():
    assert checkpoint_callback_frequency(500_000, 32) == 15_625
    assert checkpoint_callback_frequency(16_000, 8) == 2_000
    assert checkpoint_callback_frequency(2, 8) == 1


def test_continuation_log_appends_session_without_erasing_prior_audit(tmp_path):
    log_path = tmp_path / "training_log.txt"
    log_path.write_text("prior accepted run\n", encoding="utf-8")
    args = SimpleNamespace(
        agent="mlp",
        load_model="checkpoint.zip",
        total_timesteps=64,
        num_envs=2,
    )

    first = open_training_log(log_path, args)
    first.close()
    second = open_training_log(log_path, args)
    second.close()

    content = log_path.read_text(encoding="utf-8")
    assert content.startswith("prior accepted run\n")
    assert content.count("=== training session ") == 2
    assert "load_model=checkpoint.zip" in content


def test_cli_paired_gate_requires_enough_training_and_both_behavior_suites():
    accepted = paired_evaluation_decision(
        _metric(1.0, food=1.0),
        _metric(2.0, food=2.0),
        _metric(1.0, food=1.0),
        _metric(2.0, food=2.0),
        min_delta=0.001,
        attempted_timesteps=4096,
        min_training_timesteps=4096,
    )
    short = paired_evaluation_decision(
        _metric(1.0, food=1.0),
        _metric(2.0, food=2.0),
        _metric(1.0, food=1.0),
        _metric(2.0, food=2.0),
        min_delta=0.001,
        attempted_timesteps=8,
        min_training_timesteps=4096,
    )
    holdout_flat = paired_evaluation_decision(
        _metric(1.0, food=1.0),
        _metric(2.0, food=2.0),
        _metric(1.0, food=1.0),
        _metric(1.0, food=1.0),
        min_delta=0.001,
        attempted_timesteps=4096,
        min_training_timesteps=4096,
    )

    assert accepted["accepted"] is True
    assert short["accepted"] is False
    assert short["reason"] == "insufficient_training_evidence"
    assert holdout_flat["accepted"] is False
    assert holdout_flat["reason"] == "fixed_holdout_not_improved"


def test_terminal_collision_preserves_length_and_uses_true_terminal_penalty():
    env = SnakeMlpEnv(seed=3, board_size=4, silent_mode=True, limit_step=True)
    try:
        env.reset(seed=3)
        initial_length = len(env.game.snake)
        _obs, reward, terminated, truncated, info = env.step(3)  # DOWN into the wall

        assert terminated is True
        assert truncated is False
        assert len(env.game.snake) == initial_length
        assert info["snake_size"] == initial_length
        assert tuple(info["attempted_head_pos"]) == (4, 2)
        assert reward == pytest.approx(env._terminal_penalty({"snake_size": initial_length}))
    finally:
        env.close()


def test_food_budget_failure_is_terminal_not_time_limit_bootstrap():
    env = SnakeMlpEnv(seed=4, board_size=4, silent_mode=True, limit_step=True)
    try:
        env.reset(seed=4)
        env.step_limit = 0
        _obs, reward, terminated, truncated, info = env.step(1)  # safe LEFT, then starve
        assert terminated is True
        assert truncated is False
        assert info["starved"] is True
        assert reward == pytest.approx(env._terminal_penalty(info))
    finally:
        env.close()

    vec = DummyVecEnv(
        [lambda: SnakeMlpEnv(seed=4, board_size=4, silent_mode=True, limit_step=True)]
    )
    try:
        vec.envs[0].step_limit = 0
        vec.reset()
        _obs, _rewards, dones, infos = vec.step(np.asarray([1]))
        assert bool(dones[0]) is True
        assert infos[0]["TimeLimit.truncated"] is False
        assert infos[0]["starved"] is True
    finally:
        vec.close()


class _DownPolicy:
    def predict(self, _obs, **_kwargs):
        return 3, None


class _ConfigMutatingDownPolicy:
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.mutated = False

    def predict(self, _obs, **_kwargs):
        if not self.mutated:
            self.dashboard.config["board_size"] = 10
            self.dashboard.config["distance_reward_scale"] = 99.0
            self.mutated = True
        return 3, None


class _GreedyFoodPolicy:
    def predict(self, obs, *, action_masks, **_kwargs):
        head = np.argwhere(obs == 1.0)[0]
        food = np.argwhere(obs == -1.0)[0]
        moves = ((-1, 0), (0, -1), (0, 1), (1, 0))
        valid_actions = np.flatnonzero(np.asarray(action_masks, dtype=bool))
        action = min(
            valid_actions,
            key=lambda item: (
                abs(head[0] + moves[item][0] - food[0])
                + abs(head[1] + moves[item][1] - food[1]),
                int(item),
            ),
        )
        return int(action), None


def test_guard_evaluation_is_deterministic_and_config_frozen():
    dashboard = TrainingDashboard()
    dashboard._merge_config({"agent": "mlp", "board_size": 6, "seed": 23})
    frozen = dict(dashboard.config)

    expected = dashboard._evaluate_model_score(
        _DownPolicy(),
        seed_base=1000,
        episodes=3,
        max_steps=30,
        eval_config=frozen,
    )
    dashboard.config = dict(frozen)
    actual = dashboard._evaluate_model_score(
        _ConfigMutatingDownPolicy(dashboard),
        seed_base=1000,
        episodes=3,
        max_steps=30,
    )

    assert actual == expected
    assert [row["seed"] for row in actual["episode_results"]] == [1000, 1001, 1002]


def test_behavior_gate_rejects_weight_only_change_and_holdout_regression():
    unchanged = TrainingDashboard._guard_decision(
        _metric(2.0),
        _metric(2.0),
        min_delta=0.0,
        holdout_baseline=_metric(1.0),
        holdout_candidate=_metric(1.0),
    )
    improved = TrainingDashboard._guard_decision(
        _metric(2.0),
        _metric(2.1),
        min_delta=0.01,
        holdout_baseline=_metric(1.0),
        holdout_candidate=_metric(1.0),
    )
    overfit = TrainingDashboard._guard_decision(
        _metric(2.0),
        _metric(2.1),
        min_delta=0.01,
        holdout_baseline=_metric(1.0),
        holdout_candidate=_metric(0.9),
    )

    assert unchanged["accepted"] is False
    assert unchanged["reason"] == "no_measured_behavior_improvement"
    assert improved["accepted"] is True
    assert overfit["accepted"] is False
    assert overfit["reason"] == "fixed_holdout_regression"


def test_behavior_gate_accepts_measured_food_collection_improvement():
    dashboard = TrainingDashboard()
    dashboard._merge_config({"agent": "mlp", "board_size": 6, "seed": 23})
    baseline = dashboard._evaluate_model_score(
        _DownPolicy(), seed_base=1000, episodes=4, max_steps=100
    )
    candidate = dashboard._evaluate_model_score(
        _GreedyFoodPolicy(), seed_base=1000, episodes=4, max_steps=100
    )
    holdout_baseline = dashboard._evaluate_model_score(
        _DownPolicy(), seed_base=7000, episodes=4, max_steps=100
    )
    holdout_candidate = dashboard._evaluate_model_score(
        _GreedyFoodPolicy(), seed_base=7000, episodes=4, max_steps=100
    )

    decision = dashboard._guard_decision(
        baseline,
        candidate,
        min_delta=0.001,
        holdout_baseline=holdout_baseline,
        holdout_candidate=holdout_candidate,
    )
    assert candidate["avg_food"] > baseline["avg_food"]
    assert holdout_candidate["avg_food"] > holdout_baseline["avg_food"]
    assert decision["accepted"] is True
    assert decision["reason"] == "behavior_improved_and_holdout_preserved"


def test_bundle_restores_best_guard_threshold():
    dashboard = TrainingDashboard()
    benchmark = _fixed_benchmark(dashboard)
    dashboard._restore_best(
        {
            "guard_benchmark": benchmark,
            "best": {
                "score": 3,
                "steps": 40,
                "trained_steps": 128,
                "iteration": 2,
                "guard_objective": 4.25,
                "guard_objective_kind": FIXED_HOLDOUT_KIND,
            }
        }
    )

    assert dashboard.best_guard_objective == 4.25
    assert dashboard._is_new_best_guard(4.24) is False
    assert dashboard._is_new_best_guard(4.25) is False
    assert dashboard._is_new_best_guard(4.26) is True

    dashboard.history = [{"guard": {"accepted": False, "reason": "regression"}}]
    dashboard._restore_last_guard({})
    assert dashboard.last_guard == {"accepted": False, "reason": "regression"}


def test_guard_cannot_be_disabled_and_holdout_protocol_is_strict():
    dashboard = TrainingDashboard()
    dashboard._merge_config(
        {
            "guard_enabled": False,
            "guard_holdout_max_drop": 99.0,
            "guard_holdout_episodes": 2,
        }
    )
    assert dashboard.config["training_enabled"] is True
    assert dashboard.config["guard_enabled"] is True
    assert dashboard.config["guard_holdout_max_drop"] == 0.0
    assert dashboard.config["guard_holdout_episodes"] == 8
    assert dashboard._holdout_protocol()["episodes"] == 8


def test_fixed_holdout_best_promotion_ignores_incomparable_development_score(
    tmp_path, monkeypatch
):
    best_path = tmp_path / "best.snakeai.zip"
    monkeypatch.setattr(web_dashboard, "BEST_MODEL_PATH", best_path)
    dashboard = TrainingDashboard()
    protocol = dashboard._holdout_protocol()
    dashboard.guard_benchmark = _fixed_benchmark(dashboard, objective=5.0)
    dashboard.best_guard_objective = 5.0
    dashboard.trained_steps = 8
    writes = []

    def write_bundle(destination, metadata):
        writes.append((destination, metadata))
        _write_fingerprinted_dashboard_bundle(destination, metadata)

    monkeypatch.setattr(dashboard, "_write_model_bundle", write_bundle)

    # A hypothetical very high drifting development result is irrelevant: the
    # fixed holdout objective is lower, so protected best must not be replaced.
    assert dashboard._promote_fixed_holdout_best(
        protocol=protocol,
        metrics={**_metric(4.9), "episodes": 8},
        guard={"candidate": _metric(1000.0)},
    ) is False
    assert writes == []

    assert dashboard._promote_fixed_holdout_best(
        protocol=protocol,
        metrics={**_metric(5.1), "episodes": 8},
        guard=_accepted_dashboard_guard(
            {
                "kind": FIXED_HOLDOUT_KIND,
                "protocol": protocol,
                "objective": 5.1,
                "metrics": {**_metric(5.1), "episodes": 8},
            }
        ),
    ) is True
    assert len(writes) == 1
    assert dashboard.best_guard_objective == 5.1


def test_protected_bundle_is_discovered_and_model_reloaded_after_restart(
    tmp_path, monkeypatch
):
    best_path = tmp_path / "best.snakeai.zip"
    monkeypatch.setattr(web_dashboard, "BEST_MODEL_PATH", best_path)
    first = TrainingDashboard()
    first._merge_config(
        {
            "agent": "mlp",
            "model_profile": "new",
            "board_size": 4,
            "num_envs": 1,
            "n_steps": 8,
            "batch_size": 8,
            "device": "cpu",
            "guard_holdout_episodes": 8,
        }
    )
    first._ensure_model()
    first.model.learn(total_timesteps=8, reset_num_timesteps=False, progress_bar=False)
    expected = {
        key: value.detach().cpu().clone()
        for key, value in first.model.policy.state_dict().items()
    }
    protocol = first._holdout_protocol()
    metrics = first._evaluate_model_score(
        first.model,
        seed_base=protocol["seed_base"],
        episodes=protocol["episodes"],
        max_steps=protocol["max_steps"],
        eval_config=protocol["eval_config"],
    )
    benchmark = {
        "kind": FIXED_HOLDOUT_KIND,
        "protocol": protocol,
        "objective": metrics["objective"],
        "metrics": metrics,
    }
    assert first._promote_fixed_holdout_best(
        protocol=protocol,
        metrics=metrics,
        guard=_accepted_dashboard_guard(benchmark),
    ) is True
    first.close()

    restarted = TrainingDashboard()
    assert restarted.best_guard_objective == metrics["objective"]
    assert restarted.resume_best_bundle == best_path
    restarted._ensure_model()
    for key, value in restarted.model.policy.state_dict().items():
        assert torch.equal(value.detach().cpu(), expected[key])
    restarted.close()

    # Even a self-consistent model+metadata rewrite cannot be served merely by
    # updating hashes: startup recomputes the fixed validation behavior.
    with zipfile.ZipFile(best_path) as persisted:
        model_payload = persisted.read("model.zip")
        forged_metadata = json.loads(persisted.read("metadata.json"))
    forged_metrics = dict(forged_metadata["guard_benchmark"]["metrics"])
    forged_metrics["avg_score"] = float(forged_metrics["avg_score"]) + 10.0
    forged_metrics["objective"] = float(forged_metrics["objective"]) + 10.0
    forged_metadata["guard_benchmark"]["metrics"] = forged_metrics
    forged_metadata["guard_benchmark"]["objective"] = forged_metrics["objective"]
    forged_metadata["guard"]["holdout_candidate"] = dict(forged_metrics)
    forged_metadata["guard"]["holdout_baseline"] = dict(forged_metrics)
    _write_fingerprinted_dashboard_bundle(
        best_path,
        forged_metadata,
        model_payload,
    )

    forged = TrainingDashboard()
    assert forged.resume_best_bundle == best_path
    forged._ensure_model()
    assert forged.resume_best_bundle is None
    assert forged.guard_benchmark == {}
    assert not best_path.exists()
    assert "fixed-holdout revalidation" in forged.last_event
    forged.close()


def test_current_bundle_rejects_embedded_model_replacement_and_future_format():
    best_path = web_dashboard.BEST_MODEL_PATH
    builder = TrainingDashboard()
    benchmark = _fixed_benchmark(builder, objective=2.0)
    builder.guard_benchmark = benchmark
    builder.holdout_protocol = dict(benchmark["protocol"])
    builder.best_guard_objective = 2.0
    builder.trained_steps = 8
    metadata = builder._model_bundle_metadata(
        extra={"guard": _accepted_dashboard_guard(benchmark)}
    )
    builder.close()
    _write_fingerprinted_dashboard_bundle(best_path, metadata, b"original model")

    with zipfile.ZipFile(best_path) as persisted:
        unchanged_metadata = json.loads(persisted.read("metadata.json"))
    _write_dashboard_bundle(best_path, unchanged_metadata, b"replaced model")
    replaced = TrainingDashboard()
    assert replaced.resume_best_bundle is None
    assert not best_path.exists()
    assert "fingerprint or promotion provenance" in replaced.last_event
    replaced.close()

    future_metadata = dict(metadata)
    future_metadata["format_version"] = web_dashboard.DASHBOARD_BUNDLE_FORMAT_VERSION + 1
    _write_fingerprinted_dashboard_bundle(best_path, future_metadata, b"future model")
    future = TrainingDashboard()
    assert future.resume_best_bundle is None
    assert not best_path.exists()
    assert "legacy or stale format" in future.last_event
    future.close()


def test_real_cnn_bundle_rejects_self_consistent_tampered_architecture_metadata():
    best_path = web_dashboard.BEST_MODEL_PATH
    builder = TrainingDashboard()
    builder._merge_config(
        {
            "agent": "cnn",
            "model_profile": "new",
            "board_size": 6,
            "cnn_channel_first": False,
            "num_envs": 1,
            "n_steps": 8,
            "batch_size": 8,
            "n_epochs": 1,
            "device": "cpu",
        }
    )
    builder._ensure_model()
    builder.model.learn(total_timesteps=8, reset_num_timesteps=False, progress_bar=False)
    protocol = builder._holdout_protocol()
    metrics = {
        "episodes": protocol["episodes"],
        "avg_score": 0.0,
        "avg_food": 0.0,
        "avg_reward": 0.0,
        "objective": 0.0,
        "episode_results": [],
    }
    benchmark = {
        "kind": FIXED_HOLDOUT_KIND,
        "protocol": protocol,
        "objective": 0.0,
        "metrics": metrics,
    }
    assert builder._promote_fixed_holdout_best(
        protocol=protocol,
        metrics=metrics,
        guard=_accepted_dashboard_guard(benchmark),
    )
    snapshot = builder._read_bundle_snapshot(best_path)
    provenance = builder._validated_bundle_provenance(
        snapshot["metadata"],
        model_sha256=snapshot["model_sha256"],
    )
    assert builder._validate_loaded_model_architecture(builder.model, provenance)

    with zipfile.ZipFile(best_path) as persisted:
        model_payload = persisted.read("model.zip")
        metadata = json.loads(persisted.read("metadata.json"))
    tampered_architecture = {
        "cnn_channels": "16,32",
        "cnn_kernel_sizes": "5,3",
        "cnn_strides": "2,1",
        "cnn_features_dim": 256,
    }
    metadata["config"].update(tampered_architecture)
    metadata["guard_benchmark"]["protocol"]["eval_config"].update(
        tampered_architecture
    )
    metadata["guard"]["holdout_protocol"] = json.loads(
        json.dumps(metadata["guard_benchmark"]["protocol"])
    )
    tampered_provenance = builder._validated_bundle_provenance(
        metadata,
        model_sha256=snapshot["model_sha256"],
    )
    assert tampered_provenance
    builder._merge_config(tampered_architecture)
    with pytest.raises(ValueError, match="do not match metadata architecture"):
        builder._validate_loaded_model_architecture(
            builder.model,
            tampered_provenance,
        )
    builder.close()
    _write_fingerprinted_dashboard_bundle(best_path, metadata, model_payload)

    restarted = TrainingDashboard()
    assert restarted.resume_best_bundle == best_path
    assert restarted.config["cnn_channels"] == "16,32"
    restarted._ensure_model()

    assert restarted.resume_best_bundle is None
    assert restarted.guard_benchmark == {}
    assert not best_path.exists()
    assert "fixed-holdout revalidation" in restarted.last_event
    assert restarted.model.policy.features_extractor.architecture["channels"] == (
        16,
        32,
    )
    restarted.close()


def test_startup_quarantines_legacy_protected_bundle_even_with_weights_and_guard(
    tmp_path, monkeypatch
):
    best_path = web_dashboard.BEST_MODEL_PATH
    builder = TrainingDashboard()
    benchmark = _fixed_benchmark(builder, objective=9.0)
    metadata = builder._model_bundle_metadata(
        extra={
            "format_version": 1,
            "guard_benchmark": benchmark,
            "guard": {
                "accepted": True,
                "promoted_to_best": True,
                "reason": "legacy claim must not be trusted",
            },
        }
    )
    builder.close()
    _write_dashboard_bundle(best_path, metadata, b"legacy weights must not be served")

    dashboard = TrainingDashboard()

    assert dashboard.resume_best_bundle is None
    assert dashboard.guard_benchmark == {}
    assert dashboard.last_guard == {}
    assert dashboard.best_guard_objective == float("-inf")
    assert not best_path.exists()
    quarantined = list(tmp_path.glob("*quarantine-legacy-or-stale-format-*.zip"))
    assert len(quarantined) == 1
    with zipfile.ZipFile(quarantined[0]) as bundle:
        assert bundle.read("model.zip") == b"legacy weights must not be served"
    assert "quarantined stale protected bundle" in dashboard.last_event
    assert "using baseline fallback" in dashboard.last_event

    fake_env = SimpleNamespace(envs=[], close=lambda: None)
    fallback_model = SimpleNamespace(
        num_timesteps=23,
        device="cpu",
        policy=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{}])),
    )
    fallback_path = web_dashboard.MAIN_DIR / "original_models" / "fallback.zip"
    monkeypatch.setattr(dashboard, "_make_train_env", lambda: fake_env)
    monkeypatch.setattr(dashboard, "_initial_model_path", lambda _device: fallback_path)
    monkeypatch.setattr(
        dashboard,
        "_load_model_from_bundle",
        lambda *_args, **_kwargs: pytest.fail("legacy bundle load was attempted"),
    )
    monkeypatch.setattr(
        dashboard,
        "_load_model",
        lambda model_path, _env, _device: (
            fallback_model
            if model_path == fallback_path
            else pytest.fail(f"unexpected model source: {model_path}")
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "_validate_loaded_model_architecture",
        lambda model, *_args, **_kwargs: (
            True
            if model is fallback_model
            else pytest.fail("unexpected fallback model was attested")
        ),
    )
    dashboard._ensure_model()
    assert dashboard.model is fallback_model
    assert dashboard.trained_steps == 23
    assert "using baseline fallback" in dashboard.last_event
    assert "loaded original model" in dashboard.last_event
    dashboard.close()


def test_startup_quarantines_v2_bundle_with_noncurrent_holdout_protocol(tmp_path):
    best_path = web_dashboard.BEST_MODEL_PATH
    builder = TrainingDashboard()
    benchmark = _fixed_benchmark(builder, objective=4.0)
    benchmark["protocol"]["seed_base"] += 1
    builder.trained_steps = 8
    metadata = builder._model_bundle_metadata(
        extra={
            "guard_benchmark": benchmark,
            "guard": _accepted_dashboard_guard(benchmark),
        }
    )
    builder.close()
    _write_fingerprinted_dashboard_bundle(best_path, metadata)

    dashboard = TrainingDashboard()

    assert dashboard.resume_best_bundle is None
    assert dashboard.guard_benchmark == {}
    assert not best_path.exists()
    assert list(tmp_path.glob("*quarantine-fixed-holdout-protocol-*.zip"))
    assert "protocol does not match configuration" in dashboard.last_event
    assert "using baseline fallback" in dashboard.last_event


def test_trusted_legacy_import_loads_weights_but_clears_claimed_provenance(
    tmp_path, monkeypatch
):
    dashboard = TrainingDashboard()
    benchmark = _fixed_benchmark(dashboard, objective=99.0)
    metadata = {
        "format": "snake-ai-dashboard-bundle",
        "format_version": 1,
        "config": dict(dashboard.config),
        "trained_steps": 999_999,
        "iteration": 81,
        "history": [{"guard": {"accepted": True, "reason": "legacy claim"}}],
        "best": {
            "score": 999,
            "steps": 1,
            "trained_steps": 999_999,
            "iteration": 81,
            "guard_objective": 99.0,
        },
        "guard_benchmark": benchmark,
        "guard": {"accepted": True, "promoted_to_best": True},
    }
    upload_path = tmp_path / "trusted-legacy.snakeai.zip"
    _write_dashboard_bundle(upload_path, metadata)

    fake_env = SimpleNamespace(envs=[], close=lambda: None)
    fake_model = SimpleNamespace(
        num_timesteps=17,
        device="cpu",
        policy=SimpleNamespace(
            optimizer=SimpleNamespace(param_groups=[{"lr": 123.0}])
        ),
    )
    monkeypatch.setattr(web_dashboard, "MODEL_UPLOAD_ENABLED", True)
    monkeypatch.setattr(dashboard, "_make_train_env", lambda: fake_env)
    monkeypatch.setattr(
        dashboard,
        "_load_model",
        lambda _model_path, _env, _device: fake_model,
    )

    class UploadedBundle:
        def save(self, destination):
            Path(destination).write_bytes(upload_path.read_bytes())

    snapshot = dashboard.import_model_bundle(UploadedBundle())

    assert dashboard.model is fake_model
    assert snapshot["trained_steps"] == 17
    assert snapshot["iteration"] == 0
    assert snapshot["history"] == []
    assert snapshot["guard"] == {}
    assert snapshot["best"]["score"] == 0
    assert snapshot["best"]["guard_objective"] is None
    assert dashboard.resume_best_bundle is None
    assert "unverified baseline" in snapshot["last_event"]
    assert "provenance cleared" in snapshot["last_event"]
    dashboard.close()


def test_candidate_evaluation_exception_rolls_back_precommit_model(tmp_path):
    dashboard = TrainingDashboard()
    dashboard._merge_config(
        {
            "agent": "mlp",
            "model_profile": "new",
            "board_size": 4,
            "num_envs": 1,
            "n_steps": 8,
            "batch_size": 8,
            "n_epochs": 1,
            "device": "cpu",
            "guard_eval_episodes": 4,
            "guard_holdout_episodes": 8,
        }
    )
    dashboard._ensure_model()
    baseline_steps = int(dashboard.model.num_timesteps)
    baseline = {
        key: value.detach().cpu().clone()
        for key, value in dashboard.model.policy.state_dict().items()
    }
    calls = 0

    def injected_evaluation(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:  # first candidate evaluation, after model.learn()
            raise RuntimeError("injected candidate evaluation failure")
        return {**_metric(1.0), "episodes": 8, "episode_results": []}

    dashboard._evaluate_model_score = injected_evaluation
    with pytest.raises(RuntimeError, match="injected candidate evaluation failure"):
        dashboard._run_guarded_training_transaction(8)

    assert int(dashboard.model.num_timesteps) == baseline_steps
    for key, value in dashboard.model.policy.state_dict().items():
        assert torch.equal(value.detach().cpu(), baseline[key])
    dashboard.close()


def test_best_promotion_exception_also_rolls_back_candidate():
    dashboard = TrainingDashboard()
    dashboard._merge_config(
        {
            "agent": "mlp",
            "model_profile": "new",
            "board_size": 4,
            "num_envs": 1,
            "n_steps": 8,
            "batch_size": 8,
            "n_epochs": 1,
            "device": "cpu",
            "guard_eval_episodes": 4,
            "guard_holdout_episodes": 8,
        }
    )
    dashboard._ensure_model()
    baseline_steps = int(dashboard.model.num_timesteps)
    baseline = {
        key: value.detach().cpu().clone()
        for key, value in dashboard.model.policy.state_dict().items()
    }
    results = iter(
        [
            {**_metric(1.0, food=1.0), "episodes": 4},
            {**_metric(1.0, food=1.0), "episodes": 8},
            {**_metric(2.0, food=2.0), "episodes": 4},
            {**_metric(2.0, food=2.0), "episodes": 8},
        ]
    )
    dashboard._evaluate_model_score = lambda *_args, **_kwargs: next(results)

    def fail_promotion(**_kwargs):
        raise OSError("injected atomic promotion failure")

    dashboard._promote_fixed_holdout_best = fail_promotion
    with pytest.raises(OSError, match="injected atomic promotion failure"):
        dashboard._run_guarded_training_transaction(8)

    assert int(dashboard.model.num_timesteps) == baseline_steps
    for key, value in dashboard.model.policy.state_dict().items():
        assert torch.equal(value.detach().cpu(), baseline[key])
    dashboard.close()


def test_ui_labels_real_ppo_preview_source_and_guard_evidence():
    html = (REPO_ROOT / "snake-ai" / "web" / "index.html").read_text(encoding="utf-8")
    app = (REPO_ROOT / "snake-ai" / "web" / "app.js").read_text(encoding="utf-8")
    assert "Start PPO Training" in html
    assert "Enable real PPO weight training" in html
    assert 'id="guardEvidence"' in html
    assert "Hamiltonian oracle (not PPO evidence)" in app
    assert "Trusted imports load as unverified baselines" in html
    assert "Imported as an unverified baseline" in app
    assert '"preview_strategy": strategy' in (
        REPO_ROOT / "snake-ai" / "main" / "web_dashboard.py"
    ).read_text(encoding="utf-8")


def test_close_keeps_stop_requested_until_busy_worker_reaches_boundary(monkeypatch):
    dashboard = TrainingDashboard()
    release = threading.Event()
    worker = threading.Thread(target=release.wait, daemon=True)
    dashboard.thread = worker
    dashboard.running = True
    worker.start()
    monkeypatch.setattr(web_dashboard, "THREAD_JOIN_TIMEOUT_SECONDS", 0.01)

    assert dashboard.close() is False
    assert dashboard.running is False
    assert dashboard.stop_requested is True

    release.set()
    worker.join(timeout=1)
    assert dashboard.close() is True
    assert dashboard.thread is None
    assert dashboard.stop_requested is False


def test_rejected_weight_change_rolls_back_to_reload_equivalent_policy(tmp_path):
    dashboard = TrainingDashboard()
    dashboard._merge_config(
        {
            "agent": "mlp",
            "model_profile": "new",
            "board_size": 4,
            "num_envs": 1,
            "n_steps": 8,
            "batch_size": 8,
            "device": "cpu",
        }
    )
    dashboard._ensure_model()
    try:
        baseline_eval = dashboard._evaluate_model_score(
            dashboard.model,
            seed_base=700,
            episodes=2,
            max_steps=30,
        )
        baseline_steps = int(dashboard.model.num_timesteps)
        baseline_state = {
            key: value.detach().cpu().clone()
            for key, value in dashboard.model.policy.state_dict().items()
        }
        checkpoint = tmp_path / "before.zip"
        dashboard.model.save(checkpoint)

        # Change critic weights only: the file changed, but deterministic policy
        # behavior did not.  This must not count as training improvement.
        value_key = next(key for key in baseline_state if "value_net" in key)
        with torch.no_grad():
            dashboard.model.policy.state_dict()[value_key].add_(0.25)
        changed_state = dashboard.model.policy.state_dict()
        assert not torch.equal(changed_state[value_key].cpu(), baseline_state[value_key])

        candidate_eval = dashboard._evaluate_model_score(
            dashboard.model,
            seed_base=700,
            episodes=2,
            max_steps=30,
        )
        decision = dashboard._guard_decision(
            baseline_eval,
            candidate_eval,
            min_delta=0.0,
            holdout_baseline=baseline_eval,
            holdout_candidate=candidate_eval,
        )
        assert candidate_eval == baseline_eval
        assert decision["accepted"] is False

        dashboard._rollback_guard_candidate(checkpoint)
        reloaded_eval = dashboard._evaluate_model_score(
            dashboard.model,
            seed_base=700,
            episodes=2,
            max_steps=30,
        )
        assert reloaded_eval == baseline_eval
        assert int(dashboard.model.num_timesteps) == baseline_steps
        for key, expected in baseline_state.items():
            assert torch.equal(dashboard.model.policy.state_dict()[key].cpu(), expected)
    finally:
        if dashboard.train_env is not None:
            dashboard.train_env.close()


def test_real_minimal_mlp_cli_training_starts_and_saves(tmp_path):
    save_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    command = [
        sys.executable,
        str(SNAKE_MAIN / "train.py"),
        "--agent",
        "mlp",
        "--board-size",
        "6",
        "--num-envs",
        "1",
        "--total-timesteps",
        "8",
        "--n-steps",
        "8",
        "--batch-size",
        "8",
        "--n-epochs",
        "1",
        "--device",
        "cpu",
        "--save-dir",
        str(save_dir),
        "--log-dir",
        str(log_dir),
        "--checkpoint-interval-timesteps",
        "8",
        "--guard-eval-episodes",
        "4",
        "--guard-holdout-episodes",
        "8",
        "--guard-max-steps",
        "30",
        "--no-stdout-log",
    ]
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    result = subprocess.run(
        command,
        cwd=SNAKE_MAIN,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    report = json.loads((save_dir / "training_guard_report.json").read_text(encoding="utf-8"))
    assert report["decision"]["verified"] is False
    assert report["decision"]["reason"] == "insufficient_training_evidence"
    assert report["artifact_status"] == "unverified_candidate"
    assert not (save_dir / "ppo_snake_final.zip").exists()
    candidate_path = save_dir / "ppo_snake_candidate_unverified.zip"
    assert candidate_path.exists()
    with zipfile.ZipFile(candidate_path) as candidate_bundle:
        embedded = json.loads(candidate_bundle.read("training_guard.json"))
    assert embedded["artifact_status"] == "unverified_candidate"
    assert embedded["decision"]["verified"] is False
    assert list(save_dir.glob("ppo_snake_mlp_*_steps.zip"))
