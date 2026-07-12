from __future__ import annotations

import importlib.util
import copy
import json
import random
import sys
from pathlib import Path

import chess
import pytest


ROOT = Path(__file__).resolve().parents[1]
CHESS_DIR = ROOT / "chess-ai"


@pytest.fixture(scope="module")
def chess_app():
    sys.path.insert(0, str(CHESS_DIR))
    try:
        spec = importlib.util.spec_from_file_location("test_chess_integrity_app", CHESS_DIR / "app.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(CHESS_DIR))


def isolated_trainer(chess_app, tmp_path: Path):
    trainer = chess_app.Trainer()
    trainer.checkpoint_path = tmp_path / "chess_policy.json"
    trainer.best_checkpoint_path = tmp_path / "chess_policy.best.json"
    trainer.reset(load_checkpoint=False)
    trainer.stockfish_path = ""
    trainer.exploration = 0.0
    return trainer


def checkpoint_payload(chess_app, weights: dict[str, float], **config) -> dict:
    fingerprint = chess_app.policy_fingerprint(weights)
    return {
        "checkpoint_version": chess_app.CHECKPOINT_VERSION,
        "training_protocol": chess_app.TRAINING_PROTOCOL,
        "created_at": "2026-07-12T00:00:00+0000",
        "teacher": "test",
        "weights": weights,
        "policy_fingerprint": fingerprint,
        "best_weights": weights,
        "learning": {
            "accepted_chunks": 3,
            "rejected_chunks": 2,
            "policy_updates": 5,
            "teacher_updates": 2,
            "rl_samples": 3,
        },
        "accepted_guard": {
            "accepted": True,
            "behavior_changed": True,
            "candidate_fingerprint": fingerprint,
            "baseline": {"avg_gap": 10.0},
            "candidate": {"avg_gap": 1.0},
            "holdout_baseline": {"avg_gap": 10.0},
            "holdout_candidate": {"avg_gap": 1.0},
        },
        "guard": {"accepted": True, "behavior_changed": True},
        "config": config,
    }


def test_guard_and_holdout_match_the_student_side(chess_app):
    assert set(chess_app.GUARD_FENS).isdisjoint(chess_app.HOLDOUT_FENS)
    assert set(chess_app.BENCHMARK_FENS).isdisjoint(chess_app.AUDIT_FENS)
    assert set(chess_app.PROMOTION_FENS).isdisjoint(chess_app.INDEPENDENT_AUDIT_FENS)
    assert chess_app.PROMOTION_FENS == chess_app.BENCHMARK_FENS + chess_app.AUDIT_FENS
    assert all(
        chess.Board(fen).turn == chess.WHITE
        for fen in chess_app.PROMOTION_FENS + chess_app.INDEPENDENT_AUDIT_FENS
    )


def test_gap_uses_the_chosen_move_full_ranking(chess_app, tmp_path):
    trainer = isolated_trainer(chess_app, tmp_path)
    bad = {key: -1.5 for key in chess_app.FEATURE_KEYS}
    metrics = trainer.evaluate_teacher_gap(bad)
    assert any(row["rank"] > 5 for row in metrics["choices"])
    for row in metrics["choices"]:
        ranked = dict(chess_app.benchmark_ranked_moves(row["fen"]))
        expected = max(0.0, ranked[row["best"]] - ranked[row["chosen"]])
        assert row["gap"] == expected


def test_loader_rejects_a_checkpoint_below_verified_default(chess_app, tmp_path):
    trainer = isolated_trainer(chess_app, tmp_path)
    degraded = {
        "material": 0.25,
        "mobility": -0.01,
        "center": 0.21,
        "king_safety": 0.14,
        "reply_safety": 0.0,
    }
    trainer.best_checkpoint_path.write_text(
        json.dumps(checkpoint_payload(chess_app, degraded)),
        encoding="utf-8",
    )
    trainer.reset(load_checkpoint=True)
    assert trainer.weights == chess_app.DEFAULT_WEIGHTS
    assert trainer.loaded_checkpoint["path"] == ""
    assert trainer.loaded_checkpoint["rejected"]


def test_loader_does_not_call_behavior_equivalent_weights_training(chess_app, tmp_path):
    trainer = isolated_trainer(chess_app, tmp_path)
    trainer.checkpoint_path.write_text(
        json.dumps(checkpoint_payload(chess_app, dict(chess_app.DEFAULT_WEIGHTS))),
        encoding="utf-8",
    )
    trainer.reset(load_checkpoint=True)
    assert trainer.weights == chess_app.DEFAULT_WEIGHTS
    assert trainer.loaded_checkpoint["path"] == ""
    assert trainer.loaded_checkpoint["rejected"]


def test_loader_restores_a_better_verified_checkpoint_and_config(chess_app, tmp_path):
    trainer = isolated_trainer(chess_app, tmp_path)
    improved = dict(chess_app.DEFAULT_WEIGHTS)
    improved["reply_safety"] = 2.0
    trainer.checkpoint_path.write_text(
        json.dumps(
            checkpoint_payload(
                chess_app,
                improved,
                chunk_moves=64,
                learning_rate=0.01,
                teacher_learning_rate=0.02,
                exploration=0.03,
                mutation=0.04,
                guard_enabled=True,
                guard_min_gap_delta=0.0,
                guard_holdout_tolerance=0.0,
                discount=0.9,
            )
        ),
        encoding="utf-8",
    )
    trainer.reset(load_checkpoint=True)
    assert trainer.weights == improved
    assert trainer.loaded_checkpoint["path"] == str(trainer.checkpoint_path)
    assert trainer.chunk_moves == 64
    assert trainer.learning_rate == 0.01
    assert trainer.accepted_chunks == 3
    assert trainer.rejected_chunks == 2
    assert trainer.guard_enabled is True


def test_loader_quarantines_better_weights_without_current_provenance(chess_app, tmp_path):
    trainer = isolated_trainer(chess_app, tmp_path)
    improved = dict(chess_app.DEFAULT_WEIGHTS)
    improved["reply_safety"] = 2.0
    trainer.checkpoint_path.write_text(
        json.dumps({"weights": improved, "learning": {"accepted_chunks": 99}}),
        encoding="utf-8",
    )
    trainer.reset(load_checkpoint=True)
    assert trainer.weights == chess_app.DEFAULT_WEIGHTS
    assert trainer.loaded_checkpoint["path"] == ""
    assert "missing current schema" in trainer.last_error


def test_legacy_config_cannot_disable_acceptance_guard(chess_app, tmp_path):
    trainer = isolated_trainer(chess_app, tmp_path)
    improved = dict(chess_app.DEFAULT_WEIGHTS)
    improved["reply_safety"] = 2.0
    trainer.checkpoint_path.write_text(
        json.dumps(checkpoint_payload(chess_app, improved, guard_enabled=False)),
        encoding="utf-8",
    )
    trainer.reset(load_checkpoint=True)
    assert trainer.guard_enabled is True
    assert trainer.loaded_checkpoint["path"] == ""


def test_failed_candidate_is_rolled_back_and_never_persisted(chess_app, tmp_path, monkeypatch):
    trainer = isolated_trainer(chess_app, tmp_path)
    with trainer.lock:
        trainer._save_checkpoint_locked()
    saved_before = json.loads(trainer.checkpoint_path.read_text(encoding="utf-8"))
    weights_before = dict(trainer.weights)

    def fail_mid_candidate():
        with trainer.lock:
            trainer.weights["reply_safety"] = 2.0
            trainer._save_checkpoint_locked()
        raise RuntimeError("injected candidate failure")

    monkeypatch.setattr(trainer, "_step_once", fail_mid_candidate)
    with pytest.raises(RuntimeError, match="injected candidate failure"):
        trainer.step_guarded(1)

    saved_after = json.loads(trainer.checkpoint_path.read_text(encoding="utf-8"))
    assert trainer.weights == weights_before
    assert trainer.guard_in_progress is False
    assert trainer.rejected_chunks == 1
    assert saved_after["weights"] == saved_before["weights"]
    assert saved_after["guard"]["accepted"] is False


def test_behavior_identical_candidate_rolls_back(chess_app, tmp_path):
    trainer = isolated_trainer(chess_app, tmp_path)
    before = trainer._capture_state_locked()
    result = trainer.step_guarded(1)
    assert result["accepted"] is False
    assert result["behavior_changed"] is False
    assert trainer.weights == before["weights"]
    assert trainer.ply == before["ply"]
    assert trainer.policy_updates == before["policy_updates"]
    assert trainer.rejected_chunks == 1


def test_public_step_is_guarded_and_private_candidate_is_not_in_snapshot(chess_app, tmp_path):
    trainer = isolated_trainer(chess_app, tmp_path)
    result = trainer.step_once()
    assert result["accepted"] is False
    assert trainer.weights == chess_app.DEFAULT_WEIGHTS

    with trainer.lock:
        accepted = trainer._capture_state_locked()
        trainer.guard_in_progress = True
        trainer.guard_public_state = copy.deepcopy(accepted)
        trainer.weights["reply_safety"] = 2.0
    snapshot = trainer.snapshot()
    assert snapshot["guard_in_progress"] is True
    assert snapshot["weights"] == accepted["weights"]
    assert snapshot["weights"] != trainer.weights
    with trainer.lock:
        trainer._restore_state_locked(accepted)
        trainer.guard_in_progress = False
        trainer.guard_public_state = None


def test_manual_weight_update_cannot_reuse_training_provenance(chess_app, tmp_path):
    trainer = isolated_trainer(chess_app, tmp_path)
    config_before = trainer.chunk_moves
    with pytest.raises(ValueError, match="manual policy-weight changes are disabled"):
        trainer.update_config(
            {
                "chunk_moves": 64,
                "weights": {**trainer.weights, "reply_safety": 2.0},
            }
        )
    assert trainer.weights == chess_app.DEFAULT_WEIGHTS
    assert trainer.chunk_moves == config_before


def test_real_teacher_training_improves_guard_and_holdout_then_reloads(chess_app, tmp_path):
    trainer = isolated_trainer(chess_app, tmp_path)
    audit_before = trainer.evaluate_teacher_gap(fens=chess_app.INDEPENDENT_AUDIT_FENS)
    random.seed(9)
    result = trainer.step_guarded(80)
    assert result["accepted"] is True
    assert result["behavior_changed"] is True
    assert result["candidate"]["avg_gap"] < result["baseline"]["avg_gap"]
    assert result["holdout_candidate"]["avg_gap"] <= result["holdout_baseline"]["avg_gap"]
    assert trainer._quality_key(result["holdout_candidate"]) < trainer._quality_key(
        result["holdout_baseline"]
    )
    assert trainer._quality_key(result["audit_candidate"]) <= trainer._quality_key(
        result["audit_baseline"]
    )
    assert trainer.accepted_chunks == 1
    audit_after = trainer.evaluate_teacher_gap(fens=chess_app.INDEPENDENT_AUDIT_FENS)
    assert trainer._quality_key(audit_after) <= trainer._quality_key(audit_before)
    trained_weights = dict(trainer.weights)

    rejected = trainer.step_guarded(1)
    assert rejected["accepted"] is False
    assert trainer.weights == trained_weights
    assert trainer.accepted_guard["candidate_fingerprint"] == chess_app.policy_fingerprint(
        trained_weights
    )

    reloaded = chess_app.Trainer()
    reloaded.checkpoint_path = trainer.checkpoint_path
    reloaded.best_checkpoint_path = trainer.best_checkpoint_path
    reloaded.reset(load_checkpoint=True)
    assert reloaded.weights == trained_weights
    assert reloaded.loaded_checkpoint["path"] in {
        str(trainer.checkpoint_path),
        str(trainer.best_checkpoint_path),
    }


def test_snapshot_names_the_actual_update_signal(chess_app, tmp_path):
    trainer = isolated_trainer(chess_app, tmp_path)
    learning = trainer.snapshot()["learning"]
    assert learning["mode"] == "reward-shaped teacher ranker"
    assert "update_signal_ema" in learning
    assert "td_error" not in learning
    assert "strength" not in learning
