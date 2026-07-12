import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAKE_MAIN = REPO_ROOT / "snake-ai" / "main"
sys.path.insert(0, str(SNAKE_MAIN))

import web_dashboard  # noqa: E402
from cnn_features import ConfigurableCNN, validate_cnn_spatial_shape  # noqa: E402
from web_dashboard import DEFAULT_CONFIG, TrainingDashboard  # noqa: E402


def _config_only_dashboard():
    dashboard = TrainingDashboard.__new__(TrainingDashboard)
    dashboard.config = dict(DEFAULT_CONFIG)
    return dashboard


def test_dashboard_accepts_valid_cnn_board_and_architecture():
    dashboard = _config_only_dashboard()

    dashboard._merge_config(
        {
            "agent": "cnn",
            "board_size": 12,
            "cnn_channels": "16,32",
            "cnn_kernel_sizes": "5,3",
            "cnn_strides": "2,1",
            "cnn_channel_first": False,
        }
    )

    assert dashboard.config["board_size"] == 12
    assert dashboard.config["cnn_channel_first"] is False
    assert dashboard.config["cnn_channels"] == "16,32"


def test_dashboard_rejects_invalid_cnn_board_transactionally():
    dashboard = _config_only_dashboard()
    original = dict(dashboard.config)

    with pytest.raises(ValueError, match="must divide 84 exactly"):
        dashboard._merge_config({"agent": "cnn", "board_size": 10})

    assert dashboard.config == original


def test_mlp_can_use_board_that_is_not_a_cnn_image_divisor():
    dashboard = _config_only_dashboard()

    dashboard._merge_config({"agent": "mlp", "board_size": 10})

    assert dashboard.config["board_size"] == 10


def test_dashboard_rejects_impossible_convolution_transactionally():
    dashboard = _config_only_dashboard()
    original = dict(dashboard.config)

    with pytest.raises(ValueError, match="kernel 85 does not fit"):
        dashboard._merge_config(
            {
                "agent": "cnn",
                "cnn_channels": "32",
                "cnn_kernel_sizes": "85",
                "cnn_strides": "1",
            }
        )

    assert dashboard.config == original


@pytest.mark.parametrize(
    "updates",
    [
        {"training_enabled": "false"},
        {"board_size": 6.0},
        {"cnn_channels": [16.0, 32.0]},
        {"cnn_features_dim": True},
    ],
)
def test_dashboard_rejects_coercive_setting_types_transactionally(updates):
    dashboard = _config_only_dashboard()
    original = dict(dashboard.config)

    with pytest.raises(TypeError):
        dashboard._merge_config(updates)

    assert dashboard.config == original


def test_dashboard_rejects_cnn_architecture_that_exceeds_memory_budget():
    dashboard = _config_only_dashboard()

    with pytest.raises(ValueError, match="architecture is too large"):
        dashboard._merge_config(
            {
                "cnn_channels": "512",
                "cnn_kernel_sizes": "1",
                "cnn_strides": "1",
                "cnn_features_dim": 2048,
            }
        )


def test_cnn_architecture_and_layout_change_protected_protocol_namespace():
    dashboard = _config_only_dashboard()
    default_protocol = dashboard._expected_holdout_protocol()
    default_id = dashboard._protocol_id(default_protocol)

    dashboard._merge_config(
        {
            "cnn_channel_first": False,
            "cnn_channels": "16,32",
            "cnn_kernel_sizes": "5,3",
            "cnn_strides": "2,1",
            "cnn_features_dim": 256,
        }
    )
    custom_protocol = dashboard._expected_holdout_protocol()

    assert custom_protocol["eval_config"]["cnn_channel_first"] is False
    assert custom_protocol["eval_config"]["cnn_channels"] == "16,32"
    assert custom_protocol["eval_config"]["cnn_features_dim"] == 256
    assert dashboard._protocol_id(custom_protocol) != default_id


def test_spatial_validator_reports_the_exact_failing_layer():
    with pytest.raises(ValueError, match="layer 2 kernel 40"):
        validate_cnn_spatial_shape(84, 84, (8, 40), (4, 1))


def test_settings_api_returns_validation_message(monkeypatch):
    def reject(_updates):
        raise ValueError("invalid CNN configuration")

    monkeypatch.setattr(
        web_dashboard,
        "dashboard",
        SimpleNamespace(update_config=reject),
    )
    response = web_dashboard.app.test_client().post(
        "/api/settings",
        json={"agent": "cnn", "board_size": 10},
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "ok": False,
        "error": "invalid CNN configuration",
    }


def test_real_repo_mlp_legacy_space_fallback_loads_and_attests(
    tmp_path, monkeypatch
):
    legacy_path = tmp_path / "legacy-best.snakeai.zip"
    monkeypatch.setattr(web_dashboard, "BEST_MODEL_PATH", legacy_path)
    monkeypatch.setattr(
        web_dashboard,
        "_DEFAULT_LEGACY_BEST_MODEL_PATH",
        legacy_path,
    )
    monkeypatch.setattr(
        web_dashboard,
        "PROTECTED_BEST_DIR",
        tmp_path / "protected",
    )
    dashboard = TrainingDashboard()
    try:
        dashboard._merge_config(
            {
                "agent": "mlp",
                "model_profile": "repo_original",
                "board_size": 12,
                "device": "cpu",
                "num_envs": 1,
                "n_steps": 8,
                "batch_size": 8,
            }
        )
        dashboard._ensure_model()

        assert dashboard._validate_loaded_model_architecture(dashboard.model)
        assert type(dashboard.model.policy).__name__ == "MaskableActorCriticPolicy"
        assert tuple(dashboard.model.observation_space.shape) == (12, 12)
        assert "loaded original model from" in dashboard.last_event
        assert "trained_models_mlp/ppo_snake_final.zip" in dashboard.last_event
    finally:
        dashboard.close()


def test_real_fullboard_legacy_nature_cnn_is_structurally_attested_only_for_fallback(
    tmp_path, monkeypatch
):
    legacy_path = tmp_path / "legacy-best.snakeai.zip"
    monkeypatch.setattr(web_dashboard, "BEST_MODEL_PATH", legacy_path)
    monkeypatch.setattr(
        web_dashboard,
        "_DEFAULT_LEGACY_BEST_MODEL_PATH",
        legacy_path,
    )
    monkeypatch.setattr(
        web_dashboard,
        "PROTECTED_BEST_DIR",
        tmp_path / "protected",
    )
    dashboard = TrainingDashboard()
    try:
        dashboard._merge_config(
            {
                "agent": "cnn",
                "model_profile": "fullboard_12x12",
                "board_size": 12,
                "cnn_channel_first": True,
                "device": "cpu",
                "num_envs": 1,
                "n_steps": 8,
                "batch_size": 8,
            }
        )
        dashboard._ensure_model()

        assert type(dashboard.model.policy.features_extractor).__name__ == "NatureCNN"
        assert dashboard._validate_loaded_model_architecture(
            dashboard.model,
            allow_equivalent_legacy_extractor=True,
        )
        with pytest.raises(ValueError, match="metadata architecture"):
            dashboard._validate_loaded_model_architecture(dashboard.model)
        assert "loaded original model from" in dashboard.last_event
        assert "ppo_snake_bc_final_12x12.zip" in dashboard.last_event
    finally:
        dashboard.close()


def test_dashboard_builds_and_predicts_with_channel_last_cnn(tmp_path, monkeypatch):
    legacy_path = tmp_path / "legacy-best.snakeai.zip"
    monkeypatch.setattr(web_dashboard, "BEST_MODEL_PATH", legacy_path)
    monkeypatch.setattr(
        web_dashboard,
        "_DEFAULT_LEGACY_BEST_MODEL_PATH",
        legacy_path,
    )
    monkeypatch.setattr(
        web_dashboard,
        "PROTECTED_BEST_DIR",
        tmp_path / "protected",
    )
    dashboard = TrainingDashboard()
    try:
        dashboard._merge_config(
            {
                "agent": "cnn",
                "board_size": 6,
                "model_profile": "new",
                "device": "cpu",
                "num_envs": 1,
                "n_steps": 8,
                "batch_size": 8,
                "n_epochs": 1,
                "cnn_channel_first": False,
            }
        )
        dashboard._ensure_model()
        env = dashboard._make_eval_env(31)
        try:
            observation, _info = env.reset(seed=31)
            action, _state = dashboard.model.predict(
                observation,
                action_masks=env.get_action_mask(),
                deterministic=True,
            )
        finally:
            env.close()

        assert observation.shape == (84, 84, 3)
        assert 0 <= int(action) < 4
        assert isinstance(
            dashboard.model.policy.features_extractor,
            ConfigurableCNN,
        )

        evaluations = iter(
            [
                {"episodes": 4, "avg_score": 0.0, "avg_food": 0.0, "avg_reward": 0.0, "objective": 0.0},
                {"episodes": 8, "avg_score": 0.0, "avg_food": 0.0, "avg_reward": 0.0, "objective": 0.0},
                {"episodes": 4, "avg_score": 1.0, "avg_food": 1.0, "avg_reward": 0.0, "objective": 1.5},
                {"episodes": 8, "avg_score": 1.0, "avg_food": 1.0, "avg_reward": 0.0, "objective": 1.5},
            ]
        )
        monkeypatch.setattr(
            dashboard,
            "_evaluate_model_score",
            lambda *_args, **_kwargs: next(evaluations),
        )
        monkeypatch.setattr(
            dashboard,
            "_promote_fixed_holdout_best",
            lambda **_kwargs: True,
        )
        guard = dashboard._run_guarded_training_transaction(8)

        assert guard["accepted"] is True
        assert guard["promoted_to_best"] is True
        assert guard["attempted_timesteps"] == 8
        assert dashboard.model.num_timesteps == 8

        accepted_steps = int(dashboard.model.num_timesteps)
        accepted_state = {
            key: value.detach().cpu().clone()
            for key, value in dashboard.model.policy.state_dict().items()
        }
        flat = {
            "episodes": 4,
            "avg_score": 0.0,
            "avg_food": 0.0,
            "avg_reward": 0.0,
            "objective": 0.0,
        }
        holdout_flat = {**flat, "episodes": 8}
        rejected_evaluations = iter([flat, holdout_flat, flat, holdout_flat])
        monkeypatch.setattr(
            dashboard,
            "_evaluate_model_score",
            lambda *_args, **_kwargs: next(rejected_evaluations),
        )
        rejected = dashboard._run_guarded_training_transaction(8)

        assert rejected["accepted"] is False
        assert int(dashboard.model.num_timesteps) == accepted_steps
        for key, expected in accepted_state.items():
            assert torch.equal(
                dashboard.model.policy.state_dict()[key].detach().cpu(),
                expected,
            )
    finally:
        dashboard.close()
