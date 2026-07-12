import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAKE_MAIN = REPO_ROOT / "snake-ai" / "main"
sys.path.insert(0, str(SNAKE_MAIN))

from snake_env import (  # noqa: E402
    SnakeCnnEnv,
    validate_cnn_board_size,
    validate_cnn_channel_mode,
)


def _cnn_cli_command(save_dir, log_dir, *, board_size, channel_first=True):
    command = [
        sys.executable,
        str(SNAKE_MAIN / "train.py"),
        "--agent",
        "cnn",
        "--board-size",
        str(board_size),
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
        "--seed",
        "123",
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
        "10",
        "--no-stdout-log",
    ]
    if not channel_first:
        command.append("--no-cnn-channel-first")
    return command


def _subprocess_env():
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["SNAKE_NATIVE_THREADS"] = "1"
    return env


def test_cnn_board_and_channel_validation_matches_observation_layouts():
    assert validate_cnn_board_size(6) == 6
    with pytest.raises(ValueError, match=r"board_size=10.*image_size=84.*Compatible values"):
        validate_cnn_board_size(10)
    with pytest.raises(ValueError, match="channel_first must be a boolean"):
        validate_cnn_channel_mode("CHW")
    with pytest.raises(ValueError, match="image_size must be an integer"):
        SnakeCnnEnv(seed=7, board_size=6, image_size="84")
    with pytest.raises(ValueError, match="image_size must be an integer"):
        SnakeCnnEnv(seed=7, board_size=6, image_size=True)
    with pytest.raises(ValueError, match="channel_first must be a boolean"):
        SnakeCnnEnv(seed=7, board_size=6, channel_first=1)

    positional_env = SnakeCnnEnv(7, 12)
    try:
        positional_observation, _info = positional_env.reset(seed=7)
        assert positional_env.board_size == 12
        assert positional_observation.shape == (84, 84, 3)
        assert positional_observation.dtype.name == "uint8"
    finally:
        positional_env.close()

    for channel_first, expected_shape in (
        (True, (3, 84, 84)),
        (False, (84, 84, 3)),
    ):
        env = SnakeCnnEnv(
            seed=7,
            board_size=6,
            silent_mode=True,
            channel_first=channel_first,
        )
        try:
            observation, _info = env.reset(seed=7)
            assert observation.shape == expected_shape
            assert env.observation_space.shape == expected_shape
            assert env.observation_space.contains(observation)
        finally:
            env.close()


def test_invalid_cnn_board_fails_before_creating_training_outputs(tmp_path):
    save_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    result = subprocess.run(
        _cnn_cli_command(save_dir, log_dir, board_size=10),
        cwd=SNAKE_MAIN,
        env=_subprocess_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "CNN board_size=10 is incompatible with image_size=84" in result.stdout
    assert "Compatible values:" in result.stdout
    assert not save_dir.exists()
    assert not log_dir.exists()


def test_help_bypasses_training_preflight_and_creates_no_outputs(tmp_path):
    save_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    command = _cnn_cli_command(save_dir, log_dir, board_size=10)
    command.append("--help")
    result = subprocess.run(
        command,
        cwd=SNAKE_MAIN,
        env=_subprocess_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert "Train a MaskablePPO Snake agent" in result.stdout
    assert "--cnn-channel-first" in result.stdout
    assert "--no-cnn-channel-first" in result.stdout
    assert not save_dir.exists()
    assert not log_dir.exists()


@pytest.mark.parametrize("channel_first", [True, False], ids=["chw", "hwc"])
def test_real_minimal_cnn_cli_saves_only_unverified_candidate(
    tmp_path,
    channel_first,
):
    save_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    result = subprocess.run(
        _cnn_cli_command(
            save_dir,
            log_dir,
            board_size=6,
            channel_first=channel_first,
        ),
        cwd=SNAKE_MAIN,
        env=_subprocess_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    report = json.loads(
        (save_dir / "training_guard_report.json").read_text(encoding="utf-8")
    )
    assert report["agent"] == "cnn"
    assert report["board_size"] == 6
    assert report["cnn_channel_first"] is channel_first
    assert report["decision"]["verified"] is False
    assert report["decision"]["reason"] == "insufficient_training_evidence"
    assert report["artifact_status"] == "unverified_candidate"
    assert not (save_dir / "ppo_snake_final.zip").exists()

    candidate_path = save_dir / "ppo_snake_candidate_unverified.zip"
    assert candidate_path.exists()
    with zipfile.ZipFile(candidate_path) as candidate_bundle:
        embedded = json.loads(candidate_bundle.read("training_guard.json"))
    assert embedded["agent"] == "cnn"
    assert embedded["decision"]["verified"] is False
    assert embedded["artifact_status"] == "unverified_candidate"
    assert list(save_dir.glob("ppo_snake_cnn_candidate_unverified_*_steps.zip"))
