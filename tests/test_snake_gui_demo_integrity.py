import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAKE_MAIN = REPO_ROOT / "snake-ai" / "main"
sys.path.insert(0, str(SNAKE_MAIN))

import gui_train_demo  # noqa: E402


class ZipWritingModel:
    def save(self, path):
        with zipfile.ZipFile(path, mode="w") as bundle:
            bundle.writestr("fake-policy", b"candidate weights")


def _args(save_dir, *, agent="mlp"):
    return SimpleNamespace(
        save_dir=str(save_dir),
        agent=agent,
        board_size=12,
        total_timesteps=4096,
        chunk_timesteps=512,
    )


def test_gui_demo_saves_only_explicitly_unverified_candidate_with_evidence(tmp_path):
    candidate_path, report_path, report = gui_train_demo.save_unverified_candidate(
        ZipWritingModel(),
        _args(tmp_path),
        trained_steps=2048,
    )

    assert candidate_path.name == "ppo_snake_mlp_gui_demo_candidate_unverified.zip"
    assert report_path.name == (
        "ppo_snake_mlp_gui_demo_candidate_unverified.guard.json"
    )
    assert report["artifact_status"] == "unverified_candidate"
    assert report["decision"] == {
        "accepted": False,
        "verified": False,
        "reason": "gui_demo_has_no_promotion_evaluation",
    }
    assert report["promotion_evaluation"] is None
    assert report["attempted_timesteps"] == 2048
    assert report["termination_reason"] == "completed_requested_budget"

    sidecar = json.loads(report_path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(candidate_path) as bundle:
        embedded = json.loads(bundle.read("training_guard.json"))
    assert embedded == sidecar == report

    assert not (tmp_path / "ppo_snake_final.zip").exists()
    assert not list(tmp_path.glob("*.snakeai.zip"))
    assert set(path.name for path in tmp_path.iterdir()) == {
        candidate_path.name,
        report_path.name,
    }


def test_gui_demo_labels_help_and_saved_path_as_unverified(tmp_path):
    help_text = " ".join(gui_train_demo.build_arg_parser().format_help().split())
    candidate_path, report_path = gui_train_demo.unverified_candidate_paths(
        tmp_path,
        "cnn",
    )

    assert "unverified" in help_text.lower()
    assert "never produces an official/final model" in help_text.lower()
    assert "UNVERIFIED CANDIDATE" in gui_train_demo.GUI_DEMO_WARNING
    assert candidate_path.name.endswith("_candidate_unverified.zip")
    assert report_path.name.endswith("_candidate_unverified.guard.json")
    assert "final" not in candidate_path.name
    assert "protected" not in candidate_path.name


class OvershootingModel:
    def __init__(self, start=100):
        self.num_timesteps = start
        self.learn_calls = []

    def learn(self, *, total_timesteps, reset_num_timesteps, progress_bar):
        self.learn_calls.append(
            (total_timesteps, reset_num_timesteps, progress_bar)
        )
        # Simulate SB3 completing a full rollout beyond the requested chunk.
        self.num_timesteps += int(total_timesteps) + 37


def test_gui_loop_counts_actual_model_timestep_delta(monkeypatch, tmp_path):
    args = _args(tmp_path)
    args.total_timesteps = 512
    model = OvershootingModel()
    previews = []
    monkeypatch.setattr(
        gui_train_demo,
        "preview_policy",
        lambda _model, _env_cls, _args, trained_steps: previews.append(trained_steps),
    )

    trained_steps, reason = gui_train_demo.train_preview_loop(model, object, args)

    assert trained_steps == 549
    assert reason == "completed_requested_budget"
    assert previews == [549]
    assert model.learn_calls == [(512, False, False)]


class InterruptingZipModel(ZipWritingModel):
    def __init__(self):
        self.num_timesteps = 0

    def learn(self, *, total_timesteps, reset_num_timesteps, progress_bar):
        self.num_timesteps += 19
        raise KeyboardInterrupt


class FakeVecEnv:
    instances = []

    def __init__(self, _factories):
        self.closed = False
        self.__class__.instances.append(self)

    def close(self):
        self.closed = True


def test_ctrl_c_is_normal_stop_and_saves_actual_partial_candidate(
    monkeypatch,
    tmp_path,
):
    model = InterruptingZipModel()
    FakeVecEnv.instances.clear()
    monkeypatch.setattr(gui_train_demo, "DummyVecEnv", FakeVecEnv)
    monkeypatch.setattr(gui_train_demo, "MaskablePPO", lambda *_args, **_kwargs: model)

    gui_train_demo.main(
        [
            "--agent",
            "mlp",
            "--total-timesteps",
            "100",
            "--chunk-timesteps",
            "50",
            "--save-dir",
            str(tmp_path),
            "--device",
            "cpu",
        ]
    )

    candidate_path, report_path = gui_train_demo.unverified_candidate_paths(
        tmp_path,
        "mlp",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert candidate_path.exists()
    assert report["attempted_timesteps"] == 19
    assert report["termination_reason"] == "keyboard_interrupt"
    assert report["artifact_status"] == "unverified_candidate"
    assert report["decision"]["verified"] is False
    assert FakeVecEnv.instances[-1].closed is True
    assert not (tmp_path / "ppo_snake_final.zip").exists()


def test_gui_loop_does_not_swallow_non_interrupt_errors(monkeypatch, tmp_path):
    args = _args(tmp_path)
    model = OvershootingModel()

    def fail_learn(**_kwargs):
        raise RuntimeError("trainer failed")

    model.learn = fail_learn
    monkeypatch.setattr(gui_train_demo, "preview_policy", lambda *_args: None)

    with pytest.raises(RuntimeError, match="trainer failed"):
        gui_train_demo.train_preview_loop(model, object, args)


def test_gui_main_propagates_failures_without_writing_candidate(
    monkeypatch,
    tmp_path,
):
    class FailingModel(ZipWritingModel):
        num_timesteps = 0

        def learn(self, **_kwargs):
            raise RuntimeError("optimizer failed")

    FakeVecEnv.instances.clear()
    monkeypatch.setattr(gui_train_demo, "DummyVecEnv", FakeVecEnv)
    monkeypatch.setattr(
        gui_train_demo,
        "MaskablePPO",
        lambda *_args, **_kwargs: FailingModel(),
    )

    with pytest.raises(RuntimeError, match="optimizer failed"):
        gui_train_demo.main(
            [
                "--agent",
                "mlp",
                "--total-timesteps",
                "100",
                "--chunk-timesteps",
                "50",
                "--save-dir",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )

    assert FakeVecEnv.instances[-1].closed is True
    assert not list(tmp_path.iterdir())
