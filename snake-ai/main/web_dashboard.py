import io
import json
import random
import tempfile
import threading
import time
import zipfile
import os
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import torch
from flask import Flask, jsonify, request, send_file, send_from_directory
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from cnn_features import ConfigurableCNN
from snake_env import SnakeCnnEnv, SnakeMlpEnv
from train import select_device

sys.modules.setdefault("gym", gym)
sys.modules.setdefault("gym.spaces", gym.spaces)


ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
MAIN_DIR = ROOT_DIR / "main"
ORIGINAL_MODEL_DIR = MAIN_DIR / "original_models"
FULLBOARD_CNN_MODEL = MAIN_DIR / "trained_models_cnn_oracle_bc" / "ppo_snake_bc_final_12x12.zip"
RUNTIME_DIR = ROOT_DIR / "runtime"
BEST_MODEL_PATH = RUNTIME_DIR / "snake_policy.best.zip"
MAX_MODEL_UPLOAD_BYTES = int(os.environ.get("SNAKE_MAX_MODEL_UPLOAD_BYTES", str(128 * 1024 * 1024)))
MODEL_UPLOAD_ENABLED = os.environ.get("SNAKE_ENABLE_MODEL_UPLOAD", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
THREAD_JOIN_TIMEOUT_SECONDS = float(os.environ.get("SNAKE_THREAD_JOIN_TIMEOUT_SECONDS", "5"))


DEFAULT_CONFIG = {
    "agent": "cnn",
    "model_profile": "fullboard_12x12",
    "board_size": 12,
    "device": "cpu",
    "seed": 7,
    "learning_rate": 2.5e-4,
    "clip_range": 0.15,
    "gamma": 0.94,
    "ent_coef": 0.0,
    "n_steps": 2048,
    "batch_size": 512,
    "num_envs": 32,
    "n_epochs": 4,
    "chunk_timesteps": 65536,
    "preview_steps": 1200,
    "strategy": "model",
    "complete_episode_preview": True,
    "deterministic_preview": True,
    "training_enabled": False,
    "guard_enabled": True,
    "guard_eval_episodes": 8,
    "guard_eval_steps": 600,
    "guard_min_delta": 0.0,
    "food_time_penalty": 0.0,
    "food_step_limit_multiplier": 4.0,
    "food_reward_bonus": 0.0,
    "distance_reward_scale": 0.1,
    "loop_penalty": 0.0,
    "loop_window": 16,
    "oscillation_penalty": 0.0,
    "oscillation_window": 12,
    "cnn_channels": "32,64,64",
    "cnn_kernel_sizes": "8,4,3",
    "cnn_strides": "4,2,1",
    "cnn_features_dim": 512,
    "cnn_channel_first": True,
}


def default_original_model_path(agent, device):
    candidates = []
    if agent == "cnn" and device == "mps":
        candidates.append(ORIGINAL_MODEL_DIR / "trained_models_cnn_mps" / "ppo_snake_final.zip")
        candidates.append(MAIN_DIR / "trained_models_cnn_mps" / "ppo_snake_final.zip")
    else:
        candidates.append(ORIGINAL_MODEL_DIR / f"trained_models_{agent}" / "ppo_snake_final.zip")
        candidates.append(MAIN_DIR / f"trained_models_{agent}" / "ppo_snake_final.zip")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class TrainingDashboard:
    def __init__(self):
        self.lock = threading.RLock()
        self.config = dict(DEFAULT_CONFIG)
        self.model_io_lock = threading.RLock()
        self.model = None
        self.train_env = None
        self.thread = None
        self.running = False
        self.stop_requested = False
        self.trained_steps = 0
        self.iteration = 0
        self.frames = []
        self.frame_version = 0
        self.history = []
        self.best_score = 0
        self.best_score_steps = 0
        self.best_score_trained_steps = 0
        self.best_score_iteration = 0
        self.best_guard_objective = float("-inf")
        self.last_guard = {}
        self.actual_device = None
        self.last_error = None
        self.last_event = "idle"

    def _stop_active_thread(self):
        with self.lock:
            self.running = False
            self.stop_requested = True
            active_thread = self.thread
        if active_thread is not None and active_thread.is_alive():
            active_thread.join(timeout=THREAD_JOIN_TIMEOUT_SECONDS)
            if active_thread.is_alive():
                with self.lock:
                    self.stop_requested = False
                    self.last_event = "waiting for training chunk to finish"
                    self.last_error = (
                        "Training is still finishing its current chunk. "
                        "Pause and retry after the current chunk completes."
                    )
                raise RuntimeError(self.last_error)

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "has_model": self.model is not None,
                "trained_steps": self.trained_steps,
                "iteration": self.iteration,
                "frame_version": self.frame_version,
                "frames": self.frames,
                "history": self.history[-80:],
                "guard": dict(self.last_guard),
                "best": {
                    "score": self.best_score,
                    "steps": self.best_score_steps,
                    "trained_steps": self.best_score_trained_steps,
                    "iteration": self.best_score_iteration,
                    "guard_objective": None if self.best_guard_objective == float("-inf") else self.best_guard_objective,
                    "model_path": str(BEST_MODEL_PATH),
                },
                "architecture": {
                    "cnn": self._cnn_architecture_summary(),
                },
                "device_info": self._device_info(),
                "actual_device": self.actual_device,
                "config": dict(self.config),
                "last_error": self.last_error,
                "last_event": self.last_event,
            }

    def start(self):
        with self.lock:
            self.running = True
            self.stop_requested = False
            self.last_event = "starting"
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._loop, daemon=True)
                self.thread.start()

    def pause(self):
        with self.lock:
            self.running = False
            self.last_event = "paused"

    def reset(self, updates=None):
        self._stop_active_thread()

        with self.lock:
            if updates:
                self._merge_config(updates)
            if self.train_env is not None:
                self.train_env.close()
            self.model = None
            self.train_env = None
            self.thread = None
            self.stop_requested = False
            self.trained_steps = 0
            self.iteration = 0
            self.frames = []
            self.frame_version += 1
            self.history = []
            self.best_score = 0
            self.best_score_steps = 0
            self.best_score_trained_steps = 0
            self.best_score_iteration = 0
            self.best_guard_objective = float("-inf")
            self.last_guard = {}
            self.actual_device = None
            self.last_error = None
            self.last_event = "reset"

    def export_model_bundle(self):
        with self.model_io_lock:
            with self.lock:
                if self.model is None:
                    raise ValueError("No model has been created yet. Start training before downloading.")
                metadata = {
                    "format": "snake-ai-dashboard-bundle",
                    "format_version": 1,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "config": dict(self.config),
                    "trained_steps": self.trained_steps,
                    "iteration": self.iteration,
                    "history": self.history[-200:],
                    "best": {
                        "score": self.best_score,
                        "steps": self.best_score_steps,
                        "trained_steps": self.best_score_trained_steps,
                        "iteration": self.best_score_iteration,
                    },
                    "last_event": self.last_event,
                }

            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "model.zip"
                self.model.save(model_path)
                payload = io.BytesIO()
                with zipfile.ZipFile(payload, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
                    bundle.write(model_path, "model.zip")
                    bundle.writestr("metadata.json", json.dumps(metadata, indent=2))
                payload.seek(0)

        filename = f"snake-ai-{metadata['config']['agent']}-{metadata['trained_steps']}-steps.snakeai.zip"
        return payload, filename

    def import_model_bundle(self, uploaded_file):
        if not MODEL_UPLOAD_ENABLED:
            raise PermissionError(
                "Model upload is disabled. Set SNAKE_ENABLE_MODEL_UPLOAD=1 only for trusted local use."
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            upload_path = Path(tmpdir) / "upload.zip"
            uploaded_file.save(upload_path)
            if upload_path.stat().st_size > MAX_MODEL_UPLOAD_BYTES:
                raise ValueError("Uploaded model bundle is too large.")
            model_path = Path(tmpdir) / "model.zip"
            metadata = {}

            with zipfile.ZipFile(upload_path) as uploaded_zip:
                names = set(uploaded_zip.namelist())
                if {"model.zip", "metadata.json"}.issubset(names):
                    model_info = uploaded_zip.getinfo("model.zip")
                    metadata_info = uploaded_zip.getinfo("metadata.json")
                    if (
                        model_info.file_size > MAX_MODEL_UPLOAD_BYTES
                        or metadata_info.file_size > 1024 * 1024
                    ):
                        raise ValueError("Uploaded model bundle contents are too large.")
                    uploaded_zip.extract("model.zip", tmpdir)
                    metadata = json.loads(uploaded_zip.read("metadata.json").decode("utf-8"))
                else:
                    model_path.write_bytes(upload_path.read_bytes())

            self._replace_model_from_file(model_path, metadata)

        return self.snapshot()

    def move_to_device(self, device):
        with self.lock:
            if self.model is None:
                raise ValueError("No model has been created yet.")
        self._stop_active_thread()

        with self.model_io_lock:
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "model.zip"
                self.model.save(model_path)

                with self.lock:
                    self._merge_config({"device": device})
                    if self.train_env is not None:
                        self.train_env.close()
                    self.train_env = self._make_train_env()
                    resolved_device = select_device(self.config["device"])

                model = self._load_model(model_path, self.train_env, resolved_device)

            with self.lock:
                self.model = model
                self._apply_live_config()
                self.thread = None
                self.running = False
                self.stop_requested = False
                self.actual_device = str(self.model.device)
                self.last_error = None
                self.last_event = f"moved model to {self.actual_device}"

        return self.snapshot()

    def _replace_model_from_file(self, model_path, metadata):
        self._stop_active_thread()

        with self.model_io_lock:
            with self.lock:
                if self.train_env is not None:
                    self.train_env.close()
                bundle_config = metadata.get("config") if isinstance(metadata, dict) else None
                if isinstance(bundle_config, dict):
                    self._merge_config(bundle_config)
                self.train_env = self._make_train_env()
                device = select_device(self.config["device"])

            model = self._load_model(model_path, self.train_env, device)

            with self.lock:
                self.model = model
                self._apply_live_config()
                self.thread = None
                self.running = False
                self.stop_requested = False
                self.trained_steps = int(metadata.get("trained_steps") or self.model.num_timesteps or 0)
                self.iteration = int(metadata.get("iteration") or 0)
                self.history = list(metadata.get("history") or [])
                self._restore_best(metadata)
                self.frames = []
                self.frame_version += 1
                self.last_error = None
                self.actual_device = str(self.model.device)
                self.last_event = f"imported model at {self.trained_steps} steps"

    def _restore_best(self, metadata):
        best = metadata.get("best") if isinstance(metadata, dict) else None
        if isinstance(best, dict):
            self.best_score = int(best.get("score") or 0)
            self.best_score_steps = int(best.get("steps") or 0)
            self.best_score_trained_steps = int(best.get("trained_steps") or 0)
            self.best_score_iteration = int(best.get("iteration") or 0)
            return

        self.best_score = 0
        self.best_score_steps = 0
        self.best_score_trained_steps = 0
        self.best_score_iteration = 0
        for item in self.history:
            self._update_best(item)

    def _update_best(self, summary):
        score = int(summary.get("preview_score") or 0)
        steps = int(summary.get("preview_steps") or 0)
        if score < self.best_score:
            return
        if score == self.best_score and self.best_score_steps and steps >= self.best_score_steps:
            return
        self.best_score = score
        self.best_score_steps = steps
        self.best_score_trained_steps = int(summary.get("trained_steps") or self.trained_steps)
        self.best_score_iteration = int(summary.get("iteration") or self.iteration)

    def _load_model(self, model_path, env, device):
        return MaskablePPO.load(
            model_path,
            env=env,
            device=device,
            custom_objects={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
            },
        )

    def update_config(self, updates):
        with self.lock:
            self._merge_config(updates)
            self._apply_live_config()
            self.last_event = "settings updated"
            return dict(self.config)

    def _merge_config(self, updates):
        for key, value in updates.items():
            if key not in self.config:
                continue
            if key in {"agent", "device", "strategy", "model_profile"}:
                self.config[key] = str(value)
            elif key in {"cnn_channels", "cnn_kernel_sizes", "cnn_strides"}:
                self.config[key] = self._normalize_int_list(value)
            elif key in {"deterministic_preview", "complete_episode_preview", "cnn_channel_first", "training_enabled", "guard_enabled"}:
                self.config[key] = bool(value)
            elif key in {"board_size", "seed", "n_steps", "batch_size", "num_envs", "n_epochs", "chunk_timesteps", "preview_steps", "loop_window", "oscillation_window", "cnn_features_dim", "guard_eval_episodes", "guard_eval_steps"}:
                self.config[key] = max(1, int(value))
            else:
                self.config[key] = float(value)

        self.config["agent"] = "cnn" if self.config["agent"] == "cnn" else "mlp"
        if self.config["strategy"] not in {"model", "hamiltonian"}:
            self.config["strategy"] = "model"
        if self.config["model_profile"] not in {"custom", "repo_original", "fullboard_12x12", "new"}:
            self.config["model_profile"] = "custom"
        if self.config["device"] not in {"auto", "cpu", "cuda", "mps"}:
            self.config["device"] = "cpu"
        self.config["board_size"] = max(4, self.config["board_size"])
        self.config["chunk_timesteps"] = max(16, self.config["chunk_timesteps"])
        self.config["preview_steps"] = max(10, self.config["preview_steps"])
        self.config["guard_eval_episodes"] = max(2, min(64, self.config["guard_eval_episodes"]))
        self.config["guard_eval_steps"] = max(50, min(5000, self.config["guard_eval_steps"]))
        self.config["guard_min_delta"] = max(-1000.0, min(1000.0, self.config["guard_min_delta"]))
        self.config["batch_size"] = max(8, self.config["batch_size"])
        self.config["n_steps"] = max(8, self.config["n_steps"])
        self.config["num_envs"] = max(1, self.config["num_envs"])
        self.config["loop_window"] = max(2, self.config["loop_window"])
        self.config["oscillation_window"] = max(4, self.config["oscillation_window"])
        self.config["food_time_penalty"] = max(0.0, self.config["food_time_penalty"])
        self.config["food_step_limit_multiplier"] = max(0.25, self.config["food_step_limit_multiplier"])
        self.config["food_reward_bonus"] = max(0.0, self.config["food_reward_bonus"])
        self.config["distance_reward_scale"] = max(0.0, self.config["distance_reward_scale"])
        self.config["loop_penalty"] = max(0.0, self.config["loop_penalty"])
        self.config["oscillation_penalty"] = max(0.0, self.config["oscillation_penalty"])
        self.config["cnn_features_dim"] = max(16, int(self.config["cnn_features_dim"]))
        self._validate_cnn_config()

    def _normalize_int_list(self, value):
        if isinstance(value, (list, tuple)):
            values = [int(item) for item in value]
        else:
            values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
        if not values:
            raise ValueError("CNN layer list cannot be empty")
        if any(item <= 0 for item in values):
            raise ValueError("CNN layer values must be positive")
        return ",".join(str(item) for item in values)

    def _parse_int_list(self, key):
        return tuple(int(item.strip()) for item in str(self.config[key]).split(",") if item.strip())

    def _validate_cnn_config(self):
        channels = self._parse_int_list("cnn_channels")
        kernels = self._parse_int_list("cnn_kernel_sizes")
        strides = self._parse_int_list("cnn_strides")
        if not (len(channels) == len(kernels) == len(strides)):
            raise ValueError("CNN channels, kernel sizes, and strides must have the same number of values")

    def _cnn_policy_kwargs(self):
        return {
            "features_extractor_class": ConfigurableCNN,
            "features_extractor_kwargs": {
                "channels": self._parse_int_list("cnn_channels"),
                "kernel_sizes": self._parse_int_list("cnn_kernel_sizes"),
                "strides": self._parse_int_list("cnn_strides"),
                "features_dim": int(self.config["cnn_features_dim"]),
            },
        }

    def _uses_default_cnn_architecture(self):
        return (
            self.config["agent"] == "cnn"
            and self.config["board_size"] == 12
            and self.config["cnn_channel_first"]
            and self._parse_int_list("cnn_channels") == (32, 64, 64)
            and self._parse_int_list("cnn_kernel_sizes") == (8, 4, 3)
            and self._parse_int_list("cnn_strides") == (4, 2, 1)
            and int(self.config["cnn_features_dim"]) == 512
        )

    def _cnn_architecture_summary(self):
        channels = self._parse_int_list("cnn_channels")
        kernels = self._parse_int_list("cnn_kernel_sizes")
        strides = self._parse_int_list("cnn_strides")
        layers = []
        in_channels = 3
        for index, (out_channels, kernel_size, stride) in enumerate(zip(channels, kernels, strides), start=1):
            layers.append(
                {
                    "index": index,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "kernel_size": kernel_size,
                    "stride": stride,
                }
            )
            in_channels = out_channels
        return {
            "layers": layers,
            "features_dim": int(self.config["cnn_features_dim"]),
        }

    def _device_info(self):
        cuda_available = bool(torch.cuda.is_available())
        cuda_devices = int(torch.cuda.device_count()) if cuda_available else 0
        cuda_name = torch.cuda.get_device_name(0) if cuda_available and cuda_devices else None
        mps_available = bool(torch.backends.mps.is_available())
        return {
            "cuda_available": cuda_available,
            "cuda_devices": cuda_devices,
            "cuda_name": cuda_name,
            "mps_available": mps_available,
        }

    def _env_class(self):
        return SnakeCnnEnv if self.config["agent"] == "cnn" else SnakeMlpEnv

    def _policy(self):
        return "CnnPolicy" if self.config["agent"] == "cnn" else "MlpPolicy"

    def _hamiltonian_cycle(self, board_size):
        if board_size % 2 != 0:
            raise ValueError("Hamiltonian strategy requires an even board size.")

        path = [(0, col) for col in range(board_size)]
        for row in range(1, board_size):
            cols = range(board_size - 1, 0, -1) if row % 2 else range(1, board_size)
            for col in cols:
                path.append((row, col))
        for row in range(board_size - 1, 0, -1):
            path.append((row, 0))
        return path

    def _hamiltonian_action(self, env):
        cycle = self._hamiltonian_cycle(env.board_size)
        cycle_index = {cell: index for index, cell in enumerate(cycle)}
        head = env.game.snake[0]
        next_position = cycle[(cycle_index[head] + 1) % len(cycle)]
        row_diff = next_position[0] - head[0]
        col_diff = next_position[1] - head[1]
        if row_diff == -1 and col_diff == 0:
            return 0
        if row_diff == 0 and col_diff == -1:
            return 1
        if row_diff == 0 and col_diff == 1:
            return 2
        if row_diff == 1 and col_diff == 0:
            return 3
        raise ValueError(f"Invalid Hamiltonian edge: {head} -> {next_position}")

    def _initial_model_path(self, device):
        profile = self.config["model_profile"]
        if profile in {"new", "custom"}:
            return None
        if profile == "fullboard_12x12":
            if self.config["agent"] == "cnn" and self._uses_default_cnn_architecture() and FULLBOARD_CNN_MODEL.exists():
                return FULLBOARD_CNN_MODEL
            return None
        if profile == "repo_original":
            if self.config["agent"] == "cnn" and not self._uses_default_cnn_architecture():
                return None
            if self.config["agent"] == "mlp" and int(self.config["board_size"]) != 12:
                return None
            return default_original_model_path(self.config["agent"], device)
        return None

    def _make_train_env(self):
        env_cls = self._env_class()
        base_seed = self.config["seed"]
        board_size = self.config["board_size"]

        def _init(rank):
            seed = base_seed + rank * 1009
            env_kwargs = {
                "seed": seed,
                "board_size": board_size,
                "silent_mode": True,
                "limit_step": True,
                "food_time_penalty": self.config["food_time_penalty"],
                "food_step_limit_multiplier": self.config["food_step_limit_multiplier"],
                "food_reward_bonus": self.config["food_reward_bonus"],
                "distance_reward_scale": self.config["distance_reward_scale"],
                "loop_penalty": self.config["loop_penalty"],
                "loop_window": self.config["loop_window"],
                "oscillation_penalty": self.config["oscillation_penalty"],
                "oscillation_window": self.config["oscillation_window"],
            }
            if env_cls is SnakeCnnEnv:
                env_kwargs["channel_first"] = self.config["cnn_channel_first"]
            env = env_cls(**env_kwargs)
            env = Monitor(env)
            env = ActionMasker(env, lambda wrapped_env: wrapped_env.unwrapped.get_action_mask())
            env.reset(seed=seed)
            return env

        return DummyVecEnv([lambda rank=rank: _init(rank) for rank in range(self.config["num_envs"])])

    def _make_eval_env(self, seed):
        env_cls = self._env_class()
        env_kwargs = {
            "seed": seed,
            "board_size": self.config["board_size"],
            "silent_mode": True,
            "limit_step": True,
            "food_time_penalty": self.config["food_time_penalty"],
            "food_step_limit_multiplier": self.config["food_step_limit_multiplier"],
            "food_reward_bonus": self.config["food_reward_bonus"],
            "distance_reward_scale": self.config["distance_reward_scale"],
            "loop_penalty": self.config["loop_penalty"],
            "loop_window": self.config["loop_window"],
            "oscillation_penalty": self.config["oscillation_penalty"],
            "oscillation_window": self.config["oscillation_window"],
        }
        if env_cls is SnakeCnnEnv:
            env_kwargs["channel_first"] = self.config["cnn_channel_first"]
        return env_cls(**env_kwargs)

    def _guard_objective(self, metrics):
        return (
            float(metrics.get("avg_score", 0.0))
            + float(metrics.get("avg_food", 0.0)) * 0.5
            + float(metrics.get("avg_reward", 0.0)) * 0.05
        )

    def _evaluate_model_score(self, model, *, seed_base, episodes, max_steps):
        scores = []
        foods = []
        rewards = []
        for index in range(int(episodes)):
            seed = seed_base + index
            env = self._make_eval_env(seed)
            try:
                obs, _info = env.reset(seed=seed)
                done = False
                total_reward = 0.0
                food_count = 0
                steps = 0
                while not done and steps < max_steps:
                    action, _state = model.predict(
                        obs,
                        deterministic=True,
                        action_masks=env.get_action_mask(),
                    )
                    obs, reward, terminated, truncated, info = env.step(int(action))
                    done = bool(terminated or truncated)
                    total_reward += float(reward)
                    food_count += int(bool(info.get("food_obtained")))
                    steps += 1
                scores.append(len(env.game.snake) - env.init_snake_size)
                foods.append(food_count)
                rewards.append(total_reward)
            finally:
                env.close()
        count = max(1, len(scores))
        metrics = {
            "episodes": count,
            "avg_score": round(sum(scores) / count, 4),
            "avg_food": round(sum(foods) / count, 4),
            "avg_reward": round(sum(rewards) / count, 5),
        }
        metrics["objective"] = round(self._guard_objective(metrics), 5)
        return metrics

    def _ensure_model(self):
        if self.model is not None:
            return

        seed = self.config["seed"]
        random.seed(seed)
        torch.manual_seed(seed)
        self.train_env = self._make_train_env()
        device = select_device(self.config["device"])
        original_model_path = self._initial_model_path(device)
        if original_model_path is not None:
            self.model = self._load_model(original_model_path, self.train_env, device)
            if self.trained_steps == 0:
                self.trained_steps = int(self.model.num_timesteps or 0)
            source_event = f"loaded original model from {original_model_path.relative_to(MAIN_DIR)}"
        else:
            self.model = MaskablePPO(
                self._policy(),
                self.train_env,
                device=device,
                verbose=0,
                n_steps=self.config["n_steps"],
                batch_size=self.config["batch_size"],
                n_epochs=self.config["n_epochs"],
                gamma=self.config["gamma"],
                learning_rate=self.config["learning_rate"],
                clip_range=self.config["clip_range"],
                ent_coef=self.config["ent_coef"],
                policy_kwargs=self._cnn_policy_kwargs() if self.config["agent"] == "cnn" else None,
            )
            source_event = "created new model"
        self._apply_live_config()
        self.actual_device = str(self.model.device)
        if self.actual_device != str(device):
            self.last_event = f"{source_event}; ready on {self.actual_device} (requested {device})"
        else:
            self.last_event = f"{source_event}; ready on {self.actual_device}"

    def _apply_live_config(self):
        self._apply_env_config()
        if self.model is None:
            return
        lr = float(self.config["learning_rate"])
        clip = float(self.config["clip_range"])
        self.model.learning_rate = lr
        self.model.lr_schedule = lambda _: lr
        self.model.clip_range = lambda _: clip
        self.model.gamma = float(self.config["gamma"])
        self.model.ent_coef = float(self.config["ent_coef"])
        self.model.n_epochs = int(self.config["n_epochs"])
        for group in self.model.policy.optimizer.param_groups:
            group["lr"] = lr

    def _apply_env_config(self):
        if self.train_env is None:
            return
        loop_window = max(2, int(self.config["loop_window"]))
        oscillation_window = max(4, int(self.config["oscillation_window"]))
        for wrapped_env in getattr(self.train_env, "envs", []):
            env = wrapped_env.unwrapped
            env.food_time_penalty = float(self.config["food_time_penalty"])
            env.food_step_limit_multiplier = float(self.config["food_step_limit_multiplier"])
            env.step_limit = env._make_step_limit(True)
            env.food_reward_bonus = float(self.config["food_reward_bonus"])
            env.distance_reward_scale = float(self.config["distance_reward_scale"])
            env.loop_penalty = float(self.config["loop_penalty"])
            env.oscillation_penalty = float(self.config["oscillation_penalty"])
            env.oscillation_window = oscillation_window
            if getattr(env, "loop_window", loop_window) != loop_window:
                positions = list(getattr(env, "recent_head_positions", []))[-loop_window:]
                env.loop_window = loop_window
                env.recent_head_positions = deque(positions, maxlen=loop_window)

    def _loop(self):
        while True:
            with self.lock:
                if self.stop_requested:
                    return
                is_running = self.running
            if not is_running:
                time.sleep(0.1)
                continue

            try:
                with self.lock:
                    training_enabled = bool(self.config["training_enabled"])
                    strategy = self.config["strategy"]
                    if training_enabled or strategy == "model":
                        self._ensure_model()
                    self._apply_live_config()
                    chunk = max(
                        int(self.config["chunk_timesteps"]),
                        int(self.config["n_steps"]) * int(self.config["num_envs"]),
                    )

                start = time.time()
                start_steps = int(self.model.num_timesteps) if self.model is not None else self.trained_steps
                guard = {}
                if training_enabled:
                    with self.model_io_lock:
                        guard_enabled = bool(self.config["guard_enabled"])
                        guard_seed = int(self.config["seed"]) + self.iteration * 10_007 + 50_000
                        guard_episodes = int(self.config["guard_eval_episodes"])
                        guard_steps = int(self.config["guard_eval_steps"])
                        min_delta = float(self.config["guard_min_delta"])
                        with tempfile.TemporaryDirectory() as tmpdir:
                            checkpoint = Path(tmpdir) / "model_before.zip"
                            baseline = {}
                            if guard_enabled:
                                self.model.save(checkpoint)
                                baseline = self._evaluate_model_score(
                                    self.model,
                                    seed_base=guard_seed,
                                    episodes=guard_episodes,
                                    max_steps=guard_steps,
                                )
                            self.model.learn(
                                total_timesteps=chunk,
                                reset_num_timesteps=False,
                                progress_bar=False,
                            )
                            candidate = {}
                            accepted = True
                            if guard_enabled:
                                candidate = self._evaluate_model_score(
                                    self.model,
                                    seed_base=guard_seed,
                                    episodes=guard_episodes,
                                    max_steps=guard_steps,
                                )
                                accepted = candidate["objective"] >= baseline["objective"] + min_delta
                                if not accepted:
                                    self.model = self._load_model(checkpoint, self.train_env, self.model.device)
                                elif candidate["objective"] >= self.best_guard_objective:
                                    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
                                    self.model.save(BEST_MODEL_PATH)
                                    self.best_guard_objective = candidate["objective"]
                                guard = {
                                    "accepted": accepted,
                                    "episodes": guard_episodes,
                                    "max_steps": guard_steps,
                                    "min_delta": min_delta,
                                    "baseline": baseline,
                                    "candidate": candidate,
                                }
                elapsed = max(0.001, time.time() - start)
                end_steps = int(self.model.num_timesteps) if self.model is not None else start_steps
                trained_delta = max(0, end_steps - start_steps)

                with self.lock:
                    self.trained_steps += trained_delta
                    if guard:
                        self.last_guard = guard
                    self.iteration += 1
                    preview_steps = int(self.config["preview_steps"])
                    deterministic = bool(self.config["deterministic_preview"])
                    complete_episode = bool(self.config["complete_episode_preview"])
                    strategy = self.config["strategy"]

                frames, summary = self._preview(preview_steps, deterministic, complete_episode, strategy)

                with self.lock:
                    summary.update(
                        {
                            "iteration": self.iteration,
                            "trained_steps": self.trained_steps,
                            "chunk_timesteps": trained_delta,
                            "target_timesteps": chunk,
                            "num_envs": self.config["num_envs"],
                            "train_seconds": round(elapsed, 3),
                            "fps": round(trained_delta / elapsed, 2),
                            "time": time.strftime("%H:%M:%S"),
                        }
                    )
                    self.frames = frames
                    self.frame_version += 1
                    self.history.append(summary)
                    self._update_best(summary)
                    self.last_error = None
                    if training_enabled:
                        if guard and not guard.get("accepted", True):
                            self.last_event = f"guard rejected candidate at {self.trained_steps} steps"
                        else:
                            self.last_event = f"trained to {self.trained_steps} steps"
                    else:
                        self.running = False
                        self.last_event = "previewed without training"
            except Exception as exc:
                with self.lock:
                    self.running = False
                    self.last_error = repr(exc)
                    self.last_event = "error"
                return

    def _preview(self, max_steps, deterministic, complete_episode, strategy):
        env_cls = self._env_class()
        seed = self.config["seed"] + self.iteration * 1009
        env_kwargs = {
            "seed": seed,
            "board_size": self.config["board_size"],
            "silent_mode": True,
            "limit_step": complete_episode,
            "food_time_penalty": self.config["food_time_penalty"],
            "food_step_limit_multiplier": self.config["food_step_limit_multiplier"],
            "food_reward_bonus": self.config["food_reward_bonus"],
            "distance_reward_scale": self.config["distance_reward_scale"],
            "loop_penalty": self.config["loop_penalty"],
            "loop_window": self.config["loop_window"],
            "oscillation_penalty": self.config["oscillation_penalty"],
            "oscillation_window": self.config["oscillation_window"],
        }
        if env_cls is SnakeCnnEnv:
            env_kwargs["channel_first"] = self.config["cnn_channel_first"]
        env = env_cls(**env_kwargs)
        obs, _ = env.reset(seed=seed)
        frames = []
        max_score = 0
        food_count = 0
        last_food_step = 0
        food_step_counts = []
        loop_revisits = 0
        oscillations = 0
        done = False

        with self.model_io_lock:
            for step in range(max_steps):
                if strategy == "hamiltonian":
                    action = self._hamiltonian_action(env)
                else:
                    action, _ = self.model.predict(
                        obs,
                        action_masks=env.get_action_mask(),
                        deterministic=deterministic,
                    )
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                max_score = max(max_score, env.game.score)
                ate = bool(info["food_obtained"])
                if ate:
                    food_count += 1
                    steps_to_food = step + 1 - last_food_step
                    food_step_counts.append(steps_to_food)
                    last_food_step = step + 1
                if info.get("loop_revisit"):
                    loop_revisits += 1
                if info.get("oscillation"):
                    oscillations += 1
                frames.append(self._frame(env, step, reward, action, done, ate, food_count))
                if done:
                    break

        avg_steps_per_food = (
            sum(food_step_counts) / len(food_step_counts)
            if food_step_counts
            else 0
        )

        summary = {
            "preview_score": env.game.score,
            "preview_max_score": max_score,
            "preview_steps": len(frames),
            "food_count": food_count,
            "snake_size": len(env.game.snake),
            "avg_steps_per_food": round(avg_steps_per_food, 2),
            "loop_revisits": loop_revisits,
            "oscillations": oscillations,
            "done": done,
            "hit_preview_cap": not done and len(frames) >= max_steps,
        }
        env.close()
        return frames, summary

    def _frame(self, env, step, reward, action, done, ate, food_count):
        return {
            "step": step,
            "board_size": env.board_size,
            "snake": [list(cell) for cell in env.game.snake],
            "food": list(env.game.food),
            "score": env.game.score,
            "length": len(env.game.snake),
            "ate": ate,
            "food_count": food_count,
            "reward": float(reward),
            "action": int(action),
            "done": bool(done),
            "steps_since_food": int(getattr(env, "steps_since_food", 0)),
            "loop_revisit_count": int(getattr(env, "loop_revisit_count", 0)),
            "oscillation_count": int(getattr(env, "oscillation_count", 0)),
        }


dashboard = TrainingDashboard()
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")


def json_error(message, status=400):
    return jsonify({"ok": False, "error": str(message)}), status


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/status")
def status():
    return jsonify(dashboard.snapshot())


@app.post("/api/start")
def start():
    dashboard.start()
    return jsonify({"ok": True, **dashboard.snapshot()})


@app.post("/api/pause")
def pause():
    dashboard.pause()
    return jsonify({"ok": True, **dashboard.snapshot()})


@app.post("/api/reset")
def reset():
    try:
        dashboard.reset(request.get_json(silent=True) or {})
    except RuntimeError as exc:
        return json_error(exc, 409)
    except (TypeError, ValueError) as exc:
        return json_error(exc, 400)
    return jsonify({"ok": True, **dashboard.snapshot()})


@app.post("/api/settings")
def settings():
    try:
        config = dashboard.update_config(request.get_json(silent=True) or {})
    except (TypeError, ValueError) as exc:
        return json_error(exc, 400)
    return jsonify({"ok": True, "config": config})


@app.get("/api/model/download")
def download_model():
    try:
        payload, filename = dashboard.export_model_bundle()
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409
    return send_file(
        payload,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename,
    )


@app.post("/api/model/upload")
def upload_model():
    if not MODEL_UPLOAD_ENABLED:
        return json_error(
            "Model upload is disabled. Set SNAKE_ENABLE_MODEL_UPLOAD=1 only for trusted local use.",
            403,
        )
    if request.content_length and request.content_length > MAX_MODEL_UPLOAD_BYTES:
        return json_error("Uploaded model bundle is too large.", 413)
    uploaded_file = request.files.get("model")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"ok": False, "error": "Upload a .snakeai.zip or Stable-Baselines3 .zip model file."}), 400
    try:
        snapshot = dashboard.import_model_bundle(uploaded_file)
    except PermissionError as exc:
        return json_error(exc, 403)
    except RuntimeError as exc:
        return json_error(exc, 409)
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        return jsonify({"ok": False, "error": f"Could not import model: {exc}"}), 400
    return jsonify({"ok": True, **snapshot})


@app.post("/api/model/device")
def move_model_device():
    payload = request.get_json(silent=True) or {}
    try:
        snapshot = dashboard.move_to_device(payload.get("device", dashboard.config["device"]))
    except RuntimeError as exc:
        return json_error(exc, 409)
    except (OSError, ValueError) as exc:
        return jsonify({"ok": False, "error": f"Could not move model device: {exc}"}), 400
    return jsonify({"ok": True, **snapshot})


if __name__ == "__main__":
    host = os.environ.get("SNAKE_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("SNAKE_DASHBOARD_PORT", "7860"))
    app.run(host=host, port=port, threaded=True)
