import io
import json
import math
import random
import tempfile
import threading
import time
import zipfile
import os
import sys
import atexit
import hashlib
import shutil
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
BEST_MODEL_PATH = RUNTIME_DIR / "snake_policy.best.snakeai.zip"
MAX_MODEL_UPLOAD_BYTES = int(os.environ.get("SNAKE_MAX_MODEL_UPLOAD_BYTES", str(128 * 1024 * 1024)))
MODEL_UPLOAD_ENABLED = os.environ.get("SNAKE_ENABLE_MODEL_UPLOAD", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
THREAD_JOIN_TIMEOUT_SECONDS = float(os.environ.get("SNAKE_THREAD_JOIN_TIMEOUT_SECONDS", "5"))
DASHBOARD_BUNDLE_FORMAT = "snake-ai-dashboard-bundle"
DASHBOARD_BUNDLE_FORMAT_VERSION = 2
FIXED_HOLDOUT_KIND = "fixed_holdout_v1"
FIXED_HOLDOUT_PROTOCOL_VERSION = 1
HOLDOUT_EVAL_CONFIG_KEYS = (
    "agent",
    "board_size",
    "cnn_channel_first",
    "food_time_penalty",
    "food_step_limit_multiplier",
    "food_reward_bonus",
    "distance_reward_scale",
    "loop_penalty",
    "loop_window",
    "oscillation_penalty",
    "oscillation_window",
)


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
    "training_enabled": True,
    "guard_enabled": True,
    "guard_eval_episodes": 8,
    "guard_eval_steps": 600,
    "guard_min_delta": 0.001,
    "guard_holdout_episodes": 8,
    "guard_holdout_max_drop": 0.0,
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
        self.guard_benchmark = {}
        self.holdout_protocol = {}
        self.resume_best_bundle = None
        self.last_guard = {}
        self.actual_device = None
        self.last_error = None
        self.last_event = "idle"
        self.startup_notice = None
        self.closed = False
        self._restore_persisted_best_checkpoint()

    def _stop_active_thread(self):
        with self.lock:
            self.running = False
            self.stop_requested = True
            active_thread = self.thread
        if active_thread is not None and active_thread.is_alive():
            active_thread.join(timeout=THREAD_JOIN_TIMEOUT_SECONDS)
            if active_thread.is_alive():
                with self.lock:
                    self.last_event = "waiting for training chunk to finish"
                    self.last_error = (
                        "Training is still finishing its current chunk. "
                        "Pause and retry after the current chunk completes."
                    )
                raise RuntimeError(self.last_error)
        with self.lock:
            if self.thread is active_thread:
                self.thread = None
            self.stop_requested = False

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
                "model_upload_enabled": MODEL_UPLOAD_ENABLED,
            }

    def start(self):
        with self.lock:
            self.closed = False
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

        # Reset participates in the same model-I/O -> state-lock ordering as
        # export/import/training, so a concurrent download cannot observe a
        # half-reset model.
        with self.model_io_lock:
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
                protected_preserved = self._benchmark_matches_model_config(
                    self.guard_benchmark
                ) and BEST_MODEL_PATH.exists()
                if not protected_preserved:
                    self.best_guard_objective = float("-inf")
                    self.guard_benchmark = {}
                    self.holdout_protocol = {}
                    self.resume_best_bundle = None
                else:
                    self.resume_best_bundle = BEST_MODEL_PATH
                self.last_guard = {}
                self.actual_device = None
                self.last_error = None
                self.last_event = (
                    "reset; protected fixed-holdout best preserved"
                    if protected_preserved
                    else "reset; starting a new fixed-holdout benchmark"
                )

    def close(self):
        """Stop the worker and close its VecEnv without racing model I/O."""

        try:
            self._stop_active_thread()
        except RuntimeError:
            # Keep stop_requested set; the daemon worker will exit at the next
            # transaction boundary rather than silently resuming.
            return False
        with self.model_io_lock:
            with self.lock:
                if self.train_env is not None:
                    self.train_env.close()
                self.train_env = None
                self.model = None
                self.thread = None
                self.running = False
                self.closed = True
                self.last_event = "closed"
        return True

    def _model_bundle_metadata(self, *, extra=None):
        metadata = {
            "format": DASHBOARD_BUNDLE_FORMAT,
            "format_version": DASHBOARD_BUNDLE_FORMAT_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": dict(self.config),
            "trained_steps": int(self.model.num_timesteps) if self.model is not None else self.trained_steps,
            "iteration": self.iteration,
            "history": self.history[-200:],
            "best": {
                "score": self.best_score,
                "steps": self.best_score_steps,
                "trained_steps": self.best_score_trained_steps,
                "iteration": self.best_score_iteration,
                "guard_objective": None if self.best_guard_objective == float("-inf") else self.best_guard_objective,
                "guard_objective_kind": FIXED_HOLDOUT_KIND,
            },
            "guard_benchmark": dict(self.guard_benchmark),
            "last_event": self.last_event,
        }
        if extra:
            metadata.update(extra)
        return metadata

    @staticmethod
    def _read_bundle_metadata(bundle_path):
        path = Path(bundle_path)
        if not path.exists():
            return None
        try:
            with zipfile.ZipFile(path) as bundle:
                if "metadata.json" not in bundle.namelist():
                    return None
                info = bundle.getinfo("metadata.json")
                if info.file_size > 1024 * 1024:
                    return None
                metadata = json.loads(bundle.read("metadata.json").decode("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError, zipfile.BadZipFile, json.JSONDecodeError):
            return None
        return metadata if isinstance(metadata, dict) else None

    @staticmethod
    def _validated_guard_benchmark(metadata):
        benchmark = metadata.get("guard_benchmark") if isinstance(metadata, dict) else None
        if not isinstance(benchmark, dict) or benchmark.get("kind") != FIXED_HOLDOUT_KIND:
            return {}
        protocol = benchmark.get("protocol")
        try:
            objective = float(benchmark.get("objective"))
            protocol_version = int(protocol["version"])
            seed_base = int(protocol["seed_base"])
            episodes = int(protocol["episodes"])
            max_steps = int(protocol["max_steps"])
            eval_config = dict(protocol["eval_config"])
            metrics = dict(benchmark["metrics"])
            metrics_objective = float(metrics["objective"])
            metrics_episodes = int(metrics["episodes"])
        except (TypeError, ValueError, KeyError):
            return {}
        if (
            protocol_version != FIXED_HOLDOUT_PROTOCOL_VERSION
            or not math.isfinite(objective)
            or not math.isfinite(metrics_objective)
            or not math.isclose(objective, metrics_objective, rel_tol=0.0, abs_tol=1e-9)
            or seed_base < 0
            or episodes < 8
            or metrics_episodes != episodes
            or max_steps < 1
        ):
            return {}
        required = set(HOLDOUT_EVAL_CONFIG_KEYS)
        if not required.issubset(eval_config):
            return {}
        clean_protocol = {
            "version": FIXED_HOLDOUT_PROTOCOL_VERSION,
            "seed_base": seed_base,
            "episodes": episodes,
            "max_steps": max_steps,
            "eval_config": {key: eval_config[key] for key in HOLDOUT_EVAL_CONFIG_KEYS},
        }
        return {
            "kind": FIXED_HOLDOUT_KIND,
            "protocol": clean_protocol,
            "objective": objective,
            "metrics": metrics,
        }

    @staticmethod
    def _has_resumable_bundle_format(metadata):
        if not isinstance(metadata, dict) or metadata.get("format") != DASHBOARD_BUNDLE_FORMAT:
            return False
        version = metadata.get("format_version")
        return isinstance(version, int) and not isinstance(version, bool) and version >= 2

    def _clear_verified_provenance(self, *, reset_progress=False):
        self.guard_benchmark = {}
        self.holdout_protocol = {}
        self.best_guard_objective = float("-inf")
        self.resume_best_bundle = None
        self.last_guard = {}
        if reset_progress:
            self.trained_steps = 0
            self.iteration = 0
            self.history = []
            self.best_score = 0
            self.best_score_steps = 0
            self.best_score_trained_steps = 0
            self.best_score_iteration = 0

    def _quarantine_persisted_best(self, reason):
        """Move an untrusted/stale protected path aside before model loading."""

        self._clear_verified_provenance(reset_progress=True)
        safe_reason = "".join(
            character if character.isalnum() else "-" for character in str(reason).lower()
        ).strip("-") or "invalid"
        quarantine_path = None
        if BEST_MODEL_PATH.exists():
            quarantine_path = BEST_MODEL_PATH.with_name(
                f"{BEST_MODEL_PATH.stem}.quarantine-{safe_reason}-{time.time_ns()}"
                f"{BEST_MODEL_PATH.suffix}"
            )
            try:
                os.replace(BEST_MODEL_PATH, quarantine_path)
            except OSError as exc:
                quarantine_path = None
                self.last_error = f"Could not quarantine stale protected bundle: {exc}"

        if quarantine_path is not None:
            notice = (
                f"quarantined stale protected bundle ({reason}) as "
                f"{quarantine_path.name}; using baseline fallback"
            )
        else:
            notice = (
                f"ignored stale protected bundle ({reason}); using baseline fallback"
            )
        self.startup_notice = notice
        self.last_event = notice
        return quarantine_path

    def _restore_persisted_best_checkpoint(self):
        if not BEST_MODEL_PATH.exists():
            return
        metadata = self._read_bundle_metadata(BEST_MODEL_PATH)
        if not metadata:
            self._quarantine_persisted_best("missing or invalid metadata")
            return
        if not self._has_resumable_bundle_format(metadata):
            self._quarantine_persisted_best("legacy or stale format")
            return
        benchmark = self._validated_guard_benchmark(metadata)
        if not benchmark:
            self._quarantine_persisted_best("missing or invalid fixed-holdout benchmark")
            return
        bundle_config = metadata.get("config")
        if not isinstance(bundle_config, dict):
            self._quarantine_persisted_best("missing model configuration")
            return
        original_config = dict(self.config)
        try:
            self._merge_config(bundle_config)
        except (TypeError, ValueError):
            self.config = original_config
            self._quarantine_persisted_best("invalid model configuration")
            return
        if not self._benchmark_matches_model_config(benchmark):
            self.config = original_config
            self._quarantine_persisted_best("fixed-holdout protocol does not match configuration")
            return
        try:
            history = metadata.get("history") or []
            if not isinstance(history, list):
                raise TypeError("history must be a list")
            trained_steps = max(0, int(metadata.get("trained_steps") or 0))
            iteration = max(0, int(metadata.get("iteration") or 0))
            best = metadata.get("best") if isinstance(metadata.get("best"), dict) else {}
            best_score = max(0, int(best.get("score") or 0))
            best_score_steps = max(0, int(best.get("steps") or 0))
            best_score_trained_steps = max(0, int(best.get("trained_steps") or 0))
            best_score_iteration = max(0, int(best.get("iteration") or 0))
        except (TypeError, ValueError):
            self.config = original_config
            self._quarantine_persisted_best("invalid persisted training state")
            return

        self.resume_best_bundle = BEST_MODEL_PATH
        self.trained_steps = trained_steps
        self.iteration = iteration
        self.history = history
        self.best_score = best_score
        self.best_score_steps = best_score_steps
        self.best_score_trained_steps = best_score_trained_steps
        self.best_score_iteration = best_score_iteration
        self._restore_last_guard(metadata)
        self.guard_benchmark = benchmark
        self.holdout_protocol = dict(benchmark["protocol"])
        self.best_guard_objective = float(benchmark["objective"])
        self.last_event = (
            f"protected best ready ({self.best_guard_objective:.5f} fixed holdout)"
        )

    def _expected_holdout_protocol(self):
        return {
            "version": FIXED_HOLDOUT_PROTOCOL_VERSION,
            "seed_base": int(self.config["seed"]) + 5_000_000,
            "episodes": max(8, int(self.config["guard_holdout_episodes"])),
            "max_steps": int(self.config["guard_eval_steps"]),
            "eval_config": {
                key: self.config[key] for key in HOLDOUT_EVAL_CONFIG_KEYS
            },
        }

    def _benchmark_matches_model_config(self, benchmark):
        benchmark = benchmark if isinstance(benchmark, dict) else {}
        protocol = benchmark.get("protocol") if isinstance(benchmark, dict) else None
        return isinstance(protocol, dict) and protocol == self._expected_holdout_protocol()

    def _protocol_matches_model_config(self, protocol):
        return self._benchmark_matches_model_config({"protocol": protocol})

    def _holdout_protocol(self):
        if self._protocol_matches_model_config(self.holdout_protocol):
            return dict(self.holdout_protocol)
        if self._benchmark_matches_model_config(self.guard_benchmark):
            self.holdout_protocol = dict(self.guard_benchmark["protocol"])
            return dict(self.holdout_protocol)
        self.guard_benchmark = {}
        self.best_guard_objective = float("-inf")
        self.resume_best_bundle = None
        self.holdout_protocol = self._expected_holdout_protocol()
        return dict(self.holdout_protocol)

    @staticmethod
    def _protocol_id(protocol):
        raw = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]

    def _write_model_bundle(self, destination: Path, metadata: dict) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(file_descriptor)
        temporary_destination = Path(temporary_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                model_path = Path(tmpdir) / "model.zip"
                self.model.save(model_path)
                with zipfile.ZipFile(
                    temporary_destination,
                    mode="w",
                    compression=zipfile.ZIP_DEFLATED,
                ) as bundle:
                    bundle.write(model_path, "model.zip")
                    bundle.writestr("metadata.json", json.dumps(metadata, indent=2))
                os.replace(temporary_destination, destination)
            finally:
                temporary_destination.unlink(missing_ok=True)

    def _promote_fixed_holdout_best(self, *, protocol, metrics, guard):
        """Atomically promote only a strictly better comparable holdout result."""

        objective = float(metrics["objective"])
        existing_metadata = self._read_bundle_metadata(BEST_MODEL_PATH) or {}
        existing_benchmark = self._validated_guard_benchmark(existing_metadata)
        same_protocol = bool(existing_benchmark) and existing_benchmark.get("protocol") == protocol
        existing_objective = (
            float(existing_benchmark["objective"])
            if same_protocol
            else float("-inf")
        )
        in_memory_objective = (
            self.best_guard_objective
            if self.guard_benchmark.get("protocol") == protocol
            else float("-inf")
        )
        if objective <= max(existing_objective, in_memory_objective):
            return False

        # Preserve an incompatible protocol's best under a stable archive name;
        # objectives from different boards/observation layouts are not compared.
        if existing_benchmark and not same_protocol and BEST_MODEL_PATH.exists():
            old_id = self._protocol_id(existing_benchmark["protocol"])
            archive = BEST_MODEL_PATH.with_name(
                f"{BEST_MODEL_PATH.stem}.{old_id}{BEST_MODEL_PATH.suffix}"
            )
            if not archive.exists():
                shutil.copy2(BEST_MODEL_PATH, archive)

        previous_benchmark = dict(self.guard_benchmark)
        previous_protocol = dict(self.holdout_protocol)
        previous_objective = self.best_guard_objective
        self.guard_benchmark = {
            "kind": FIXED_HOLDOUT_KIND,
            "protocol": dict(protocol),
            "objective": objective,
            "metrics": dict(metrics),
        }
        self.holdout_protocol = dict(protocol)
        self.best_guard_objective = objective
        try:
            metadata = self._model_bundle_metadata(
                extra={
                    "guard_objective": objective,
                    "guard_objective_kind": FIXED_HOLDOUT_KIND,
                    "guard": dict(guard, promoted_to_best=True),
                }
            )
            self._write_model_bundle(BEST_MODEL_PATH, metadata)
        except Exception:
            self.guard_benchmark = previous_benchmark
            self.holdout_protocol = previous_protocol
            self.best_guard_objective = previous_objective
            raise
        self.resume_best_bundle = BEST_MODEL_PATH
        return True

    def _load_model_from_bundle(self, bundle_path, env, device):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.zip"
            with zipfile.ZipFile(bundle_path) as bundle:
                info = bundle.getinfo("model.zip")
                if info.file_size > MAX_MODEL_UPLOAD_BYTES:
                    raise ValueError("Protected model bundle is too large")
                model_path.write_bytes(bundle.read("model.zip"))
            return self._load_model(model_path, env, device)

    def export_model_bundle(self):
        with self.model_io_lock:
            with self.lock:
                if self.model is None:
                    raise ValueError("No model has been created yet. Start training before downloading.")
                metadata = self._model_bundle_metadata()

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
                    model_path.write_bytes(uploaded_zip.read("model.zip"))
                    metadata = json.loads(uploaded_zip.read("metadata.json").decode("utf-8"))
                    if not isinstance(metadata, dict):
                        raise ValueError("Bundle metadata must be a JSON object")
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
        """Load a trusted manual import as an explicitly unverified baseline.

        SB3 archives are trusted-code inputs, but dashboard metadata is not
        proof that the imported weights produced its claimed evaluation.  In
        particular, legacy guard/history fields must never enter the protected
        fixed-holdout provenance chain.
        """

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
                self.trained_steps = max(0, int(self.model.num_timesteps or 0))
                self.iteration = 0
                self.history = []
                self._clear_verified_provenance()
                self.best_score = 0
                self.best_score_steps = 0
                self.best_score_trained_steps = 0
                self.best_score_iteration = 0
                self.frames = []
                self.frame_version += 1
                self.last_error = None
                self.actual_device = str(self.model.device)
                self.startup_notice = None
                self.last_event = (
                    f"trusted model imported as unverified baseline at "
                    f"{self.trained_steps} model steps; provenance cleared"
                )

    def _restore_last_guard(self, metadata):
        guard = metadata.get("guard") if isinstance(metadata, dict) else None
        if not isinstance(guard, dict):
            for item in reversed(self.history):
                candidate = item.get("guard") if isinstance(item, dict) else None
                if isinstance(candidate, dict):
                    guard = candidate
                    break
        self.last_guard = dict(guard) if isinstance(guard, dict) else {}

    def _restore_best(self, metadata):
        benchmark = self._validated_guard_benchmark(metadata if isinstance(metadata, dict) else {})
        self.guard_benchmark = benchmark
        self.holdout_protocol = dict(benchmark.get("protocol") or {})
        self.best_guard_objective = (
            float(benchmark["objective"]) if benchmark else float("-inf")
        )
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
        if summary.get("preview_strategy", "model") != "model":
            return
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

    def _rollback_guard_candidate(self, checkpoint):
        """Restore policy, optimizer, schedules, and timestep count."""

        self.model = self._load_model(checkpoint, self.train_env, self.model.device)

    def _is_new_best_guard(self, objective):
        return float(objective) > self.best_guard_objective

    def update_config(self, updates):
        # Match the model-I/O -> state-lock ordering used by training/export.
        # A live update then cannot alter the policy or environment halfway
        # through a guarded baseline/train/candidate transaction.
        with self.model_io_lock:
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
            elif key in {"board_size", "seed", "n_steps", "batch_size", "num_envs", "n_epochs", "chunk_timesteps", "preview_steps", "loop_window", "oscillation_window", "cnn_features_dim", "guard_eval_episodes", "guard_eval_steps", "guard_holdout_episodes"}:
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
        self.config["guard_eval_episodes"] = max(4, min(64, self.config["guard_eval_episodes"]))
        self.config["guard_eval_steps"] = max(50, min(5000, self.config["guard_eval_steps"]))
        self.config["guard_min_delta"] = max(0.0, min(1000.0, self.config["guard_min_delta"]))
        # Production training is always guarded; preview-only mode is controlled
        # by training_enabled rather than silently accepting unverified PPO.
        self.config["guard_enabled"] = True
        self.config["guard_holdout_episodes"] = max(
            8, min(32, self.config["guard_holdout_episodes"])
        )
        self.config["guard_holdout_max_drop"] = 0.0
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

    def _make_eval_env(self, seed, eval_config=None):
        # A guard comparison must use one immutable environment specification.
        # Copying here also protects direct evaluator callers from live dashboard
        # updates that arrive between episodes.
        config = dict(eval_config) if eval_config is not None else dict(self.config)
        env_cls = SnakeCnnEnv if config["agent"] == "cnn" else SnakeMlpEnv
        env_kwargs = {
            "seed": seed,
            "board_size": config["board_size"],
            "silent_mode": True,
            "limit_step": True,
            "food_time_penalty": config["food_time_penalty"],
            "food_step_limit_multiplier": config["food_step_limit_multiplier"],
            "food_reward_bonus": config["food_reward_bonus"],
            "distance_reward_scale": config["distance_reward_scale"],
            "loop_penalty": config["loop_penalty"],
            "loop_window": config["loop_window"],
            "oscillation_penalty": config["oscillation_penalty"],
            "oscillation_window": config["oscillation_window"],
        }
        if env_cls is SnakeCnnEnv:
            env_kwargs["channel_first"] = config["cnn_channel_first"]
        return env_cls(**env_kwargs)

    def _guard_objective(self, metrics):
        return (
            float(metrics.get("avg_score", 0.0))
            + float(metrics.get("avg_food", 0.0)) * 0.5
            + float(metrics.get("avg_reward", 0.0)) * 0.05
        )

    @staticmethod
    def _guard_decision(
        baseline,
        candidate,
        *,
        min_delta,
        holdout_baseline=None,
        holdout_candidate=None,
        holdout_max_drop=0.0,
    ):
        """Return an evidence-based behavior gate decision.

        A weight change alone cannot pass: the candidate must improve the
        deterministic, same-seed development objective by a non-zero practical
        effect while not reducing food collection.  A fixed holdout, when
        supplied, additionally prevents accepting a development-only gain that
        regresses on the stable reference set.
        """

        required_delta = max(float(min_delta), 1e-5)
        development_delta = float(candidate["objective"]) - float(baseline["objective"])
        food_delta = float(candidate.get("avg_food", 0.0)) - float(
            baseline.get("avg_food", 0.0)
        )
        development_pass = development_delta + 1e-12 >= required_delta and food_delta >= 0.0

        holdout_delta = None
        holdout_food_delta = None
        holdout_pass = True
        if holdout_baseline is not None and holdout_candidate is not None:
            holdout_delta = float(holdout_candidate["objective"]) - float(
                holdout_baseline["objective"]
            )
            holdout_food_delta = float(holdout_candidate.get("avg_food", 0.0)) - float(
                holdout_baseline.get("avg_food", 0.0)
            )
            max_drop = max(0.0, float(holdout_max_drop))
            holdout_pass = holdout_delta + 1e-12 >= -max_drop and holdout_food_delta >= 0.0

        accepted = bool(development_pass and holdout_pass)
        if not development_pass:
            reason = "no_measured_behavior_improvement"
        elif not holdout_pass:
            reason = "fixed_holdout_regression"
        else:
            reason = "behavior_improved_and_holdout_preserved"
        return {
            "accepted": accepted,
            "reason": reason,
            "required_delta": required_delta,
            "development_delta": round(development_delta, 5),
            "development_food_delta": round(food_delta, 5),
            "holdout_delta": None if holdout_delta is None else round(holdout_delta, 5),
            "holdout_food_delta": (
                None if holdout_food_delta is None else round(holdout_food_delta, 5)
            ),
        }

    def _evaluate_model_score(
        self,
        model,
        *,
        seed_base,
        episodes,
        max_steps,
        eval_config=None,
    ):
        eval_config = dict(eval_config) if eval_config is not None else dict(self.config)
        scores = []
        foods = []
        rewards = []
        episode_results = []
        for index in range(int(episodes)):
            seed = seed_base + index
            env = self._make_eval_env(seed, eval_config)
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
                episode_results.append(
                    {
                        "seed": seed,
                        "score": scores[-1],
                        "food": food_count,
                        "reward": round(total_reward, 6),
                        "steps": steps,
                    }
                )
            finally:
                env.close()
        count = max(1, len(scores))
        metrics = {
            "episodes": count,
            "avg_score": round(sum(scores) / count, 4),
            "avg_food": round(sum(foods) / count, 4),
            "avg_reward": round(sum(rewards) / count, 5),
            "episode_results": episode_results,
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
        protected_bundle = (
            Path(self.resume_best_bundle)
            if self.resume_best_bundle and Path(self.resume_best_bundle).exists()
            else None
        )
        original_model_path = self._initial_model_path(device)
        if protected_bundle is not None:
            try:
                self.model = self._load_model_from_bundle(
                    protected_bundle,
                    self.train_env,
                    device,
                )
                if self.trained_steps == 0:
                    self.trained_steps = int(self.model.num_timesteps or 0)
                source_event = f"resumed protected fixed-holdout best from {protected_bundle.name}"
            except (OSError, ValueError, KeyError, RuntimeError, EOFError, zipfile.BadZipFile):
                self._quarantine_persisted_best("protected model weights failed to load")
                self.model = None
        if self.model is None and original_model_path is not None:
            self.model = self._load_model(original_model_path, self.train_env, device)
            if self.trained_steps == 0:
                self.trained_steps = int(self.model.num_timesteps or 0)
            source_event = f"loaded original model from {original_model_path.relative_to(MAIN_DIR)}"
        elif self.model is None:
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
        if self.startup_notice:
            source_event = f"{self.startup_notice}; {source_event}"
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

    def _run_guarded_training_transaction(self, chunk):
        """Train/evaluate one candidate, rolling back every pre-commit failure."""

        guard_seed = int(self.config["seed"]) + self.iteration * 10_007 + 50_000
        guard_episodes = int(self.config["guard_eval_episodes"])
        guard_steps = int(self.config["guard_eval_steps"])
        min_delta = float(self.config["guard_min_delta"])
        holdout_protocol = self._holdout_protocol()
        holdout_seed = int(holdout_protocol["seed_base"])
        holdout_episodes = int(holdout_protocol["episodes"])
        holdout_steps = int(holdout_protocol["max_steps"])
        holdout_eval_config = dict(holdout_protocol["eval_config"])
        eval_config = dict(self.config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model_before.zip"
            self.model.save(checkpoint)
            candidate_started = False
            candidate_committed = False
            try:
                baseline = self._evaluate_model_score(
                    self.model,
                    seed_base=guard_seed,
                    episodes=guard_episodes,
                    max_steps=guard_steps,
                    eval_config=eval_config,
                )
                holdout_baseline = self._evaluate_model_score(
                    self.model,
                    seed_base=holdout_seed,
                    episodes=holdout_episodes,
                    max_steps=holdout_steps,
                    eval_config=holdout_eval_config,
                )
                candidate_start_steps = int(self.model.num_timesteps)
                candidate_started = True
                self.model.learn(
                    total_timesteps=chunk,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                attempted_timesteps = max(
                    0, int(self.model.num_timesteps) - candidate_start_steps
                )
                candidate = self._evaluate_model_score(
                    self.model,
                    seed_base=guard_seed,
                    episodes=guard_episodes,
                    max_steps=guard_steps,
                    eval_config=eval_config,
                )
                holdout_candidate = self._evaluate_model_score(
                    self.model,
                    seed_base=holdout_seed,
                    episodes=holdout_episodes,
                    max_steps=holdout_steps,
                    eval_config=holdout_eval_config,
                )
                decision = self._guard_decision(
                    baseline,
                    candidate,
                    min_delta=min_delta,
                    holdout_baseline=holdout_baseline,
                    holdout_candidate=holdout_candidate,
                    holdout_max_drop=0.0,
                )
                accepted = decision["accepted"]
                guard = {
                    "accepted": accepted,
                    "reason": decision["reason"],
                    "episodes": guard_episodes,
                    "max_steps": guard_steps,
                    "min_delta": min_delta,
                    "baseline": baseline,
                    "candidate": candidate,
                    "attempted_timesteps": attempted_timesteps,
                    "decision": decision,
                    "eval_seed_base": guard_seed,
                    "holdout_seed_base": holdout_seed,
                    "holdout_episodes": holdout_episodes,
                    "holdout_max_steps": holdout_steps,
                    "holdout_max_drop": 0.0,
                    "holdout_baseline": holdout_baseline,
                    "holdout_candidate": holdout_candidate,
                    "holdout_protocol": holdout_protocol,
                    "promotion_basis": FIXED_HOLDOUT_KIND,
                    "candidate_isolated_by_model_io": True,
                    "evaluation_frozen": True,
                    "eval_env_wrappers": ["raw_env", "manual_action_mask"],
                    "promoted_to_best": False,
                }
                if not accepted:
                    self._rollback_guard_candidate(checkpoint)
                    candidate_started = False
                    return guard
                guard["promoted_to_best"] = self._promote_fixed_holdout_best(
                    protocol=holdout_protocol,
                    metrics=holdout_candidate,
                    guard=guard,
                )
                candidate_committed = True
                return guard
            except BaseException as exc:
                if candidate_started and not candidate_committed:
                    try:
                        self._rollback_guard_candidate(checkpoint)
                    except Exception as rollback_exc:
                        raise RuntimeError(
                            f"candidate failed and rollback also failed: {rollback_exc!r}"
                        ) from exc
                raise

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
                        guard = self._run_guarded_training_transaction(chunk)
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
                            "preview_strategy": strategy,
                            "time": time.strftime("%H:%M:%S"),
                        }
                    )
                    if guard:
                        # Keep rejected attempts in exported history too; a
                        # rollback should never erase its audit evidence.
                        summary["guard"] = guard
                    self.frames = frames
                    self.frame_version += 1
                    self.history.append(summary)
                    self._update_best(summary)
                    self.last_error = None
                    if training_enabled:
                        if guard and not guard.get("accepted", True):
                            self.last_event = f"guard rejected candidate at {self.trained_steps} steps"
                        elif strategy == "hamiltonian":
                            self.last_event = (
                                f"PPO trained to {self.trained_steps} steps; "
                                "canvas is Hamiltonian preview only"
                            )
                        else:
                            self.last_event = f"PPO trained to {self.trained_steps} accepted steps"
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
atexit.register(dashboard.close)
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
