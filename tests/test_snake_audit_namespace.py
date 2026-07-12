import importlib.util
import json
import random
import zipfile
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "scripts" / "audit_usable_models.py"
SPEC = importlib.util.spec_from_file_location("snake_namespace_audit_tests", AUDIT_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def _write_bundle(path, metadata, model=b"policy"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("model.zip", model)
        bundle.writestr("metadata.json", json.dumps(metadata))


def test_stage_copies_all_namespaces_and_legacy_without_aliasing_sources(tmp_path):
    source = tmp_path / "source" / "snake-ai"
    destination = tmp_path / "private" / "snake-ai"
    (source / "main").mkdir(parents=True)
    (source / "main" / "web_dashboard.py").write_text("VALUE = 1\n")
    cnn = source / "runtime/protected_best/snake_policy.cnn.aaa.best.snakeai.zip"
    mlp = source / "runtime/protected_best/snake_policy.mlp.bbb.best.snakeai.zip"
    legacy = source / "runtime/snake_policy.best.snakeai.zip"
    mlp_fallback = (
        source
        / "main/original_models/trained_models_mlp/ppo_snake_final.zip"
    )
    cnn_original = (
        source
        / "main/original_models/trained_models_cnn/ppo_snake_final.zip"
    )
    cnn_fullboard = (
        source
        / "main/trained_models_cnn_oracle_bc/ppo_snake_bc_final_12x12.zip"
    )
    _write_bundle(cnn, {"config": {"agent": "cnn"}}, b"cnn")
    _write_bundle(mlp, {"config": {"agent": "mlp"}}, b"mlp")
    _write_bundle(legacy, {"config": {"agent": "mlp"}}, b"legacy")
    for path, payload in (
        (mlp_fallback, b"mlp fallback"),
        (cnn_original, b"cnn original"),
        (cnn_fullboard, b"cnn fullboard"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    production_before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (cnn, mlp, legacy, mlp_fallback, cnn_original, cnn_fullboard)
    }

    staged = audit._stage_snake_audit_tree(source, destination)

    assert set(staged["source_relatives"]) == {
        cnn.relative_to(source),
        mlp.relative_to(source),
        legacy.relative_to(source),
    }
    copied_cnn = destination / cnn.relative_to(source)
    copied_mlp = destination / mlp.relative_to(source)
    copied_legacy = destination / legacy.relative_to(source)
    copied_mlp_fallback = destination / mlp_fallback.relative_to(source)
    copied_cnn_original = destination / cnn_original.relative_to(source)
    copied_cnn_fullboard = destination / cnn_fullboard.relative_to(source)
    copied_cnn.unlink()
    copied_mlp.rename(copied_mlp.with_suffix(".quarantine"))
    copied_legacy.write_bytes(b"isolated mutation")
    copied_mlp_fallback.write_bytes(b"isolated mlp mutation")
    copied_cnn_original.unlink()
    copied_cnn_fullboard.write_bytes(b"isolated cnn mutation")

    assert all(row["staged_verified"] for row in staged["fallback_artifacts"])
    for path, (content, mtime_ns) in production_before.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == mtime_ns
    with zipfile.ZipFile(legacy) as bundle:
        assert bundle.read("model.zip") == b"legacy"


def test_private_migration_maps_back_to_legacy_source(tmp_path):
    source = tmp_path / "source" / "snake-ai"
    isolated = tmp_path / "isolated" / "snake-ai"
    legacy_relative = Path("runtime/snake_policy.best.snakeai.zip")
    migrated = isolated / "runtime/protected_best/snake_policy.mlp.abc.best.snakeai.zip"
    migrated.parent.mkdir(parents=True)
    migrated.write_bytes(b"migrated copy")

    original, was_migrated = audit._snake_original_checkpoint(
        migrated,
        source_root=source,
        isolated_root=isolated,
        staged={
            "protected_relatives": [],
            "legacy_relative": legacy_relative,
        },
    )

    assert original == source / legacy_relative
    assert was_migrated is True


class _FakeModel:
    def __init__(self, quality):
        self.quality = quality


class _FakeDashboard:
    def __init__(self, path, metadata):
        self.path = path
        self.metadata = metadata
        self.config = dict(metadata["config"])
        self.resume_best_bundle = None
        self.resume_best_model_sha256 = None
        self.train_env = object()
        self.model = None

    def _validated_guard_benchmark(self, metadata):
        return dict(metadata.get("guard_benchmark") or {})

    def _read_bundle_model_sha256(self, _path):
        return "a" * 64

    def _validated_bundle_provenance(self, metadata, *, model_sha256):
        if (
            model_sha256 == metadata.get("model_sha256")
            and (metadata.get("guard") or {}).get("promoted_to_best") is True
        ):
            return {
                "model_sha256": model_sha256,
                "benchmark": dict(metadata.get("guard_benchmark") or {}),
                "guard": dict(metadata.get("guard") or {}),
            }
        return {}

    def _has_resumable_bundle_format(self, metadata):
        return (
            metadata.get("format") == "snake-ai-dashboard-bundle"
            and metadata.get("format_version") == 3
        )

    def _protocol_id(self, _protocol):
        return "protocol"

    def _protected_best_path(self, _protocol):
        return self.path

    def reset(self, updates):
        self.config.update(updates)
        self.resume_best_bundle = self.path
        self.resume_best_model_sha256 = "a" * 64

    def update_config(self, updates):
        self.config.update(updates)

    def _ensure_model(self):
        self.model = _FakeModel(2.0)

    def _evaluate_model_score(self, model, **_kwargs):
        return {
            "episodes": 4,
            "avg_score": model.quality,
            "avg_food": model.quality,
            "avg_reward": 0.0,
            "objective": model.quality,
        }


def test_selected_policy_needs_runtime_selection_improvement_and_promotion(
    tmp_path, monkeypatch
):
    path = tmp_path / "snake_policy.mlp.protocol.best.snakeai.zip"
    path.write_bytes(b"kept selected")
    protocol = {"eval_config": {"agent": "mlp"}}
    metadata = {
        "format": "snake-ai-dashboard-bundle",
        "format_version": 3,
        "model_sha256": "a" * 64,
        "config": {
            "agent": "mlp",
            "seed": 7,
            "guard_eval_steps": 120,
            "n_steps": 8,
            "batch_size": 8,
            "n_epochs": 1,
            "gamma": 0.94,
            "learning_rate": 0.001,
            "clip_range": 0.15,
            "ent_coef": 0.0,
        },
        "trained_steps": 100,
        "guard_benchmark": {"protocol": protocol, "objective": 2.0},
        "guard": {"accepted": True, "promoted_to_best": True},
    }
    dashboard = _FakeDashboard(path, metadata)
    module = SimpleNamespace(
        DEFAULT_CONFIG=dict(metadata["config"]),
        DASHBOARD_BUNDLE_FORMAT="snake-ai-dashboard-bundle",
        DASHBOARD_BUNDLE_FORMAT_VERSION=3,
        random=random,
        torch=SimpleNamespace(manual_seed=lambda _seed: None),
    )
    monkeypatch.setattr(audit, "_new_snake_baseline_model", lambda *_args: _FakeModel(0.0))

    result = audit._audit_selected_snake_bundle(
        module,
        dashboard,
        local_path=path,
        source_path=path,
        metadata=metadata,
        episodes=4,
        migrated_from_legacy=False,
    )

    assert result["runtime_selected"] is True
    assert result["independent_comparison"]["strictly_better"] is True
    assert result["training_verified"] is True

    metadata["guard"].pop("promoted_to_best")
    dashboard = _FakeDashboard(path, metadata)
    result = audit._audit_selected_snake_bundle(
        module,
        dashboard,
        local_path=path,
        source_path=path,
        metadata=metadata,
        episodes=4,
        migrated_from_legacy=False,
    )
    assert result["runtime_selected"] is False
    assert result["accepted_training_evidence"] is False
    assert result["training_verified"] is False
    assert "provenance validator rejected" in result["loader_disposition"]

    metadata["guard"]["promoted_to_best"] = True
    metadata["model_sha256"] = "b" * 64
    dashboard = _FakeDashboard(path, metadata)
    result = audit._audit_selected_snake_bundle(
        module,
        dashboard,
        local_path=path,
        source_path=path,
        metadata=metadata,
        episodes=4,
        migrated_from_legacy=False,
    )
    assert result["runtime_selected"] is False
    assert result["accepted_training_evidence"] is False
    assert result["training_verified"] is False


def test_agent_summary_never_treats_an_absent_agent_as_safe_or_verified():
    rows = [
        {
            "agent": "mlp",
            "runtime_selected": True,
            "safe_to_serve": True,
            "training_verified": True,
        }
    ]

    summary = audit._snake_agent_summary(rows)

    assert summary["mlp"]["runtime_selected"] is True
    assert summary["mlp"]["safe_to_serve"] is True
    assert summary["mlp"]["training_verified"] is True
    assert summary["cnn"]["selected_policies"] == 0
    assert summary["cnn"]["runtime_selected"] is False
    assert summary["cnn"]["safe_to_serve"] is False
    assert summary["cnn"]["training_verified"] is False


def test_agent_summary_requires_namespaced_mlp_and_cnn_loader_selections():
    rows = [
        {
            "agent": agent,
            "runtime_kind": "protected",
            "runtime_selected": True,
            "safe_to_serve": True,
            "training_verified": True,
            "checkpoint": f"snake_policy.{agent}.protocol.best.snakeai.zip",
        }
        for agent in ("mlp", "cnn")
    ]

    summary = audit._snake_agent_summary(rows)

    assert all(summary[agent]["selected_policies"] == 1 for agent in ("mlp", "cnn"))
    assert all(summary[agent]["runtime_selected"] for agent in ("mlp", "cnn"))
    assert all(summary[agent]["safe_to_serve"] for agent in ("mlp", "cnn"))
    assert all(summary[agent]["training_verified"] for agent in ("mlp", "cnn"))


class _BrokenFallbackDashboard:
    def __init__(self, local_path):
        self.local_path = local_path
        self.resume_best_bundle = None
        self.model = None
        self.config = {}

    def reset(self, updates):
        self.config = dict(updates)
        self.resume_best_bundle = None

    def _initial_model_path(self, _device):
        return self.local_path

    def _ensure_model(self):
        raise ValueError("corrupt SB3 archive")


def test_corrupt_production_fallback_is_an_explicit_unsafe_loader_error(tmp_path):
    source_root = tmp_path / "production" / "snake-ai"
    isolated_root = tmp_path / "private" / "snake-ai"
    local_path = (
        isolated_root
        / "main/original_models/trained_models_mlp/ppo_snake_final.zip"
    )
    local_path.parent.mkdir(parents=True)
    local_path.write_bytes(b"not an SB3 model")
    dashboard = _BrokenFallbackDashboard(local_path)
    module = SimpleNamespace(DEFAULT_CONFIG={})

    result = audit._audit_snake_runtime_fallback(
        module,
        dashboard,
        source_root=source_root,
        isolated_root=isolated_root,
        agent="mlp",
        episodes=4,
    )

    assert result["runtime_selected"] is False
    assert result["safe_to_serve"] is False
    assert result["training_verified"] is False
    assert "corrupt SB3 archive" in result["loader_disposition"]
