from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path, import_dir: Path):
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


def audit_chess() -> dict:
    chess_dir = ROOT / "chess-ai"
    checkpoint = chess_dir / "runtime" / "chess_policy.best.json"
    app = load_module("audit_chess_app", chess_dir / "app.py", chess_dir)
    trainer = app.Trainer()
    data = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    final = dict(data.get("final") or {})
    baseline = dict(data.get("baseline") or {})
    loaded = trainer.snapshot().get("checkpoint", {}).get("loaded", {})
    return {
        "checkpoint": str(checkpoint),
        "exists": checkpoint.exists(),
        "loaded_by_dashboard": Path(str(loaded.get("path", ""))).resolve() == checkpoint.resolve() if loaded.get("path") else False,
        "teacher": data.get("teacher", ""),
        "baseline": {"avg_gap": baseline.get("avg_gap"), "match_rate": baseline.get("match_rate")},
        "final": {"avg_gap": final.get("avg_gap"), "match_rate": final.get("match_rate")},
        "improved": float(final.get("avg_gap", 1e18)) < float(baseline.get("avg_gap", 1e18)),
    }


def audit_snake(episodes: int) -> dict:
    snake_dir = ROOT / "snake-ai" / "main"
    sys.path.insert(0, str(snake_dir))
    try:
        import web_dashboard as dashboard
    finally:
        try:
            sys.path.remove(str(snake_dir))
        except ValueError:
            pass
    checkpoint = ROOT / "snake-ai" / "runtime" / "snake_policy.best.snakeai.zip"
    with zipfile.ZipFile(checkpoint) as bundle:
        metadata = json.loads(bundle.read("metadata.json").decode("utf-8"))
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.zip"
            model_path.write_bytes(bundle.read("model.zip"))
            trainer = dashboard.TrainingDashboard()
            trainer.update_config(metadata.get("config", {}))
            trainer._ensure_model()
            trainer.model = trainer._load_model(model_path, trainer.train_env, trainer.model.device)
            evaluation = trainer._evaluate_model_score(
                trainer.model,
                seed_base=1_230_000,
                episodes=episodes,
                max_steps=360,
            )
    return {
        "checkpoint": str(checkpoint),
        "exists": checkpoint.exists(),
        "config": {
            "agent": metadata.get("config", {}).get("agent"),
            "board_size": metadata.get("config", {}).get("board_size"),
        },
        "guard_objective": metadata.get("guard_objective"),
        "guard_accepted": bool((metadata.get("guard") or {}).get("accepted")),
        "evaluation": evaluation,
        "usable": evaluation.get("avg_score", 0) > 0 and bool((metadata.get("guard") or {}).get("accepted")),
    }


def audit_soccer(episodes: int) -> dict:
    soccer_dir = ROOT / "soccer-ai"
    for key in ("rl_trainer", "soccer_env"):
        sys.modules.pop(key, None)
    sys.path.insert(0, str(soccer_dir))
    try:
        from rl_trainer import RLTrainer
    finally:
        try:
            sys.path.remove(str(soccer_dir))
        except ValueError:
            pass
    trainer = RLTrainer(soccer_dir)
    evaluation = trainer.evaluate(episodes=episodes, opponent="mixed")
    objective = trainer._guard_objective(evaluation)
    best = soccer_dir / "runtime" / "soccer_policy.best.json"
    best_payload = json.loads(best.read_text(encoding="utf-8")) if best.exists() else {}
    return {
        "checkpoint": str(trainer.checkpoint_path),
        "best_checkpoint": str(best),
        "exists": trainer.checkpoint_path.exists(),
        "best_exists": best.exists(),
        "best_objective": best_payload.get("objective"),
        "evaluation": {
            "objective": round(objective, 3),
            "win_rate": evaluation.get("record", {}).get("win_rate"),
            "avg_goal_diff": evaluation.get("avg_goal_diff"),
            "opponent_distribution": evaluation.get("opponent_distribution"),
        },
        "usable": objective > 0,
    }


def _eval_tetris_policy(policy, trainer, seeds: list[int], max_pieces: int) -> dict:
    from tetris_env import TetrisEnv

    rows = []
    for seed in seeds:
        env = TetrisEnv(seed=seed, max_pieces=max_pieces)
        env.reset()
        done = False
        while not done:
            move = policy.choose(
                env.legal_moves(),
                epsilon=0.0,
                temperature=0.0,
                next_moves_provider=lambda row: env.future_legal_moves_for(row, include_hold=False),
                lookahead_weight=trainer.lookahead_weight,
                lookahead_candidates=trainer.lookahead_candidates,
                lookahead_include_hold=False,
            )
            _obs, _reward, done, _info = env.step(move or {}, check_terminal=False)
        info = env.info()
        rows.append({"score": info["score"], "tetrises": info["tetrises"], "pieces": info["pieces"]})
    count = max(1, len(rows))
    return {
        "avg_score": round(sum(row["score"] for row in rows) / count, 2),
        "avg_tetrises": round(sum(row["tetrises"] for row in rows) / count, 3),
        "piece_cap_hit_rate": round(sum(row["pieces"] >= max_pieces for row in rows) / count, 3),
        "rows": rows,
    }


def audit_tetris() -> dict:
    tetris_dir = ROOT / "tetris-ai"
    for key in ("rl_trainer", "tetris_env"):
        sys.modules.pop(key, None)
    sys.path.insert(0, str(tetris_dir))
    try:
        from rl_trainer import AfterstateValue, RLTrainer
    finally:
        try:
            sys.path.remove(str(tetris_dir))
        except ValueError:
            pass
    trainer = RLTrainer(tetris_dir)
    runtime = tetris_dir / "runtime"
    seeds = list(range(2_000, 2_005))
    max_pieces = 200
    results = {"current": _eval_tetris_policy(trainer.policy, trainer, seeds, max_pieces)}
    for name, filename in (("guard_best", "tetris_policy.best.json"), ("score_best", "tetris_policy.best_score.json")):
        payload = json.loads((runtime / filename).read_text(encoding="utf-8"))
        policy = AfterstateValue(trainer.policy.dim)
        policy.load_json(payload.get("policy") or payload.get("best_policy") or {})
        results[name] = _eval_tetris_policy(policy, trainer, seeds, max_pieces)
    preferred = max(results, key=lambda key: (results[key]["avg_score"], results[key]["avg_tetrises"]))
    return {
        "checkpoint": str(trainer.checkpoint_path),
        "guard_best": str(runtime / "tetris_policy.best.json"),
        "score_best": str(runtime / "tetris_policy.best_score.json"),
        "seeds": seeds,
        "max_pieces": max_pieces,
        "evaluations": results,
        "preferred": preferred,
        "usable": results[preferred]["piece_cap_hit_rate"] >= 0.8,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit locally trained game AI runtime models.")
    parser.add_argument("--snake-episodes", type=int, default=20)
    parser.add_argument("--soccer-episodes", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime" / "usable_model_audit_latest.json")
    args = parser.parse_args()

    report = {
        "chess": audit_chess(),
        "snake": audit_snake(max(2, args.snake_episodes)),
        "soccer": audit_soccer(max(4, args.soccer_episodes)),
        "tetris": audit_tetris(),
    }
    report["overall_pass"] = all(item.get("usable", item.get("improved", False)) for item in report.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
