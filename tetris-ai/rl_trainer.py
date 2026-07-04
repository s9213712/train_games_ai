from __future__ import annotations

import json
import math
import random
import threading
import time
from pathlib import Path

from tetris_env import UNKNOWN_NEXT_PIECE, TetrisEnv


class AfterstateValue:
    def __init__(self, dim: int, *, seed: int = 11) -> None:
        self.random = random.Random(seed)
        self.dim = dim
        self.weights = [self.random.uniform(-0.02, 0.02) for _ in range(dim)]
        if dim >= 9:
            self.weights[1] = 2.2
            self.weights[2] = -0.8
            self.weights[3] = -1.2
            self.weights[4] = -4.0
            if dim >= 15:
                self.weights[5] = -1.2
                self.weights[6] = -0.65
                self.weights[7] = -0.35
                self.weights[8] = 0.8
                self.weights[9] = 0.55
                self.weights[10] = 0.7
                self.weights[11] = -0.35
                self.weights[12] = -0.35
            else:
                self.weights[5] = -0.7
                self.weights[6] = -0.5
                self.weights[7] = -0.4
                self.weights[8] = -0.4

    def value(self, vector: list[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, vector))

    def score_move(self, move: dict) -> float:
        return float(move.get("immediate_reward", 0.0)) + self.value(move["vector"])

    def score_moves(
        self,
        moves: list[dict],
        *,
        next_moves_provider=None,
        lookahead_weight: float = 0.0,
        lookahead_candidates: int = 0,
        lookahead_include_hold: bool = False,
    ) -> list[tuple[float, dict]]:
        scored = [(self.score_move(move), move) for move in moves]
        if not scored or lookahead_weight <= 0.0 or next_moves_provider is None or lookahead_candidates <= 0:
            return scored
        limit = min(len(scored), lookahead_candidates)
        candidate_indexes = sorted(range(len(scored)), key=lambda index: scored[index][0], reverse=True)[:limit]
        rescored = []
        future_cache: dict[tuple, list[dict]] = {}
        for index in candidate_indexes:
            base_score, move = scored[index]
            key = (
                int(move.get("next_current", -1)),
                int(move.get("next_piece", -1)),
                move.get("next_hold_piece"),
                bool(lookahead_include_hold),
                tuple(tuple(row) for row in move["board"]),
            )
            if key not in future_cache:
                future_cache[key] = next_moves_provider(move)
            future_moves = future_cache[key]
            if future_moves:
                future_score = max(self.score_move(next_move) for next_move in future_moves)
            else:
                future_score = -8.0
            rescored.append((base_score + lookahead_weight * future_score, move))
        return rescored

    def choose(
        self,
        moves: list[dict],
        *,
        epsilon: float,
        temperature: float,
        next_moves_provider=None,
        lookahead_weight: float = 0.0,
        lookahead_candidates: int = 0,
        lookahead_include_hold: bool = False,
    ) -> dict | None:
        if not moves:
            return None
        if self.random.random() < epsilon:
            return self.random.choice(moves)
        scored = self.score_moves(
            moves,
            next_moves_provider=next_moves_provider,
            lookahead_weight=lookahead_weight,
            lookahead_candidates=lookahead_candidates,
            lookahead_include_hold=lookahead_include_hold,
        )
        if temperature > 0.02 and self.random.random() < min(0.4, temperature):
            offset = max(score for score, _ in scored)
            probs = [math.exp(max(-30.0, min(30.0, (score - offset) / max(0.05, temperature)))) for score, _ in scored]
            total = sum(probs) or 1.0
            r = self.random.random()
            acc = 0.0
            for prob, (_, move) in zip(probs, scored):
                acc += prob / total
                if r <= acc:
                    return move
        return max(scored, key=lambda item: item[0])[1]

    def update(self, vector: list[float], target: float, *, learning_rate: float) -> float:
        pred = self.value(vector)
        error = max(-10.0, min(10.0, target - pred))
        for i, value in enumerate(vector):
            self.weights[i] += learning_rate * error * value
            self.weights[i] = self._clamp_weight(i, self.weights[i])
        return error

    def clone(self) -> "AfterstateValue":
        policy = AfterstateValue(self.dim)
        policy.weights = [float(v) for v in self.weights]
        return policy

    def soft_update_from(self, source: "AfterstateValue", tau: float) -> None:
        for i in range(self.dim):
            self.weights[i] = self._clamp_weight(i, self.weights[i] * (1.0 - tau) + source.weights[i] * tau)

    def to_json(self) -> dict:
        return {"dim": self.dim, "weights": self.weights}

    def load_json(self, payload: dict) -> None:
        weights = list(payload.get("weights") or [])
        if weights:
            original_len = len(weights)
            if original_len == 11 and self.dim >= 15:
                old = [float(v) for v in weights]
                values = [
                    old[0],
                    old[1],
                    old[2],
                    old[3],
                    old[4],
                    -1.2,
                    old[5],
                    old[6],
                    1.25,
                    0.85,
                    1.2,
                    old[7],
                    old[8],
                    old[9],
                    old[10],
                ]
            else:
                values = [float(v) for v in weights[: self.dim]]
            if len(values) < self.dim:
                values.extend(0.0 for _ in range(self.dim - len(values)))
            if original_len < self.dim and self.dim >= 15 and original_len != 11:
                values[5] = -1.2
                values[8] = 1.25
                values[9] = 0.85
                values[10] = 1.2
                values[11] = -0.35
                values[12] = -0.35
            self.weights = [self._clamp_weight(index, value) for index, value in enumerate(values)]

    def _clamp_weight(self, index: int, value: float) -> float:
        if self.dim >= 15:
            limits = (
                (-8.0, 8.0),
                (-2.0, 12.0),
                (-10.0, 1.0),
                (-10.0, 1.0),
                (-12.0, 0.5),
                (-8.0, 0.5),
                (-4.0, 1.0),
                (-3.0, 2.0),
                (-1.0, 5.0),
                (-1.0, 6.0),
                (-1.0, 6.0),
                (-6.0, 1.0),
                (-6.0, 1.0),
                (-2.0, 2.0),
                (-2.0, 2.0),
            )
            lo, hi = limits[index] if index < len(limits) else (-10.0, 10.0)
            return max(lo, min(hi, value))
        return max(-10.0, min(10.0, value))


class RLTrainer:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir(parents=True, exist_ok=True)
        probe = TetrisEnv(seed=1)
        obs = probe.reset()
        self.policy = AfterstateValue(len(obs))
        self.target_policy = self.policy.clone()
        self.best_policy = self.policy.clone()
        self.checkpoint_path = self.runtime / "tetris_policy.json"
        self.metrics_path = self.runtime / "training_metrics.jsonl"
        self.replay_path = self.runtime / "latest_replay.json"
        self.eval_path = self.runtime / "latest_evaluation.json"
        self.lock = threading.RLock()
        self.training_lock = threading.RLock()
        self.running = False
        self.stop_requested = False
        self.thread: threading.Thread | None = None
        self.episode = 0
        self.learning_rate = 0.0005
        self.gamma = 0.985
        self.epsilon = 0.015
        self.temperature = 0.05
        self.target_tau = 0.006
        self.elite_anchor = 0.035
        self.lookahead_weight = 0.1
        self.lookahead_candidates = 4
        self.lookahead_include_hold = False
        self.best_score = 0
        self.best_lines = 0
        self.history: list[dict] = []
        self.latest_info: dict = {}
        self.latest_replay: list[dict] = []
        self.last_eval: dict = {}
        self.last_guard: dict = {}
        self.last_event = "ready"
        self._load_checkpoint()

    def train_episode(self, *, persist: bool = True) -> dict:
        with self.training_lock:
            return self._train_episode(persist=persist)

    def _train_episode(self, *, persist: bool = True) -> dict:
        with self.lock:
            episode = self.episode
            self.episode += 1
            lr = self.learning_rate
            gamma = self.gamma
            epsilon = self.epsilon
            temperature = self.temperature
            lookahead_weight = self.lookahead_weight
            lookahead_candidates = self.lookahead_candidates
            lookahead_include_hold = self.lookahead_include_hold
            target_policy = self.target_policy.clone()
        env = TetrisEnv(seed=100_000 + episode)
        env.reset()
        total_reward = 0.0
        errors = []
        done = False
        while not done:
            moves = env.legal_moves()
            move = self.policy.choose(
                moves,
                epsilon=epsilon,
                temperature=temperature,
                next_moves_provider=lambda row: self._future_moves(env, row, include_hold=lookahead_include_hold),
                lookahead_weight=lookahead_weight,
                lookahead_candidates=lookahead_candidates,
                lookahead_include_hold=lookahead_include_hold,
            )
            if move is None:
                obs, reward, done, info = env.step({})
                total_reward += reward
                break
            vector = list(move["vector"])
            obs, reward, done, info = env.step(move, check_terminal=False)
            next_moves = env.legal_moves() if not done else []
            if not done and not next_moves and env.state.stats.pieces < env.max_pieces:
                done = True
                env.state.done = True
                env.state.last_event = "top out"
                reward -= 8.02
                env.last_reward_terms["survival"] = -8.0
                if env.replay:
                    env.replay[-1]["done"] = True
                    env.replay[-1]["last_event"] = "top out"
                    env.replay[-1]["reward"] = env.replay[-1].get("reward", 0.0) - 8.02
                    env.replay[-1]["reward_terms"] = dict(env.last_reward_terms)
            bootstrap = max((target_policy.score_move(row) for row in next_moves), default=0.0)
            target = reward + (0.0 if done else gamma * bootstrap)
            errors.append(abs(self.policy.update(vector, target, learning_rate=lr)))
            self.target_policy.soft_update_from(self.policy, self.target_tau)
            total_reward += reward
        final = env.info()
        final.update(
            {
                "episode": episode,
                "reward": round(total_reward, 3),
                "td_error": round(sum(errors) / max(1, len(errors)), 4),
                "weights": self.weight_summary(),
                "replay_frames": len(env.replay),
            }
        )
        with self.lock:
            previous_best_score = self.best_score
            self.latest_info = final
            self.latest_replay = list(env.replay)
            if final["score"] > self.best_score:
                self.best_score = final["score"]
                self.best_lines = max(self.best_lines, final["lines"])
                self.best_policy = self.policy.clone()
                final["new_best"] = True
            elif previous_best_score and final["score"] < previous_best_score * 0.55:
                self.policy.soft_update_from(self.best_policy, self.elite_anchor)
                self.target_policy.soft_update_from(self.best_policy, self.elite_anchor * 0.5)
                final["elite_anchor"] = True
            else:
                self.best_lines = max(self.best_lines, final["lines"])
            self.history.append(final)
            self.history = self.history[-250:]
            self.last_event = f"episode {episode}: {final['score']} score / {final['lines']} lines"
        if persist:
            self._persist(final, env.replay)
        return final

    def evaluate(self, episodes: int = 20, *, lookahead_include_hold: bool | None = None) -> dict:
        with self.training_lock:
            return self._evaluate(episodes, lookahead_include_hold=lookahead_include_hold)

    def _evaluate(self, episodes: int = 20, *, lookahead_include_hold: bool | None = None) -> dict:
        episodes = max(1, min(200, int(episodes)))
        with self.lock:
            policy = self.policy.clone()
            start = self.episode
            lookahead_weight = self.lookahead_weight
            lookahead_candidates = self.lookahead_candidates
            include_hold = self.lookahead_include_hold if lookahead_include_hold is None else bool(lookahead_include_hold)
        evaluation = self._evaluate_policy(policy, episodes, start, lookahead_weight, lookahead_candidates, include_hold, capture_replay=True)
        evaluation["future_hold"] = include_hold
        latest_replay = evaluation.pop("_latest_replay", [])
        with self.lock:
            self.last_eval = evaluation
            self.latest_replay = latest_replay
            self.latest_info = {"evaluation": True, **(evaluation["rows"][-1] if evaluation.get("rows") else {})}
            self.last_event = f"evaluated {episodes}"
        self.eval_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        self.replay_path.write_text(json.dumps({"latest": self.latest_info, "frames": latest_replay}, ensure_ascii=False, indent=2), encoding="utf-8")
        return evaluation

    def train_guarded_batch(self, episodes: int, *, eval_episodes: int = 4, accept_ratio: float = 0.98) -> dict:
        with self.training_lock:
            return self._train_guarded_batch(
                episodes,
                eval_episodes=eval_episodes,
                accept_ratio=accept_ratio,
            )

    def _train_guarded_batch(self, episodes: int, *, eval_episodes: int = 4, accept_ratio: float = 0.98) -> dict:
        episodes = max(1, min(500, int(episodes)))
        eval_episodes = max(2, min(24, int(eval_episodes)))
        accept_ratio = max(0.7, min(1.1, float(accept_ratio)))
        with self.lock:
            original = {
                "policy": self.policy.clone(),
                "target_policy": self.target_policy.clone(),
                "best_policy": self.best_policy.clone(),
                "episode": self.episode,
                "best_score": self.best_score,
                "best_lines": self.best_lines,
                "history": list(self.history),
                "latest_info": dict(self.latest_info),
                "latest_replay": list(self.latest_replay),
                "last_eval": dict(self.last_eval),
                "last_guard": dict(self.last_guard),
            }
            start = self.episode
            lookahead_weight = self.lookahead_weight
            lookahead_candidates = self.lookahead_candidates
            lookahead_include_hold = self.lookahead_include_hold
        baseline = self._evaluate_policy(original["policy"], eval_episodes, start, lookahead_weight, lookahead_candidates, lookahead_include_hold)
        latest = None
        for _ in range(episodes):
            latest = self.train_episode(persist=False)
        with self.lock:
            candidate_policy = self.policy.clone()
            candidate_replay = list(self.latest_replay)
            candidate_latest = dict(self.latest_info)
        candidate = self._evaluate_policy(candidate_policy, eval_episodes, start, lookahead_weight, lookahead_candidates, lookahead_include_hold)
        baseline_objective = self._guard_objective(baseline)
        candidate_objective = self._guard_objective(candidate)
        accepted = candidate_objective >= baseline_objective * accept_ratio
        guard = {
            "accepted": accepted,
            "episodes": episodes,
            "eval_episodes": eval_episodes,
            "accept_ratio": accept_ratio,
            "baseline_score": baseline["avg_score"],
            "candidate_score": candidate["avg_score"],
            "baseline_tetrises": baseline["avg_tetrises"],
            "candidate_tetrises": candidate["avg_tetrises"],
            "baseline_objective": round(baseline_objective, 2),
            "candidate_objective": round(candidate_objective, 2),
        }
        with self.lock:
            if accepted:
                self.last_eval = candidate
                self.last_guard = guard
                self.last_event = f"guard accepted {episodes}: {candidate['avg_score']} avg score"
                self.eval_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
                if latest is None:
                    latest = candidate_latest
                latest["guard"] = guard
                self._persist(latest, candidate_replay)
            else:
                self.policy = original["policy"]
                self.target_policy = original["target_policy"]
                self.best_policy = original["best_policy"]
                self.episode = original["episode"]
                self.best_score = original["best_score"]
                self.best_lines = original["best_lines"]
                self.history = original["history"]
                self.latest_info = dict(original["latest_info"])
                self.latest_replay = list(original["latest_replay"])
                self.last_eval = dict(original["last_eval"] or baseline)
                self.last_guard = guard
                self.last_event = f"guard rejected {episodes}: {candidate['avg_score']} < {baseline['avg_score']}"
                self._save_checkpoint()
        return {"accepted": accepted, "guard": guard, "baseline": baseline, "candidate": candidate, "latest": latest or {}}

    @staticmethod
    def _guard_objective(evaluation: dict) -> float:
        return float(evaluation.get("avg_score", 0.0)) + float(evaluation.get("avg_tetrises", 0.0)) * 1200.0

    def _evaluate_policy(
        self,
        policy: AfterstateValue,
        episodes: int,
        start: int,
        lookahead_weight: float,
        lookahead_candidates: int,
        lookahead_include_hold: bool,
        capture_replay: bool = False,
    ) -> dict:
        rows = []
        latest_replay = []
        for index in range(episodes):
            env = TetrisEnv(seed=800_000 + start + index)
            env.reset()
            done = False
            total_reward = 0.0
            while not done:
                move = policy.choose(
                    env.legal_moves(),
                    epsilon=0.0,
                    temperature=0.0,
                    next_moves_provider=lambda row: self._future_moves(env, row, include_hold=lookahead_include_hold),
                    lookahead_weight=lookahead_weight,
                    lookahead_candidates=lookahead_candidates,
                    lookahead_include_hold=lookahead_include_hold,
                )
                _obs, reward, done, _info = env.step(move or {}, check_terminal=False)
                total_reward += reward
            info = env.info()
            rows.append(
                {
                    "score": info["score"],
                    "lines": info["lines"],
                    "pieces": info["pieces"],
                    "tetrises": info["tetrises"],
                    "holds": info.get("holds", 0),
                    "reward": round(total_reward, 3),
                }
            )
            latest_replay = list(env.replay)
        avg_score = sum(row["score"] for row in rows) / episodes
        avg_lines = sum(row["lines"] for row in rows) / episodes
        avg_pieces = sum(row["pieces"] for row in rows) / episodes
        avg_tetrises = sum(row["tetrises"] for row in rows) / episodes
        avg_holds = sum(row.get("holds", 0) for row in rows) / episodes
        evaluation = {
            "episodes": episodes,
            "avg_score": round(avg_score, 2),
            "avg_lines": round(avg_lines, 2),
            "avg_pieces": round(avg_pieces, 2),
            "avg_tetrises": round(avg_tetrises, 3),
            "avg_holds": round(avg_holds, 2),
            "best_score": max(row["score"] for row in rows),
            "best_lines": max(row["lines"] for row in rows),
            "best_tetrises": max(row["tetrises"] for row in rows),
            "rows": rows[-60:],
            "replay_frames": len(latest_replay),
        }
        if capture_replay:
            evaluation["_latest_replay"] = latest_replay
        return evaluation

    @staticmethod
    def _future_moves(env: TetrisEnv, move: dict, *, include_hold: bool = False) -> list[dict]:
        return env.future_legal_moves_for(move, include_hold=include_hold)

    def start(self) -> None:
        with self.lock:
            self.running = True
            self.stop_requested = False
            self.last_event = "training"
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._loop, daemon=True)
                self.thread.start()

    def pause(self) -> None:
        with self.lock:
            self.running = False
            self.last_event = "paused"

    def reset(self) -> None:
        with self.training_lock:
            self._reset()

    def _reset(self) -> None:
        with self.lock:
            self.policy = AfterstateValue(self.policy.dim)
            self.target_policy = self.policy.clone()
            self.best_policy = self.policy.clone()
            self.episode = 0
            self.best_score = 0
            self.best_lines = 0
            self.history = []
            self.latest_info = {}
            self.latest_replay = []
            self.last_eval = {}
            self.last_guard = {}
            self.last_event = "reset"
        self._save_checkpoint()

    def update_config(self, payload: dict) -> None:
        with self.lock:
            self.learning_rate = max(0.0005, min(0.2, float(payload.get("learning_rate", self.learning_rate))))
            self.gamma = max(0.8, min(0.999, float(payload.get("gamma", self.gamma))))
            self.epsilon = max(0.0, min(0.5, float(payload.get("epsilon", self.epsilon))))
            self.temperature = max(0.0, min(1.0, float(payload.get("temperature", self.temperature))))
            self.target_tau = max(0.001, min(0.25, float(payload.get("target_tau", self.target_tau))))
            self.elite_anchor = max(0.0, min(0.25, float(payload.get("elite_anchor", self.elite_anchor))))
            self.lookahead_weight = max(0.0, min(0.95, float(payload.get("lookahead_weight", self.lookahead_weight))))
            self.lookahead_candidates = max(0, min(40, int(payload.get("lookahead_candidates", self.lookahead_candidates))))
            self.lookahead_include_hold = bool(payload.get("lookahead_include_hold", self.lookahead_include_hold))
            self.last_event = "settings updated"

    def snapshot(self) -> dict:
        with self.lock:
            played = len(self.history)
            avg_score = sum(row.get("score", 0) for row in self.history) / played if played else 0.0
            avg_lines = sum(row.get("lines", 0) for row in self.history) / played if played else 0.0
            avg_tetrises = sum(row.get("tetrises", 0) for row in self.history) / played if played else 0.0
            return {
                "running": self.running,
                "episode": self.episode,
                "last_event": self.last_event,
                "config": {
                    "learning_rate": self.learning_rate,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "temperature": self.temperature,
                    "target_tau": self.target_tau,
                    "elite_anchor": self.elite_anchor,
                    "lookahead_weight": self.lookahead_weight,
                    "lookahead_candidates": self.lookahead_candidates,
                    "lookahead_include_hold": self.lookahead_include_hold,
                },
                "guard": dict(self.last_guard),
                "record": {
                    "played": played,
                    "avg_score": round(avg_score, 2),
                    "avg_lines": round(avg_lines, 2),
                    "avg_tetrises": round(avg_tetrises, 3),
                    "best_score": self.best_score,
                    "best_lines": self.best_lines,
                },
                "latest": dict(self.latest_info),
                "evaluation": dict(self.last_eval),
                "history": self.history[-120:],
                "weights": self.weight_summary(),
                "checkpoint": str(self.checkpoint_path),
                "replay_frames": len(self.latest_replay),
            }

    def latest_replay_payload(self) -> dict:
        with self.lock:
            return {"frames": list(self.latest_replay), "latest": dict(self.latest_info)}

    def weight_summary(self) -> dict:
        names = ["bias", "lines", "height", "max_height", "holes", "bumpiness", "wells", "row_trans", "col_trans", "current", "next"]
        if self.policy.dim >= 15:
            names = [
                "bias",
                "lines",
                "height",
                "max_height",
                "holes",
                "covered_holes",
                "bumpiness",
                "wells",
                "right_well",
                "eroded",
                "tetris_ready",
                "row_trans",
                "col_trans",
                "current",
                "next",
            ]
        return {name: round(value, 4) for name, value in zip(names, self.policy.weights)}

    def close(self) -> None:
        with self.lock:
            self.stop_requested = True
            self.running = False
        self._save_checkpoint()

    def _loop(self) -> None:
        while True:
            with self.lock:
                if self.stop_requested:
                    return
                active = self.running
            if not active:
                time.sleep(0.08)
                continue
            self.train_episode()
            time.sleep(0.01)

    def _persist(self, metrics: dict, replay: list[dict]) -> None:
        self._save_checkpoint()
        self.replay_path.write_text(json.dumps({"latest": metrics, "frames": replay}, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")

    def _save_checkpoint(self) -> None:
        payload = {
            "episode": self.episode,
            "best_score": self.best_score,
            "best_lines": self.best_lines,
            "config": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "temperature": self.temperature,
                "target_tau": self.target_tau,
                "elite_anchor": self.elite_anchor,
                "lookahead_weight": self.lookahead_weight,
                "lookahead_candidates": self.lookahead_candidates,
                "lookahead_include_hold": self.lookahead_include_hold,
            },
            "policy": self.policy.to_json(),
            "target_policy": self.target_policy.to_json(),
            "best_policy": self.best_policy.to_json(),
        }
        self.checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        try:
            payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            self.episode = int(payload.get("episode", 0))
            self.best_score = int(payload.get("best_score", 0))
            self.best_lines = int(payload.get("best_lines", 0))
            config = dict(payload.get("config") or {})
            migrating_old_config = "elite_anchor" not in config
            self.learning_rate = float(config.get("learning_rate", self.learning_rate))
            self.gamma = float(config.get("gamma", self.gamma))
            self.epsilon = float(config.get("epsilon", self.epsilon))
            self.temperature = float(config.get("temperature", self.temperature))
            self.target_tau = float(config.get("target_tau", self.target_tau))
            self.elite_anchor = float(config.get("elite_anchor", self.elite_anchor))
            self.lookahead_weight = float(config.get("lookahead_weight", self.lookahead_weight))
            self.lookahead_candidates = int(config.get("lookahead_candidates", self.lookahead_candidates))
            self.lookahead_include_hold = bool(config.get("lookahead_include_hold", self.lookahead_include_hold))
            if migrating_old_config:
                self.learning_rate = min(self.learning_rate, 0.0005)
                self.epsilon = min(self.epsilon, 0.015)
                self.target_tau = min(self.target_tau, 0.006)
                self.lookahead_weight = max(self.lookahead_weight, 0.1)
                self.lookahead_candidates = max(self.lookahead_candidates, 4)
            self.policy.load_json(dict(payload.get("policy") or {}))
            self.target_policy = self.policy.clone()
            self.target_policy.load_json(dict(payload.get("target_policy") or payload.get("policy") or {}))
            self.best_policy = self.policy.clone()
            self.best_policy.load_json(dict(payload.get("best_policy") or payload.get("policy") or {}))
        except Exception:
            self.episode = 0
