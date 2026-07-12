from __future__ import annotations

import copy
import json
import math
import os
import random
import threading
import time
from pathlib import Path

from tetris_env import UNKNOWN_NEXT_PIECE, TetrisEnv


AFTERSTATE_SEMANTICS = "immediate_reward_once_plus_future_afterstate_v2"
CHECKPOINT_VERSION = 3


class AfterstateValue:
    """Linear value of rewards that happen *after* a candidate placement.

    ``score_move`` adds the reward produced by the candidate placement exactly
    once.  The learned value therefore represents only later rewards (plus any
    terminal correction that was not visible while enumerating the move).
    """

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
        # A greedy evaluation must be a frozen/read-only operation.  Avoid even
        # advancing the policy RNG when exploration is disabled.
        if epsilon > 0.0 and self.random.random() < epsilon:
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

    def clone(self, *, preserve_rng: bool = False) -> "AfterstateValue":
        policy = AfterstateValue(self.dim)
        policy.weights = [float(v) for v in self.weights]
        if preserve_rng:
            policy.random.setstate(self.random.getstate())
        return policy

    def soft_update_from(self, source: "AfterstateValue", tau: float) -> None:
        for i in range(self.dim):
            self.weights[i] = self._clamp_weight(i, self.weights[i] * (1.0 - tau) + source.weights[i] * tau)

    def to_json(self) -> dict:
        return {"dim": self.dim, "weights": [float(value) for value in self.weights]}

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
        self.best_checkpoint_path = self.runtime / "tetris_policy.best.json"
        self.best_score_checkpoint_path = self.runtime / "tetris_policy.best_score.json"
        self.metrics_path = self.runtime / "training_metrics.jsonl"
        self.replay_path = self.runtime / "latest_replay.json"
        self.eval_path = self.runtime / "latest_evaluation.json"
        self.lock = threading.RLock()
        self.training_lock = threading.RLock()
        self.running = False
        self.stop_requested = False
        self.thread: threading.Thread | None = None
        self.episode = 0
        self.rollout = 0
        self.learning_rate = 0.0005
        self.gamma = 0.985
        self.epsilon = 0.015
        self.temperature = 0.05
        self.target_tau = 0.006
        self.elite_anchor = 0.035
        self.lookahead_weight = 0.1
        self.lookahead_candidates = 4
        self.lookahead_include_hold = False
        self.train_max_pieces = 360
        self.eval_max_pieces = 900
        self.guard_max_pieces = 240
        self.background_guard_episodes = 10
        self.background_eval_episodes = 4
        self.guard_seed_base = 1_000_000_000
        self.promotion_seed_base = 930_000
        self.evaluation_seed_base = 980_000
        self.guard_min_effect = 1.0
        self.promotion_eval_episodes = 8
        self.promotion_min_effect = 1.0
        self.best_score = 0
        self.best_lines = 0
        self.best_guard_objective = -math.inf
        self.policy_guard_objective = -math.inf
        self.guard_benchmark: dict = {}
        self.promotion_benchmark: dict = {}
        self.history: list[dict] = []
        self.latest_info: dict = {}
        self.latest_replay: list[dict] = []
        self.last_eval: dict = {}
        self.last_guard: dict = {}
        self.last_event = "ready"
        self.loaded_checkpoint: dict = {
            "source": "built_in_default",
            "path": "",
            "protected": False,
            "policy": self.policy.to_json(),
            "rejected": [],
        }
        self._load_checkpoint()

    def train_episode(self, *, persist: bool = True) -> dict:
        """Train one candidate episode through the mandatory acceptance gates.

        The old public entry point mutated the served policy directly and made
        ``persist=False`` an easy guard bypass.  Keep the convenience method,
        but route every call through the same isolated transaction as background
        and manual batch training.
        """

        if not persist:
            raise ValueError("unguarded non-persistent training is disabled")
        with self.lock:
            eval_episodes = self.background_eval_episodes
        return self.train_guarded_batch(1, eval_episodes=eval_episodes)

    def _train_candidate_episode(
        self,
        state: dict,
        config: dict,
        *,
        rollout_index: int,
    ) -> tuple[dict, list[dict]]:
        """Train one isolated candidate episode without touching served state."""
        episode = int(state["episode"])
        state["episode"] = episode + 1
        policy: AfterstateValue = state["policy"]
        target_policy: AfterstateValue = state["target_policy"]
        bootstrap_policy = target_policy.clone()
        lookahead_include_hold = bool(config["lookahead_include_hold"])
        env = TetrisEnv(
            seed=self._training_seed(rollout_index),
            max_pieces=int(config["train_max_pieces"]),
        )
        env.reset()
        total_reward = 0.0
        errors = []
        done = False
        while not done:
            moves = env.legal_moves()
            move = policy.choose(
                moves,
                epsilon=float(config["epsilon"]),
                temperature=float(config["temperature"]),
                next_moves_provider=lambda row: self._future_moves(
                    env,
                    row,
                    include_hold=lookahead_include_hold,
                ),
                lookahead_weight=float(config["lookahead_weight"]),
                lookahead_candidates=int(config["lookahead_candidates"]),
                lookahead_include_hold=lookahead_include_hold,
            )
            if move is None:
                _obs, reward, done, _info = env.step({})
                total_reward += reward
                break
            vector = list(move["vector"])
            _obs, reward, done, _info = env.step(move, check_terminal=False)
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
            bootstrap = max((bootstrap_policy.score_move(row) for row in next_moves), default=0.0)
            target = self.afterstate_td_target(
                reward=reward,
                enumerated_reward=float(move.get("immediate_reward", 0.0)),
                done=done,
                gamma=float(config["gamma"]),
                bootstrap=bootstrap,
            )
            errors.append(
                abs(
                    policy.update(
                        vector,
                        target,
                        learning_rate=float(config["learning_rate"]),
                    )
                )
            )
            target_policy.soft_update_from(policy, float(config["target_tau"]))
            total_reward += reward

        final = env.info()
        final.update(
            {
                "episode": episode,
                "rollout": rollout_index,
                "reward": round(total_reward, 3),
                "td_error": round(sum(errors) / max(1, len(errors)), 4),
                "weights": self._weight_summary_for(policy),
                "replay_frames": len(env.replay),
            }
        )
        previous_best_score = int(state["best_score"])
        if final["score"] > previous_best_score:
            state["best_score"] = final["score"]
            state["best_lines"] = max(int(state["best_lines"]), final["lines"])
            state["best_policy"] = policy.clone(preserve_rng=True)
            final["new_best"] = True
        elif previous_best_score and final["score"] < previous_best_score * 0.55:
            policy.soft_update_from(state["best_policy"], float(config["elite_anchor"]))
            target_policy.soft_update_from(
                state["best_policy"],
                float(config["elite_anchor"]) * 0.5,
            )
            final["elite_anchor"] = True
        else:
            state["best_lines"] = max(int(state["best_lines"]), final["lines"])
        state["latest_info"] = final
        state["latest_replay"] = list(env.replay)
        state["history"].append(final)
        state["history"] = state["history"][-250:]
        return final, list(env.replay)

    @staticmethod
    def afterstate_td_target(
        *,
        reward: float,
        enumerated_reward: float,
        done: bool,
        gamma: float,
        bootstrap: float,
    ) -> float:
        """Return the target for a value attached to the chosen afterstate.

        The placement reward is already included by :meth:`score_move`, so it
        must not also be learned into the afterstate value.  ``reward`` can
        differ from the reward calculated during move enumeration when stepping
        reveals a terminal/top-out penalty; that explicit residual belongs in
        the afterstate value and is retained here.
        """

        transition_correction = float(reward) - float(enumerated_reward)
        future_return = 0.0 if done else float(gamma) * float(bootstrap)
        return transition_correction + future_return

    @staticmethod
    def _training_seed(episode: int) -> int:
        # Training stays below the reserved validation namespace (>= 800,000), even
        # for very long-running jobs.
        return 100_000 + int(episode) % 700_000

    def evaluate(self, episodes: int = 20, *, lookahead_include_hold: bool | None = None) -> dict:
        with self.training_lock:
            return self._evaluate(episodes, lookahead_include_hold=lookahead_include_hold)

    def _evaluate(self, episodes: int = 20, *, lookahead_include_hold: bool | None = None) -> dict:
        episodes = max(1, min(200, int(episodes)))
        with self.lock:
            policy = self.policy.clone()
            lookahead_weight = self.lookahead_weight
            lookahead_candidates = self.lookahead_candidates
            include_hold = self.lookahead_include_hold if lookahead_include_hold is None else bool(lookahead_include_hold)
            eval_max_pieces = self.eval_max_pieces
            seeds = [self.evaluation_seed_base + index for index in range(episodes)]
        evaluation = self._evaluate_policy(
            policy,
            episodes,
            0,
            lookahead_weight,
            lookahead_candidates,
            include_hold,
            capture_replay=True,
            max_pieces=eval_max_pieces,
            seeds=seeds,
        )
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

    def train_guarded_batch(self, episodes: int, *, eval_episodes: int = 4, accept_min_delta: float = 0.0) -> dict:
        with self.training_lock:
            return self._train_guarded_batch(
                episodes,
                eval_episodes=eval_episodes,
                accept_min_delta=accept_min_delta,
            )

    def _train_guarded_batch(self, episodes: int, *, eval_episodes: int = 4, accept_min_delta: float = 0.0) -> dict:
        episodes = max(1, min(500, int(episodes)))
        requested_eval_episodes = max(4, min(24, int(eval_episodes)))
        accept_min_delta = max(0.0, min(100000.0, float(accept_min_delta)))
        with self.lock:
            original = {
                "policy": self.policy.clone(preserve_rng=True),
                "target_policy": self.target_policy.clone(preserve_rng=True),
                "best_policy": self.best_policy.clone(preserve_rng=True),
                "episode": self.episode,
                "best_score": self.best_score,
                "best_lines": self.best_lines,
                "history": list(self.history),
                "latest_info": dict(self.latest_info),
                "latest_replay": list(self.latest_replay),
                "last_eval": dict(self.last_eval),
                "last_guard": dict(self.last_guard),
                "best_guard_objective": self.best_guard_objective,
                "policy_guard_objective": self.policy_guard_objective,
                "guard_benchmark": dict(self.guard_benchmark),
                "promotion_benchmark": dict(self.promotion_benchmark),
            }
            start_rollout = self.rollout
            # Rollouts are attempted work, not accepted model state. Reserving
            # them here guarantees fresh training seeds after every rejection.
            self.rollout += episodes
            protocol = self._guard_protocol(requested_eval_episodes, start_rollout=start_rollout)
            promotion_protocol = self._promotion_protocol()
            promotion_benchmark = (
                dict(original["promotion_benchmark"])
                if dict((original["promotion_benchmark"] or {}).get("protocol") or {}) == dict(promotion_protocol)
                else {}
            )
            minimum_effect = max(self.guard_min_effect, accept_min_delta)
            config = self._config_payload()
        candidate_state = {
            "policy": original["policy"].clone(preserve_rng=True),
            "target_policy": original["target_policy"].clone(preserve_rng=True),
            "best_policy": original["best_policy"].clone(preserve_rng=True),
            "episode": original["episode"],
            "best_score": original["best_score"],
            "best_lines": original["best_lines"],
            "history": list(original["history"]),
            "latest_info": dict(original["latest_info"]),
            "latest_replay": list(original["latest_replay"]),
        }
        guard_seeds = list(protocol["seeds"])
        eval_episodes = len(guard_seeds)
        baseline = self._evaluate_policy(
            original["policy"],
            eval_episodes,
            0,
            float(protocol["lookahead_weight"]),
            int(protocol["lookahead_candidates"]),
            bool(protocol["lookahead_include_hold"]),
            max_pieces=int(protocol["max_pieces"]),
            seeds=guard_seeds,
        )
        latest = None
        candidate_replay: list[dict] = []
        for offset in range(episodes):
            latest, candidate_replay = self._train_candidate_episode(
                candidate_state,
                config,
                rollout_index=start_rollout + offset,
            )
        candidate_policy = candidate_state["policy"].clone(preserve_rng=True)
        candidate_latest = dict(candidate_state["latest_info"])
        candidate = self._evaluate_policy(
            candidate_policy,
            eval_episodes,
            0,
            float(protocol["lookahead_weight"]),
            int(protocol["lookahead_candidates"]),
            bool(protocol["lookahead_include_hold"]),
            max_pieces=int(protocol["max_pieces"]),
            seeds=guard_seeds,
        )
        baseline_objective = self._guard_objective(baseline)
        candidate_objective = self._guard_objective(candidate)
        reference_objective = baseline_objective
        improvement = candidate_objective - reference_objective
        behavior_changed = self._evaluation_signature(candidate) != self._evaluation_signature(baseline)
        gate_passed = behavior_changed and candidate_objective > reference_objective and improvement >= minimum_effect
        promotion = self._promotion_result(
            original_policy=original["policy"],
            original_target_policy=original["target_policy"],
            candidate_policy=candidate_policy,
            protocol=promotion_protocol,
            benchmark=promotion_benchmark,
        ) if gate_passed else {"evaluated": False, "promoted": False}
        accepted = gate_passed and bool(promotion.get("promoted"))
        guard = {
            "accepted": accepted,
            "episodes": episodes,
            "eval_episodes": eval_episodes,
            "requested_eval_episodes": requested_eval_episodes,
            "accept_min_delta": accept_min_delta,
            "minimum_effect": minimum_effect,
            "behavior_changed": behavior_changed,
            "gate_passed": gate_passed,
            "holdout_seeds": guard_seeds,
            "training_seeds": [self._training_seed(start_rollout + index) for index in range(episodes)],
            "baseline_score": baseline["avg_score"],
            "candidate_score": candidate["avg_score"],
            "baseline_tetrises": baseline["avg_tetrises"],
            "candidate_tetrises": candidate["avg_tetrises"],
            "baseline_objective": round(baseline_objective, 2),
            "candidate_objective": round(candidate_objective, 2),
            "reference_objective": round(reference_objective, 2),
            "objective_improvement": round(improvement, 2),
            "rollout_start": start_rollout,
            "promotion": self._public_promotion_result(promotion),
        }
        with self.lock:
            if accepted:
                self.policy = candidate_state["policy"]
                self.target_policy = candidate_state["target_policy"]
                self.best_policy = candidate_state["best_policy"]
                self.episode = int(candidate_state["episode"])
                self.best_score = int(candidate_state["best_score"])
                self.best_lines = int(candidate_state["best_lines"])
                self.history = list(candidate_state["history"])
                self.latest_info = dict(candidate_state["latest_info"])
                self.latest_replay = list(candidate_state["latest_replay"])
                self.policy_guard_objective = candidate_objective
                self.guard_benchmark = self._benchmark_payload(protocol, candidate_objective, candidate)
                promotion_objective = float(promotion["candidate_objective"])
                self.best_guard_objective = promotion_objective
                self.promotion_benchmark = self._benchmark_payload(
                    promotion_protocol,
                    promotion_objective,
                    promotion["candidate"],
                )
                self._save_best_checkpoint(
                    objective=promotion_objective,
                    policy=self.policy,
                    target_policy=self.target_policy,
                    allow_protocol_migration=True,
                )
                self.last_eval = candidate
                self.last_guard = guard
                self.last_event = f"guard accepted {episodes}: {candidate['avg_score']} avg score"
                self.eval_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
                if latest is None:
                    latest = candidate_latest
                latest["guard"] = guard
                self._persist(latest, candidate_replay)
            else:
                self.last_eval = baseline
                self.policy_guard_objective = baseline_objective
                self.guard_benchmark = self._benchmark_payload(protocol, baseline_objective, baseline)
                if (
                    gate_passed
                    and not promotion_benchmark
                    and promotion.get("reference_evaluation")
                ):
                    reference_objective_value = float(promotion["reference_objective"])
                    self.best_guard_objective = reference_objective_value
                    self.promotion_benchmark = self._benchmark_payload(
                        promotion_protocol,
                        reference_objective_value,
                        promotion["reference_evaluation"],
                    )
                    self._save_best_checkpoint(
                        objective=reference_objective_value,
                        policy=promotion["reference_policy"],
                        target_policy=promotion["reference_target_policy"],
                        allow_protocol_migration=True,
                    )
                else:
                    self.best_guard_objective = original["best_guard_objective"]
                    self.promotion_benchmark = dict(original["promotion_benchmark"])
                self.last_guard = guard
                reason = "rotating gate" if not gate_passed else "fixed canonical promotion validation"
                self.last_event = f"rejected by {reason}: candidate {candidate_objective:.2f}"
                self.eval_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
                self._save_checkpoint()
        return {"accepted": accepted, "guard": guard, "baseline": baseline, "candidate": candidate, "latest": latest or {}}

    def _promotion_protocol(self) -> dict:
        saved = dict((self.promotion_benchmark or {}).get("protocol") or {})
        saved_seeds = list(saved.get("seeds") or [])
        try:
            normalized = [int(seed) for seed in saved_seeds]
            if (
                len(normalized) >= 4
                and len(set(normalized)) == len(normalized)
                and all(900_000 <= seed < 950_000 for seed in normalized)
            ):
                saved["version"] = 1
                saved["kind"] = "fixed_canonical_promotion_validation"
                saved["seeds"] = normalized
                saved["max_pieces"] = max(40, min(1200, int(saved["max_pieces"])))
                saved["lookahead_weight"] = max(0.0, min(0.95, float(saved["lookahead_weight"])))
                saved["lookahead_candidates"] = max(0, min(40, int(saved["lookahead_candidates"])))
                saved["lookahead_include_hold"] = bool(saved["lookahead_include_hold"])
                saved["minimum_effect"] = max(
                    0.0001,
                    min(100000.0, float(saved.get("minimum_effect", self.promotion_min_effect))),
                )
                return saved
        except (KeyError, TypeError, ValueError):
            pass
        episodes = max(4, min(24, int(self.promotion_eval_episodes)))
        return {
            "version": 1,
            "kind": "fixed_canonical_promotion_validation",
            "seeds": [self.promotion_seed_base + index for index in range(episodes)],
            "max_pieces": self.guard_max_pieces,
            "lookahead_weight": self.lookahead_weight,
            "lookahead_candidates": self.lookahead_candidates,
            "lookahead_include_hold": self.lookahead_include_hold,
            "minimum_effect": self.promotion_min_effect,
        }

    def _promotion_result(
        self,
        *,
        original_policy: AfterstateValue,
        original_target_policy: AfterstateValue,
        candidate_policy: AfterstateValue,
        protocol: dict,
        benchmark: dict,
    ) -> dict:
        if dict((benchmark or {}).get("protocol") or {}) != dict(protocol):
            benchmark = {}
        seeds = list(protocol["seeds"])
        args = {
            "episodes": len(seeds),
            "start": 0,
            "lookahead_weight": float(protocol["lookahead_weight"]),
            "lookahead_candidates": int(protocol["lookahead_candidates"]),
            "lookahead_include_hold": bool(protocol["lookahead_include_hold"]),
            "max_pieces": int(protocol["max_pieces"]),
            "seeds": seeds,
        }
        baseline = self._evaluate_policy(original_policy, **args)
        candidate = self._evaluate_policy(candidate_policy, **args)
        baseline_objective = self._guard_objective(baseline)
        candidate_objective = self._guard_objective(candidate)
        reference_objective = baseline_objective
        reference_evaluation = baseline
        reference_policy = original_policy.clone(preserve_rng=True)
        reference_target_policy = original_target_policy.clone(preserve_rng=True)

        saved_objective = self._benchmark_objective(benchmark)
        saved_evaluation = dict((benchmark or {}).get("evaluation") or {})
        if math.isfinite(saved_objective) and saved_objective > reference_objective:
            reference_objective = saved_objective
            if saved_evaluation:
                reference_evaluation = saved_evaluation
        elif not benchmark:
            # During a v2/legacy migration, evaluate the existing protected file
            # on the new promotion protocol instead of comparing old objectives.
            protected = self._read_checkpoint(self.best_checkpoint_path, compatible_only=True)
            if protected:
                protected_policy = AfterstateValue(original_policy.dim)
                protected_policy.load_json(dict(protected.get("policy") or {}))
                protected_target = protected_policy.clone()
                protected_target.load_json(
                    dict(protected.get("target_policy") or protected.get("policy") or {})
                )
                protected_evaluation = self._evaluate_policy(protected_policy, **args)
                protected_objective = self._guard_objective(protected_evaluation)
                if protected_objective > reference_objective:
                    reference_objective = protected_objective
                    reference_evaluation = protected_evaluation
                    reference_policy = protected_policy
                    reference_target_policy = protected_target

        minimum_effect = float(protocol["minimum_effect"])
        improvement = candidate_objective - reference_objective
        behavior_changed = (
            self._evaluation_signature(candidate)
            != self._evaluation_signature(reference_evaluation)
        )
        promoted = (
            behavior_changed
            and candidate_objective > reference_objective
            and improvement >= minimum_effect
        )
        return {
            "evaluated": True,
            "promoted": promoted,
            "seeds": seeds,
            "minimum_effect": minimum_effect,
            "behavior_changed": behavior_changed,
            "baseline": baseline,
            "candidate": candidate,
            "baseline_objective": baseline_objective,
            "candidate_objective": candidate_objective,
            "reference_objective": reference_objective,
            "objective_improvement": improvement,
            "reference_evaluation": reference_evaluation,
            "reference_policy": reference_policy,
            "reference_target_policy": reference_target_policy,
        }

    @staticmethod
    def _public_promotion_result(result: dict) -> dict:
        if not result.get("evaluated"):
            return {"evaluated": False, "promoted": False}
        return {
            "evaluated": True,
            "promoted": bool(result.get("promoted")),
            "behavior_changed": bool(result.get("behavior_changed")),
            "holdout_seeds": list(result.get("seeds") or []),
            "minimum_effect": round(float(result.get("minimum_effect", 0.0)), 4),
            "baseline_objective": round(float(result.get("baseline_objective", 0.0)), 2),
            "candidate_objective": round(float(result.get("candidate_objective", 0.0)), 2),
            "reference_objective": round(float(result.get("reference_objective", 0.0)), 2),
            "objective_improvement": round(float(result.get("objective_improvement", 0.0)), 2),
        }

    def _guard_protocol(self, requested_eval_episodes: int, *, start_rollout: int) -> dict:
        eval_episodes = max(4, min(24, int(requested_eval_episodes)))
        # Allocate a 24-seed namespace block per attempted rollout. Rejected
        # candidates therefore never see the same paired gate again, while the
        # fixed canonical promotion validation below remains comparable.
        seeds = [
            self.guard_seed_base + int(start_rollout) * 24 + index
            for index in range(eval_episodes)
        ]
        return {
            "version": 1,
            "kind": "rotating_paired_gate",
            "rollout": int(start_rollout),
            "seeds": seeds,
            "max_pieces": self.guard_max_pieces,
            "lookahead_weight": self.lookahead_weight,
            "lookahead_candidates": self.lookahead_candidates,
            "lookahead_include_hold": self.lookahead_include_hold,
        }

    @staticmethod
    def _benchmark_objective(benchmark: dict) -> float:
        try:
            objective = float((benchmark or {}).get("objective"))
        except (TypeError, ValueError):
            return -math.inf
        return objective if math.isfinite(objective) else -math.inf

    @staticmethod
    def _benchmark_payload(protocol: dict, objective: float, evaluation: dict) -> dict:
        return {
            "protocol": dict(protocol),
            "objective": float(objective),
            "evaluation": dict(evaluation),
        }

    @staticmethod
    def _evaluation_signature(evaluation: dict) -> tuple:
        return tuple(
            (row.get("score"), row.get("lines"), row.get("pieces"), row.get("tetrises"))
            for row in evaluation.get("rows", [])
        )

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
        max_pieces: int | None = None,
        seeds: list[int] | tuple[int, ...] | None = None,
    ) -> dict:
        selected_seeds = list(seeds) if seeds is not None else [800_000 + int(start) + index for index in range(episodes)]
        if not selected_seeds:
            raise ValueError("evaluation requires at least one seed")
        episodes = len(selected_seeds)
        # ``choose`` owns an RNG for exploration.  Evaluate a clone so even a
        # future scoring change cannot mutate the training policy or its RNG.
        eval_policy = policy.clone()
        rows = []
        latest_replay = []
        piece_limit = max(40, min(1200, int(max_pieces or 900)))
        for seed in selected_seeds:
            env = TetrisEnv(seed=int(seed), max_pieces=piece_limit)
            env.reset()
            done = False
            total_reward = 0.0
            while not done:
                move = eval_policy.choose(
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
                    "seed": int(seed),
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
            "seeds": [int(seed) for seed in selected_seeds],
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
            "max_pieces": piece_limit,
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
            self.rollout = 0
            self.best_score = 0
            self.best_lines = 0
            self.best_guard_objective = -math.inf
            self.policy_guard_objective = -math.inf
            self.guard_benchmark = {}
            self.promotion_benchmark = {}
            self.history = []
            self.latest_info = {}
            self.latest_replay = []
            self.last_eval = {}
            self.last_guard = {}
            self.last_event = "reset"
            self.loaded_checkpoint = {
                "source": "built_in_default",
                "path": "",
                "protected": False,
                "policy": self.policy.to_json(),
                "rejected": [],
            }
        self.best_checkpoint_path.unlink(missing_ok=True)
        self.best_score_checkpoint_path.unlink(missing_ok=True)
        self._save_checkpoint()

    def update_config(self, payload: dict) -> None:
        with self.training_lock:
            with self.lock:
                self._apply_config(payload)
                self.last_event = "settings updated"
            # Candidate policies live only in local guarded-batch state, so this
            # can persist settings without ever serializing an unaccepted model.
            self._save_checkpoint()

    def _apply_config(self, payload: dict) -> None:
        self._commit_config(self._normalized_config(payload))

    def _normalized_config(self, payload: dict, *, base: dict | None = None) -> dict:
        """Parse every setting before any live field is changed."""

        if not isinstance(payload, dict):
            raise TypeError("checkpoint config must be an object")
        values = dict(base or self._config_payload())

        def bounded_float(name: str, low: float, high: float) -> float:
            value = float(payload.get(name, values[name]))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            return max(low, min(high, value))

        def bounded_int(name: str, low: int, high: int) -> int:
            value = int(payload.get(name, values[name]))
            return max(low, min(high, value))

        values.update(
            {
                "learning_rate": bounded_float("learning_rate", 0.0005, 0.2),
                "gamma": bounded_float("gamma", 0.8, 0.999),
                "epsilon": bounded_float("epsilon", 0.0, 0.5),
                "temperature": bounded_float("temperature", 0.0, 1.0),
                "target_tau": bounded_float("target_tau", 0.001, 0.25),
                "elite_anchor": bounded_float("elite_anchor", 0.0, 0.25),
                "lookahead_weight": bounded_float("lookahead_weight", 0.0, 0.95),
                "lookahead_candidates": bounded_int("lookahead_candidates", 0, 40),
                "lookahead_include_hold": bool(
                    payload.get("lookahead_include_hold", values["lookahead_include_hold"])
                ),
                "train_max_pieces": bounded_int("train_max_pieces", 40, 1200),
                "eval_max_pieces": bounded_int("eval_max_pieces", 40, 1200),
                "guard_max_pieces": bounded_int("guard_max_pieces", 40, 1200),
                "background_guard_episodes": bounded_int("background_guard_episodes", 4, 20),
                "background_eval_episodes": bounded_int("background_eval_episodes", 4, 24),
                "guard_seed_base": bounded_int("guard_seed_base", 1_000_000_000, 2_000_000_000),
                "promotion_seed_base": bounded_int("promotion_seed_base", 900_000, 949_000),
                "evaluation_seed_base": bounded_int("evaluation_seed_base", 960_000, 2_000_000_000),
                "guard_min_effect": bounded_float("guard_min_effect", 0.0001, 100000.0),
                "promotion_eval_episodes": bounded_int("promotion_eval_episodes", 4, 24),
                "promotion_min_effect": bounded_float("promotion_min_effect", 0.0001, 100000.0),
            }
        )
        return values

    def _commit_config(self, values: dict) -> None:
        for name in self._config_payload():
            setattr(self, name, values[name])

    def _config_payload(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "temperature": self.temperature,
            "target_tau": self.target_tau,
            "elite_anchor": self.elite_anchor,
            "lookahead_weight": self.lookahead_weight,
            "lookahead_candidates": self.lookahead_candidates,
            "lookahead_include_hold": self.lookahead_include_hold,
            "train_max_pieces": self.train_max_pieces,
            "eval_max_pieces": self.eval_max_pieces,
            "guard_max_pieces": self.guard_max_pieces,
            "background_guard_episodes": self.background_guard_episodes,
            "background_eval_episodes": self.background_eval_episodes,
            "guard_seed_base": self.guard_seed_base,
            "promotion_seed_base": self.promotion_seed_base,
            "evaluation_seed_base": self.evaluation_seed_base,
            "guard_min_effect": self.guard_min_effect,
            "promotion_eval_episodes": self.promotion_eval_episodes,
            "promotion_min_effect": self.promotion_min_effect,
        }

    def snapshot(self) -> dict:
        with self.lock:
            played = len(self.history)
            avg_score = sum(row.get("score", 0) for row in self.history) / played if played else 0.0
            avg_lines = sum(row.get("lines", 0) for row in self.history) / played if played else 0.0
            avg_tetrises = sum(row.get("tetrises", 0) for row in self.history) / played if played else 0.0
            return {
                "running": self.running,
                "episode": self.episode,
                "rollout": self.rollout,
                "last_event": self.last_event,
                "training_semantics": AFTERSTATE_SEMANTICS,
                "config": self._config_payload(),
                "guard": dict(self.last_guard),
                "guard_benchmark": dict(self.guard_benchmark),
                "promotion_benchmark": dict(self.promotion_benchmark),
                "record": {
                    "played": played,
                    "avg_score": round(avg_score, 2),
                    "avg_lines": round(avg_lines, 2),
                    "avg_tetrises": round(avg_tetrises, 3),
                    "best_score": self.best_score,
                    "best_lines": self.best_lines,
                    "best_guard_objective": None if not math.isfinite(self.best_guard_objective) else round(self.best_guard_objective, 2),
                    "policy_guard_objective": None if not math.isfinite(self.policy_guard_objective) else round(self.policy_guard_objective, 2),
                },
                "latest": dict(self.latest_info),
                "evaluation": dict(self.last_eval),
                "history": self.history[-120:],
                "weights": self.weight_summary(),
                "checkpoint": str(self.checkpoint_path),
                "best_checkpoint": str(self.best_checkpoint_path),
                "best_score_checkpoint": str(self.best_score_checkpoint_path),
                "loaded_checkpoint": self.checkpoint_selection(),
                "replay_frames": len(self.latest_replay),
            }

    def checkpoint_selection(self) -> dict:
        """Describe the exact artifact/policy chosen by the real loader."""

        with self.lock:
            selected = dict(self.loaded_checkpoint)
            selected["policy"] = dict(selected.get("policy") or {})
            selected["rejected"] = [
                dict(row) for row in list(selected.get("rejected") or [])
            ]
            return selected

    def latest_replay_payload(self) -> dict:
        with self.lock:
            return {"frames": list(self.latest_replay), "latest": dict(self.latest_info)}

    def weight_summary(self) -> dict:
        return self._weight_summary_for(self.policy)

    @staticmethod
    def _weight_summary_for(policy: AfterstateValue) -> dict:
        names = ["bias", "lines", "height", "max_height", "holes", "bumpiness", "wells", "row_trans", "col_trans", "current", "next"]
        if policy.dim >= 15:
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
        return {name: round(value, 4) for name, value in zip(names, policy.weights)}

    def close(self) -> None:
        with self.lock:
            self.stop_requested = True
            self.running = False
            thread = self.thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join()
        with self.training_lock:
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
            with self.lock:
                batch_episodes = self.background_guard_episodes
                eval_episodes = self.background_eval_episodes
                official = self._capture_served_state_locked()
            try:
                self.train_guarded_batch(
                    batch_episodes,
                    eval_episodes=eval_episodes,
                    accept_min_delta=0.0,
                )
            except Exception as exc:
                with self.lock:
                    self._restore_served_state_locked(official)
                    self.running = False
                    self.last_event = "error"
                    self.last_guard = {
                        "accepted": False,
                        "reason": f"background transaction failed: {type(exc).__name__}: {exc}",
                    }
                try:
                    self._save_checkpoint()
                except Exception:
                    # The in-memory served policy is already restored and the
                    # previous atomic checkpoint remains authoritative.
                    pass
                return
            time.sleep(0.01)

    def _capture_served_state_locked(self) -> dict:
        return {
            "policy": self.policy.clone(preserve_rng=True),
            "target_policy": self.target_policy.clone(preserve_rng=True),
            "best_policy": self.best_policy.clone(preserve_rng=True),
            "episode": self.episode,
            "rollout": self.rollout,
            "best_score": self.best_score,
            "best_lines": self.best_lines,
            "best_guard_objective": self.best_guard_objective,
            "policy_guard_objective": self.policy_guard_objective,
            "guard_benchmark": copy.deepcopy(self.guard_benchmark),
            "promotion_benchmark": copy.deepcopy(self.promotion_benchmark),
            "history": copy.deepcopy(self.history),
            "latest_info": copy.deepcopy(self.latest_info),
            "latest_replay": copy.deepcopy(self.latest_replay),
            "last_eval": copy.deepcopy(self.last_eval),
            "last_guard": copy.deepcopy(self.last_guard),
        }

    def _restore_served_state_locked(self, state: dict) -> None:
        self.policy = state["policy"].clone(preserve_rng=True)
        self.target_policy = state["target_policy"].clone(preserve_rng=True)
        self.best_policy = state["best_policy"].clone(preserve_rng=True)
        self.episode = int(state["episode"])
        self.rollout = int(state["rollout"])
        self.best_score = int(state["best_score"])
        self.best_lines = int(state["best_lines"])
        self.best_guard_objective = float(state["best_guard_objective"])
        self.policy_guard_objective = float(state["policy_guard_objective"])
        self.guard_benchmark = copy.deepcopy(state["guard_benchmark"])
        self.promotion_benchmark = copy.deepcopy(state["promotion_benchmark"])
        self.history = copy.deepcopy(state["history"])
        self.latest_info = copy.deepcopy(state["latest_info"])
        self.latest_replay = copy.deepcopy(state["latest_replay"])
        self.last_eval = copy.deepcopy(state["last_eval"])
        self.last_guard = copy.deepcopy(state["last_guard"])

    def _persist(self, metrics: dict, replay: list[dict]) -> None:
        self._save_checkpoint()
        self.replay_path.write_text(json.dumps({"latest": metrics, "frames": replay}, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict) -> None:
        temp_path = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _save_checkpoint(self) -> None:
        with self.lock:
            payload = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "training_semantics": AFTERSTATE_SEMANTICS,
                "episode": self.episode,
                "rollout": self.rollout,
                "best_score": self.best_score,
                "best_lines": self.best_lines,
                "best_guard_objective": None if not math.isfinite(self.best_guard_objective) else self.best_guard_objective,
                "policy_guard_objective": None if not math.isfinite(self.policy_guard_objective) else self.policy_guard_objective,
                "guard_benchmark": self.guard_benchmark,
                "promotion_benchmark": self.promotion_benchmark,
                "config": self._config_payload(),
                "policy": self.policy.to_json(),
                "target_policy": self.target_policy.to_json(),
                "best_policy": self.best_policy.to_json(),
            }
        self._atomic_write_json(self.checkpoint_path, payload)
        if self.best_score > 0:
            self._save_best_score_checkpoint()

    def _save_best_checkpoint(
        self,
        *,
        objective: float | None = None,
        policy: AfterstateValue | None = None,
        target_policy: AfterstateValue | None = None,
        allow_protocol_migration: bool = False,
    ) -> bool:
        try:
            objective_value = float(objective)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(objective_value):
            return False
        existing = self._read_checkpoint(self.best_checkpoint_path, compatible_only=True)
        existing_objective = self._checkpoint_objective(existing or {}, prefer_policy=False)
        existing_protocol = dict(((existing or {}).get("promotion_benchmark") or {}).get("protocol") or {})
        current_protocol = dict((self.promotion_benchmark or {}).get("protocol") or {})
        # The protected file is monotonic.  A caller cannot replace it with an
        # equal or worse policy under the same benchmark.  A legacy/different
        # protocol may be replaced only after the guard has established and
        # stored the new protocol by comparing both policies on it.
        same_protocol = existing_protocol == current_protocol and bool(current_protocol)
        migrating_protocol = bool(allow_protocol_migration and current_protocol and not same_protocol)
        if math.isfinite(existing_objective) and not migrating_protocol and objective_value <= existing_objective:
            return False
        save_policy = policy or self.policy
        save_target = target_policy or self.target_policy
        saves_current_policy = save_policy.weights == self.policy.weights
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "training_semantics": AFTERSTATE_SEMANTICS,
            "episode": self.episode,
            "rollout": self.rollout,
            "objective": objective_value,
            "best_score": self.best_score,
            "best_lines": self.best_lines,
            "best_guard_objective": max(self.best_guard_objective, objective_value),
            "policy_guard_objective": self.policy_guard_objective if saves_current_policy else None,
            "guard_benchmark": self.guard_benchmark if saves_current_policy else {},
            "promotion_benchmark": self.promotion_benchmark,
            "config": self._config_payload(),
            "policy": save_policy.to_json(),
            "target_policy": save_target.to_json(),
            "best_policy": save_policy.to_json(),
        }
        self._atomic_write_json(self.best_checkpoint_path, payload)
        return True

    def _save_best_score_checkpoint(self) -> None:
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "training_semantics": AFTERSTATE_SEMANTICS,
            "episode": self.episode,
            "rollout": self.rollout,
            "best_score": self.best_score,
            "best_lines": self.best_lines,
            "config": self._config_payload(),
            "guard_benchmark": self.guard_benchmark,
            "promotion_benchmark": self.promotion_benchmark,
            "policy": self.best_policy.to_json(),
            "target_policy": self.best_policy.to_json(),
            "best_policy": self.best_policy.to_json(),
        }
        self._atomic_write_json(self.best_score_checkpoint_path, payload)

    def _load_checkpoint(self) -> None:
        entries = (
            ("main", self.checkpoint_path, False),
            ("protected_best", self.best_checkpoint_path, True),
        )
        parsed: dict[str, dict] = {}
        rejected: list[dict] = []
        for source, path, protected in entries:
            raw = self._read_checkpoint(path)
            if raw is None:
                if path.exists():
                    rejected.append({"source": source, "path": str(path), "reason": "unreadable JSON"})
                continue
            if not self._checkpoint_is_compatible(raw, require_promotion=protected):
                rejected.append({"source": source, "path": str(path), "reason": "incompatible schema or semantics"})
                continue
            try:
                parsed[source] = self._parse_checkpoint_payload(raw, protected=protected)
                parsed[source]["path"] = str(path)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                rejected.append(
                    {
                        "source": source,
                        "path": str(path),
                        "reason": f"invalid payload: {type(exc).__name__}: {exc}",
                    }
                )

        selected = parsed.get("protected_best") or parsed.get("main")
        if selected is None:
            self.loaded_checkpoint = {
                "source": "built_in_default",
                "path": "",
                "protected": False,
                "policy": self.policy.to_json(),
                "rejected": rejected,
            }
            if rejected:
                self.last_event = "legacy/incompatible checkpoint quarantined"
            return

        valid_states = list(parsed.values())
        effective_config = dict((parsed.get("main") or selected)["config"])
        if "elite_anchor" not in selected["raw_config"]:
            effective_config["learning_rate"] = min(effective_config["learning_rate"], 0.0005)
            effective_config["epsilon"] = min(effective_config["epsilon"], 0.015)
            effective_config["target_tau"] = min(effective_config["target_tau"], 0.006)
            effective_config["lookahead_weight"] = max(effective_config["lookahead_weight"], 0.1)
            effective_config["lookahead_candidates"] = max(effective_config["lookahead_candidates"], 4)

        # Nothing above mutates the trainer. Commit only after every selected
        # field and every aggregate counter has been parsed successfully.
        episode = max(row["episode"] for row in valid_states)
        rollout = max(row["rollout"] for row in valid_states)
        best_score = max(row["best_score"] for row in valid_states)
        best_lines = max(row["best_lines"] for row in valid_states)
        with self.lock:
            self.policy = selected["policy"]
            self.target_policy = selected["target_policy"]
            self.best_policy = selected["best_policy"]
            self.episode = episode
            self.rollout = rollout
            self.best_score = best_score
            self.best_lines = best_lines
            self.guard_benchmark = selected["guard_benchmark"]
            self.promotion_benchmark = selected["promotion_benchmark"]
            self.policy_guard_objective = selected["policy_guard_objective"]
            self.best_guard_objective = selected["best_guard_objective"]
            self._commit_config(effective_config)
            source = "protected_best" if selected is parsed.get("protected_best") else "main"
            self.loaded_checkpoint = {
                "source": source,
                "path": selected["path"],
                "protected": source == "protected_best",
                "policy": self.policy.to_json(),
                "rejected": rejected,
            }
            self.last_event = f"loaded {source} checkpoint"

    def _parse_checkpoint_payload(self, payload: dict, *, protected: bool) -> dict:
        raw_config = payload.get("config")
        if not isinstance(raw_config, dict):
            raise TypeError("config must be an object")
        config = self._normalized_config(raw_config, base=self._config_payload())
        policy = self._policy_from_checkpoint(payload.get("policy"), field="policy")
        target_policy = self._policy_from_checkpoint(
            payload.get("target_policy"),
            field="target_policy",
        )
        best_policy = self._policy_from_checkpoint(
            payload.get("best_policy"),
            field="best_policy",
        )

        guard_benchmark = self._validated_benchmark(
            payload.get("guard_benchmark"),
            field="guard_benchmark",
            required=False,
            promotion=False,
        )
        promotion_benchmark = self._validated_benchmark(
            payload.get("promotion_benchmark"),
            field="promotion_benchmark",
            required=protected,
            promotion=True,
        )

        episode = self._nonnegative_checkpoint_int(payload, "episode")
        rollout = max(episode, self._nonnegative_checkpoint_int(payload, "rollout", default=episode))
        best_score = self._nonnegative_checkpoint_int(payload, "best_score")
        best_lines = self._nonnegative_checkpoint_int(payload, "best_lines")
        policy_guard_objective = self._optional_checkpoint_objective(
            payload,
            "policy_guard_objective",
        )
        best_guard_objective = self._checkpoint_objective(payload, prefer_policy=False)
        if promotion_benchmark and not math.isfinite(best_guard_objective):
            raise ValueError("promotion benchmark requires a finite best objective")
        if protected:
            artifact_objective = float(payload.get("objective"))
            benchmark_objective = float(promotion_benchmark["objective"])
            if not math.isfinite(artifact_objective) or not math.isclose(
                artifact_objective,
                benchmark_objective,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("protected objective does not match promotion benchmark")

        return {
            "raw_config": dict(raw_config),
            "config": config,
            "policy": policy,
            "target_policy": target_policy,
            "best_policy": best_policy,
            "episode": episode,
            "rollout": rollout,
            "best_score": best_score,
            "best_lines": best_lines,
            "guard_benchmark": guard_benchmark,
            "promotion_benchmark": promotion_benchmark,
            "policy_guard_objective": policy_guard_objective,
            "best_guard_objective": (
                best_guard_objective if promotion_benchmark else -math.inf
            ),
        }

    def _policy_from_checkpoint(self, payload, *, field: str) -> AfterstateValue:
        if not isinstance(payload, dict):
            raise TypeError(f"{field} must be an object")
        if int(payload.get("dim", -1)) != self.policy.dim:
            raise ValueError(f"{field} dimension mismatch")
        raw_weights = payload.get("weights")
        if not isinstance(raw_weights, list) or len(raw_weights) != self.policy.dim:
            raise ValueError(f"{field} must contain exactly {self.policy.dim} weights")
        weights = [float(value) for value in raw_weights]
        if not all(math.isfinite(value) for value in weights):
            raise ValueError(f"{field} weights must be finite")
        policy = AfterstateValue(self.policy.dim)
        policy.load_json({"dim": self.policy.dim, "weights": weights})
        return policy

    @staticmethod
    def _nonnegative_checkpoint_int(payload: dict, field: str, *, default: int = 0) -> int:
        value = int(payload.get(field, default))
        if value < 0:
            raise ValueError(f"{field} must be non-negative")
        return value

    @staticmethod
    def _optional_checkpoint_objective(payload: dict, field: str) -> float:
        raw = payload.get(field)
        if raw is None:
            return -math.inf
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite or null")
        return value

    @staticmethod
    def _validated_benchmark(
        payload,
        *,
        field: str,
        required: bool,
        promotion: bool,
    ) -> dict:
        if payload in (None, {}):
            if required:
                raise ValueError(f"{field} is required")
            return {}
        if not isinstance(payload, dict):
            raise TypeError(f"{field} must be an object")
        protocol = payload.get("protocol")
        evaluation = payload.get("evaluation")
        objective = float(payload.get("objective"))
        if not isinstance(protocol, dict) or not isinstance(evaluation, dict):
            raise TypeError(f"{field} protocol/evaluation must be objects")
        if not math.isfinite(objective):
            raise ValueError(f"{field} objective must be finite")
        if promotion:
            seeds = [int(seed) for seed in list(protocol.get("seeds") or [])]
            if (
                len(seeds) < 4
                or len(set(seeds)) != len(seeds)
                or not all(900_000 <= seed < 950_000 for seed in seeds)
            ):
                raise ValueError("promotion validation seeds are invalid")
            protocol = dict(protocol)
            protocol["version"] = 1
            protocol["kind"] = "fixed_canonical_promotion_validation"
            protocol["seeds"] = seeds
        return {
            "protocol": dict(protocol),
            "objective": objective,
            "evaluation": dict(evaluation),
        }

    @staticmethod
    def _checkpoint_is_compatible(payload: dict | None, *, require_promotion: bool = False) -> bool:
        if not isinstance(payload, dict):
            return False
        try:
            version = int(payload.get("checkpoint_version", 0))
        except (TypeError, ValueError):
            return False
        if version != CHECKPOINT_VERSION:
            return False
        if payload.get("training_semantics") != AFTERSTATE_SEMANTICS:
            return False
        if require_promotion:
            promotion = payload.get("promotion_benchmark")
            if not isinstance(promotion, dict) or not promotion.get("protocol"):
                return False
        return True

    @classmethod
    def _read_checkpoint(cls, path: Path, *, compatible_only: bool = False) -> dict | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if compatible_only and not cls._checkpoint_is_compatible(payload):
            return None
        return payload

    @staticmethod
    def _checkpoint_objective(payload: dict, *, prefer_policy: bool) -> float:
        keys = ("policy_guard_objective", "objective") if prefer_policy else (
            "objective",
            "best_guard_objective",
            "policy_guard_objective",
        )
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                objective = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(objective):
                return objective
        return -math.inf
