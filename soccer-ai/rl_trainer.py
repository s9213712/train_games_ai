from __future__ import annotations

import json
import math
import random
import threading
import time
from pathlib import Path

from soccer_env import ACTION_NAMES, TacticalSoccerEnv


class SoftmaxPolicy:
    def __init__(self, obs_dim: int, action_dim: int, *, seed: int = 7, init_scale: float = 0.0) -> None:
        self.random = random.Random(seed)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        scale = max(0.0, float(init_scale))
        self.weights = []
        for _ in range(action_dim):
            if scale > 0.0:
                self.weights.append([self.random.uniform(-scale, scale) for _ in range(obs_dim)])
            else:
                self.weights.append([0.0 for _ in range(obs_dim)])

    def probabilities(self, obs: list[float]) -> list[float]:
        logits = [sum(w * x for w, x in zip(row, obs)) for row in self.weights]
        offset = max(logits)
        exps = [math.exp(max(-40.0, min(40.0, value - offset))) for value in logits]
        total = sum(exps) or 1.0
        return [value / total for value in exps]

    def sample(self, obs: list[float], *, temperature: float = 1.0) -> tuple[int, list[float]]:
        probs = self.probabilities(obs)
        if temperature != 1.0:
            adjusted = [pow(max(1e-8, p), 1.0 / max(0.05, temperature)) for p in probs]
            total = sum(adjusted)
            probs = [p / total for p in adjusted]
        r = self.random.random()
        acc = 0.0
        for index, prob in enumerate(probs):
            acc += prob
            if r <= acc:
                return index, probs
        return len(probs) - 1, probs

    def greedy(self, obs: list[float]) -> int:
        probs = self.probabilities(obs)
        return max(range(len(probs)), key=lambda idx: probs[idx])

    def update(self, trajectory: list[dict], *, learning_rate: float, gamma: float, baseline: float) -> float:
        returns: list[float] = []
        running = 0.0
        for row in reversed(trajectory):
            running = float(row["reward"]) + gamma * running
            returns.append(running)
        returns.reverse()
        if not returns:
            return baseline
        for row, ret in zip(trajectory, returns):
            advantage = max(-8.0, min(8.0, ret - baseline))
            obs = row["obs"]
            action = int(row["action"])
            probs = row["probs"]
            for action_index in range(self.action_dim):
                scale = (1.0 if action_index == action else 0.0) - probs[action_index]
                for obs_index in range(self.obs_dim):
                    self.weights[action_index][obs_index] += learning_rate * advantage * scale * obs[obs_index]
        episode_return = returns[0]
        return baseline * 0.94 + episode_return * 0.06

    def imitate(self, trajectory: list[dict], *, learning_rate: float, target_action) -> float:
        if learning_rate <= 0.0 or not trajectory:
            return 0.0
        total_loss = 0.0
        for row in trajectory:
            obs = row["obs"]
            action = int(target_action(obs))
            probs = self.probabilities(obs)
            total_loss += -math.log(max(1e-8, probs[action]))
            for action_index in range(self.action_dim):
                scale = (1.0 if action_index == action else 0.0) - probs[action_index]
                for obs_index in range(self.obs_dim):
                    self.weights[action_index][obs_index] += learning_rate * scale * obs[obs_index]
        return total_loss / len(trajectory)

    def top_actions(self, obs: list[float], limit: int = 4) -> list[dict]:
        probs = self.probabilities(obs)
        rows = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
        return [{"action": ACTION_NAMES[index], "prob": round(prob, 3)} for index, prob in rows[:limit]]

    def to_json(self) -> dict:
        return {"obs_dim": self.obs_dim, "action_dim": self.action_dim, "weights": self.weights}

    def clone(self, *, seed: int | None = None) -> "SoftmaxPolicy":
        policy = SoftmaxPolicy(self.obs_dim, self.action_dim, seed=seed if seed is not None else 17)
        policy.weights = [[float(value) for value in row] for row in self.weights]
        return policy

    def load_json(self, payload: dict) -> None:
        weights = payload.get("weights")
        if isinstance(weights, list) and len(weights) == self.action_dim:
            loaded = []
            for row in weights:
                values = [float(v) for v in list(row)[: self.obs_dim]]
                if len(values) < self.obs_dim:
                    values.extend(0.0 for _ in range(self.obs_dim - len(values)))
                loaded.append(values)
            self.weights = loaded


class RLTrainer:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.runtime / "soccer_policy.json"
        self.best_checkpoint_path = self.runtime / "soccer_policy.best.json"
        self.replay_path = self.runtime / "latest_replay.json"
        self.metrics_path = self.runtime / "training_metrics.jsonl"
        self.eval_path = self.runtime / "latest_evaluation.json"
        probe_env = TacticalSoccerEnv(seed=1)
        obs = probe_env.reset()
        self.policy = SoftmaxPolicy(len(obs), len(ACTION_NAMES))
        self.red_policy = SoftmaxPolicy(len(obs), len(ACTION_NAMES), seed=13)
        self.lock = threading.RLock()
        self.training_lock = threading.RLock()
        self.running = False
        self.stop_requested = False
        self.thread: threading.Thread | None = None
        self.episode = 0
        self.baseline = 0.0
        self.learning_rate = 0.0008
        self.gamma = 0.985
        self.self_play = False
        self.league_enabled = True
        self.temperature = 1.15
        self.guard_enabled = True
        self.guard_batch_episodes = 20
        self.guard_eval_episodes = 32
        self.guard_accept_margin = 0.0
        self.guard_opponent = "mixed"
        self.best_guard_objective = -math.inf
        self.coach_enabled = True
        self.coach_rate = 0.001
        self.elo = 1000.0
        self.scripted_elo = 950.0
        self.league_pool: list[dict] = []
        self.last_opponent: dict = {"kind": "scripted", "name": "scripted red", "elo": self.scripted_elo}
        self.history: list[dict] = []
        self.latest_replay: list[dict] = []
        self.latest_info: dict = {}
        self.last_eval: dict = {}
        self.last_guard: dict = {}
        self.last_event = "ready"
        self._load_checkpoint()

    def train_episode(self, *, persist: bool = True) -> dict:
        with self.training_lock:
            return self._train_episode(persist=persist)

    def _train_episode(self, *, persist: bool = True) -> dict:
        with self.lock:
            episode_index = self.episode
            self.episode += 1
            learning_rate = self.learning_rate
            gamma = self.gamma
            self_play = self.self_play
            temperature = self.temperature
            baseline = self.baseline
            coach_enabled = self.coach_enabled
            coach_rate = self.coach_rate
            opponent = self._select_opponent_locked(episode_index, self_play)
        env = TacticalSoccerEnv(seed=10_000 + episode_index)
        obs = env.reset()
        trajectory: list[dict] = []
        red_trajectory: list[dict] = []
        total_reward = 0.0
        done = False
        while not done:
            blue_action, blue_probs = self.policy.sample(obs, temperature=temperature)
            if opponent["kind"] == "current_red":
                red_obs = self._invert_obs(obs)
                red_action, red_probs = self.red_policy.sample(red_obs, temperature=temperature)
            elif opponent["kind"] == "league":
                red_obs = self._invert_obs(obs)
                red_action, _red_probs = opponent["policy"].sample(red_obs, temperature=max(0.55, temperature * 0.8))
                red_probs = []
            else:
                red_action = self._scripted_red(obs)
                red_probs = []
            next_obs, reward, done, info = env.step(blue_action, red_action)
            trajectory.append({"obs": obs, "action": blue_action, "probs": blue_probs, "reward": reward, "info": info})
            if opponent["kind"] == "current_red":
                red_trajectory.append({"obs": self._invert_obs(obs), "action": red_action, "probs": red_probs, "reward": -reward})
            total_reward += reward
            obs = next_obs

        new_baseline = self.policy.update(trajectory, learning_rate=learning_rate, gamma=gamma, baseline=baseline)
        coach_loss = (
            self.policy.imitate(trajectory, learning_rate=coach_rate, target_action=self._coach_blue)
            if coach_enabled
            else 0.0
        )
        if opponent["kind"] == "current_red":
            self.red_policy.update(red_trajectory, learning_rate=learning_rate * 0.7, gamma=gamma, baseline=-baseline)
        final = env.info()
        result = self._result(final)
        reward_totals = self._sum_reward_terms(trajectory)
        final.update(
            {
                "episode": episode_index,
                "reward": round(total_reward, 3),
                "reward_terms": reward_totals,
                "baseline": round(new_baseline, 3),
                "coach_loss": round(coach_loss, 4),
                "result": result,
                "opponent": self._opponent_public(opponent),
                "top_actions": self.policy.top_actions(env.observation()),
            }
        )
        with self.lock:
            self.baseline = new_baseline
            self._update_elo_locked(opponent, result)
            self._maybe_add_league_snapshot_locked(episode_index, final)
            self.latest_info = final
            self.latest_replay = list(env.replay)
            self.history.append(final)
            self.history = self.history[-200:]
            self.last_event = f"episode {episode_index}: {final['result']}"
        if persist:
            self._persist(final, env.replay)
        return final

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
            self.policy = SoftmaxPolicy(self.policy.obs_dim, self.policy.action_dim)
            self.red_policy = SoftmaxPolicy(self.red_policy.obs_dim, self.red_policy.action_dim, seed=13)
            self.episode = 0
            self.baseline = 0.0
            self.elo = 1000.0
            self.scripted_elo = 950.0
            self.league_pool = []
            self.last_opponent = {"kind": "scripted", "name": "scripted red", "elo": self.scripted_elo}
            self.history = []
            self.latest_replay = []
            self.latest_info = {}
            self.last_eval = {}
            self.last_guard = {}
            self.last_event = "reset"
        self._save_checkpoint()

    def update_config(self, payload: dict) -> None:
        with self.lock:
            self.learning_rate = max(0.0001, min(0.02, float(payload.get("learning_rate", self.learning_rate))))
            self.gamma = max(0.8, min(0.999, float(payload.get("gamma", self.gamma))))
            self.temperature = max(0.25, min(2.5, float(payload.get("temperature", self.temperature))))
            self.self_play = bool(payload.get("self_play", self.self_play))
            self.league_enabled = bool(payload.get("league_enabled", self.league_enabled))
            self.guard_enabled = bool(payload.get("guard_enabled", self.guard_enabled))
            self.guard_batch_episodes = max(1, min(100, int(payload.get("guard_batch_episodes", self.guard_batch_episodes))))
            self.guard_eval_episodes = max(4, min(120, int(payload.get("guard_eval_episodes", self.guard_eval_episodes))))
            self.guard_accept_margin = max(-10.0, min(25.0, float(payload.get("guard_accept_margin", self.guard_accept_margin))))
            guard_opponent = str(payload.get("guard_opponent", self.guard_opponent) or self.guard_opponent)
            self.guard_opponent = guard_opponent if guard_opponent in {"mixed", "scripted", "current_red", "league"} else "mixed"
            self.coach_enabled = bool(payload.get("coach_enabled", self.coach_enabled))
            self.coach_rate = max(0.0, min(0.05, float(payload.get("coach_rate", self.coach_rate))))
            self.last_event = "settings updated"

    def snapshot(self) -> dict:
        with self.lock:
            played = len(self.history)
            wins = sum(1 for row in self.history if row.get("result") == "win")
            losses = sum(1 for row in self.history if row.get("result") == "loss")
            draws = sum(1 for row in self.history if row.get("result") == "draw")
            return {
                "running": self.running,
                "episode": self.episode,
                "last_event": self.last_event,
                "config": {
                    "learning_rate": self.learning_rate,
                    "gamma": self.gamma,
                    "temperature": self.temperature,
                    "self_play": self.self_play,
                    "league_enabled": self.league_enabled,
                    "guard_enabled": self.guard_enabled,
                    "guard_batch_episodes": self.guard_batch_episodes,
                    "guard_eval_episodes": self.guard_eval_episodes,
                    "guard_accept_margin": self.guard_accept_margin,
                    "guard_opponent": self.guard_opponent,
                    "coach_enabled": self.coach_enabled,
                    "coach_rate": self.coach_rate,
                },
                "guard": dict(self.last_guard),
                "league": {
                    "elo": round(self.elo, 1),
                    "scripted_elo": round(self.scripted_elo, 1),
                    "pool_size": len(self.league_pool),
                    "last_opponent": dict(self.last_opponent),
                    "pool": [
                        {
                            "id": row["id"],
                            "name": row["name"],
                            "elo": round(row["elo"], 1),
                            "games": row.get("games", 0),
                            "wins": row.get("wins", 0),
                            "losses": row.get("losses", 0),
                            "draws": row.get("draws", 0),
                        }
                        for row in self.league_pool[-6:]
                    ],
                },
                "record": {"wins": wins, "losses": losses, "draws": draws, "played": played, "win_rate": round(wins / played, 3) if played else 0.0},
                "latest": dict(self.latest_info),
                "history": self.history[-80:],
                "evaluation": dict(self.last_eval),
                "checkpoint": str(self.checkpoint_path),
                "best_checkpoint": str(self.best_checkpoint_path),
                "replay_frames": len(self.latest_replay),
            }

    def latest_replay_payload(self) -> dict:
        with self.lock:
            return {"frames": list(self.latest_replay), "latest": dict(self.latest_info)}

    def train_guarded_batch(self, episodes: int, *, eval_episodes: int | None = None, accept_margin: float | None = None) -> dict:
        with self.training_lock:
            return self._train_guarded_batch(episodes, eval_episodes=eval_episodes, accept_margin=accept_margin)

    def _train_guarded_batch(self, episodes: int, *, eval_episodes: int | None = None, accept_margin: float | None = None) -> dict:
        episodes = max(1, min(250, int(episodes)))
        eval_episodes = max(4, min(120, int(eval_episodes if eval_episodes is not None else self.guard_eval_episodes)))
        accept_margin = float(self.guard_accept_margin if accept_margin is None else accept_margin)
        with self.lock:
            original = self._capture_state_locked()
            start_episode = self.episode
            seeds = [830_000 + start_episode * 37 + index for index in range(eval_episodes)]
            guard_opponent = self.guard_opponent
        baseline = self._evaluate_policy(
            original["policy"],
            original["red_policy"],
            original["league_pool"],
            seeds,
            opponent=guard_opponent,
        )
        latest = None
        for _ in range(episodes):
            latest = self.train_episode(persist=False)
        with self.lock:
            candidate_policy = self.policy.clone(seed=71)
            candidate_red = self.red_policy.clone(seed=73)
            candidate_pool = self._clone_league_pool_locked()
            candidate_replay = list(self.latest_replay)
            candidate_latest = dict(self.latest_info)
        candidate = self._evaluate_policy(candidate_policy, candidate_red, candidate_pool, seeds, opponent=guard_opponent)
        baseline_objective = self._guard_objective(baseline)
        candidate_objective = self._guard_objective(candidate)
        accepted = candidate_objective >= baseline_objective + accept_margin
        guard = {
            "accepted": accepted,
            "episodes": episodes,
            "eval_episodes": eval_episodes,
            "opponent": guard_opponent,
            "opponent_distribution": candidate.get("opponent_distribution", {}),
            "accept_margin": round(accept_margin, 3),
            "baseline_objective": round(baseline_objective, 3),
            "candidate_objective": round(candidate_objective, 3),
            "baseline_reward": baseline["avg_reward"],
            "candidate_reward": candidate["avg_reward"],
            "baseline_goal_diff": baseline["avg_goal_diff"],
            "candidate_goal_diff": candidate["avg_goal_diff"],
            "baseline_win_rate": baseline["record"]["win_rate"],
            "candidate_win_rate": candidate["record"]["win_rate"],
            "reward_audit": self._reward_audit(candidate),
        }
        with self.lock:
            if accepted:
                self.last_eval = candidate
                self.last_guard = guard
                if latest is None:
                    latest = candidate_latest
                latest["guard"] = guard
                self.last_event = f"guard accepted {episodes}: {baseline_objective:.2f} -> {candidate_objective:.2f}"
                self._save_best_checkpoint(candidate_objective)
                self._persist(latest, candidate_replay)
            else:
                self._restore_state_locked(original)
                self.last_guard = guard
                self.last_event = f"guard rejected {episodes}: {candidate_objective:.2f} < {baseline_objective:.2f}"
                self._save_checkpoint()
        return {"accepted": accepted, "guard": guard, "baseline": baseline, "candidate": candidate, "latest": latest or {}}

    def _capture_state_locked(self) -> dict:
        return {
            "policy": self.policy.clone(seed=61),
            "red_policy": self.red_policy.clone(seed=63),
            "episode": self.episode,
            "baseline": self.baseline,
            "elo": self.elo,
            "scripted_elo": self.scripted_elo,
            "league_pool": self._clone_league_pool_locked(),
            "last_opponent": dict(self.last_opponent),
            "history": list(self.history),
            "latest_replay": list(self.latest_replay),
            "latest_info": dict(self.latest_info),
            "last_eval": dict(self.last_eval),
            "last_guard": dict(self.last_guard),
            "last_event": self.last_event,
        }

    def _restore_state_locked(self, state: dict) -> None:
        self.policy = state["policy"]
        self.red_policy = state["red_policy"]
        self.episode = state["episode"]
        self.baseline = state["baseline"]
        self.elo = state["elo"]
        self.scripted_elo = state["scripted_elo"]
        self.league_pool = state["league_pool"]
        self.last_opponent = dict(state["last_opponent"])
        self.history = list(state["history"])
        self.latest_replay = list(state["latest_replay"])
        self.latest_info = dict(state["latest_info"])
        self.last_eval = dict(state["last_eval"])
        self.last_guard = dict(state["last_guard"])
        self.last_event = state["last_event"]

    def _clone_league_pool_locked(self) -> list[dict]:
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "elo": row["elo"],
                "games": row.get("games", 0),
                "wins": row.get("wins", 0),
                "losses": row.get("losses", 0),
                "draws": row.get("draws", 0),
                "policy": row["policy"].clone(seed=81 + index),
            }
            for index, row in enumerate(self.league_pool)
        ]

    @staticmethod
    def _guard_objective(evaluation: dict) -> float:
        record = evaluation.get("record", {})
        return (
            float(evaluation.get("avg_reward", 0.0))
            + float(evaluation.get("avg_goal_diff", 0.0)) * 6.0
            + float(evaluation.get("avg_xg_diff", 0.0)) * 3.0
            + float(record.get("win_rate", 0.0)) * 2.0
            - float(record.get("loss_rate", 0.0)) * 2.0
        )

    @staticmethod
    def _reward_audit(evaluation: dict) -> dict:
        terms = dict(evaluation.get("avg_reward_terms") or {})
        total_abs = sum(abs(float(value)) for value in terms.values()) or 1.0
        shares = {
            key: round(abs(float(value)) / total_abs, 3)
            for key, value in sorted(terms.items())
        }
        warnings = []
        if float(evaluation.get("avg_goal_diff", 0.0)) < 0.0 and (
            shares.get("territory", 0.0) + shares.get("xg", 0.0)
        ) > 0.45:
            warnings.append("objective improved while goal differential is negative; inspect shaping terms")
        return {"term_abs_share": shares, "warnings": warnings}

    def _evaluate_policy(
        self,
        blue_policy: SoftmaxPolicy,
        red_policy: SoftmaxPolicy,
        pool: list[dict],
        seeds: list[int],
        *,
        opponent: str = "mixed",
    ) -> dict:
        rows = []
        reward_totals: dict[str, float] = {}
        opponent_distribution = {"scripted": 0, "current_red": 0, "league": 0}
        latest_replay: list[dict] = []
        for index, seed in enumerate(seeds):
            env = TacticalSoccerEnv(seed=seed)
            obs = env.reset()
            selected = self._evaluation_opponent(opponent, index, pool)
            opponent_distribution[selected["kind"]] = opponent_distribution.get(selected["kind"], 0) + 1
            total_reward = 0.0
            done = False
            latest_info = {}
            while not done:
                blue_action = blue_policy.greedy(obs)
                if selected["kind"] == "current_red":
                    red_action = red_policy.greedy(self._invert_obs(obs))
                elif selected["kind"] == "league":
                    red_action = selected["policy"].greedy(self._invert_obs(obs))
                else:
                    red_action = self._scripted_red(obs)
                obs, reward, done, latest_info = env.step(blue_action, red_action)
                total_reward += reward
                for key, value in latest_info.get("reward_terms", {}).items():
                    reward_totals[key] = reward_totals.get(key, 0.0) + float(value)
            final = env.info()
            result = self._result(final)
            rows.append(
                {
                    "result": result,
                    "score": final["score"],
                    "xg": final["xg"],
                    "opponent": self._opponent_public(selected),
                    "reward": round(total_reward, 3),
                }
            )
            latest_replay = list(env.replay)
        episodes = max(1, len(rows))
        wins = sum(1 for row in rows if row["result"] == "win")
        losses = sum(1 for row in rows if row["result"] == "loss")
        draws = episodes - wins - losses
        goal_diff = sum(row["score"]["blue"] - row["score"]["red"] for row in rows) / episodes
        xg_diff = sum(row["xg"]["blue"] - row["xg"]["red"] for row in rows) / episodes
        avg_reward = sum(row["reward"] for row in rows) / episodes
        return {
            "episodes": episodes,
            "opponent": opponent,
            "record": {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": round(wins / episodes, 3),
                "loss_rate": round(losses / episodes, 3),
            },
            "avg_reward": round(avg_reward, 3),
            "avg_goal_diff": round(goal_diff, 3),
            "avg_xg_diff": round(xg_diff, 3),
            "avg_reward_terms": {key: round(value / episodes, 3) for key, value in sorted(reward_totals.items())},
            "opponent_distribution": opponent_distribution,
            "latest": rows[-1] if rows else {},
            "rows": rows[-40:],
            "replay_frames": len(latest_replay),
        }

    def evaluate(self, episodes: int = 20, opponent: str = "mixed") -> dict:
        with self.training_lock:
            return self._evaluate(episodes, opponent)

    def _evaluate(self, episodes: int = 20, opponent: str = "mixed") -> dict:
        episodes = max(1, min(200, int(episodes)))
        opponent = opponent if opponent in {"mixed", "scripted", "current_red", "league"} else "mixed"
        with self.lock:
            blue_policy = self.policy.clone(seed=31)
            red_policy = self.red_policy.clone(seed=37)
            pool = self._clone_league_pool_locked()
            start_episode = self.episode
        seeds = [700_000 + start_episode + index for index in range(episodes)]
        evaluation = self._evaluate_policy(blue_policy, red_policy, pool, seeds, opponent=opponent)
        with self.lock:
            self.last_eval = evaluation
            self.last_event = f"evaluated {episodes} vs {opponent}"
        self.eval_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        return evaluation

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
            with self.lock:
                guarded = self.guard_enabled
                batch_episodes = self.guard_batch_episodes
                eval_episodes = self.guard_eval_episodes
                accept_margin = self.guard_accept_margin
            if guarded:
                self.train_guarded_batch(batch_episodes, eval_episodes=eval_episodes, accept_margin=accept_margin)
            else:
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
            "baseline": self.baseline,
            "elo": self.elo,
            "scripted_elo": self.scripted_elo,
            "league_enabled": self.league_enabled,
            "guard_enabled": self.guard_enabled,
            "guard_batch_episodes": self.guard_batch_episodes,
            "guard_eval_episodes": self.guard_eval_episodes,
            "guard_accept_margin": self.guard_accept_margin,
            "guard_opponent": self.guard_opponent,
            "best_guard_objective": None if not math.isfinite(self.best_guard_objective) else self.best_guard_objective,
            "coach_enabled": self.coach_enabled,
            "coach_rate": self.coach_rate,
            "policy": self.policy.to_json(),
            "red_policy": self.red_policy.to_json(),
            "league_pool": [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "elo": row["elo"],
                    "games": row.get("games", 0),
                    "wins": row.get("wins", 0),
                    "losses": row.get("losses", 0),
                    "draws": row.get("draws", 0),
                    "policy": row["policy"].to_json(),
                }
                for row in self.league_pool
            ],
        }
        self.checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_best_checkpoint(self, objective: float) -> None:
        if objective < self.best_guard_objective:
            return
        self.best_guard_objective = float(objective)
        payload = {
            "episode": self.episode,
            "objective": self.best_guard_objective,
            "baseline": self.baseline,
            "elo": self.elo,
            "scripted_elo": self.scripted_elo,
            "league_enabled": self.league_enabled,
            "guard_opponent": self.guard_opponent,
            "policy": self.policy.to_json(),
            "red_policy": self.red_policy.to_json(),
            "league_pool": [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "elo": row["elo"],
                    "games": row.get("games", 0),
                    "wins": row.get("wins", 0),
                    "losses": row.get("losses", 0),
                    "draws": row.get("draws", 0),
                    "policy": row["policy"].to_json(),
                }
                for row in self.league_pool
            ],
        }
        self.best_checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        try:
            payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            self.episode = int(payload.get("episode", 0))
            self.baseline = float(payload.get("baseline", 0.0))
            self.elo = float(payload.get("elo", self.elo))
            self.scripted_elo = float(payload.get("scripted_elo", self.scripted_elo))
            self.league_enabled = bool(payload.get("league_enabled", self.league_enabled))
            self.guard_enabled = bool(payload.get("guard_enabled", self.guard_enabled))
            self.guard_batch_episodes = int(payload.get("guard_batch_episodes", self.guard_batch_episodes))
            self.guard_eval_episodes = int(payload.get("guard_eval_episodes", self.guard_eval_episodes))
            self.guard_accept_margin = float(payload.get("guard_accept_margin", self.guard_accept_margin))
            self.guard_opponent = str(payload.get("guard_opponent", self.guard_opponent) or self.guard_opponent)
            best_objective = payload.get("best_guard_objective")
            if best_objective is not None:
                self.best_guard_objective = float(best_objective)
            self.coach_enabled = bool(payload.get("coach_enabled", self.coach_enabled))
            self.coach_rate = float(payload.get("coach_rate", self.coach_rate))
            self.policy.load_json(dict(payload.get("policy") or {}))
            self.red_policy.load_json(dict(payload.get("red_policy") or {}))
            self.league_pool = []
            for row in list(payload.get("league_pool") or [])[-12:]:
                policy = SoftmaxPolicy(self.policy.obs_dim, self.policy.action_dim)
                policy.load_json(dict(row.get("policy") or {}))
                self.league_pool.append(
                    {
                        "id": int(row.get("id", len(self.league_pool))),
                        "name": str(row.get("name", f"snapshot {len(self.league_pool)}")),
                        "elo": float(row.get("elo", 1000.0)),
                        "games": int(row.get("games", 0)),
                        "wins": int(row.get("wins", 0)),
                        "losses": int(row.get("losses", 0)),
                        "draws": int(row.get("draws", 0)),
                        "policy": policy,
                    }
                )
        except Exception:
            self.episode = 0
            self.baseline = 0.0

    def _select_opponent_locked(self, episode_index: int, self_play: bool) -> dict:
        if not self_play:
            opponent = {"kind": "scripted", "name": "scripted red", "elo": self.scripted_elo}
            self.last_opponent = self._opponent_public(opponent)
            return opponent
        if not self.league_enabled:
            opponent = {"kind": "current_red", "name": "current red", "elo": self.elo}
            self.last_opponent = self._opponent_public(opponent)
            return opponent
        roll = random.Random(90_000 + episode_index).random()
        if self.league_pool and roll < 0.5:
            pool = sorted(self.league_pool, key=lambda row: abs(row["elo"] - self.elo))
            row = pool[min(len(pool) - 1, int(roll * len(pool) * 1.7))]
            opponent = {
                "kind": "league",
                "id": row["id"],
                "name": row["name"],
                "elo": row["elo"],
                "policy": row["policy"].clone(seed=episode_index + 1000),
            }
        elif roll < 0.8:
            opponent = {"kind": "current_red", "name": "current red", "elo": self.elo}
        else:
            opponent = {"kind": "scripted", "name": "scripted red", "elo": self.scripted_elo}
        self.last_opponent = self._opponent_public(opponent)
        return opponent

    def _update_elo_locked(self, opponent: dict, result: str) -> None:
        score = 1.0 if result == "win" else 0.0 if result == "loss" else 0.5
        opponent_elo = float(opponent.get("elo", self.scripted_elo))
        expected = 1.0 / (1.0 + 10 ** ((opponent_elo - self.elo) / 400.0))
        delta = 24.0 * (score - expected)
        self.elo += delta
        if opponent["kind"] == "scripted":
            self.scripted_elo -= delta * 0.35
        elif opponent["kind"] == "league":
            for row in self.league_pool:
                if row["id"] == opponent.get("id"):
                    row["elo"] -= delta * 0.6
                    row["games"] = row.get("games", 0) + 1
                    row["wins"] = row.get("wins", 0) + (1 if result == "loss" else 0)
                    row["losses"] = row.get("losses", 0) + (1 if result == "win" else 0)
                    row["draws"] = row.get("draws", 0) + (1 if result == "draw" else 0)
                    break

    def _maybe_add_league_snapshot_locked(self, episode_index: int, final: dict) -> None:
        if not self.league_enabled:
            return
        if episode_index == 0 or episode_index % 20 != 0:
            return
        row = {
            "id": episode_index,
            "name": f"blue@{episode_index}",
            "elo": self.elo,
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "policy": self.policy.clone(seed=episode_index + 2000),
        }
        self.league_pool.append(row)
        self.league_pool = sorted(self.league_pool, key=lambda item: item["id"])[-12:]
        final["league_snapshot"] = row["name"]

    @staticmethod
    def _opponent_public(opponent: dict) -> dict:
        return {
            "kind": opponent.get("kind", "scripted"),
            "name": opponent.get("name", "scripted red"),
            "elo": round(float(opponent.get("elo", 0.0)), 1),
        }

    @staticmethod
    def _sum_reward_terms(trajectory: list[dict]) -> dict:
        totals: dict[str, float] = {}
        for row in trajectory:
            terms = row.get("info", {}).get("reward_terms", {})
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value)
        return {key: round(value, 3) for key, value in sorted(totals.items())}

    @staticmethod
    def _evaluation_opponent(opponent: str, index: int, pool: list[dict]) -> dict:
        if opponent == "scripted":
            return {"kind": "scripted", "name": "scripted red", "elo": 950.0}
        if opponent == "current_red":
            return {"kind": "current_red", "name": "current red", "elo": 1000.0}
        if opponent == "league" and pool:
            row = pool[index % len(pool)]
            return {"kind": "league", "id": row["id"], "name": row["name"], "elo": row["elo"], "policy": row["policy"]}
        slot = index % 4
        if slot < 2:
            return {"kind": "scripted", "name": "scripted red", "elo": 950.0}
        if slot == 2:
            return {"kind": "current_red", "name": "current red", "elo": 1000.0}
        if pool:
            row = pool[index % len(pool)]
            return {"kind": "league", "id": row["id"], "name": row["name"], "elo": row["elo"], "policy": row["policy"]}
        return {"kind": "scripted", "name": "scripted red", "elo": 950.0}

    @staticmethod
    def _scripted_red(obs: list[float]) -> int:
        minute, ball_x, _ball_y, possession, goal_diff, _shot_diff, blue_stamina, red_stamina, *_ = obs
        if red_stamina < 0.38:
            return ACTION_NAMES.index("conserve")
        if goal_diff > 0.34:
            return ACTION_NAMES.index("direct_attack")
        if possession < 0 and ball_x < -0.35:
            return ACTION_NAMES.index("direct_attack")
        if possession > 0 and ball_x > 0.25:
            return ACTION_NAMES.index("low_block")
        if minute > 0.72 and goal_diff < 0:
            return ACTION_NAMES.index("high_press")
        if blue_stamina < 0.45:
            return ACTION_NAMES.index("high_press")
        return ACTION_NAMES.index("balanced")

    @staticmethod
    def _coach_blue(_obs: list[float]) -> int:
        return ACTION_NAMES.index("possession")

    @staticmethod
    def _invert_obs(obs: list[float]) -> list[float]:
        out = list(obs)
        out[1] = -out[1]
        out[2] = -out[2]
        out[3] = -out[3]
        out[4] = -out[4]
        out[5] = -out[5]
        out[6], out[7] = out[7], out[6]
        out[8] = -out[8]
        out[9] = -out[9]
        if len(out) > 13:
            out[12], out[13] = out[13], out[12]
        if len(out) > 14:
            out[14] = -out[14]
        if len(out) > 16:
            out[16] = -out[16]
        if len(out) > 17:
            out[17] = -out[17]
        if len(out) > 18:
            out[18] = -out[18]
        return out

    @staticmethod
    def _result(info: dict) -> str:
        blue = int(info.get("score", {}).get("blue", 0))
        red = int(info.get("score", {}).get("red", 0))
        if blue > red:
            return "win"
        if red > blue:
            return "loss"
        return "draw"
