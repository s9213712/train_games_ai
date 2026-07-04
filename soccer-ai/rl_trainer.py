from __future__ import annotations

import json
import math
import random
import threading
import time
from pathlib import Path

from soccer_env import ACTION_NAMES, TacticalSoccerEnv


class SoftmaxPolicy:
    def __init__(self, obs_dim: int, action_dim: int, *, seed: int = 7) -> None:
        self.random = random.Random(seed)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.weights = [
            [self.random.uniform(-0.035, 0.035) for _ in range(obs_dim)]
            for _ in range(action_dim)
        ]

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
        self.replay_path = self.runtime / "latest_replay.json"
        self.metrics_path = self.runtime / "training_metrics.jsonl"
        self.eval_path = self.runtime / "latest_evaluation.json"
        probe_env = TacticalSoccerEnv(seed=1)
        obs = probe_env.reset()
        self.policy = SoftmaxPolicy(len(obs), len(ACTION_NAMES))
        self.red_policy = SoftmaxPolicy(len(obs), len(ACTION_NAMES), seed=13)
        self.lock = threading.RLock()
        self.running = False
        self.stop_requested = False
        self.thread: threading.Thread | None = None
        self.episode = 0
        self.baseline = 0.0
        self.learning_rate = 0.012
        self.gamma = 0.985
        self.self_play = False
        self.league_enabled = True
        self.temperature = 1.15
        self.elo = 1000.0
        self.scripted_elo = 950.0
        self.league_pool: list[dict] = []
        self.last_opponent: dict = {"kind": "scripted", "name": "scripted red", "elo": self.scripted_elo}
        self.history: list[dict] = []
        self.latest_replay: list[dict] = []
        self.latest_info: dict = {}
        self.last_eval: dict = {}
        self.last_event = "ready"
        self._load_checkpoint()

    def train_episode(self) -> dict:
        with self.lock:
            episode_index = self.episode
            self.episode += 1
            learning_rate = self.learning_rate
            gamma = self.gamma
            self_play = self.self_play
            temperature = self.temperature
            baseline = self.baseline
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
            self.last_event = "reset"
        self._save_checkpoint()

    def update_config(self, payload: dict) -> None:
        with self.lock:
            self.learning_rate = max(0.0005, min(0.08, float(payload.get("learning_rate", self.learning_rate))))
            self.gamma = max(0.8, min(0.999, float(payload.get("gamma", self.gamma))))
            self.temperature = max(0.25, min(2.5, float(payload.get("temperature", self.temperature))))
            self.self_play = bool(payload.get("self_play", self.self_play))
            self.league_enabled = bool(payload.get("league_enabled", self.league_enabled))
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
                },
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
                "replay_frames": len(self.latest_replay),
            }

    def latest_replay_payload(self) -> dict:
        with self.lock:
            return {"frames": list(self.latest_replay), "latest": dict(self.latest_info)}

    def evaluate(self, episodes: int = 20, opponent: str = "mixed") -> dict:
        episodes = max(1, min(200, int(episodes)))
        opponent = opponent if opponent in {"mixed", "scripted", "current_red", "league"} else "mixed"
        with self.lock:
            blue_policy = self.policy.clone(seed=31)
            red_policy = self.red_policy.clone(seed=37)
            pool = [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "elo": row["elo"],
                    "policy": row["policy"].clone(seed=41 + index),
                }
                for index, row in enumerate(self.league_pool)
            ]
            start_episode = self.episode
        rows = []
        reward_totals: dict[str, float] = {}
        latest_replay: list[dict] = []
        for index in range(episodes):
            env = TacticalSoccerEnv(seed=700_000 + start_episode + index)
            obs = env.reset()
            selected = self._evaluation_opponent(opponent, index, pool)
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
        wins = sum(1 for row in rows if row["result"] == "win")
        losses = sum(1 for row in rows if row["result"] == "loss")
        draws = episodes - wins - losses
        goal_diff = sum(row["score"]["blue"] - row["score"]["red"] for row in rows) / episodes
        xg_diff = sum(row["xg"]["blue"] - row["xg"]["red"] for row in rows) / episodes
        evaluation = {
            "episodes": episodes,
            "opponent": opponent,
            "record": {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": round(wins / episodes, 3),
            },
            "avg_goal_diff": round(goal_diff, 3),
            "avg_xg_diff": round(xg_diff, 3),
            "avg_reward_terms": {key: round(value / episodes, 3) for key, value in sorted(reward_totals.items())},
            "latest": rows[-1] if rows else {},
            "rows": rows[-40:],
            "replay_frames": len(latest_replay),
        }
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
        if pool and index % 3 == 0:
            row = pool[index % len(pool)]
            return {"kind": "league", "id": row["id"], "name": row["name"], "elo": row["elo"], "policy": row["policy"]}
        if index % 3 == 1:
            return {"kind": "current_red", "name": "current red", "elo": 1000.0}
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
