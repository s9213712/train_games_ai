from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TETRIS_ROOT = ROOT / "tetris-ai"
sys.path.insert(0, str(TETRIS_ROOT))
SPEC = importlib.util.spec_from_file_location("tetris_rl_training_integrity", TETRIS_ROOT / "rl_trainer.py")
assert SPEC and SPEC.loader
rl_trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rl_trainer
SPEC.loader.exec_module(rl_trainer)


def evaluation(score: int, seeds: list[int], *, tetrises: int = 0) -> dict:
    rows = [
        {
            "seed": seed,
            "score": score,
            "lines": tetrises * 4,
            "pieces": 40,
            "tetrises": tetrises,
            "holds": 0,
            "reward": float(score),
        }
        for seed in seeds
    ]
    return {
        "episodes": len(rows),
        "seeds": list(seeds),
        "avg_score": float(score),
        "avg_lines": float(tetrises * 4),
        "avg_pieces": 40.0,
        "avg_tetrises": float(tetrises),
        "avg_holds": 0.0,
        "best_score": score,
        "best_lines": tetrises * 4,
        "best_tetrises": tetrises,
        "rows": rows,
        "replay_frames": 1,
        "max_pieces": 40,
    }


class OneStepHoldoutEnv:
    """A deterministic environment where choosing action 1 is truly better."""

    def __init__(self, *, seed: int | None = None, max_pieces: int = 40) -> None:
        self.seed = int(seed or 0)
        self.max_pieces = max_pieces
        self.done = False
        self.action = 0
        self.replay: list[dict] = []

    def reset(self) -> list[float]:
        self.done = False
        self.action = 0
        self.replay = []
        return [1.0, 0.0]

    def legal_moves(self) -> list[dict]:
        if self.done:
            return []
        return [
            {"action": 0, "vector": [1.0, 0.0], "immediate_reward": 0.0},
            {"action": 1, "vector": [1.0, 1.0], "immediate_reward": 0.0},
        ]

    def future_legal_moves_for(self, _move: dict, *, include_hold: bool = False) -> list[dict]:
        return []

    def step(self, move: dict, *, check_terminal: bool = False) -> tuple[list[float], float, bool, dict]:
        self.action = int(move.get("action", 0))
        self.done = True
        row = self.info()
        self.replay = [dict(row, board=[])]
        return [1.0, float(self.action)], float(row["score"]), True, row

    def info(self) -> dict:
        good = self.action == 1
        return {
            "score": (100 if good else 10) + self.seed % 3,
            "lines": 4 if good else 0,
            "pieces": 1,
            "tetrises": 1 if good else 0,
            "holds": 0,
        }


class TetrisTrainingIntegrityTests(unittest.TestCase):
    def make_trainer(self, root: Path):
        return rl_trainer.RLTrainer(root)

    def test_afterstate_target_does_not_learn_enumerated_reward_twice(self) -> None:
        policy = rl_trainer.AfterstateValue(1)
        policy.weights = [2.0]
        move = {"vector": [1.0], "immediate_reward": 3.0}
        self.assertEqual(policy.score_move(move), 5.0)

        target = rl_trainer.RLTrainer.afterstate_td_target(
            reward=3.0,
            enumerated_reward=3.0,
            done=False,
            gamma=0.9,
            bootstrap=5.0,
        )
        self.assertAlmostEqual(target, 4.5)
        terminal_target = rl_trainer.RLTrainer.afterstate_td_target(
            reward=-5.0,
            enumerated_reward=3.0,
            done=True,
            gamma=0.9,
            bootstrap=999.0,
        )
        self.assertAlmostEqual(terminal_target, -8.0)
        self.assertLess(rl_trainer.RLTrainer._training_seed(880_000), 800_000)

    def test_evaluation_is_frozen_repeatable_and_does_not_advance_policy_rng(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            policy = rl_trainer.AfterstateValue(2)
            policy.weights = [0.0, 1.0]
            weights_before = list(policy.weights)
            rng_before = policy.random.getstate()
            kwargs = {
                "policy": policy,
                "episodes": 4,
                "start": 99,
                "lookahead_weight": 0.0,
                "lookahead_candidates": 0,
                "lookahead_include_hold": False,
                "max_pieces": 40,
                "seeds": [880_000, 880_001, 880_002, 880_003],
            }
            with mock.patch.object(rl_trainer, "TetrisEnv", OneStepHoldoutEnv):
                first = trainer._evaluate_policy(**kwargs)
                second = trainer._evaluate_policy(**kwargs)

            self.assertEqual(first, second)
            self.assertEqual(policy.weights, weights_before)
            self.assertEqual(policy.random.getstate(), rng_before)
            self.assertEqual(trainer.episode, 0)
            self.assertEqual(trainer.history, [])

    def test_guard_rolls_back_candidate_that_does_not_improve_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            original_weights = list(trainer.policy.weights)
            seen_seed_sets: list[tuple[int, ...]] = []
            attempted_rollouts: list[int] = []

            def fake_evaluate(
                _self,
                policy,
                episodes,
                start,
                lookahead_weight,
                lookahead_candidates,
                lookahead_include_hold,
                capture_replay=False,
                max_pieces=None,
                seeds=None,
            ):
                selected = list(seeds or [])
                seen_seed_sets.append(tuple(selected))
                # Deliberately return identical behavior even though training
                # below changes weights: weight motion alone must not pass.
                return evaluation(100, selected)

            def fake_train(_self, state, config, *, rollout_index):
                del config
                attempted_rollouts.append(rollout_index)
                state["policy"].weights[0] += 1.0 + rollout_index
                state["target_policy"] = state["policy"].clone()
                state["episode"] += 1
                state["latest_info"] = {"score": 80, "lines": 0, "episode": state["episode"]}
                state["latest_replay"] = []
                return dict(state["latest_info"]), []

            trainer._evaluate_policy = types.MethodType(fake_evaluate, trainer)
            trainer._train_candidate_episode = types.MethodType(fake_train, trainer)
            result = trainer.train_guarded_batch(2, eval_episodes=2, accept_min_delta=0.0)
            second = trainer.train_guarded_batch(2, eval_episodes=2, accept_min_delta=0.0)

            self.assertFalse(result["accepted"])
            self.assertFalse(result["guard"]["behavior_changed"])
            self.assertEqual(trainer.policy.weights, original_weights)
            self.assertEqual(trainer.episode, 0)
            self.assertEqual(trainer.rollout, 4)
            self.assertEqual(result["guard"]["eval_episodes"], 4)
            self.assertEqual(seen_seed_sets[0], seen_seed_sets[1])
            self.assertEqual(seen_seed_sets[0], tuple(range(1_000_000_000, 1_000_000_004)))
            self.assertEqual(seen_seed_sets[2], seen_seed_sets[3])
            self.assertNotEqual(seen_seed_sets[0], seen_seed_sets[2])
            self.assertEqual(attempted_rollouts, [0, 1, 2, 3])
            self.assertFalse(second["accepted"])
            saved = json.loads(trainer.checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["training_semantics"], rl_trainer.AFTERSTATE_SEMANTICS)
            self.assertEqual(saved["policy"]["weights"], original_weights)
            self.assertEqual(saved["rollout"], 4)
            self.assertEqual(saved["guard_benchmark"]["protocol"]["seeds"], list(seen_seed_sets[2]))

    def test_public_train_episode_cannot_bypass_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            original_weights = list(trainer.policy.weights)
            guarded_result = {"accepted": False, "guard": {"accepted": False}}
            with mock.patch.object(
                trainer,
                "train_guarded_batch",
                return_value=guarded_result,
            ) as guarded:
                self.assertIs(trainer.train_episode(), guarded_result)
                guarded.assert_called_once_with(
                    1,
                    eval_episodes=trainer.background_eval_episodes,
                )
            with self.assertRaisesRegex(ValueError, "unguarded"):
                trainer.train_episode(persist=False)
            self.assertEqual(trainer.policy.weights, original_weights)

    def test_reload_prefers_promoted_policy_but_preserves_latest_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trainer = self.make_trainer(root)
            trainer.gamma = 0.91
            trainer.epsilon = 0.12
            trainer.background_eval_episodes = 6
            trainer.guard_min_effect = 2.5
            good_weights = list(trainer.policy.weights)
            gate_protocol = trainer._guard_protocol(4, start_rollout=0)
            protocol = trainer._promotion_protocol()
            benchmark_eval = evaluation(120, list(protocol["seeds"]))
            trainer.guard_benchmark = trainer._benchmark_payload(gate_protocol, 110.0, evaluation(110, list(gate_protocol["seeds"])))
            trainer.promotion_benchmark = trainer._benchmark_payload(protocol, 120.0, benchmark_eval)
            trainer.best_guard_objective = 120.0
            trainer.policy_guard_objective = 120.0
            self.assertTrue(trainer._save_best_checkpoint(objective=120.0))
            trainer._save_checkpoint()

            protected_before = trainer.best_checkpoint_path.read_bytes()
            worse = trainer.policy.clone()
            worse.weights[0] += 5.0
            self.assertFalse(trainer._save_best_checkpoint(objective=100.0, policy=worse, target_policy=worse))
            self.assertEqual(trainer.best_checkpoint_path.read_bytes(), protected_before)

            trainer.policy = worse
            trainer.target_policy = worse.clone()
            trainer.policy_guard_objective = float("-inf")
            trainer.gamma = 0.999
            trainer.epsilon = 0.5
            trainer.background_eval_episodes = 4
            trainer.guard_min_effect = 0.1
            trainer._save_checkpoint()

            reloaded = self.make_trainer(root)
            self.assertEqual(reloaded.policy.weights, good_weights)
            self.assertAlmostEqual(reloaded.gamma, 0.999)
            self.assertAlmostEqual(reloaded.epsilon, 0.5)
            self.assertEqual(reloaded.background_eval_episodes, 4)
            self.assertAlmostEqual(reloaded.guard_min_effect, 0.1)
            self.assertEqual(reloaded.promotion_benchmark["protocol"], protocol)
            self.assertAlmostEqual(reloaded.policy_guard_objective, 120.0)

    def test_controlled_candidate_passes_fixed_canonical_promotion_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            trainer.policy = rl_trainer.AfterstateValue(2)
            trainer.policy.weights = [0.0, -1.0]
            trainer.target_policy = trainer.policy.clone()
            trainer.best_policy = trainer.policy.clone()
            trainer.lookahead_weight = 0.0
            trainer.lookahead_candidates = 0
            training_seeds: list[int] = []

            trainer.promotion_eval_episodes = 4

            def controlled_train(_self, state, config, *, rollout_index):
                del config
                training_seeds.append(_self._training_seed(rollout_index))
                state["policy"].weights[1] = 1.0
                state["target_policy"] = state["policy"].clone()
                state["episode"] += 1
                state["latest_info"] = {"score": 100, "lines": 4, "episode": state["episode"]}
                state["latest_replay"] = []
                return dict(state["latest_info"]), []

            trainer._train_candidate_episode = types.MethodType(controlled_train, trainer)
            with mock.patch.object(rl_trainer, "TetrisEnv", OneStepHoldoutEnv):
                result = trainer.train_guarded_batch(1, eval_episodes=4)

            holdout = result["baseline"]["seeds"]
            self.assertTrue(result["accepted"])
            self.assertTrue(result["guard"]["gate_passed"])
            self.assertTrue(result["guard"]["behavior_changed"])
            self.assertTrue(result["guard"]["promotion"]["promoted"])
            self.assertGreater(result["candidate"]["avg_score"], result["baseline"]["avg_score"])
            self.assertGreater(result["candidate"]["avg_tetrises"], result["baseline"]["avg_tetrises"])
            self.assertTrue(set(training_seeds).isdisjoint(holdout))
            promotion_holdout = result["guard"]["promotion"]["holdout_seeds"]
            self.assertTrue(set(training_seeds).isdisjoint(promotion_holdout))
            self.assertTrue(set(holdout).isdisjoint(promotion_holdout))
            self.assertEqual(holdout, list(range(1_000_000_000, 1_000_000_004)))
            self.assertEqual(trainer.guard_benchmark["protocol"]["seeds"], holdout)
            self.assertEqual(trainer.promotion_benchmark["protocol"]["seeds"], promotion_holdout)
            self.assertTrue(trainer.best_checkpoint_path.exists())

    def test_gate_pass_without_canonical_promotion_rolls_back_live_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            original_weights = list(trainer.policy.weights)
            trainer.promotion_eval_episodes = 4

            def controlled_train(_self, state, config, *, rollout_index):
                del _self, config, rollout_index
                state["policy"].weights[0] += 4.0
                state["target_policy"] = state["policy"].clone()
                state["episode"] += 1
                state["latest_info"] = {"score": 200, "lines": 4, "episode": state["episode"]}
                return dict(state["latest_info"]), []

            def split_evaluate(
                _self,
                policy,
                episodes,
                start,
                lookahead_weight,
                lookahead_candidates,
                lookahead_include_hold,
                capture_replay=False,
                max_pieces=None,
                seeds=None,
            ):
                del _self, episodes, start, lookahead_weight, lookahead_candidates
                del lookahead_include_hold, capture_replay, max_pieces
                selected = list(seeds or [])
                candidate = policy.weights[0] != original_weights[0]
                # Candidate wins the rotating gate but loses the fixed canonical
                # promotion validation, so it must never become the live policy.
                score = (200 if candidate else 100) if selected[0] >= 1_000_000_000 else (90 if candidate else 100)
                return evaluation(score, selected)

            trainer._train_candidate_episode = types.MethodType(controlled_train, trainer)
            trainer._evaluate_policy = types.MethodType(split_evaluate, trainer)
            result = trainer.train_guarded_batch(1, eval_episodes=4)

            self.assertTrue(result["guard"]["gate_passed"])
            self.assertFalse(result["guard"]["promotion"]["promoted"])
            self.assertFalse(result["accepted"])
            self.assertEqual(trainer.policy.weights, original_weights)
            self.assertEqual(trainer.episode, 0)
            self.assertEqual(trainer.rollout, 1)
            protected = json.loads(trainer.best_checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(protected["policy"]["weights"], original_weights)
            self.assertEqual(
                protected["promotion_benchmark"]["protocol"]["seeds"],
                result["guard"]["promotion"]["holdout_seeds"],
            )

    def test_candidate_isolation_blocks_config_and_close_until_transaction_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            original_weights = list(trainer.policy.weights)
            candidate_started = threading.Event()
            release_candidate = threading.Event()
            config_started = threading.Event()
            close_started = threading.Event()
            result_box: dict = {}

            def blocking_train(_self, state, config, *, rollout_index):
                del _self, config, rollout_index
                state["policy"].weights[0] += 9.0
                state["episode"] += 1
                state["latest_info"] = {"score": 9, "lines": 0, "episode": state["episode"]}
                candidate_started.set()
                self.assertTrue(release_candidate.wait(5.0))
                return dict(state["latest_info"]), []

            def unchanged_evaluate(
                _self,
                policy,
                episodes,
                start,
                lookahead_weight,
                lookahead_candidates,
                lookahead_include_hold,
                capture_replay=False,
                max_pieces=None,
                seeds=None,
            ):
                del _self, policy, episodes, start, lookahead_weight, lookahead_candidates
                del lookahead_include_hold, capture_replay, max_pieces
                return evaluation(100, list(seeds or []))

            trainer._train_candidate_episode = types.MethodType(blocking_train, trainer)
            trainer._evaluate_policy = types.MethodType(unchanged_evaluate, trainer)

            guard_thread = threading.Thread(
                target=lambda: result_box.update(result=trainer.train_guarded_batch(1, eval_episodes=4))
            )
            guard_thread.start()
            self.assertTrue(candidate_started.wait(5.0))

            # Even an explicit checkpoint during candidate work serializes only
            # the accepted/live policy.
            trainer._save_checkpoint()
            during = json.loads(trainer.checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(during["policy"]["weights"], original_weights)

            def update_settings():
                config_started.set()
                trainer.update_config({"gamma": 0.91})

            def close_trainer():
                close_started.set()
                trainer.close()

            config_thread = threading.Thread(target=update_settings)
            close_thread = threading.Thread(target=close_trainer)
            config_thread.start()
            close_thread.start()
            self.assertTrue(config_started.wait(2.0))
            self.assertTrue(close_started.wait(2.0))
            config_thread.join(0.05)
            close_thread.join(0.05)
            self.assertTrue(config_thread.is_alive())
            self.assertTrue(close_thread.is_alive())

            release_candidate.set()
            guard_thread.join(5.0)
            config_thread.join(5.0)
            close_thread.join(5.0)
            self.assertFalse(guard_thread.is_alive())
            self.assertFalse(config_thread.is_alive())
            self.assertFalse(close_thread.is_alive())
            self.assertFalse(result_box["result"]["accepted"])
            self.assertEqual(trainer.policy.weights, original_weights)
            final = json.loads(trainer.checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(final["policy"]["weights"], original_weights)
            self.assertAlmostEqual(final["config"]["gamma"], 0.91)

    def test_loader_is_transactional_and_invalid_best_falls_back_to_main(self) -> None:
        corruptions = {
            "late_target": lambda payload: payload.__setitem__(
                "target_policy",
                {
                    "dim": payload["policy"]["dim"],
                    "weights": payload["policy"]["weights"][:-1] + ["not-a-number"],
                },
            ),
            "late_config": lambda payload: payload["config"].__setitem__("gamma", "nan"),
        }
        for label, corrupt in corruptions.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                trainer = self.make_trainer(root)
                trainer.policy.weights[0] = 0.25
                trainer.target_policy = trainer.policy.clone()
                trainer.best_policy = trainer.policy.clone()
                trainer.episode = 3
                trainer.rollout = 5
                trainer.gamma = 0.91
                trainer._save_checkpoint()
                main_weights = list(trainer.policy.weights)

                trainer.policy.weights[0] = 1.25
                trainer.target_policy = trainer.policy.clone()
                trainer.best_policy = trainer.policy.clone()
                protocol = trainer._promotion_protocol()
                promoted_eval = evaluation(200, list(protocol["seeds"]))
                trainer.promotion_benchmark = trainer._benchmark_payload(
                    protocol,
                    200.0,
                    promoted_eval,
                )
                trainer.best_guard_objective = 200.0
                trainer.policy_guard_objective = 200.0
                self.assertTrue(trainer._save_best_checkpoint(objective=200.0))
                best_payload = json.loads(
                    trainer.best_checkpoint_path.read_text(encoding="utf-8")
                )
                best_payload["episode"] = 999
                best_payload["rollout"] = 999
                corrupt(best_payload)
                trainer.best_checkpoint_path.write_text(
                    json.dumps(best_payload),
                    encoding="utf-8",
                )

                reloaded = self.make_trainer(root)
                selection = reloaded.checkpoint_selection()
                self.assertEqual(reloaded.policy.weights, main_weights)
                self.assertEqual(reloaded.target_policy.weights, main_weights)
                self.assertEqual(reloaded.episode, 3)
                self.assertEqual(reloaded.rollout, 5)
                self.assertAlmostEqual(reloaded.gamma, 0.91)
                self.assertEqual(selection["source"], "main")
                self.assertEqual(selection["path"], str(reloaded.checkpoint_path))
                self.assertEqual(selection["policy"]["weights"], main_weights)
                self.assertTrue(
                    any(row["source"] == "protected_best" for row in selection["rejected"])
                )
                self.assertEqual(
                    reloaded.snapshot()["loaded_checkpoint"],
                    selection,
                )

    def test_loader_invalid_main_cannot_partially_mutate_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trainer = self.make_trainer(root)
            defaults = {
                "weights": list(trainer.policy.weights),
                "target": list(trainer.target_policy.weights),
                "gamma": trainer.gamma,
            }
            trainer._save_checkpoint()
            payload = json.loads(trainer.checkpoint_path.read_text(encoding="utf-8"))
            payload["policy"]["weights"][0] = 7.0
            payload["episode"] = 777
            payload["config"]["gamma"] = 0.9
            payload["best_policy"]["weights"][-1] = "late-corruption"
            trainer.checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

            reloaded = self.make_trainer(root)
            self.assertEqual(reloaded.policy.weights, defaults["weights"])
            self.assertEqual(reloaded.target_policy.weights, defaults["target"])
            self.assertEqual(reloaded.episode, 0)
            self.assertEqual(reloaded.gamma, defaults["gamma"])
            self.assertEqual(reloaded.loaded_checkpoint["source"], "built_in_default")
            self.assertTrue(reloaded.loaded_checkpoint["rejected"])

    def test_background_exception_restores_served_policy_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            trainer._save_checkpoint()
            original_policy = list(trainer.policy.weights)
            original_target = list(trainer.target_policy.weights)
            original_episode = trainer.episode
            trainer.running = True

            def corrupt_then_fail(_episodes, *, eval_episodes, accept_min_delta):
                del _episodes, eval_episodes, accept_min_delta
                with trainer.lock:
                    trainer.policy.weights[0] += 9.0
                    trainer.target_policy.weights[0] -= 9.0
                    trainer.episode = 999
                raise RuntimeError("injected background failure")

            trainer.train_guarded_batch = corrupt_then_fail
            trainer._loop()

            self.assertFalse(trainer.running)
            self.assertEqual(trainer.last_event, "error")
            self.assertEqual(trainer.policy.weights, original_policy)
            self.assertEqual(trainer.target_policy.weights, original_target)
            self.assertEqual(trainer.episode, original_episode)
            self.assertIn("injected background failure", trainer.last_guard["reason"])
            saved = json.loads(trainer.checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["policy"]["weights"], original_policy)

    def test_legacy_or_wrong_semantics_checkpoints_are_quarantined(self) -> None:
        for version, semantics in ((2, rl_trainer.AFTERSTATE_SEMANTICS), (3, "old_double_reward")):
            with self.subTest(version=version, semantics=semantics), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                runtime = root / "runtime"
                runtime.mkdir()
                payload = {
                    "checkpoint_version": version,
                    "training_semantics": semantics,
                    "episode": 999,
                    "policy": {"weights": [7.0] * 15},
                }
                (runtime / "tetris_policy.json").write_text(json.dumps(payload), encoding="utf-8")

                trainer = self.make_trainer(root)
                self.assertEqual(trainer.episode, 0)
                self.assertNotEqual(trainer.policy.weights, [7.0] * 15)
                self.assertIn("quarantined", trainer.last_event)

    def test_checkpoint_replace_is_atomic_and_failed_replace_preserves_previous_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            trainer._save_checkpoint()
            previous = trainer.checkpoint_path.read_bytes()
            trainer.policy.weights[0] += 3.0

            with mock.patch.object(rl_trainer.os, "replace", side_effect=OSError("simulated interruption")):
                with self.assertRaises(OSError):
                    trainer._save_checkpoint()

            self.assertEqual(trainer.checkpoint_path.read_bytes(), previous)
            self.assertEqual(list(trainer.runtime.glob(".*.tmp")), [])

    def test_ui_and_backend_share_guard_and_learning_rate_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            trainer.update_config({"learning_rate": 0.0001, "background_eval_episodes": 2})
            self.assertEqual(trainer.learning_rate, 0.0005)
            self.assertEqual(trainer.background_eval_episodes, 4)

        html = (TETRIS_ROOT / "web" / "index.html").read_text(encoding="utf-8")
        app = (TETRIS_ROOT / "web" / "app.js").read_text(encoding="utf-8")
        server = (TETRIS_ROOT / "server.py").read_text(encoding="utf-8")
        readme = (TETRIS_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn('id="rlRate" type="number" min="0.0005" max="0.2"', html)
        self.assertIn("eval_episodes: 4", app)
        self.assertIn('eval_episodes=max(4, min(24, int(payload.get("eval_episodes", 4)', server)
        self.assertIn("Fixed canonical promotion validation", app)
        self.assertIn("fixed canonical promotion validation", html)
        self.assertNotIn("independent", readme.lower())
        self.assertEqual(
            trainer._promotion_protocol()["kind"],
            "fixed_canonical_promotion_validation",
        )


if __name__ == "__main__":
    unittest.main()
