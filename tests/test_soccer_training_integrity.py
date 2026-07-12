from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOCCER_ROOT = ROOT / "soccer-ai"
sys.path.insert(0, str(SOCCER_ROOT))
SPEC = importlib.util.spec_from_file_location("soccer_rl_training_integrity", SOCCER_ROOT / "rl_trainer.py")
assert SPEC and SPEC.loader
rl_trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rl_trainer
SPEC.loader.exec_module(rl_trainer)

ACTION_NAMES = rl_trainer.ACTION_NAMES
CHECKPOINT_VERSION = rl_trainer.CHECKPOINT_VERSION
GUARD_GATE_SEED_BASE = rl_trainer.GUARD_GATE_SEED_BASE
GUARD_GATE_SEED_NAMESPACE = rl_trainer.GUARD_GATE_SEED_NAMESPACE
GUARD_PROMOTION_EPISODES = rl_trainer.GUARD_PROMOTION_EPISODES
GUARD_PROMOTION_SEED_BASE = rl_trainer.GUARD_PROMOTION_SEED_BASE
GUARD_PROMOTION_SEED_NAMESPACE = rl_trainer.GUARD_PROMOTION_SEED_NAMESPACE
TRAIN_SEED_NAMESPACE = rl_trainer.TRAIN_SEED_NAMESPACE
RLTrainer = rl_trainer.RLTrainer
SoftmaxPolicy = rl_trainer.SoftmaxPolicy


def probe_observations(count: int = 12) -> list[list[float]]:
    rows = []
    for index in range(count):
        obs = [0.0] * 19
        obs[0] = index / max(1, count - 1)
        obs[1] = -0.8 + index * 0.12
        obs[3] = 1.0 if index % 2 else -1.0
        obs[6] = 0.9
        obs[7] = 0.8
        obs[11] = 1.0
        obs[12] = 1.0
        obs[13] = 1.0
        rows.append(obs)
    return rows


def evaluation(*, improved: bool, context: dict, capture: bool = False) -> dict:
    if improved:
        wins, losses, draws = 2, 1, 1
        reward, goal_diff, xg_diff = 1.0, 0.25, 0.3
    else:
        wins, losses, draws = 1, 1, 2
        reward, goal_diff, xg_diff = 0.0, 0.0, 0.0
    result = {
        "episodes": 4,
        "record": {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / 4,
            "loss_rate": losses / 4,
        },
        "avg_reward": reward,
        "avg_goal_diff": goal_diff,
        "avg_xg_diff": xg_diff,
        "avg_reward_terms": {"score": reward},
        "opponent_distribution": {"scripted": 2, "current_red": 1, "league": 1},
        "evaluation_context": dict(context),
        "latest": {},
        "rows": [],
        "replay_frames": 0,
    }
    if capture:
        result["_observations"] = probe_observations()
    return result


class SoccerTrainingIntegrityTests(unittest.TestCase):
    def make_trainer(self) -> tuple[tempfile.TemporaryDirectory, RLTrainer]:
        tempdir = tempfile.TemporaryDirectory()
        return tempdir, RLTrainer(Path(tempdir.name))

    def test_default_is_pure_rl_and_optional_coach_is_state_dependent(self) -> None:
        tempdir, trainer = self.make_trainer()
        self.addCleanup(tempdir.cleanup)

        self.assertFalse(trainer.coach_enabled)
        self.assertEqual(trainer.snapshot()["config"]["coach_mode"], "off_pure_rl")

        states = probe_observations(8)
        states[0][6] = 0.2  # low blue stamina -> conserve
        states[1][0], states[1][4] = 0.9, 0.3  # late lead -> low block
        states[2][4] = -0.7  # trailing -> direct attack
        states[3][3], states[3][1] = -1.0, -0.7  # defend deep -> counter
        states[4][3], states[4][1] = -1.0, 0.2  # defend higher -> press
        states[5][3], states[5][1] = 1.0, 0.7  # possession high -> attack
        actions = {ACTION_NAMES[trainer._coach_blue(obs)] for obs in states}

        self.assertGreaterEqual(len(actions), 5)
        self.assertIn("conserve", actions)
        self.assertIn("direct_attack", actions)
        self.assertNotEqual(actions, {"possession"})

        def forbidden_imitation(*args, **kwargs):
            del args, kwargs
            raise AssertionError("default pure-RL episode called the optional imitation update")

        trainer.policy.imitate = forbidden_imitation
        trainer._train_episode(persist=False)
        with self.assertRaisesRegex(ValueError, "cannot be disabled"):
            trainer.update_config({"guard_enabled": False})
        with self.assertRaisesRegex(RuntimeError, "live-policy mutation"):
            trainer.train_episode(persist=False)
        with self.assertRaisesRegex(RuntimeError, "live-policy mutation"):
            trainer.train_episode(persist=True)

    def test_all_training_config_is_persisted_and_reloaded_with_exact_clamps(self) -> None:
        tempdir, trainer = self.make_trainer()
        self.addCleanup(tempdir.cleanup)
        trainer.update_config(
            {
                "learning_rate": 0.0015,
                "gamma": 0.91,
                "temperature": 0.75,
                "self_play": True,
                "league_enabled": False,
                "guard_enabled": True,
                "guard_batch_episodes": 7,
                "guard_eval_episodes": 12,
                "guard_accept_margin": 0.2,
                "guard_min_effect": 1.25,
                "guard_min_action_change_rate": 0.02,
                "guard_opponent": "current_red",
                "coach_enabled": True,
                "coach_rate": 0.0007,
            }
        )
        trainer.rollout = 23
        rng_obs = probe_observations(1)[0]
        for _ in range(5):
            trainer.policy.sample(rng_obs, temperature=0.75)
        expected_continuation = trainer.policy.clone(preserve_rng=True)
        trainer._save_checkpoint()

        payload = json.loads(trainer.checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["checkpoint_version"], CHECKPOINT_VERSION)
        self.assertEqual(payload["config"]["learning_rate"], 0.0015)
        self.assertEqual(payload["config"]["gamma"], 0.91)
        self.assertEqual(payload["config"]["temperature"], 0.75)
        self.assertTrue(payload["config"]["self_play"])

        restored = RLTrainer(Path(tempdir.name))
        self.assertEqual(restored.learning_rate, 0.0015)
        self.assertEqual(restored.gamma, 0.91)
        self.assertEqual(restored.temperature, 0.75)
        self.assertTrue(restored.self_play)
        self.assertFalse(restored.league_enabled)
        self.assertEqual(restored.guard_batch_episodes, 7)
        self.assertEqual(restored.guard_eval_episodes, 32)
        self.assertEqual(restored.guard_accept_margin, 0.2)
        self.assertEqual(restored.guard_min_effect, 1.25)
        self.assertEqual(restored.guard_min_action_change_rate, 0.02)
        self.assertEqual(restored.guard_opponent, "current_red")
        self.assertTrue(restored.coach_enabled)
        self.assertEqual(restored.coach_rate, 0.0007)
        self.assertEqual(restored.rollout, 23)
        for _ in range(5):
            self.assertEqual(
                restored.policy.sample(rng_obs, temperature=0.75)[0],
                expected_continuation.sample(rng_obs, temperature=0.75)[0],
            )
        # The auxiliary update can never dominate the policy-gradient update.
        self.assertEqual(restored.snapshot()["config"]["coach_effective_rate"], 0.0015 * 0.25)
        restored.update_config({"guard_min_effect": 0.0})
        self.assertEqual(restored.guard_min_effect, rl_trainer.GUARD_MIN_EFFECT)

    def test_guard_uses_same_frozen_opponents_and_accepts_action_holdout_improvement(self) -> None:
        tempdir, trainer = self.make_trainer()
        self.addCleanup(tempdir.cleanup)
        trainer.guard_opponent = "mixed"
        trainer.guard_min_effect = 0.05
        trainer.guard_min_action_change_rate = 0.001
        league_policy = SoftmaxPolicy(trainer.policy.obs_dim, trainer.policy.action_dim, seed=99)
        trainer.league_pool = [
            {
                "id": 4,
                "name": "frozen@4",
                "elo": 1000.0,
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "policy": league_policy,
            }
        ]
        calls: list[dict] = []

        def fake_train(this, *, persist: bool = True):
            del persist
            with this.lock:
                episode = this.episode
                this.episode += 1
                this.rollout += 1
                # Blue changes to a distinct greedy action. Red and league are
                # also mutated to prove they cannot leak into candidate eval.
                this.policy.weights[ACTION_NAMES.index("direct_attack")][11] = 5.0
                this.red_policy.weights[ACTION_NAMES.index("low_block")][11] = 8.0
                this.league_pool[0]["policy"].weights[ACTION_NAMES.index("possession")][11] = 9.0
                this.latest_info = {"episode": episode}
                this.latest_replay = []
                return dict(this.latest_info)

        def fake_evaluate(this, blue_policy, red_policy, pool, seeds, **kwargs):
            del this
            calls.append(
                {
                    "blue": RLTrainer._policy_fingerprint(blue_policy),
                    "red": RLTrainer._policy_fingerprint(red_policy),
                    "pool": [RLTrainer._policy_fingerprint(row["policy"]) for row in pool],
                    "schedule_ids": [id(row) for row in kwargs["opponent_schedule"]],
                    "schedule": [
                        (row.get("kind"), row.get("id"), row.get("name"))
                        for row in kwargs["opponent_schedule"]
                    ],
                    "seeds": list(seeds),
                    "context": dict(kwargs["evaluation_context"]),
                }
            )
            return evaluation(
                improved=len(calls) in {2, 4},
                context=kwargs["evaluation_context"],
                capture=bool(kwargs.get("capture_observations")),
            )

        trainer._train_episode = types.MethodType(fake_train, trainer)
        trainer._evaluate_policy = types.MethodType(fake_evaluate, trainer)
        result = trainer.train_guarded_batch(1, eval_episodes=32)

        self.assertTrue(result["accepted"])
        self.assertEqual(len(calls), 4)
        self.assertNotEqual(calls[0]["blue"], calls[1]["blue"])
        self.assertEqual(calls[0]["red"], calls[1]["red"])
        self.assertEqual(calls[0]["pool"], calls[1]["pool"])
        self.assertEqual(calls[0]["schedule_ids"], calls[1]["schedule_ids"])
        self.assertEqual(calls[0]["schedule"], calls[1]["schedule"])
        self.assertEqual(calls[0]["context"], calls[1]["context"])
        self.assertEqual(
            calls[0]["seeds"],
            [rl_trainer.namespaced_seed(GUARD_GATE_SEED_NAMESPACE, i) for i in range(32)],
        )
        self.assertEqual(
            calls[2]["seeds"],
            [
                rl_trainer.namespaced_seed(GUARD_PROMOTION_SEED_NAMESPACE, i)
                for i in range(GUARD_PROMOTION_EPISODES)
            ],
        )
        self.assertEqual(calls[2]["context"], calls[3]["context"])
        self.assertNotEqual(calls[0]["context"], calls[2]["context"])
        self.assertGreater(result["guard"]["behavior"]["changed_actions"], 0)
        self.assertGreaterEqual(result["guard"]["objective_delta"], result["guard"]["min_effect"])
        self.assertTrue(result["guard"]["outcome_non_regression"])
        self.assertTrue(result["guard"]["promotion"]["beats_current"])
        self.assertTrue(result["guard"]["promotion"]["promoted"])

        best_payload = json.loads(trainer.best_checkpoint_path.read_text(encoding="utf-8"))
        context_id = result["guard"]["promotion"]["context_id"]
        self.assertEqual(best_payload["objective_context"]["context_id"], context_id)
        restored = RLTrainer(Path(tempdir.name))
        self.assertEqual(restored.best_guard_context, context_id)
        self.assertEqual(restored.best_guard_objective, trainer.best_guard_objective)

    def test_canonical_holdout_failure_cannot_be_accepted_or_served(self) -> None:
        tempdir, trainer = self.make_trainer()
        self.addCleanup(tempdir.cleanup)
        original = trainer.policy.to_json()
        calls = 0

        def fake_train(this, *, persist: bool = False):
            del persist
            this.policy.weights[ACTION_NAMES.index("direct_attack")][11] = 5.0
            this.episode += 1
            this.rollout += 1
            this.latest_info = {"episode": 0}
            this.latest_replay = []
            return dict(this.latest_info)

        def fake_evaluate(this, blue_policy, red_policy, pool, seeds, **kwargs):
            nonlocal calls
            del this, blue_policy, red_policy, pool, seeds
            calls += 1
            # Rotating gate passes (call 2); canonical candidate deliberately
            # ties its baseline (calls 3/4), so it must still be rejected.
            return evaluation(
                improved=calls == 2,
                context=kwargs["evaluation_context"],
                capture=bool(kwargs.get("capture_observations")),
            )

        trainer._train_episode = types.MethodType(fake_train, trainer)
        trainer._evaluate_policy = types.MethodType(fake_evaluate, trainer)
        result = trainer.train_guarded_batch(1, eval_episodes=32)

        self.assertFalse(result["accepted"])
        self.assertIn(
            "canonical_holdout_did_not_confirm_improvement",
            result["guard"]["rejection_reasons"],
        )
        self.assertEqual(trainer.policy.to_json(), original)
        self.assertFalse(trainer.best_checkpoint_path.exists())
        self.assertIsNone(trainer.staged_policy)

    def test_incomparable_canonical_history_directly_rejects_candidate(self) -> None:
        tempdir, trainer = self.make_trainer()
        self.addCleanup(tempdir.cleanup)
        original = trainer.policy.to_json()
        trainer.best_guard_objective = 10.0
        trainer.best_guard_context = "prior-canonical-protocol"
        calls = 0

        def fake_train(this, *, persist: bool = False):
            del persist
            this.policy.weights[ACTION_NAMES.index("direct_attack")][11] = 5.0
            this.episode += 1
            this.rollout += 1
            this.latest_info = {"episode": 0}
            this.latest_replay = []
            return dict(this.latest_info)

        def fake_evaluate(this, blue_policy, red_policy, pool, seeds, **kwargs):
            nonlocal calls
            del this, blue_policy, red_policy, pool, seeds
            calls += 1
            return evaluation(
                improved=calls in {2, 4},
                context=kwargs["evaluation_context"],
                capture=bool(kwargs.get("capture_observations")),
            )

        # An old in-memory context is not a persistable checkpoint, but the
        # guard must still reject it explicitly before any promotion attempt.
        trainer._save_checkpoint = types.MethodType(lambda this: None, trainer)
        trainer._train_episode = types.MethodType(fake_train, trainer)
        trainer._evaluate_policy = types.MethodType(fake_evaluate, trainer)
        result = trainer.train_guarded_batch(1, eval_episodes=32)

        self.assertFalse(result["accepted"])
        self.assertEqual(
            result["guard"]["rejection_reasons"],
            ["canonical_history_context_incomparable"],
        )
        self.assertFalse(result["guard"]["promotion"]["comparable_history"])
        self.assertFalse(result["guard"]["promotion"]["promoted"])
        self.assertEqual(trainer.policy.to_json(), original)
        self.assertIsNone(trainer.staged_policy)

    def test_weight_only_candidate_is_rejected_then_rollback_checkpoint_reloads(self) -> None:
        tempdir, trainer = self.make_trainer()
        self.addCleanup(tempdir.cleanup)
        trainer.update_config({"learning_rate": 0.0012, "gamma": 0.93, "temperature": 0.8})
        original_policy = trainer.policy.to_json()
        original_red = trainer.red_policy.to_json()
        calls = 0
        attempted_weights = None

        def fake_train(this, *, persist: bool = True):
            nonlocal attempted_weights
            del persist
            with this.lock:
                this.episode += 1
                this.rollout += 1
                # Adding the same logit offset to every action changes weights
                # but leaves probabilities and greedy behavior identical.
                for row in this.policy.weights:
                    row[11] += 5.0
                this.red_policy.weights[2][11] = 7.0
                this.elo += 80.0
                this.history.append({"result": "win"})
                this.latest_info = {"episode": this.episode - 1}
                attempted_weights = this.policy.to_json()
                return dict(this.latest_info)

        def fake_evaluate(this, blue_policy, red_policy, pool, seeds, **kwargs):
            nonlocal calls
            del this, blue_policy, red_policy, pool, seeds
            calls += 1
            return evaluation(
                improved=calls == 2,
                context=kwargs["evaluation_context"],
                capture=bool(kwargs.get("capture_observations")),
            )

        trainer._train_episode = types.MethodType(fake_train, trainer)
        trainer._evaluate_policy = types.MethodType(fake_evaluate, trainer)
        result = trainer.train_guarded_batch(1, eval_episodes=32)

        self.assertFalse(result["accepted"])
        self.assertIn("no_material_action_change_on_holdout_states", result["guard"]["rejection_reasons"])
        self.assertEqual(
            result["guard"]["rejection_reasons"],
            ["no_material_action_change_on_holdout_states"],
        )
        self.assertNotEqual(attempted_weights, original_policy)
        self.assertEqual(trainer.policy.to_json(), original_policy)
        self.assertEqual(trainer.red_policy.to_json(), original_red)
        self.assertEqual(trainer.episode, 0)
        self.assertEqual(trainer.elo, 1000.0)
        self.assertEqual(trainer.history, [])
        self.assertEqual(trainer.rollout, 1)
        self.assertIsNotNone(trainer.staged_policy)
        self.assertEqual(trainer.staged_batches, 1)
        self.assertFalse(trainer.snapshot()["staged_candidate"]["served"])
        durable_payload = json.loads(
            trainer.checkpoint_path.read_text(encoding="utf-8")
        )
        self.assertNotIn("staged_candidate", durable_payload)

        restored = RLTrainer(Path(tempdir.name))
        self.assertEqual(restored.policy.to_json(), original_policy)
        self.assertEqual(restored.red_policy.to_json(), original_red)
        self.assertEqual(restored.episode, 0)
        self.assertEqual(restored.rollout, 1)
        self.assertEqual(restored.learning_rate, 0.0012)
        self.assertEqual(restored.gamma, 0.93)
        self.assertEqual(restored.temperature, 0.8)
        self.assertIsNone(restored.staged_policy)
        self.assertEqual(restored.staged_batches, 0)

    def test_config_save_waits_for_guard_and_never_checkpoints_candidate(self) -> None:
        tempdir, trainer = self.make_trainer()
        self.addCleanup(tempdir.cleanup)
        trainer._save_checkpoint()
        accepted_policy = trainer.policy.to_json()
        entered = threading.Event()
        release = threading.Event()

        def blocked_train(this, *, persist: bool = False):
            del persist
            this.policy.weights[1][0] = 9.0
            this.episode += 1
            this.rollout += 1
            this.history.append({"result": "win", "candidate": True})
            this.latest_info = {"episode": 0, "candidate": True}
            this.latest_replay = [{"candidate": True}]
            entered.set()
            self.assertTrue(release.wait(5))
            return dict(this.latest_info)

        def unchanged_eval(this, blue_policy, red_policy, pool, seeds, **kwargs):
            del this, blue_policy, red_policy, pool, seeds
            return evaluation(
                improved=False,
                context=kwargs["evaluation_context"],
                capture=bool(kwargs.get("capture_observations")),
            )

        trainer._train_episode = types.MethodType(blocked_train, trainer)
        trainer._evaluate_policy = types.MethodType(unchanged_eval, trainer)
        guard_thread = threading.Thread(
            target=lambda: trainer.train_guarded_batch(1, eval_episodes=32)
        )
        guard_thread.start()
        self.assertTrue(entered.wait(5))

        update_done = threading.Event()

        def update_config():
            trainer.update_config({"gamma": 0.9})
            update_done.set()

        update_thread = threading.Thread(target=update_config)
        update_thread.start()
        time.sleep(0.05)
        self.assertFalse(update_done.is_set())
        public = trainer.snapshot()
        self.assertTrue(public["guard_in_progress"])
        self.assertEqual(public["episode"], 0)
        self.assertEqual(public["history"], [])
        self.assertEqual(trainer.latest_replay_payload()["frames"], [])
        mid_guard = json.loads(trainer.checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(mid_guard["policy"]["weights"], accepted_policy["weights"])

        release.set()
        guard_thread.join(5)
        update_thread.join(5)
        self.assertFalse(guard_thread.is_alive())
        self.assertFalse(update_thread.is_alive())
        final = json.loads(trainer.checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(final["policy"]["weights"], accepted_policy["weights"])
        self.assertEqual(final["config"]["gamma"], 0.9)

    def test_rejected_attempts_use_fresh_gate_seeds(self) -> None:
        tempdir, trainer = self.make_trainer()
        self.addCleanup(tempdir.cleanup)
        seed_batches: list[list[int | str]] = []

        def weight_only_train(this, *, persist: bool = False):
            del persist
            for row in this.policy.weights:
                row[11] += 1.0
            this.episode += 1
            this.rollout += 1
            this.latest_info = {"episode": this.episode - 1}
            this.latest_replay = []
            return dict(this.latest_info)

        def unchanged_eval(this, blue_policy, red_policy, pool, seeds, **kwargs):
            del this, blue_policy, red_policy, pool
            seed_batches.append(list(seeds))
            return evaluation(
                improved=False,
                context=kwargs["evaluation_context"],
                capture=bool(kwargs.get("capture_observations")),
            )

        trainer._train_episode = types.MethodType(weight_only_train, trainer)
        trainer._evaluate_policy = types.MethodType(unchanged_eval, trainer)
        first = trainer.train_guarded_batch(1, eval_episodes=32)
        second = trainer.train_guarded_batch(1, eval_episodes=32)

        self.assertFalse(first["accepted"])
        self.assertFalse(second["accepted"])
        self.assertEqual(len(seed_batches), 4)
        self.assertEqual(seed_batches[0], seed_batches[1])
        self.assertEqual(seed_batches[2], seed_batches[3])
        self.assertNotEqual(seed_batches[0], seed_batches[2])
        self.assertEqual(seed_batches[0][0], GUARD_GATE_SEED_BASE)
        self.assertEqual(
            seed_batches[2][0],
            rl_trainer.namespaced_seed(
                GUARD_GATE_SEED_NAMESPACE, rl_trainer.GUARD_GATE_SEED_STRIDE
            ),
        )

    def test_seed_namespaces_are_disjoint_from_integer_audit_seeds(self) -> None:
        train_seeds = {
            rl_trainer.namespaced_seed(TRAIN_SEED_NAMESPACE, index)
            for index in (0, 1, 910_000, 1_500_000_000, 10**30)
        }
        gate_seeds = {
            rl_trainer.namespaced_seed(GUARD_GATE_SEED_NAMESPACE, index)
            for index in range(5)
        }
        promotion_seeds = {
            rl_trainer.namespaced_seed(GUARD_PROMOTION_SEED_NAMESPACE, index)
            for index in range(5)
        }
        audit_seeds = {910_000 + index for index in range(5)} | {
            1_500_000_000 + index for index in range(5)
        }

        self.assertTrue(all(isinstance(seed, str) for seed in train_seeds))
        self.assertTrue(train_seeds.isdisjoint(gate_seeds))
        self.assertTrue(train_seeds.isdisjoint(promotion_seeds))
        self.assertTrue(gate_seeds.isdisjoint(promotion_seeds))
        self.assertTrue(train_seeds.isdisjoint(audit_seeds))
        self.assertTrue(gate_seeds.isdisjoint(audit_seeds))
        self.assertTrue(promotion_seeds.isdisjoint(audit_seeds))
        self.assertEqual(
            [GUARD_PROMOTION_SEED_BASE + index for index in range(5)],
            [
                rl_trainer.namespaced_seed(GUARD_PROMOTION_SEED_NAMESPACE, index)
                for index in range(5)
            ],
        )

    def test_guard_exception_rolls_back_before_checkpoint(self) -> None:
        tempdir, trainer = self.make_trainer()
        self.addCleanup(tempdir.cleanup)
        trainer._save_checkpoint()
        accepted_policy = trainer.policy.to_json()

        def failing_train(this, *, persist: bool = False):
            del persist
            this.policy.weights[3][0] = 12.0
            this.rollout += 1
            raise RuntimeError("synthetic candidate failure")

        trainer._train_episode = types.MethodType(failing_train, trainer)
        with self.assertRaisesRegex(RuntimeError, "synthetic candidate failure"):
            trainer.train_guarded_batch(1, eval_episodes=32)

        self.assertFalse(trainer.guard_in_progress)
        self.assertEqual(trainer.policy.to_json(), accepted_policy)
        durable = json.loads(trainer.checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(durable["policy"]["weights"], accepted_policy["weights"])

    def test_main_checkpoint_failure_rolls_back_and_best_failure_keeps_only_verified_main(self) -> None:
        def install_improving_candidate(trainer):
            calls = {"count": 0}

            def fake_train(this, *, persist: bool = False):
                del persist
                this.policy.weights[ACTION_NAMES.index("direct_attack")][11] = 5.0
                this.episode += 1
                this.rollout += 1
                this.latest_info = {"episode": 0}
                this.latest_replay = []
                return dict(this.latest_info)

            def fake_evaluate(this, blue_policy, red_policy, pool, seeds, **kwargs):
                del this, blue_policy, red_policy, pool, seeds
                calls["count"] += 1
                return evaluation(
                    improved=calls["count"] in {2, 4},
                    context=kwargs["evaluation_context"],
                    capture=bool(kwargs.get("capture_observations")),
                )

            trainer._train_episode = types.MethodType(fake_train, trainer)
            trainer._evaluate_policy = types.MethodType(fake_evaluate, trainer)

        first_dir, first = self.make_trainer()
        self.addCleanup(first_dir.cleanup)
        first._save_checkpoint()
        first_original = first.policy.to_json()
        install_improving_candidate(first)
        original_atomic = first._atomic_write_json
        failed = {"main": False}

        def fail_first_main(path, payload, **kwargs):
            if path == first.checkpoint_path and not failed["main"]:
                failed["main"] = True
                raise OSError("synthetic main checkpoint failure")
            return original_atomic(path, payload, **kwargs)

        first._atomic_write_json = fail_first_main
        with self.assertRaisesRegex(OSError, "synthetic main checkpoint failure"):
            first.train_guarded_batch(1, eval_episodes=32)
        self.assertEqual(first.policy.to_json(), first_original)
        first_durable = json.loads(first.checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(
            first_durable["policy"]["weights"], first_original["weights"]
        )
        self.assertFalse(first.best_checkpoint_path.exists())

        second_dir, second = self.make_trainer()
        self.addCleanup(second_dir.cleanup)
        second._save_checkpoint()
        second_original = second.policy.to_json()
        install_improving_candidate(second)

        def fail_best(_payload):
            raise OSError("synthetic best checkpoint failure")

        second._write_best_checkpoint = fail_best
        result = second.train_guarded_batch(1, eval_episodes=32)
        self.assertTrue(result["accepted"])
        self.assertNotEqual(second.policy.to_json(), second_original)
        self.assertFalse(second.best_checkpoint_path.exists())
        self.assertFalse(result["guard"]["promotion"]["promoted"])
        self.assertIn("promotion_error", result["guard"]["promotion"])
        second_durable = json.loads(second.checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(
            second_durable["policy"]["weights"], second.policy.to_json()["weights"]
        )
        self.assertIsNone(second_durable["best_guard_objective"])

    def test_close_waits_for_worker_rollback_before_durable_save(self) -> None:
        tempdir, trainer = self.make_trainer()
        self.addCleanup(tempdir.cleanup)
        trainer._save_checkpoint()
        accepted_policy = trainer.policy.to_json()
        entered = threading.Event()
        release = threading.Event()

        def blocked_train(this, *, persist: bool = False):
            del persist
            this.policy.weights[2][0] = 7.0
            entered.set()
            self.assertTrue(release.wait(5))
            this.episode += 1
            this.rollout += 1
            this.latest_info = {"episode": 0}
            this.latest_replay = []
            return dict(this.latest_info)

        def unchanged_eval(this, blue_policy, red_policy, pool, seeds, **kwargs):
            del this, blue_policy, red_policy, pool, seeds
            return evaluation(
                improved=False,
                context=kwargs["evaluation_context"],
                capture=bool(kwargs.get("capture_observations")),
            )

        trainer._train_episode = types.MethodType(blocked_train, trainer)
        trainer._evaluate_policy = types.MethodType(unchanged_eval, trainer)
        worker = threading.Thread(
            target=lambda: trainer.train_guarded_batch(1, eval_episodes=32)
        )
        trainer.thread = worker
        worker.start()
        self.assertTrue(entered.wait(5))
        closer = threading.Thread(target=trainer.close)
        closer.start()
        time.sleep(0.05)
        self.assertTrue(closer.is_alive())
        mid_guard = json.loads(trainer.checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(mid_guard["policy"]["weights"], accepted_policy["weights"])

        release.set()
        closer.join(5)
        self.assertFalse(closer.is_alive())
        self.assertIsNone(trainer.thread)
        durable = json.loads(trainer.checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(durable["policy"]["weights"], accepted_policy["weights"])

    def test_legacy_checkpoint_is_not_loaded_as_a_verified_policy(self) -> None:
        tempdir, trainer = self.make_trainer()
        self.addCleanup(tempdir.cleanup)
        trainer._save_checkpoint()
        payload = json.loads(trainer.checkpoint_path.read_text(encoding="utf-8"))
        payload["checkpoint_version"] = CHECKPOINT_VERSION - 1
        payload["policy"]["weights"][1][0] = 99.0
        trainer.checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

        restored = RLTrainer(Path(tempdir.name))
        self.assertEqual(restored.policy.weights[1][0], 0.0)
        self.assertEqual(restored.episode, 0)
        self.assertIn("checkpoint invalid", restored.last_event)

        payload["checkpoint_version"] = CHECKPOINT_VERSION
        payload["guard_objective_version"] = rl_trainer.GUARD_OBJECTIVE_VERSION
        payload["best_guard_objective"] = 123.0
        payload["best_guard_context"] = "incomparable-context"
        trainer.checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
        context_corrupt = RLTrainer(Path(tempdir.name))
        self.assertEqual(context_corrupt.policy.weights[1][0], 0.0)
        self.assertIn("checkpoint invalid", context_corrupt.last_event)

    def test_canonical_promotion_context_ignores_live_red_pool_and_settings(self) -> None:
        tempdir, trainer = self.make_trainer()
        self.addCleanup(tempdir.cleanup)
        policy = trainer.policy.clone(seed=11)
        first = trainer._evaluate_promotion_policy(policy)

        trainer.guard_opponent = "league"
        trainer.red_policy.weights[2][11] = 99.0
        league_policy = SoftmaxPolicy(policy.obs_dim, policy.action_dim, seed=33)
        league_policy.weights[4][11] = 77.0
        trainer.league_pool = [
            {
                "id": 99,
                "name": "mutable-live-opponent",
                "elo": 1800.0,
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "policy": league_policy,
            }
        ]
        second = trainer._evaluate_promotion_policy(policy)

        self.assertEqual(first, second)
        context = first["evaluation_context"]
        self.assertEqual(context["purpose"], "canonical_best_promotion_only")
        self.assertTrue(context["context_id"])

    def test_real_frozen_evaluation_is_deterministic_and_read_only(self) -> None:
        tempdir, trainer = self.make_trainer()
        self.addCleanup(tempdir.cleanup)
        seeds = [
            rl_trainer.namespaced_seed(GUARD_PROMOTION_SEED_NAMESPACE, i)
            for i in range(32)
        ]
        blue = trainer.policy.clone(seed=1)
        red = trainer.red_policy.clone(seed=2)
        pool: list[dict] = []
        schedule = trainer._build_evaluation_schedule("mixed", len(seeds), pool)
        context = trainer._evaluation_context(red, pool, schedule, seeds)
        red_before = red.to_json()

        first = trainer._evaluate_policy(
            blue,
            red,
            pool,
            seeds,
            opponent="mixed",
            opponent_schedule=schedule,
            evaluation_context=context,
        )
        second = trainer._evaluate_policy(
            blue,
            red,
            pool,
            seeds,
            opponent="mixed",
            opponent_schedule=schedule,
            evaluation_context=context,
        )

        self.assertEqual(first, second)
        self.assertEqual(red.to_json(), red_before)
        self.assertEqual(first["evaluation_context"], context)

    def test_real_pure_rl_batch_must_change_actions_and_improve_fixed_holdout(self) -> None:
        tempdir, trainer = self.make_trainer()
        self.addCleanup(tempdir.cleanup)
        self.assertFalse(trainer.coach_enabled)
        starting_policy_copy = trainer.policy.clone(seed=101, preserve_rng=True)
        starting_policy = trainer._policy_fingerprint(trainer.policy)
        accepted = None

        # The sequence is deterministic. A regressing first proposal is rolled
        # back; a later rollout is accepted only after measured football gains.
        for _ in range(6):
            result = trainer.train_guarded_batch(20, eval_episodes=32)
            if result["accepted"]:
                accepted = result
                break

        self.assertIsNotNone(accepted)
        assert accepted is not None
        guard = accepted["guard"]
        self.assertNotEqual(trainer._policy_fingerprint(trainer.policy), starting_policy)
        self.assertGreater(guard["behavior"]["changed_actions"], 0)
        self.assertGreaterEqual(guard["behavior"]["change_rate"], guard["min_action_change_rate"])
        self.assertGreaterEqual(guard["objective_delta"], guard["min_effect"])
        self.assertGreaterEqual(guard["candidate_match_points"], guard["baseline_match_points"])
        self.assertEqual(guard["evaluation_context"], accepted["baseline"]["evaluation_context"])
        self.assertEqual(guard["evaluation_context"], accepted["candidate"]["evaluation_context"])

        # This range is neither the rotating gate nor canonical promotion set.
        # It prevents the test from passing merely by reasserting guard output.
        independent_seeds = [1_500_000_000 + index for index in range(128)]
        independent_red = SoftmaxPolicy(
            trainer.red_policy.obs_dim, trainer.red_policy.action_dim, seed=20260712
        )
        independent_pool: list[dict] = []
        independent_schedule = trainer._build_evaluation_schedule(
            "mixed", len(independent_seeds), independent_pool
        )
        external_baseline = trainer._evaluate_policy(
            starting_policy_copy,
            independent_red,
            independent_pool,
            independent_seeds,
            opponent="mixed",
            opponent_schedule=independent_schedule,
        )
        external_candidate = trainer._evaluate_policy(
            trainer.policy,
            independent_red,
            independent_pool,
            independent_seeds,
            opponent="mixed",
            opponent_schedule=independent_schedule,
        )
        self.assertGreaterEqual(
            trainer._match_points(external_candidate),
            trainer._match_points(external_baseline),
        )
        self.assertGreaterEqual(
            trainer._guard_objective(external_candidate)
            - trainer._guard_objective(external_baseline),
            trainer.guard_min_effect,
        )

    def test_ui_episode_count_and_input_bounds_match_backend(self) -> None:
        html = (SOCCER_ROOT / "web" / "index.html").read_text(encoding="utf-8")
        js = (SOCCER_ROOT / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn('<button id="rlStepBtn">10 Episodes</button>', html)
        self.assertIn('episodes: 10, eval_episodes: 32', js)
        self.assertIn('id="rlRate" type="number" min="0.0001" max="0.02"', html)
        self.assertIn('id="rlGamma" type="number" min="0.8" max="0.999"', html)
        self.assertIn('id="rlTemperature" type="number" min="0.25" max="2.5"', html)
        self.assertIn("Browser-only Mutation Demo", html)
        self.assertIn("Checkpointed Backend RL", html)
        self.assertIn("off = pure RL", html)


if __name__ == "__main__":
    unittest.main()
