import argparse
import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main"
sys.path.insert(0, str(MAIN))
sys.modules.setdefault("gym", gym)
sys.modules.setdefault("gym.spaces", gym.spaces)

from sb3_contrib import MaskablePPO  # noqa: E402

from snake_env import SnakeCnnEnv  # noqa: E402
from train import select_device  # noqa: E402
from safe_food_teacher import safe_food_action  # noqa: E402


DEFAULT_SOURCE = MAIN / "original_models" / "trained_models_cnn" / "ppo_snake_final.zip"


def generate_even_board_cycle(board_size):
    if board_size % 2 != 0:
        raise ValueError("Hamiltonian-cycle behavioral cloning currently requires an even board size.")

    path = [(0, col) for col in range(board_size)]
    for row in range(1, board_size):
        cols = range(board_size - 1, 0, -1) if row % 2 else range(1, board_size)
        for col in cols:
            path.append((row, col))
    for row in range(board_size - 1, 0, -1):
        path.append((row, 0))

    for current, nxt in zip(path, path[1:] + path[:1]):
        if abs(current[0] - nxt[0]) + abs(current[1] - nxt[1]) != 1:
            raise AssertionError(f"Non-adjacent cycle edge: {current} -> {nxt}")
    return path


def action_between(current, nxt):
    delta = (nxt[0] - current[0], nxt[1] - current[1])
    if delta == (-1, 0):
        return 0
    if delta == (0, -1):
        return 1
    if delta == (0, 1):
        return 2
    if delta == (1, 0):
        return 3
    raise ValueError(f"Cells are not adjacent: {current} -> {nxt}")


class CycleSampler:
    def __init__(self, board_size, seed):
        self.board_size = board_size
        self.seed = seed
        self.rng = random.Random(seed)
        self.cycle = generate_even_board_cycle(board_size)
        self.cycle_index = {cell: index for index, cell in enumerate(self.cycle)}
        self.env = SnakeCnnEnv(
            seed=seed,
            board_size=board_size,
            silent_mode=True,
            limit_step=False,
            channel_first=True,
        )
        self.obs, _ = self.env.reset(seed=seed)

    def reset(self):
        self.seed = self.rng.randint(0, 1_000_000_000)
        self.obs, _ = self.env.reset(seed=self.seed)

    def oracle_action(self):
        head = self.env.game.snake[0]
        index = self.cycle_index[head]
        nxt = self.cycle[(index + 1) % len(self.cycle)]
        return action_between(head, nxt)

    def sample(self, random_food_ratio=0.0, trap_food_ratio=0.5):
        action = self.oracle_action()
        obs = self.augmented_observation(action, random_food_ratio, trap_food_ratio)
        self.obs, _, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated or len(self.env.game.snake) == self.env.game.grid_size:
            self.reset()
        return obs, action

    def sample_policy_state(self, model, random_food_ratio=0.0, trap_food_ratio=0.5):
        oracle_action = self.oracle_action()
        obs = self.augmented_observation(oracle_action, random_food_ratio, trap_food_ratio)
        action, _ = model.predict(
            self.obs,
            action_masks=self.env.get_action_mask(),
            deterministic=True,
        )
        self.obs, _, terminated, truncated, _ = self.env.step(int(action))
        if terminated or truncated or len(self.env.game.snake) == self.env.game.grid_size:
            self.reset()
        return obs, oracle_action

    def augmented_observation(self, oracle_action, random_food_ratio, trap_food_ratio):
        if random.random() >= random_food_ratio:
            return self.obs

        original_food = self.env.game.food
        trap_food = self._trap_food(oracle_action)
        if trap_food is not None and random.random() < trap_food_ratio:
            self.env.game.food = trap_food
        elif self.env.game.non_snake:
            self.env.game.food = random.choice(tuple(self.env.game.non_snake))
        obs = self.env._generate_observation()
        self.env.game.food = original_food
        return obs

    def _trap_food(self, oracle_action):
        head_row, head_col = self.env.game.snake[0]
        candidates = []
        for action, (row_delta, col_delta) in {
            0: (-1, 0),
            1: (0, -1),
            2: (0, 1),
            3: (1, 0),
        }.items():
            if action == oracle_action:
                continue
            cell = (head_row + row_delta, head_col + col_delta)
            if cell in self.env.game.non_snake:
                candidates.append(cell)
        return random.choice(candidates) if candidates else None

    def close(self):
        self.env.close()


class SafeFoodSampler:
    def __init__(self, board_size, seed):
        self.board_size = board_size
        self.seed = seed
        self.rng = random.Random(seed)
        self.env = SnakeCnnEnv(
            seed=seed,
            board_size=board_size,
            silent_mode=True,
            limit_step=False,
            channel_first=True,
        )
        self.obs, _ = self.env.reset(seed=seed)

    def reset(self):
        self.seed = self.rng.randint(0, 1_000_000_000)
        self.obs, _ = self.env.reset(seed=self.seed)

    def oracle_action(self):
        return safe_food_action(self.env.game)

    def sample(self, random_food_ratio=0.0, trap_food_ratio=0.5):
        action = self.oracle_action()
        obs = self.augmented_observation(action, random_food_ratio, trap_food_ratio)
        self.obs, _, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated or len(self.env.game.snake) == self.env.game.grid_size:
            self.reset()
        return obs, action

    def sample_policy_state(self, model, random_food_ratio=0.0, trap_food_ratio=0.5):
        oracle_action = self.oracle_action()
        obs = self.augmented_observation(oracle_action, random_food_ratio, trap_food_ratio)
        action, _ = model.predict(
            self.obs,
            action_masks=self.env.get_action_mask(),
            deterministic=True,
        )
        self.obs, _, terminated, truncated, _ = self.env.step(int(action))
        if terminated or truncated or len(self.env.game.snake) == self.env.game.grid_size:
            self.reset()
        return obs, oracle_action

    def augmented_observation(self, oracle_action, random_food_ratio, trap_food_ratio):
        if random.random() >= random_food_ratio:
            return self.obs

        original_food = self.env.game.food
        trap_food = self._trap_food(oracle_action)
        if trap_food is not None and random.random() < trap_food_ratio:
            self.env.game.food = trap_food
        elif self.env.game.non_snake:
            self.env.game.food = random.choice(tuple(self.env.game.non_snake))
        obs = self.env._generate_observation()
        self.env.game.food = original_food
        return obs

    def _trap_food(self, oracle_action):
        head_row, head_col = self.env.game.snake[0]
        candidates = []
        for action, (row_delta, col_delta) in {
            0: (-1, 0),
            1: (0, -1),
            2: (0, 1),
            3: (1, 0),
        }.items():
            if action == oracle_action:
                continue
            cell = (head_row + row_delta, head_col + col_delta)
            if cell in self.env.game.non_snake:
                candidates.append(cell)
        return random.choice(candidates) if candidates else None

    def close(self):
        self.env.close()


def load_or_create_model(args, env, device):
    source = Path(args.source_model) if args.source_model else DEFAULT_SOURCE
    if source.exists():
        return MaskablePPO.load(
            source,
            env=env,
            device=device,
            custom_objects={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
            },
        )
    return MaskablePPO(
        "CnnPolicy",
        env,
        device=device,
        verbose=0,
        n_steps=2048,
        batch_size=512,
        n_epochs=4,
        gamma=0.94,
        learning_rate=args.learning_rate,
        clip_range=0.15,
    )


def configure_trainable_parameters(model, mode, learning_rate):
    if mode == "all":
        trainable = list(model.policy.parameters())
    else:
        prefixes = {
            "action-net": ("action_net.",),
            "heads": ("action_net.", "value_net."),
        }[mode]
        trainable = []
        for name, parameter in model.policy.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
            if parameter.requires_grad:
                trainable.append(parameter)

    if not trainable:
        raise ValueError(f"No trainable parameters selected for mode={mode}")

    optimizer_kwargs = dict(model.policy.optimizer.defaults)
    optimizer_kwargs["lr"] = learning_rate
    model.policy.optimizer = type(model.policy.optimizer)(trainable, **optimizer_kwargs)


def save_loadable_model(model, path):
    original_optimizer = model.policy.optimizer
    optimizer_kwargs = dict(original_optimizer.defaults)
    model.policy.optimizer = type(original_optimizer)(list(model.policy.parameters()), **optimizer_kwargs)
    try:
        model.save(path)
    finally:
        model.policy.optimizer = original_optimizer


def evaluate(model, board_size, seed, episodes, max_steps):
    was_training = model.policy.training
    model.policy.set_training_mode(False)
    scores = []
    lengths = []
    filled = 0
    try:
        for episode in range(episodes):
            env = SnakeCnnEnv(
                seed=seed + episode,
                board_size=board_size,
                silent_mode=True,
                limit_step=False,
                channel_first=True,
            )
            obs, _ = env.reset(seed=seed + episode)
            done = False
            steps = 0
            while not done and steps < max_steps and len(env.game.snake) < env.game.grid_size:
                action, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
            scores.append(env.game.score)
            lengths.append(len(env.game.snake))
            filled += int(len(env.game.snake) == env.game.grid_size)
            env.close()
    finally:
        model.policy.set_training_mode(was_training)
    return {
        "score_avg": float(np.mean(scores)),
        "score_max": int(np.max(scores)),
        "length_avg": float(np.mean(lengths)),
        "length_max": int(np.max(lengths)),
        "filled": filled,
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Behavior-clone a CNN policy from a Snake oracle.")
    parser.add_argument("--board-size", type=int, default=12)
    parser.add_argument("--oracle", choices=("auto", "cycle", "safe-food"), default="auto")
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--source-model", default=None)
    parser.add_argument("--output", default="main/trained_models_cnn_oracle_bc/ppo_snake_bc.zip")
    parser.add_argument("--best-output", default=None)
    parser.add_argument("--initial-best-score", type=float, default=None)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-max-steps", type=int, default=30000)
    parser.add_argument("--dagger-ratio", type=float, default=0.5)
    parser.add_argument("--dagger-warmup-batches", type=int, default=50)
    parser.add_argument("--random-food-ratio", type=float, default=0.0)
    parser.add_argument("--trap-food-ratio", type=float, default=0.5)
    parser.add_argument("--reference-kl-coef", type=float, default=0.0)
    parser.add_argument("--trainable", choices=("all", "action-net", "heads"), default="all")
    args = parser.parse_args()

    oracle = args.oracle
    if oracle == "auto":
        oracle = "cycle" if args.board_size % 2 == 0 else "safe-food"

    if oracle == "cycle" and args.board_size % 2 != 0:
        raise SystemExit(
            "Hamiltonian-cycle BC supports even board sizes only. "
            "Use --oracle safe-food for odd boards such as 21x21."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    train_env = SnakeCnnEnv(seed=args.seed, board_size=args.board_size, silent_mode=True, channel_first=True)
    model = load_or_create_model(args, train_env, device)
    model.policy.set_training_mode(True)
    configure_trainable_parameters(model, args.trainable, args.learning_rate)

    reference_model = None
    if args.reference_kl_coef > 0.0 and args.source_model:
        source = Path(args.source_model)
        if source.exists():
            reference_model = MaskablePPO.load(
                source,
                env=train_env,
                device=device,
                custom_objects={
                    "observation_space": train_env.observation_space,
                    "action_space": train_env.action_space,
                },
            )
            reference_model.policy.set_training_mode(False)
            for parameter in reference_model.policy.parameters():
                parameter.requires_grad_(False)

    sampler_cls = CycleSampler if oracle == "cycle" else SafeFoodSampler
    samplers = [sampler_cls(args.board_size, args.seed + rank * 1009) for rank in range(8)]
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 50_000
    best_score = args.initial_best_score if args.initial_best_score is not None else float("-inf")
    best_output = ROOT / args.best_output if args.best_output else None
    try:
        for batch_index in range(1, args.batches + 1):
            obs_batch = []
            action_batch = []
            for item in range(args.batch_size):
                sampler = samplers[item % len(samplers)]
                use_policy_state = (
                    batch_index > args.dagger_warmup_batches
                    and random.random() < args.dagger_ratio
                )
                if use_policy_state:
                    obs, action = sampler.sample_policy_state(
                        model,
                        args.random_food_ratio,
                        args.trap_food_ratio,
                    )
                else:
                    obs, action = sampler.sample(
                        args.random_food_ratio,
                        args.trap_food_ratio,
                    )
                obs_batch.append(obs)
                action_batch.append(action)

            obs_array = np.asarray(obs_batch)
            obs_tensor, _ = model.policy.obs_to_tensor(obs_array)
            actions = torch.as_tensor(action_batch, dtype=torch.long, device=model.policy.device)
            dist = model.policy.get_distribution(obs_tensor)
            logits = dist.distribution.logits
            bc_loss = F.cross_entropy(logits, actions)
            kl_loss = torch.zeros((), device=model.policy.device)
            if reference_model is not None:
                with torch.no_grad():
                    ref_dist = reference_model.policy.get_distribution(obs_tensor)
                    ref_logits = ref_dist.distribution.logits
                kl_loss = F.kl_div(
                    F.log_softmax(logits, dim=1),
                    F.softmax(ref_logits, dim=1),
                    reduction="batchmean",
                )
            loss = bc_loss + args.reference_kl_coef * kl_loss

            model.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()

            if batch_index == 1 or batch_index % 25 == 0:
                pred_acc = (logits.argmax(dim=1) == actions).float().mean().item()
                print(
                    f"batch={batch_index} loss={loss.item():.4f} "
                    f"bc={bc_loss.item():.4f} kl={kl_loss.item():.4f} acc={pred_acc:.3f}",
                    flush=True,
                )
            if args.eval_interval > 0 and batch_index % args.eval_interval == 0:
                metrics = evaluate(
                    model,
                    args.board_size,
                    eval_seed,
                    args.eval_episodes,
                    args.eval_max_steps,
                )
                print(f"eval batch={batch_index} metrics={metrics}", flush=True)
                if best_output and metrics["score_avg"] > best_score:
                    best_score = metrics["score_avg"]
                    best_output.parent.mkdir(parents=True, exist_ok=True)
                    save_loadable_model(model, best_output)
                    print(f"best_saved={best_output} best_score={best_score:.2f}", flush=True)
    finally:
        for sampler in samplers:
            sampler.close()
        train_env.close()

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    save_loadable_model(model, output)
    print(f"saved={output}")
    metrics = evaluate(model, args.board_size, eval_seed, args.eval_episodes, args.eval_max_steps)
    print(f"metrics={metrics}")
    if best_output and metrics["score_avg"] > best_score:
        best_output.parent.mkdir(parents=True, exist_ok=True)
        save_loadable_model(model, best_output)
        print(f"best_saved={best_output} best_score={metrics['score_avg']:.2f}")


if __name__ == "__main__":
    main()
