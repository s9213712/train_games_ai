import math
from abc import ABC, abstractmethod
from collections import deque

import gymnasium as gym
import numpy as np

from snake_game import SnakeGame


class BaseSnakeEnv(gym.Env, ABC):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        seed=0,
        board_size=12,
        silent_mode=True,
        limit_step=True,
        render_mode=None,
        food_time_penalty=0.0,
        food_step_limit_multiplier=4.0,
        loop_penalty=0.0,
        loop_window=16,
        oscillation_penalty=0.0,
        oscillation_window=12,
        food_reward_bonus=0.0,
        distance_reward_scale=0.1,
        reachable_space_penalty=0.0,
        reachable_space_min_ratio=0.35,
    ):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.board_size = board_size
        self.grid_size = board_size**2
        self.render_mode = render_mode
        self.silent_mode = silent_mode if render_mode is None else False
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=self.silent_mode)
        self.game.reset()

        self.action_space = gym.spaces.Discrete(4)  # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        self.observation_space = self._make_observation_space()

        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size
        self.food_step_limit_multiplier = float(food_step_limit_multiplier)
        self.step_limit = self._make_step_limit(limit_step)
        self.food_time_penalty = float(food_time_penalty)
        self.loop_penalty = float(loop_penalty)
        self.loop_window = max(2, int(loop_window))
        self.oscillation_penalty = float(oscillation_penalty)
        self.oscillation_window = max(4, int(oscillation_window))
        self.food_reward_bonus = float(food_reward_bonus)
        self.distance_reward_scale = float(distance_reward_scale)
        self.reachable_space_penalty = float(reachable_space_penalty)
        self.reachable_space_min_ratio = max(0.0, min(1.0, float(reachable_space_min_ratio)))
        self.reward_step_counter = 0
        self.steps_since_food = 0
        self.loop_revisit_count = 0
        self.oscillation_count = 0
        self.recent_head_positions = deque(maxlen=self.loop_window)
        self.recent_head_positions.append(tuple(self.game.snake[0]))

    def _make_step_limit(self, limit_step=True):
        if not limit_step:
            return 1_000_000_000
        return max(self.board_size * 2, int(self.grid_size * self.food_step_limit_multiplier))

    @abstractmethod
    def _make_observation_space(self):
        raise NotImplementedError

    @abstractmethod
    def _generate_observation(self):
        raise NotImplementedError

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.seed(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

        self.game.reset()
        self.reward_step_counter = 0
        self.steps_since_food = 0
        self.loop_revisit_count = 0
        self.oscillation_count = 0
        self.recent_head_positions.clear()
        self.recent_head_positions.append(tuple(self.game.snake[0]))
        return self._generate_observation(), self._info()

    def step(self, action):
        done, info = self.game.step(int(action))
        obs = self._generate_observation()
        self.reward_step_counter += 1
        self.steps_since_food += 1

        terminated = bool(done)
        truncated = False
        reward = 0.0
        head = tuple(info["snake_head_pos"])
        loop_revisit = False
        oscillation = False
        info["starved"] = False

        if info["snake_size"] == self.grid_size:
            terminated = True
            reward = self._victory_reward(info)
            if not self.silent_mode:
                self.game.sound_victory.play()
            return obs, reward, terminated, truncated, info

        if self.reward_step_counter > self.step_limit:
            self.reward_step_counter = 0
            # This is an in-MDP starvation/failure condition: the agent failed
            # to reach food within the configured budget and receives the death
            # penalty below.  Marking it as a Gym time-limit truncation would
            # make SB3 bootstrap a terminal value on top of that penalty.
            terminated = True
            info["starved"] = True

        if not terminated and not truncated and not info["food_obtained"]:
            loop_revisit = head in self.recent_head_positions
            if loop_revisit:
                self.loop_revisit_count += 1
            oscillation = self._detect_oscillation(info)
            if oscillation:
                self.oscillation_count += 1

        info["steps_since_food"] = self.steps_since_food
        info["loop_revisit"] = loop_revisit
        info["loop_revisit_count"] = self.loop_revisit_count
        info["oscillation"] = oscillation
        info["oscillation_count"] = self.oscillation_count
        self._add_reachable_space_info(info)

        if terminated or truncated:
            reward = self._terminal_penalty(info)
        elif info["food_obtained"]:
            reward = (
                self._food_reward(info)
                - self._time_to_food_penalty(info)
                - self._reachable_space_penalty(info)
            )
            self.reward_step_counter = 0
            self.steps_since_food = 0
            self.loop_revisit_count = 0
            self.oscillation_count = 0
            self.recent_head_positions.clear()
            self.recent_head_positions.append(head)
        else:
            reward = (
                self._step_reward(info)
                - self._time_to_food_penalty(info)
                - self._loop_penalty(info)
                - self._oscillation_penalty(info)
                - self._reachable_space_penalty(info)
            )
            self.recent_head_positions.append(head)

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.silent_mode:
            return None
        self.game.render()
        return None

    def close(self):
        self.game.close()

    def action_masks(self):
        return self.get_action_mask()

    def get_action_mask(self):
        mask = np.array(
            [self._check_action_validity(action) for action in range(self.action_space.n)],
            dtype=bool,
        )
        if not mask.any():
            return np.ones(self.action_space.n, dtype=bool)
        return mask

    def _info(self):
        return {
            "snake_size": len(self.game.snake),
            "snake_head_pos": np.array(self.game.snake[0]),
            "food_pos": np.array(self.game.food),
            "food_obtained": False,
            "steps_since_food": self.steps_since_food,
            "loop_revisit": False,
            "loop_revisit_count": self.loop_revisit_count,
            "oscillation": False,
            "oscillation_count": self.oscillation_count,
            "starved": False,
            "reachable_space": 0,
            "reachable_space_ratio": 1.0,
        }

    def _check_action_validity(self, action):
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]

        if action == 0:
            if current_direction == "DOWN":
                return False
            row -= 1
        elif action == 1:
            if current_direction == "RIGHT":
                return False
            col -= 1
        elif action == 2:
            if current_direction == "LEFT":
                return False
            col += 1
        elif action == 3:
            if current_direction == "UP":
                return False
            row += 1

        if (row, col) == self.game.food:
            game_over = (row, col) in snake_list
        else:
            game_over = (row, col) in snake_list[:-1]

        return not (
            game_over
            or row < 0
            or row >= self.board_size
            or col < 0
            or col >= self.board_size
        )

    def _victory_reward(self, info):
        return self.max_growth * 0.1

    def _terminal_penalty(self, info):
        return (
            -math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth)
            * 0.1
        )

    def _food_reward(self, info):
        return self.food_reward_bonus + info["snake_size"] / self.grid_size

    def _step_reward(self, info):
        previous_distance = np.abs(info["prev_snake_head_pos"] - info["food_pos"]).sum()
        current_distance = np.abs(info["snake_head_pos"] - info["food_pos"]).sum()
        if current_distance < previous_distance:
            return self.distance_reward_scale / info["snake_size"]
        return -self.distance_reward_scale / info["snake_size"]

    def _time_to_food_penalty(self, info):
        return self.food_time_penalty * (1.0 + info["steps_since_food"] / self.grid_size)

    def _loop_penalty(self, info):
        if not info["loop_revisit"]:
            return 0.0
        scale = min(1.0, info["loop_revisit_count"] / max(1, self.loop_window // 4))
        return self.loop_penalty * scale

    def _detect_oscillation(self, info):
        positions = (list(self.recent_head_positions) + [tuple(info["snake_head_pos"])])[
            -self.oscillation_window:
        ]
        if len(positions) < self.oscillation_window:
            return False

        unique_positions = len(set(positions))
        rows = [pos[0] for pos in positions]
        cols = [pos[1] for pos in positions]
        row_span = max(rows) - min(rows)
        col_span = max(cols) - min(cols)
        current_distance = np.abs(info["snake_head_pos"] - info["food_pos"]).sum()
        start_distance = np.abs(np.array(positions[0]) - info["food_pos"]).sum()
        compact_area = (row_span + 1) * (col_span + 1)

        if current_distance < start_distance:
            return False
        if unique_positions <= max(4, self.oscillation_window // 2):
            return True
        return compact_area <= self.oscillation_window and min(row_span, col_span) <= 2

    def _oscillation_penalty(self, info):
        if not info["oscillation"]:
            return 0.0
        scale = min(1.0, info["oscillation_count"] / max(1, self.oscillation_window // 3))
        return self.oscillation_penalty * scale

    def _add_reachable_space_info(self, info):
        head = tuple(info["snake_head_pos"])
        snake = list(self.game.snake)
        blocked = set(snake[:-1])
        blocked.discard(head)
        queue = deque([head])
        seen = {head}

        while queue:
            row, col = queue.popleft()
            for row_delta, col_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (row + row_delta, col + col_delta)
                if (
                    nxt in seen
                    or nxt in blocked
                    or nxt[0] < 0
                    or nxt[0] >= self.board_size
                    or nxt[1] < 0
                    or nxt[1] >= self.board_size
                ):
                    continue
                seen.add(nxt)
                queue.append(nxt)

        available = max(1, self.grid_size - len(blocked))
        info["reachable_space"] = len(seen)
        info["reachable_space_ratio"] = len(seen) / available

    def _reachable_space_penalty(self, info):
        if self.reachable_space_penalty <= 0.0:
            return 0.0
        ratio = float(info.get("reachable_space_ratio", 1.0))
        if ratio >= self.reachable_space_min_ratio:
            return 0.0
        shortage = (self.reachable_space_min_ratio - ratio) / max(self.reachable_space_min_ratio, 1e-6)
        return self.reachable_space_penalty * shortage


class SnakeCnnEnv(BaseSnakeEnv):
    def __init__(self, *args, image_size=84, channel_first=False, **kwargs):
        self.image_size = image_size
        self.channel_first = bool(channel_first)
        super().__init__(*args, **kwargs)
        if self.image_size % self.board_size != 0:
            raise ValueError("image_size must be divisible by board_size for CNN observations")

    def _make_observation_space(self):
        shape = (
            (3, self.image_size, self.image_size)
            if self.channel_first
            else (self.image_size, self.image_size, 3)
        )
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=np.uint8,
        )

    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(
            200, 50, len(self.game.snake), dtype=np.uint8
        )
        obs = np.stack((obs, obs, obs), axis=-1)
        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]
        obs[self.game.food] = [0, 0, 255]

        scale = self.image_size // self.board_size
        obs = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1)
        if self.channel_first:
            obs = np.transpose(obs, (2, 0, 1))
        return obs


class SnakeMlpEnv(BaseSnakeEnv):
    def _make_observation_space(self):
        return gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.game.board_size, self.game.board_size),
            dtype=np.float32,
        )

    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.float32)
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(
            0.8, 0.2, len(self.game.snake), dtype=np.float32
        )
        obs[tuple(self.game.snake[0])] = 1.0
        obs[tuple(self.game.food)] = -1.0
        return obs

    def _terminal_penalty(self, info):
        return (info["snake_size"] - self.grid_size) * 0.1

    def _food_reward(self, info):
        speed_bonus = math.exp((self.grid_size - self.reward_step_counter) / self.grid_size) * 0.1
        return self.food_reward_bonus + speed_bonus
