from __future__ import annotations

import torch as th
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class ConfigurableCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        channels=(32, 64, 64),
        kernel_sizes=(8, 4, 3),
        strides=(4, 2, 1),
        features_dim=512,
        normalized_image=False,
    ):
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(f"ConfigurableCNN requires a Box observation space, got {observation_space}")
        if not is_image_space(observation_space, check_channels=False, normalized_image=normalized_image):
            raise ValueError(f"ConfigurableCNN requires image observations, got {observation_space}")

        channels = tuple(int(value) for value in channels)
        kernel_sizes = tuple(int(value) for value in kernel_sizes)
        strides = tuple(int(value) for value in strides)
        if not channels:
            raise ValueError("At least one CNN layer is required")
        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("CNN channels, kernel sizes, and strides must have the same length")
        if any(value <= 0 for value in channels + kernel_sizes + strides):
            raise ValueError("CNN channels, kernel sizes, and strides must be positive")

        super().__init__(observation_space, int(features_dim))

        n_input_channels = observation_space.shape[0]
        layers = []
        in_channels = n_input_channels
        for out_channels, kernel_size, stride in zip(channels, kernel_sizes, strides):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        if n_flatten <= 0:
            raise ValueError("CNN architecture produced an empty feature map")

        self.linear = nn.Sequential(nn.Linear(n_flatten, int(features_dim)), nn.ReLU())
        self.architecture = {
            "input_channels": n_input_channels,
            "channels": channels,
            "kernel_sizes": kernel_sizes,
            "strides": strides,
            "flattened": int(n_flatten),
            "features_dim": int(features_dim),
        }

    def forward(self, observations):
        return self.linear(self.cnn(observations))
