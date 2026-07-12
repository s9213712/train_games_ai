from __future__ import annotations

import torch as th
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


CNN_MAX_LAYERS = 5
CNN_MAX_CHANNELS = 512
CNN_MAX_FEATURES_DIM = 2048
CNN_MAX_KERNEL_SIZE = 84
CNN_MAX_STRIDE = 32
CNN_MAX_ESTIMATED_PARAMETERS = 25_000_000


def _strict_positive_int_tuple(name, values):
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be an integer list")
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        raise TypeError(f"{name} must contain integers")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(values)


def validate_cnn_spatial_shape(height, width, kernel_sizes, strides):
    """Return the final feature-map size or reject an impossible CNN stack."""

    if not isinstance(height, int) or isinstance(height, bool):
        raise TypeError("CNN input height must be an integer")
    if not isinstance(width, int) or isinstance(width, bool):
        raise TypeError("CNN input width must be an integer")
    kernels = _strict_positive_int_tuple("CNN kernel sizes", kernel_sizes)
    stride_values = _strict_positive_int_tuple("CNN strides", strides)
    if height < 1 or width < 1:
        raise ValueError(f"CNN input must be positive, got {height}x{width}")
    if len(kernels) != len(stride_values):
        raise ValueError("CNN kernel sizes and strides must have the same length")

    for index, (kernel_size, stride) in enumerate(
        zip(kernels, stride_values), start=1
    ):
        if kernel_size > height or kernel_size > width:
            raise ValueError(
                f"CNN layer {index} kernel {kernel_size} does not fit "
                f"the {height}x{width} input feature map"
            )
        height = (height - kernel_size) // stride + 1
        width = (width - kernel_size) // stride + 1
        if height < 1 or width < 1:
            raise ValueError(
                f"CNN layer {index} produces an empty feature map; "
                "reduce its kernel or stride"
            )
    return height, width


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

        channels = _strict_positive_int_tuple("CNN channels", channels)
        kernel_sizes = _strict_positive_int_tuple("CNN kernel sizes", kernel_sizes)
        strides = _strict_positive_int_tuple("CNN strides", strides)
        if not isinstance(features_dim, int) or isinstance(features_dim, bool):
            raise TypeError("CNN features_dim must be an integer")
        if features_dim <= 0:
            raise ValueError("CNN features_dim must be positive")
        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("CNN channels, kernel sizes, and strides must have the same length")
        if len(channels) > CNN_MAX_LAYERS:
            raise ValueError(f"CNN supports at most {CNN_MAX_LAYERS} convolution layers")
        if any(value > CNN_MAX_CHANNELS for value in channels):
            raise ValueError(f"CNN channel counts cannot exceed {CNN_MAX_CHANNELS}")
        if any(value > CNN_MAX_KERNEL_SIZE for value in kernel_sizes):
            raise ValueError(f"CNN kernel sizes cannot exceed {CNN_MAX_KERNEL_SIZE}")
        if any(value > CNN_MAX_STRIDE for value in strides):
            raise ValueError(f"CNN strides cannot exceed {CNN_MAX_STRIDE}")
        if features_dim > CNN_MAX_FEATURES_DIM:
            raise ValueError(f"CNN features_dim cannot exceed {CNN_MAX_FEATURES_DIM}")

        input_height, input_width = observation_space.shape[-2:]
        final_height, final_width = validate_cnn_spatial_shape(
            input_height,
            input_width,
            kernel_sizes,
            strides,
        )
        parameter_count = 0
        input_channels = int(observation_space.shape[0])
        in_channels = input_channels
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            parameter_count += out_channels * (
                in_channels * kernel_size * kernel_size + 1
            )
            in_channels = out_channels
        flattened = final_height * final_width * channels[-1]
        parameter_count += (flattened + 1) * features_dim
        if parameter_count > CNN_MAX_ESTIMATED_PARAMETERS:
            raise ValueError(
                "CNN architecture is too large "
                f"({parameter_count:,} estimated parameters; "
                f"limit {CNN_MAX_ESTIMATED_PARAMETERS:,})"
            )

        super().__init__(observation_space, features_dim)

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

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.architecture = {
            "input_channels": n_input_channels,
            "channels": channels,
            "kernel_sizes": kernel_sizes,
            "strides": strides,
            "flattened": int(n_flatten),
            "features_dim": features_dim,
        }

    def forward(self, observations):
        return self.linear(self.cnn(observations))
