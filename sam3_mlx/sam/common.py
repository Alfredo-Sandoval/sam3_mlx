from __future__ import annotations

from typing import Type

import mlx.core as mx
import mlx.nn as nn


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def __call__(self, x: mx.array) -> mx.array:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=1, keepdims=True)
        variance = mx.mean((x - mean) ** 2, axis=1, keepdims=True)
        x = (x - mean) / mx.sqrt(variance + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class Conv2dNCHW(nn.Module):
    """NCHW wrapper around MLX's NHWC Conv2d."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)

    @property
    def weight(self):
        return self.conv.weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


class ConvTranspose2dNCHW(nn.Module):
    """NCHW wrapper around MLX's NHWC ConvTranspose2d."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(*args, **kwargs)

    @property
    def weight(self):
        return self.conv.weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
