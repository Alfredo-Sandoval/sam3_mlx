from __future__ import annotations

from importlib import import_module
from typing import Protocol, TypeAlias, cast

import mlx.core as mx


_SpatialArg: TypeAlias = int | tuple[int, int]


class _ArrayModule(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class _WeightedArrayModule(_ArrayModule, Protocol):
    @property
    def weight(self) -> mx.array: ...


class _ActivationFactory(Protocol):
    def __call__(self) -> _ArrayModule: ...


class _LinearFactory(Protocol):
    def __call__(
        self, input_dims: int, output_dims: int, bias: bool = True
    ) -> _ArrayModule: ...


class _Conv2dFactory(Protocol):
    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _SpatialArg,
        stride: _SpatialArg = 1,
        padding: _SpatialArg = 0,
        dilation: _SpatialArg = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> _WeightedArrayModule: ...


class _ConvTranspose2dFactory(Protocol):
    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _SpatialArg,
        stride: _SpatialArg = 1,
        padding: _SpatialArg = 0,
        dilation: _SpatialArg = 1,
        output_padding: _SpatialArg = 0,
        bias: bool = True,
    ) -> _WeightedArrayModule: ...


_nn = import_module("mlx.nn")
_Module = cast(type[object], getattr(_nn, "Module"))
_gelu = cast(_ActivationFactory, getattr(_nn, "GELU"))
_linear = cast(_LinearFactory, getattr(_nn, "Linear"))
_conv2d = cast(_Conv2dFactory, getattr(_nn, "Conv2d"))
_conv_transpose2d = cast(_ConvTranspose2dFactory, getattr(_nn, "ConvTranspose2d"))


class MLPBlock(_Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: _ActivationFactory = _gelu,
    ) -> None:
        super().__init__()
        self.lin1 = _linear(embedding_dim, mlp_dim)
        self.lin2 = _linear(mlp_dim, embedding_dim)
        self.act = act()

    def __call__(self, x: mx.array) -> mx.array:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(_Module):
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


class Conv2dNCHW(_Module):
    """NCHW wrapper around MLX's NHWC Conv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _SpatialArg,
        stride: _SpatialArg = 1,
        padding: _SpatialArg = 0,
        dilation: _SpatialArg = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = _conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    @property
    def weight(self) -> mx.array:
        return self.conv.weight

    def __call__(self, x: mx.array) -> mx.array:
        channels_last = mx.transpose(x, axes=(0, 2, 3, 1))
        return mx.transpose(self.conv(channels_last), axes=(0, 3, 1, 2))


class ConvTranspose2dNCHW(_Module):
    """NCHW wrapper around MLX's NHWC ConvTranspose2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _SpatialArg,
        stride: _SpatialArg = 1,
        padding: _SpatialArg = 0,
        dilation: _SpatialArg = 1,
        output_padding: _SpatialArg = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = _conv_transpose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            output_padding,
            bias,
        )

    @property
    def weight(self) -> mx.array:
        return self.conv.weight

    def __call__(self, x: mx.array) -> mx.array:
        channels_last = mx.transpose(x, axes=(0, 2, 3, 1))
        return mx.transpose(self.conv(channels_last), axes=(0, 3, 1, 2))
