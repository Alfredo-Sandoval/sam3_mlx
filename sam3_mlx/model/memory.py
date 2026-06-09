# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""MLX port of the SAM3 memory encoder building blocks."""

from __future__ import annotations

import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from sam3_mlx.model.data_misc import interpolate
from sam3_mlx.model.model_misc import DropPath, LayerNorm2d, get_clones


class _NCHWConv2d(nn.Module):
    """Run MLX's NHWC Conv2d behind an upstream-style NCHW module surface."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x.transpose(0, 2, 3, 1))
        return x.transpose(0, 3, 1, 2)

    @property
    def weight(self):
        return self.conv.weight


class SimpleMaskDownSampler(nn.Module):
    """
    Progressively downsample a mask by ``total_stride``.

    The official module is written in NCHW PyTorch. This MLX port keeps the same
    external tensor layout and uses small wrappers where MLX layers expect NHWC.
    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
        interpol_size=None,
        multiplex_count: int = 1,
        starting_out_chan: int = 1,
        input_channel_multiplier: int = 1,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        multiplex_count = multiplex_count * input_channel_multiplier
        assert stride**num_layers == total_stride

        self.encoder = []
        mask_in_chans, mask_out_chans = multiplex_count, starting_out_chan
        for _ in range(num_layers):
            mask_out_chans = mask_out_chans * (stride**2)
            self.encoder.append(
                _NCHWConv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(_NCHWConv2d(mask_out_chans, embed_dim, kernel_size=1))
        self.multiplex_count = multiplex_count
        self.interpol_size = interpol_size
        if self.interpol_size is not None:
            if not isinstance(self.interpol_size, (list, tuple)):
                raise AssertionError(
                    f"Unsupported type {type(self.interpol_size)}. "
                    "Should be a list or tuple."
                )
            self.interpol_size = list(interpol_size)
            assert len(self.interpol_size) == 2

    def forward(self, x: mx.array) -> mx.array:
        if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
            x = interpolate(
                x.astype(mx.float32),
                size=self.interpol_size,
                align_corners=False,
                mode="bilinear",
            )
        for layer in self.encoder:
            x = layer(x)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


class CXBlock(nn.Module):
    """ConvNeXt-style block used by the official memory fuser."""

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = _NCHWConv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            layer_scale_init_value * mx.ones((dim,))
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.transpose(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(0, 3, 1, 2)
        return residual + self.drop_path(x)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


class SimpleFuser(nn.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = get_clones(layer, num_layers)

        if input_projection:
            assert dim is not None
            self.proj = _NCHWConv2d(dim, dim, kernel_size=1)

    def forward(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


class SimpleMaskEncoder(nn.Module):
    def __init__(
        self,
        out_dim,
        mask_downsampler,
        fuser,
        position_encoding,
        in_dim=256,
    ):
        super().__init__()
        self.mask_downsampler = mask_downsampler
        self.pix_feat_proj = _NCHWConv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = _NCHWConv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: mx.array,
        masks: mx.array,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        if not skip_mask_sigmoid:
            masks = mx.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).astype(x.dtype)
        return {"vision_features": x, "vision_pos_enc": [pos]}

    def __call__(
        self,
        pix_feat: mx.array,
        masks: mx.array,
        skip_mask_sigmoid: bool = False,
    ):
        return self.forward(pix_feat, masks, skip_mask_sigmoid=skip_mask_sigmoid)


__all__ = [
    "CXBlock",
    "SimpleFuser",
    "SimpleMaskDownSampler",
    "SimpleMaskEncoder",
]
