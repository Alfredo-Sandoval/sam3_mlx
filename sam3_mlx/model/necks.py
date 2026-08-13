from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Protocol, TypeAlias, cast

import mlx.core as mx
from mlx import nn

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.data_misc import NestedTensor


class _ArrayModule(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class _ScaleHead(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class _VisionTrunk(Protocol):
    channel_list: list[int]

    def __call__(self, x: mx.array | NestedTensor) -> list[mx.array | NestedTensor]: ...


class _PositionEncoding(Protocol):
    def __call__(self, x: mx.array | tuple[int, int, int, int]) -> mx.array: ...


class _FeatureLike(Protocol):
    tensors: mx.array
    mask: mx.array | None


TrunkFeature: TypeAlias = mx.array | NestedTensor
DualOutput: TypeAlias = tuple[
    list[mx.array],
    list[mx.array],
    list[mx.array] | None,
    list[mx.array] | None,
]
TriOutput: TypeAlias = tuple[
    list[NestedTensor],
    list[mx.array],
    list[NestedTensor],
    list[mx.array],
    list[NestedTensor],
    list[mx.array],
]


def _as_array_module(module: object) -> _ArrayModule:
    return cast(_ArrayModule, module)


def _as_scale_head(module: object) -> _ScaleHead:
    return cast(_ScaleHead, module)


def _as_trunk(module: object) -> _VisionTrunk:
    return cast(_VisionTrunk, module)


def _as_position_encoding(module: object) -> _PositionEncoding:
    return cast(_PositionEncoding, module)


def _as_feature(feature: object) -> _FeatureLike:
    return cast(_FeatureLike, feature)


def kept_scale_heads(
    convs: Sequence[_ScaleHead],
    output_levels: object,
) -> list[_ScaleHead]:
    """Return the high-resolution heads that should execute for one forward."""

    if output_levels is None:
        return list(convs)
    if isinstance(output_levels, bool) or not isinstance(output_levels, int):
        raise TypeError("output_levels must be a positive integer or None")
    if output_levels < 1 or output_levels > len(convs):
        raise ValueError(
            f"output_levels must be in 1..{len(convs)}, got {output_levels}"
        )
    return list(convs[:output_levels])


def output_levels_for_scalp(total_levels: object, *, scalp: object) -> int | None:
    """Translate backbone scalp into the neck's retained-level count."""

    if isinstance(scalp, bool) or not isinstance(scalp, int):
        raise TypeError("scalp must be a non-negative integer")
    if isinstance(total_levels, bool) or not isinstance(total_levels, int):
        raise TypeError("total_levels must be a positive integer")
    if total_levels < 1:
        raise ValueError("total_levels must be at least 1")
    if scalp < 0:
        raise ValueError("scalp must be non-negative")
    if scalp >= total_levels:
        raise ValueError("scalp must leave at least one output level")
    if scalp == 0:
        return None
    return total_levels - scalp


def _build_scale_convs(
    dim: int,
    d_model: int,
    scale_factors: Sequence[float],
    use_bias: bool,
) -> list[_ScaleHead]:
    """Build scale-head modules shared by Dual and Tri necks."""
    convs: list[_ScaleHead] = []
    for scale in scale_factors:
        if scale == 4.0:
            convs.append(
                _as_scale_head(
                    Scale4FN(in_channels=dim, d_model=d_model, use_bias=use_bias)
                )
            )
        elif scale == 2.0:
            convs.append(
                _as_scale_head(
                    Scale2FN(in_channels=dim, d_model=d_model, use_bias=use_bias)
                )
            )
        elif scale == 1.0:
            convs.append(
                _as_scale_head(
                    Scale1FN(in_channels=dim, d_model=d_model, use_bias=use_bias)
                )
            )
        elif scale == 0.5:
            convs.append(
                _as_scale_head(
                    Scale0_5FN(in_channels=dim, d_model=d_model, use_bias=use_bias)
                )
            )
        else:
            # Historical feature id from the Dual path (Tri previously delegated there).
            raise_unsupported(
                f"sam3_mlx.model.necks.Sam3DualViTDetNeck(scale_factor={scale!r})",
                reason="port-gap",
                detail=f"Scale factor {scale} is not supported yet.",
                alternative="scale_factors=(4.0, 2.0, 1.0, 0.5)",
            )
    return convs


class Scale4FN(nn.Module):
    def __init__(self, in_channels: int, d_model: int, use_bias: bool = True) -> None:
        super().__init__()
        self.dconv_2x2_0 = _as_array_module(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        )
        self.gelu = _as_array_module(nn.GELU())
        self.dconv_2x2_1 = _as_array_module(
            nn.ConvTranspose2d(
                in_channels // 2, in_channels // 4, kernel_size=2, stride=2
            )
        )
        self.conv_1x1 = _as_array_module(
            nn.Conv2d(
                in_channels=in_channels // 4,
                out_channels=d_model,
                kernel_size=1,
                bias=use_bias,
            )
        )
        self.conv_3x3 = _as_array_module(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.dconv_2x2_0(x)
        x = self.gelu(x)
        x = self.dconv_2x2_1(x)
        x = self.conv_1x1(x)
        return self.conv_3x3(x)


class Scale2FN(nn.Module):
    def __init__(self, in_channels: int, d_model: int, use_bias: bool = True) -> None:
        super().__init__()
        self.dconv_2x2 = _as_array_module(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        )
        self.gelu = _as_array_module(nn.GELU())
        self.conv_1x1 = _as_array_module(
            nn.Conv2d(
                in_channels=in_channels // 2,
                out_channels=d_model,
                kernel_size=1,
                bias=use_bias,
            )
        )
        self.conv_3x3 = _as_array_module(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.dconv_2x2(x)
        x = self.conv_1x1(x)
        return self.conv_3x3(x)


class Scale1FN(nn.Module):
    def __init__(self, in_channels: int, d_model: int, use_bias: bool = True) -> None:
        super().__init__()
        self.conv_1x1 = _as_array_module(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=d_model,
                kernel_size=1,
                bias=use_bias,
            )
        )
        self.conv_3x3 = _as_array_module(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv_3x3(self.conv_1x1(x))


class Scale0_5FN(nn.Module):
    def __init__(self, in_channels: int, d_model: int, use_bias: bool = True) -> None:
        super().__init__()
        self.maxpool_2x2 = _as_array_module(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_1x1 = _as_array_module(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=d_model,
                kernel_size=1,
                bias=use_bias,
            )
        )
        self.conv_3x3 = _as_array_module(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                padding=1,
                bias=use_bias,
            )
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.maxpool_2x2(x)
        return self.conv_3x3(self.conv_1x1(x))


class Sam3DualViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
    ) -> None:
        super().__init__()
        self.trunk = _as_trunk(trunk)
        self.position_encoding = _as_position_encoding(position_encoding)
        self.scale_factors = scale_factors
        use_bias = True
        dim: int = self.trunk.channel_list[-1]

        self.convs: list[_ScaleHead] = _build_scale_convs(
            dim, d_model, scale_factors, use_bias
        )

        self.sam2_convs: list[_ScaleHead] | None = None
        if add_sam2_neck:
            self.sam2_convs = _build_scale_convs(dim, d_model, scale_factors, use_bias)

    def _apply_array_heads(
        self,
        x_nhwc: mx.array,
        convs: list[_ScaleHead],
        *,
        output_levels: int | None = None,
    ) -> tuple[list[mx.array], list[mx.array]]:
        out: list[mx.array] = []
        pos: list[mx.array] = []
        for conv in kept_scale_heads(convs, output_levels):
            head_out = conv(x_nhwc)
            nchw_shape = (
                int(head_out.shape[0]),
                int(head_out.shape[3]),
                int(head_out.shape[1]),
                int(head_out.shape[2]),
            )
            out.append(mx.transpose(head_out, axes=(0, 3, 1, 2)))
            pos.append(self.position_encoding(nchw_shape).astype(head_out.dtype))
        return out, pos

    def forward(
        self,
        x_list: mx.array | NestedTensor,
        *,
        output_levels: int | None = None,
    ) -> DualOutput:
        xs = self.trunk(x_list)
        x = xs[-1]
        if isinstance(x, NestedTensor):
            x = _as_feature(x).tensors
        x = mx.transpose(x, axes=(0, 2, 3, 1))

        sam3_out, sam3_pos = self._apply_array_heads(
            x, self.convs, output_levels=output_levels
        )
        sam2_convs = self.sam2_convs
        if sam2_convs is not None:
            sam2_out, sam2_pos = self._apply_array_heads(
                x, sam2_convs, output_levels=output_levels
            )
            return sam3_out, sam3_pos, sam2_out, sam2_pos
        return sam3_out, sam3_pos, None, None

    def __call__(
        self,
        x_list: mx.array | NestedTensor,
        *,
        output_levels: int | None = None,
    ) -> DualOutput:
        return self.forward(x_list, output_levels=output_levels)


class Sam3TriViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        neck_norm: object | None = None,
        scale_factors: Sequence[float] = (4.0, 2.0, 1.0),
    ) -> None:
        super().__init__()
        self.trunk = _as_trunk(trunk)
        self.position_encoding = _as_position_encoding(position_encoding)
        self.scale_factors = scale_factors
        use_bias = neck_norm is None
        dim: int = self.trunk.channel_list[-1]

        self.convs: list[_ScaleHead] = _build_scale_convs(
            dim, d_model, scale_factors, use_bias
        )
        self.interactive_convs: list[_ScaleHead] = deepcopy(self.convs)
        self.propagation_convs: list[_ScaleHead] = deepcopy(self.convs)

    @staticmethod
    def _feature_tensor(feature: object) -> mx.array:
        if isinstance(feature, NestedTensor):
            return _as_feature(feature).tensors
        return cast(mx.array, feature)

    @staticmethod
    def _feature_mask(feature: object) -> mx.array | None:
        if isinstance(feature, NestedTensor):
            return _as_feature(feature).mask
        return getattr(feature, "mask", None)

    @staticmethod
    def _to_nhwc(feature: object) -> mx.array:
        tensor = Sam3TriViTDetNeck._feature_tensor(feature)
        return mx.transpose(tensor, axes=(0, 2, 3, 1))

    def _apply_head(
        self,
        x_nhwc: mx.array,
        x_mask: mx.array | None,
        convs: list[_ScaleHead],
        *,
        output_levels: int | None = None,
    ) -> tuple[list[NestedTensor], list[mx.array]]:
        out: list[NestedTensor] = []
        pos: list[mx.array] = []
        for conv in kept_scale_heads(convs, output_levels):
            head_out = conv(x_nhwc)
            nchw_shape = (
                int(head_out.shape[0]),
                int(head_out.shape[3]),
                int(head_out.shape[1]),
                int(head_out.shape[2]),
            )
            head_out = mx.transpose(head_out, axes=(0, 3, 1, 2))
            out.append(NestedTensor(head_out, x_mask))
            pos.append(self.position_encoding(nchw_shape).astype(head_out.dtype))
        return out, pos

    def forward(
        self,
        tensor_list: mx.array | NestedTensor,
        *,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
        output_levels: int | None = None,
    ) -> TriOutput:
        xs = self.trunk(tensor_list)
        x_src = xs[-1]
        x = self._to_nhwc(x_src)
        x_mask = self._feature_mask(x_src)

        sam3_out: list[NestedTensor] = []
        sam3_pos: list[mx.array] = []
        interactive_out: list[NestedTensor] = []
        interactive_pos: list[mx.array] = []
        propagation_out: list[NestedTensor] = []
        propagation_pos: list[mx.array] = []

        if need_sam3_out:
            sam3_out, sam3_pos = self._apply_head(
                x, x_mask, self.convs, output_levels=output_levels
            )
        if need_interactive_out:
            interactive_out, interactive_pos = self._apply_head(
                x, x_mask, self.interactive_convs, output_levels=output_levels
            )
        if need_propagation_out:
            propagation_out, propagation_pos = self._apply_head(
                x, x_mask, self.propagation_convs, output_levels=output_levels
            )

        return (
            sam3_out,
            sam3_pos,
            interactive_out,
            interactive_pos,
            propagation_out,
            propagation_pos,
        )

    def __call__(
        self,
        tensor_list: mx.array | NestedTensor,
        *,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
        output_levels: int | None = None,
    ) -> TriOutput:
        return self.forward(
            tensor_list,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
            output_levels=output_levels,
        )
