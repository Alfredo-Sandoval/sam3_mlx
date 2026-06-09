from copy import deepcopy
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.data_misc import NestedTensor


class Scale4FN(nn.Module):
    def __init__(self, in_channels: int, d_model: int, use_bias: bool = True):
        super().__init__()
        self.dconv_2x2_0 = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.gelu = nn.GELU()
        self.dconv_2x2_1 = nn.ConvTranspose2d(
            in_channels // 2, in_channels // 4, kernel_size=2, stride=2
        )
        self.conv_1x1 = nn.Conv2d(
            in_channels=in_channels // 4,
            out_channels=d_model,
            kernel_size=1,
            bias=use_bias,
        )
        self.conv_3x3 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            bias=use_bias,
        )

    def __call__(self, x):
        x = self.dconv_2x2_0(x)
        x = self.gelu(x)
        x = self.dconv_2x2_1(x)
        x = self.conv_1x1(x)
        return self.conv_3x3(x)


class Scale2FN(nn.Module):
    def __init__(self, in_channels: int, d_model: int, use_bias: bool = True):
        super().__init__()
        self.dconv_2x2 = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.gelu = nn.GELU()
        self.conv_1x1 = nn.Conv2d(
            in_channels=in_channels // 2,
            out_channels=d_model,
            kernel_size=1,
            bias=use_bias,
        )
        self.conv_3x3 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            bias=use_bias,
        )

    def __call__(self, x):
        x = self.dconv_2x2(x)
        x = self.conv_1x1(x)
        return self.conv_3x3(x)


class Scale1FN(nn.Module):
    def __init__(self, in_channels: int, d_model: int, use_bias: bool = True):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(
            in_channels=in_channels, out_channels=d_model, kernel_size=1, bias=use_bias
        )
        self.conv_3x3 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            bias=use_bias,
        )

    def __call__(self, x):
        return self.conv_3x3(self.conv_1x1(x))


class Scale0_5FN(nn.Module):
    def __init__(self, in_channels: int, d_model: int, use_bias: bool = True):
        super().__init__()
        self.maxpool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1x1 = nn.Conv2d(
            in_channels=in_channels, out_channels=d_model, kernel_size=1, bias=use_bias
        )
        self.conv_3x3 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            bias=use_bias,
        )

    def __call__(self, x):
        x = self.maxpool_2x2(x)
        return self.conv_3x3(self.conv_1x1(x))


class Sam3DualViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
    ):
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = []

        self.scale_factors = scale_factors
        use_bias = True
        dim: int = self.trunk.channel_list[-1]

        self.convs = self._build_convs(dim, d_model, scale_factors, use_bias)

        self.sam2_convs = None
        if add_sam2_neck:
            self.sam2_convs = self._build_convs(dim, d_model, scale_factors, use_bias)

    def _build_convs(self, dim, d_model, scale_factors, use_bias):
        convs = []
        for _, scale in enumerate(scale_factors):
            if scale == 4.0:
                convs.append(
                    Scale4FN(in_channels=dim, d_model=d_model, use_bias=use_bias)
                )
            elif scale == 2.0:
                convs.append(
                    Scale2FN(in_channels=dim, d_model=d_model, use_bias=use_bias)
                )
            elif scale == 1.0:
                convs.append(
                    Scale1FN(in_channels=dim, d_model=d_model, use_bias=use_bias)
                )
            elif scale == 0.5:
                convs.append(
                    Scale0_5FN(in_channels=dim, d_model=d_model, use_bias=use_bias)
                )
            else:
                raise_unsupported(
                    f"sam3_mlx.model.necks.Sam3DualViTDetNeck(scale_factor={scale!r})",
                    reason="port-gap",
                    detail=f"Scale factor {scale} is not supported yet.",
                    alternative="scale_factors=(4.0, 2.0, 1.0, 0.5)",
                )
        return convs

    def forward(
        self, x_list: List[mx.array]
    ) -> Tuple[
        List[mx.array],
        List[mx.array],
        Optional[List[mx.array]],
        Optional[List[mx.array]],
    ]:
        xs = self.trunk(x_list)
        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []
        x = xs[-1]
        if isinstance(x, NestedTensor):
            x = x.tensors
        x = x.transpose(0, 2, 3, 1)
        for i in range(len(self.convs)):
            sam3_x_out = self.convs[i](x)
            nchw_shape = (
                sam3_x_out.shape[0],
                sam3_x_out.shape[3],
                sam3_x_out.shape[1],
                sam3_x_out.shape[2],
            )
            sam3_out.append(sam3_x_out.transpose(0, 3, 1, 2))
            sam3_pos.append(self.position_encoding(nchw_shape).astype(sam3_x_out.dtype))

            if self.sam2_convs is not None:
                sam2_x_out = self.sam2_convs[i](x)
                nchw_shape = (
                    sam2_x_out.shape[0],
                    sam2_x_out.shape[3],
                    sam2_x_out.shape[1],
                    sam2_x_out.shape[2],
                )
                sam2_out.append(sam2_x_out.transpose(0, 3, 1, 2))
                sam2_pos.append(
                    self.position_encoding(nchw_shape).astype(sam2_x_out.dtype)
                )

        return sam3_out, sam3_pos, sam2_out, sam2_pos

    def __call__(
        self, x_list: List[mx.array]
    ) -> Tuple[
        List[mx.array],
        List[mx.array],
        Optional[List[mx.array]],
        Optional[List[mx.array]],
    ]:
        return self.forward(x_list)


class Sam3TriViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        neck_norm=None,
        scale_factors=(4.0, 2.0, 1.0),
    ):
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.scale_factors = scale_factors
        use_bias = neck_norm is None
        dim: int = self.trunk.channel_list[-1]

        self.convs = self._build_convs(dim, d_model, scale_factors, use_bias)
        self.interactive_convs = deepcopy(self.convs)
        self.propagation_convs = deepcopy(self.convs)

    def _build_convs(self, dim, d_model, scale_factors, use_bias):
        return Sam3DualViTDetNeck._build_convs(
            self,
            dim=dim,
            d_model=d_model,
            scale_factors=scale_factors,
            use_bias=use_bias,
        )

    @staticmethod
    def _feature_tensor(feature):
        return getattr(feature, "tensors", feature)

    @staticmethod
    def _feature_mask(feature):
        return getattr(feature, "mask", None)

    @staticmethod
    def _to_nhwc(feature):
        tensor = Sam3TriViTDetNeck._feature_tensor(feature)
        return tensor.transpose(0, 2, 3, 1)

    def _apply_head(self, x_nhwc, x_mask, convs):
        out, pos = [], []
        for conv in convs:
            head_out = conv(x_nhwc)
            nchw_shape = (
                head_out.shape[0],
                head_out.shape[3],
                head_out.shape[1],
                head_out.shape[2],
            )
            head_out = head_out.transpose(0, 3, 1, 2)
            out.append(NestedTensor(head_out, x_mask))
            pos.append(self.position_encoding(nchw_shape).astype(head_out.dtype))
        return out, pos

    def forward(
        self,
        tensor_list,
        *,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ):
        xs = self.trunk(tensor_list)
        x_src = xs[-1]
        x = self._to_nhwc(x_src)
        x_mask = self._feature_mask(x_src)

        sam3_out, sam3_pos = [], []
        interactive_out, interactive_pos = [], []
        propagation_out, propagation_pos = [], []

        if need_sam3_out:
            sam3_out, sam3_pos = self._apply_head(x, x_mask, self.convs)
        if need_interactive_out:
            interactive_out, interactive_pos = self._apply_head(
                x, x_mask, self.interactive_convs
            )
        if need_propagation_out:
            propagation_out, propagation_pos = self._apply_head(
                x, x_mask, self.propagation_convs
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
        tensor_list,
        *,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ):
        return self.forward(
            tensor_list,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )
