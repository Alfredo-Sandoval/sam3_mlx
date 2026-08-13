from __future__ import annotations

from typing import Protocol, cast

import mlx.core as mx
import pytest
from mlx import nn

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.model.data_misc import NestedTensor
from sam3_mlx.model.necks import (
    Scale0_5FN,
    Scale1FN,
    Scale2FN,
    Scale4FN,
    Sam3DualViTDetNeck,
    Sam3TriViTDetNeck,
    kept_scale_heads,
    output_levels_for_scalp,
)
from sam3_mlx.model.vl_combiner import SAM3VLBackbone
from sam3_mlx.model.position_encoding import PositionEmbeddingSine
from sam3_mlx.model.vitdet import ViT
from tests._mlx_runtime import flat_parameters


class _NestedFeature(Protocol):
    tensors: mx.array
    mask: mx.array | None


def _as_nested(feature: object) -> _NestedFeature:
    return cast(_NestedFeature, feature)


class _Eval(Protocol):
    def __call__(self, *args: mx.array) -> None: ...


_eval = cast(_Eval, getattr(mx, "eval"))


class _StubTrunk(nn.Module):
    def __init__(self, channels: int = 8, spatial: int = 4) -> None:
        super().__init__()
        self.channel_list = [channels]
        self.spatial = spatial

    def __call__(self, x: mx.array | NestedTensor) -> list[mx.array | NestedTensor]:
        if isinstance(x, NestedTensor):
            nested = _as_nested(x)
            batch = int(nested.tensors.shape[0])
            tensors = mx.ones(
                (batch, self.channel_list[-1], self.spatial, self.spatial),
                dtype=mx.float32,
            )
            return [NestedTensor(tensors, nested.mask)]
        batch = int(x.shape[0])
        return [
            mx.ones(
                (batch, self.channel_list[-1], self.spatial, self.spatial),
                dtype=mx.float32,
            )
        ]


def _pos_enc(d_model: int = 4) -> PositionEmbeddingSine:
    return PositionEmbeddingSine(num_pos_feats=d_model, normalize=True)


def test_scale_heads_produce_expected_spatial_shapes():
    x = mx.ones((1, 4, 4, 8), dtype=mx.float32)
    cases: list[tuple[nn.Module, tuple[int, int, int, int]]] = [
        (Scale4FN(8, 4), (1, 16, 16, 4)),
        (Scale2FN(8, 4), (1, 8, 8, 4)),
        (Scale1FN(8, 4), (1, 4, 4, 4)),
        (Scale0_5FN(8, 4), (1, 2, 2, 4)),
    ]
    for head, expected in cases:
        out = cast(mx.array, head(x))
        _eval(out)
        assert out.shape == expected


def test_scale2fn_forward_does_not_call_allocated_gelu():
    head = Scale2FN(8, 4)
    assert hasattr(head, "gelu")

    def _raise_if_called(x: mx.array) -> mx.array:
        raise AssertionError("Scale2FN forward must not call gelu")

    head.gelu = _raise_if_called
    out = head(mx.ones((1, 4, 4, 8), dtype=mx.float32))
    _eval(out)
    assert out.shape == (1, 8, 8, 4)


def test_dual_neck_plain_array_and_nested_tensor_with_optional_sam2():
    trunk = _StubTrunk(channels=8, spatial=4)
    pe = _pos_enc(4)

    dual = Sam3DualViTDetNeck(
        trunk=trunk,
        position_encoding=pe,
        d_model=4,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck=False,
    )
    sam3_out, sam3_pos, sam2_out, sam2_pos = dual(mx.ones((1, 3, 16, 16)))
    _eval(*sam3_out, *sam3_pos)
    assert sam2_out is None and sam2_pos is None
    assert [t.shape for t in sam3_out] == [
        (1, 4, 16, 16),
        (1, 4, 8, 8),
        (1, 4, 4, 4),
        (1, 4, 2, 2),
    ]
    assert [p.shape for p in sam3_pos] == [t.shape for t in sam3_out]

    dual_sam2 = Sam3DualViTDetNeck(
        trunk=trunk,
        position_encoding=pe,
        d_model=4,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck=True,
    )
    mask = mx.zeros((1, 16, 16), dtype=mx.bool_)
    nested = NestedTensor(mx.ones((1, 3, 16, 16)), mask)
    s3, p3, s2, p2 = dual_sam2(nested)
    assert s2 is not None and p2 is not None
    _eval(*s3, *p3, *s2, *p2)
    assert [t.shape for t in s2] == [t.shape for t in s3]
    assert len(s2) == 4


def test_tri_neck_output_ordering_mask_propagation_and_flags():
    trunk = _StubTrunk(channels=8, spatial=4)
    pe = _pos_enc(4)
    tri = Sam3TriViTDetNeck(
        trunk=trunk,
        position_encoding=pe,
        d_model=4,
        scale_factors=(4.0, 2.0, 1.0),
    )

    mask = mx.ones((1, 16, 16), dtype=mx.bool_)
    nested = NestedTensor(mx.ones((1, 3, 16, 16)), mask)
    (
        sam3_out,
        sam3_pos,
        interactive_out,
        interactive_pos,
        propagation_out,
        propagation_pos,
    ) = tri(nested)

    assert len(sam3_out) == len(interactive_out) == len(propagation_out) == 3
    assert len(sam3_pos) == len(interactive_pos) == len(propagation_pos) == 3
    assert [_as_nested(t).tensors.shape for t in sam3_out] == [
        (1, 4, 16, 16),
        (1, 4, 8, 8),
        (1, 4, 4, 4),
    ]
    for feature in (*sam3_out, *interactive_out, *propagation_out):
        assert isinstance(feature, NestedTensor)
        assert _as_nested(feature).mask is mask

    empty = tri(
        nested,
        need_sam3_out=False,
        need_interactive_out=False,
        need_propagation_out=True,
    )
    assert empty[0] == []
    assert empty[2] == []
    assert len(empty[4]) == 3


def test_unsupported_scale_factor_guard_remains_canonical():
    with pytest.raises(Sam3MlxUnsupportedError) as excinfo:
        Sam3DualViTDetNeck(
            trunk=_StubTrunk(),
            position_encoding=_pos_enc(4),
            d_model=4,
            scale_factors=(3.0,),
        )
    message = str(excinfo.value)
    assert "scale_factor=3.0" in message
    assert "Sam3DualViTDetNeck" in message


class _CountingScaleHead(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls = 0

    def __call__(self, x: mx.array) -> mx.array:
        self.calls += 1
        return cast(mx.array, self.inner(x))


def _assert_array_lists_equal(left: list[mx.array], right: list[mx.array]) -> None:
    assert len(left) == len(right)
    for left_item, right_item in zip(left, right, strict=True):
        _eval(left_item, right_item)
        assert bool(mx.array_equal(left_item, right_item).item())
        assert left_item.shape == right_item.shape
        assert left_item.dtype == right_item.dtype


def test_output_level_helpers_keep_high_resolution_heads() -> None:
    heads = [object(), object(), object(), object()]
    assert kept_scale_heads(cast(list[object], heads), None) == heads
    assert kept_scale_heads(cast(list[object], heads), 3) == heads[:3]
    assert output_levels_for_scalp(4, scalp=0) is None
    assert output_levels_for_scalp(4, scalp=1) == 3

    with pytest.raises(ValueError, match="1..4"):
        kept_scale_heads(cast(list[object], heads), 0)
    with pytest.raises(ValueError, match="at least one output level"):
        output_levels_for_scalp(4, scalp=4)


def test_dual_neck_skips_discarded_head_and_keeps_retained_tensors() -> None:
    trunk = _StubTrunk(channels=8, spatial=4)
    pe = _pos_enc(4)
    dual = Sam3DualViTDetNeck(
        trunk=trunk,
        position_encoding=pe,
        d_model=4,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck=True,
    )
    discarded = _CountingScaleHead(cast(nn.Module, dual.convs[3]))
    discarded_sam2 = _CountingScaleHead(cast(nn.Module, dual.sam2_convs[3]))
    dual.convs[3] = discarded
    assert dual.sam2_convs is not None
    dual.sam2_convs[3] = discarded_sam2

    samples = mx.ones((1, 3, 16, 16), dtype=mx.float32)
    full_s3, full_p3, full_s2, full_p2 = dual(samples)
    pruned_s3, pruned_p3, pruned_s2, pruned_p2 = dual(samples, output_levels=3)

    assert discarded.calls == 1
    assert discarded_sam2.calls == 1
    assert full_s2 is not None and full_p2 is not None
    assert pruned_s2 is not None and pruned_p2 is not None
    assert [t.shape for t in full_s3] == [
        (1, 4, 16, 16),
        (1, 4, 8, 8),
        (1, 4, 4, 4),
        (1, 4, 2, 2),
    ]
    assert [t.shape for t in pruned_s3] == [
        (1, 4, 16, 16),
        (1, 4, 8, 8),
        (1, 4, 4, 4),
    ]
    _assert_array_lists_equal(pruned_s3, full_s3[:3])
    _assert_array_lists_equal(pruned_p3, full_p3[:3])
    _assert_array_lists_equal(pruned_s2, full_s2[:3])
    _assert_array_lists_equal(pruned_p2, full_p2[:3])


def test_dual_neck_keeps_discarded_head_checkpoint_keys() -> None:
    dual = Sam3DualViTDetNeck(
        trunk=_StubTrunk(channels=8, spatial=4),
        position_encoding=_pos_enc(4),
        d_model=4,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck=True,
    )
    _eval(*dual(mx.ones((1, 3, 16, 16)), output_levels=3)[0])
    keys = set(flat_parameters(dual))
    assert any(key.startswith("convs.3.") for key in keys)
    assert any(key.startswith("sam2_convs.3.") for key in keys)
    assert any(key.startswith("convs.0.") for key in keys)


def test_dual_neck_public_forward_still_emits_every_scale() -> None:
    dual = Sam3DualViTDetNeck(
        trunk=_StubTrunk(channels=8, spatial=4),
        position_encoding=_pos_enc(4),
        d_model=4,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
    )
    sam3_out, sam3_pos, sam2_out, sam2_pos = dual(mx.ones((1, 3, 16, 16)))
    _eval(*sam3_out, *sam3_pos)
    assert sam2_out is None and sam2_pos is None
    assert len(sam3_out) == 4
    assert [t.shape for t in sam3_out] == [p.shape for p in sam3_pos]


def test_tri_neck_skips_discarded_heads_consistently() -> None:
    tri = Sam3TriViTDetNeck(
        trunk=_StubTrunk(channels=8, spatial=4),
        position_encoding=_pos_enc(4),
        d_model=4,
        scale_factors=(4.0, 2.0, 1.0),
    )
    discarded = _CountingScaleHead(cast(nn.Module, tri.convs[2]))
    discarded_interactive = _CountingScaleHead(
        cast(nn.Module, tri.interactive_convs[2])
    )
    discarded_propagation = _CountingScaleHead(
        cast(nn.Module, tri.propagation_convs[2])
    )
    tri.convs[2] = discarded
    tri.interactive_convs[2] = discarded_interactive
    tri.propagation_convs[2] = discarded_propagation

    samples = mx.ones((1, 3, 16, 16), dtype=mx.float32)
    full = tri(samples)
    pruned = tri(samples, output_levels=2)

    assert discarded.calls == 1
    assert discarded_interactive.calls == 1
    assert discarded_propagation.calls == 1
    assert len(full[0]) == 3
    assert len(pruned[0]) == 2
    _assert_array_lists_equal(
        [_as_nested(item).tensors for item in pruned[0]],
        [_as_nested(item).tensors for item in full[0][:2]],
    )
    _assert_array_lists_equal(pruned[1], full[1][:2])


def test_vl_backbone_scalp_does_not_execute_discarded_head() -> None:
    neck = Sam3DualViTDetNeck(
        trunk=_StubTrunk(channels=8, spatial=4),
        position_encoding=_pos_enc(4),
        d_model=4,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck=True,
    )
    discarded = _CountingScaleHead(cast(nn.Module, neck.convs[3]))
    discarded_sam2 = _CountingScaleHead(cast(nn.Module, neck.sam2_convs[3]))
    neck.convs[3] = discarded
    assert neck.sam2_convs is not None
    neck.sam2_convs[3] = discarded_sam2

    full_neck = neck(mx.ones((1, 3, 16, 16), dtype=mx.float32))
    backbone = SAM3VLBackbone(visual=neck, text=None, scalp=1)
    output = backbone.forward_image(mx.ones((1, 3, 16, 16), dtype=mx.float32))
    features = cast(list[mx.array], output["backbone_fpn"])
    positions = cast(list[mx.array], output["vision_pos_enc"])
    sam2_out = cast(dict[str, object], output["sam2_backbone_out"])
    sam2_features = cast(list[mx.array], sam2_out["backbone_fpn"])

    assert discarded.calls == 1
    assert discarded_sam2.calls == 1
    assert len(features) == 3
    assert [item.shape for item in features] == [
        (1, 4, 16, 16),
        (1, 4, 8, 8),
        (1, 4, 4, 4),
    ]
    _assert_array_lists_equal(features, full_neck[0][:3])
    _assert_array_lists_equal(positions, full_neck[1][:3])
    assert full_neck[2] is not None
    _assert_array_lists_equal(sam2_features, full_neck[2][:3])
    assert bool(mx.array_equal(output["vision_features"], features[-1]).item())


def test_vit_trunk_class_identity_for_neck():
    vit = ViT(
        img_size=32,
        patch_size=16,
        embed_dim=8,
        depth=1,
        num_heads=2,
        rel_pos_blocks=(),
        global_att_blocks=(0,),
        window_size=0,
        retain_cls_token=False,
        pretrain_use_cls_token=False,
        use_abs_pos=True,
        tile_abs_pos=False,
        use_rope=False,
    )
    neck = Sam3DualViTDetNeck(
        trunk=vit,
        position_encoding=_pos_enc(4),
        d_model=4,
        scale_factors=(1.0,),
    )
    # Protocol-cast storage must preserve runtime module identity.
    assert type(neck.trunk) is ViT
    out = neck(mx.ones((1, 3, 32, 32)))
    _eval(out[0][0])
    assert out[0][0].shape == (1, 4, 2, 2)
