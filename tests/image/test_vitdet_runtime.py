from __future__ import annotations

from typing import Protocol, cast

import mlx.core as mx
from mlx import nn

from sam3_mlx.model.data_misc import NestedTensor
from sam3_mlx.model.model_misc import Mlp
from sam3_mlx.model.vitdet import (
    Attention,
    Block,
    ViT,
    get_abs_pos,
    window_partition,
    window_unpartition,
)


class _NestedFeature(Protocol):
    tensors: mx.array
    mask: mx.array | None


def _as_nested(feature: object) -> _NestedFeature:
    return cast(_NestedFeature, feature)


class _Eval(Protocol):
    def __call__(self, *args: mx.array) -> None: ...


_eval = cast(_Eval, getattr(mx, "eval"))


def test_vit_mlp_zero_dropout_uses_explicit_identity_layers():
    mlp = Mlp(in_features=4, hidden_features=8, drop=(0.0, 0.25))

    assert isinstance(mlp.drop1, nn.Identity)
    assert isinstance(mlp.drop2, nn.Dropout)


def test_vit_block_zero_residual_dropout_uses_explicit_identity_layer():
    block = Block(
        dim=4,
        num_heads=1,
        mlp_ratio=1.0,
        window_size=0,
        input_size=(2, 2),
        use_rope=False,
        dropout=0.0,
    )

    assert isinstance(block.dropout, nn.Identity)


def test_vit_attention_recomputes_rope_for_smaller_global_grid():
    attention = Attention(
        dim=8,
        num_heads=2,
        input_size=(4, 4),
        use_rope=True,
        rope_pt_size=(2, 2),
        rope_interp=True,
        cls_token=False,
    )
    x = mx.ones((1, 2, 2, 8), dtype=mx.float32)

    out = attention(x)
    _eval(out)

    assert out.shape == (1, 2, 2, 8)
    cache = cast(
        dict[tuple[int, int], mx.array], getattr(attention, "_freqs_cis_cache")
    )
    assert (2, 2) in cache
    assert cache[(2, 2)].shape == (4, 2)


def test_rel_pos_blocks_false_disables_all_blocks():
    vit = ViT(
        img_size=32,
        patch_size=16,
        embed_dim=8,
        depth=3,
        num_heads=2,
        rel_pos_blocks=False,
        global_att_blocks=(2,),
        window_size=0,
        retain_cls_token=False,
        pretrain_use_cls_token=False,
        use_abs_pos=True,
        tile_abs_pos=False,
        use_rope=False,
    )

    assert vit.rel_pos_blocks == [False, False, False]
    assert all(not block.attn.use_rel_pos for block in vit.blocks)

    out = vit(mx.ones((1, 3, 32, 32), dtype=mx.float32))
    feat = out[0]
    assert isinstance(feat, mx.array)
    _eval(feat)
    assert len(out) == 1
    assert feat.shape == (1, 8, 2, 2)


def test_window_partition_unpartition_roundtrip_with_padding():
    x = mx.reshape(mx.arange(2 * 5 * 5 * 3, dtype=mx.float32), (2, 5, 5, 3))
    windows, pad_hw = window_partition(x, window_size=2)

    assert pad_hw == (6, 6)
    assert windows.shape == (2 * 3 * 3, 2, 2, 3)

    restored = window_unpartition(windows, window_size=2, pad_hw=pad_hw, hw=(5, 5))
    _eval(restored)
    assert restored.shape == x.shape
    assert bool(mx.allclose(restored, x).item())


def test_get_abs_pos_with_and_without_class_token():
    abs_pos = mx.reshape(mx.arange(1 * 4 * 2, dtype=mx.float32), (1, 4, 2))
    no_cls = get_abs_pos(abs_pos, has_cls_token=False, hw=(3, 3), tiling=True)
    assert no_cls.shape == (1, 3, 3, 2)

    abs_pos_cls = mx.reshape(mx.arange(1 * 5 * 2, dtype=mx.float32), (1, 5, 2))
    with_cls = get_abs_pos(
        abs_pos_cls,
        has_cls_token=True,
        hw=(3, 3),
        retain_cls_token=True,
        tiling=True,
    )
    assert with_cls.shape == (1, 1 + 3 * 3, 2)
    assert bool(mx.allclose(with_cls[:, :1], abs_pos_cls[:, :1]).item())


def test_relative_position_attention_executes():
    attention = Attention(
        dim=8,
        num_heads=2,
        input_size=(2, 2),
        use_rel_pos=True,
        rel_pos_zero_init=True,
        cls_token=False,
        use_rope=False,
    )
    assert attention.rel_pos_h is not None
    assert attention.rel_pos_w is not None

    x = mx.ones((1, 2, 2, 8), dtype=mx.float32)
    out = attention(x)
    _eval(out)
    assert out.shape == (1, 2, 2, 8)


def test_vit_accepts_plain_array_and_nested_tensor():
    vit = ViT(
        img_size=32,
        patch_size=16,
        embed_dim=8,
        depth=2,
        num_heads=2,
        rel_pos_blocks=(),
        global_att_blocks=(1,),
        window_size=0,
        retain_cls_token=False,
        pretrain_use_cls_token=False,
        use_abs_pos=True,
        tile_abs_pos=False,
        use_rope=False,
    )
    plain = mx.ones((1, 3, 32, 32), dtype=mx.float32)
    plain_out = vit(plain)
    plain_feat = plain_out[0]
    assert isinstance(plain_feat, mx.array)
    _eval(plain_feat)
    assert plain_feat.shape == (1, 8, 2, 2)

    mask = mx.zeros((1, 32, 32), dtype=mx.bool_)
    nested = NestedTensor(plain, mask)
    nested_out = vit(nested)
    nested_feat = nested_out[0]
    assert isinstance(nested_feat, NestedTensor)
    nested_view = _as_nested(nested_feat)
    _eval(nested_view.tensors)
    assert nested_view.tensors.shape == (1, 8, 2, 2)
    assert nested_view.mask is not None
    assert nested_view.mask.shape[-2:] == (2, 2)
