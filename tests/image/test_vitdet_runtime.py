import mlx.core as mx
import mlx.nn as nn

from sam3_mlx.model.model_misc import Mlp
from sam3_mlx.model.vitdet import Attention, Block


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
    mx.eval(out)

    assert out.shape == (1, 2, 2, 8)
    assert (2, 2) in attention._freqs_cis_cache
    assert attention._freqs_cis_cache[(2, 2)].shape == (4, 2)
