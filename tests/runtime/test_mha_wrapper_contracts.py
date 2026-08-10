import mlx.nn as nn
import mlx.utils as mlx_utils
import pytest

from sam3_mlx.model.model_misc import MultiheadAttentionWrapper


def _param_keys(module: nn.Module) -> list[str]:
    return sorted(key for key, _ in mlx_utils.tree_flatten(module.parameters()))


@pytest.mark.parametrize("bias", [True, False])
def test_mha_wrapper_parameter_tree_matches_base_mlx_mha(bias: bool):
    dims, heads = 8, 2
    base = nn.MultiHeadAttention(dims, heads, bias=bias)
    wrapper = MultiheadAttentionWrapper(dims, heads, bias=bias)

    assert isinstance(wrapper, nn.Module)
    assert not isinstance(wrapper, nn.MultiHeadAttention)
    assert _param_keys(wrapper) == _param_keys(base)
    for key in ("query_proj", "key_proj", "value_proj", "out_proj"):
        assert getattr(wrapper, key).weight.shape == getattr(base, key).weight.shape


def test_mha_wrapper_kdim_without_vdim_matches_mlx_value_input_fallback():
    dims, heads, kdim = 8, 2, 4
    base = nn.MultiHeadAttention(dims, heads, key_input_dims=kdim, bias=True)
    wrapper = MultiheadAttentionWrapper(dims, heads, kdim=kdim, bias=True)

    assert wrapper.kdim == kdim
    assert wrapper.vdim == kdim
    assert wrapper.value_proj.weight.shape == (dims, kdim)
    assert wrapper.value_proj.weight.shape == base.value_proj.weight.shape
    assert wrapper.key_proj.weight.shape == base.key_proj.weight.shape


def test_mha_wrapper_rejects_nondivisible_embed_dim_with_value_error():
    with pytest.raises(ValueError, match="divisible by the number of heads"):
        MultiheadAttentionWrapper(7, 2)
