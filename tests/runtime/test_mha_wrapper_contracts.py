import importlib
from collections.abc import Callable
from typing import Protocol, cast

import mlx.core as mx
import mlx.utils as mlx_utils
import pytest

from sam3_mlx.model.model_misc import MultiheadAttentionWrapper


class _TreeFlatten(Protocol):
    def __call__(self, tree: object) -> list[tuple[str, object]]: ...


class _ParameterModule(Protocol):
    def parameters(self) -> object: ...


class _LinearProjection(Protocol):
    weight: mx.array


class _AttentionProjections(_ParameterModule, Protocol):
    query_proj: _LinearProjection
    key_proj: _LinearProjection
    value_proj: _LinearProjection
    out_proj: _LinearProjection


class _MHAFactory(Protocol):
    def __call__(
        self, dims: int, num_heads: int, *, bias: bool = False
    ) -> _AttentionProjections: ...


_tree_flatten = cast(_TreeFlatten, getattr(mlx_utils, "tree_flatten"))
_mlx_mha_type = cast(
    type[object], getattr(importlib.import_module("mlx.nn"), "MultiHeadAttention")
)
_mlx_mha_factory = cast(_MHAFactory, _mlx_mha_type)


def _param_keys(module: object) -> list[str]:
    tree = cast(_ParameterModule, module).parameters()
    return sorted(key for key, _ in _tree_flatten(tree))


@pytest.mark.parametrize("bias", [True, False])
def test_mha_wrapper_parameter_tree_matches_base_mlx_mha(bias: bool):
    dims, heads = 8, 2
    base = _mlx_mha_factory(dims, heads, bias=bias)
    wrapper = MultiheadAttentionWrapper(dims, heads, bias=bias)

    assert isinstance(wrapper, _mlx_mha_type)
    assert type(wrapper).__mro__[1].__name__ == "MultiHeadAttention"
    assert _param_keys(wrapper) == _param_keys(base)
    for key in ("query_proj", "key_proj", "value_proj", "out_proj"):
        wrapper_projection = cast(_LinearProjection, getattr(wrapper, key))
        base_projection = cast(_LinearProjection, getattr(base, key))
        assert wrapper_projection.weight.shape == base_projection.weight.shape


def test_mha_wrapper_retains_inherited_causal_mask_contract():
    create_causal_mask = cast(
        Callable[[int], mx.array],
        getattr(MultiheadAttentionWrapper, "create_additive_causal_mask"),
    )
    mask = create_causal_mask(3)

    assert mask.shape == (3, 3)
    assert mx.array_equal(
        mask == 0,
        mx.array(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ]
        ),
    ).item()


def test_mha_wrapper_kdim_without_vdim_matches_mlx_value_input_fallback():
    dims, heads, kdim = 8, 2, 4
    wrapper = MultiheadAttentionWrapper(dims, heads, kdim=kdim, bias=True)

    assert wrapper.kdim == kdim
    assert wrapper.vdim == kdim
    assert wrapper.value_proj.weight.shape == (dims, kdim)
    assert wrapper.key_proj.weight.shape == (dims, kdim)


def test_mha_wrapper_rejects_nondivisible_embed_dim_with_value_error():
    with pytest.raises(ValueError, match="divisible by the number of heads"):
        MultiheadAttentionWrapper(7, 2)
