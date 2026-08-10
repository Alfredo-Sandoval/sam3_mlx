from importlib import import_module
from typing import Protocol, cast

import mlx.core as mx
import pytest

from sam3_mlx.model.act_ckpt_utils import activation_ckpt_wrapper, clone_output_wrapper
from sam3_mlx.model.model_misc import (
    CloneableModule,
    LayerScale,
    get_clones,
    get_clones_seq,
)


class _SequentialLayers(Protocol):
    layers: list[object]


def test_get_clones_module_and_factory_paths():
    template = LayerScale(dim=2, init_values=2.0)
    clones = get_clones(template, 3)
    assert len(clones) == 3
    assert all(isinstance(clone, LayerScale) for clone in clones)
    assert clones[0] is not template
    assert mx.allclose(clones[0].gamma, mx.full((2,), 2.0)).item()

    made = get_clones(lambda: LayerScale(dim=2, init_values=3.0), 2)
    assert len(made) == 2
    assert all(mx.allclose(clone.gamma, mx.full((2,), 3.0)).item() for clone in made)


def test_get_clones_rejects_non_callable_non_module():
    with pytest.raises(TypeError, match="zero-argument factory"):
        get_clones(cast(CloneableModule[LayerScale], object()), 1)


def test_get_clones_seq_returns_sequential():
    seq = get_clones_seq(LayerScale(dim=2), 2)
    sequential_type = cast(type[object], getattr(import_module("mlx.nn"), "Sequential"))
    assert isinstance(seq, sequential_type)
    layers = cast(_SequentialLayers, seq).layers
    assert len(layers) == 2
    assert all(isinstance(layer, LayerScale) for layer in layers)


def test_activation_ckpt_wrapper_accepts_flags_without_forwarding():
    seen_args: tuple[object, ...] = ()
    seen_kwargs: dict[str, object] = {}

    def module(*args: object, **kwargs: object) -> object:
        nonlocal seen_args, seen_kwargs
        seen_args = args
        seen_kwargs = kwargs
        return "ok"

    wrapped = activation_ckpt_wrapper(module)
    assert wrapped(1, 2, act_ckpt_enable=True, use_reentrant=True, x=3) == "ok"
    assert seen_args == (1, 2)
    assert seen_kwargs == {"x": 3}
    assert "act_ckpt_enable" not in seen_kwargs
    assert "use_reentrant" not in seen_kwargs


def test_clone_output_wrapper_preserves_return():
    def f(x: int) -> int:
        return x + 1

    assert clone_output_wrapper(f)(4) == 5
