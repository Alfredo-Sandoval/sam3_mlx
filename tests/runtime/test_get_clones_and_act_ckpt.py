from typing import cast

import mlx.core as mx
import mlx.nn as nn
import pytest

from sam3_mlx.model.act_ckpt_utils import activation_ckpt_wrapper, clone_output_wrapper
from sam3_mlx.model.model_misc import CloneableModule, get_clones, get_clones_seq


class _Tiny(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.w = mx.ones((2, 2)) * scale

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.scale


def test_get_clones_module_and_factory_paths():
    template = _Tiny(scale=2.0)
    clones = get_clones(template, 3)
    assert len(clones) == 3
    assert all(isinstance(c, _Tiny) for c in clones)
    assert clones[0] is not template
    assert float(clones[0].scale) == 2.0

    made = get_clones(lambda: _Tiny(scale=3.0), 2)
    assert len(made) == 2
    assert all(float(c.scale) == 3.0 for c in made)


def test_get_clones_rejects_non_callable_non_module():
    with pytest.raises(TypeError, match="zero-argument factory"):
        get_clones(cast(CloneableModule[nn.Module], object()), 1)


def test_get_clones_seq_returns_sequential():
    seq = get_clones_seq(_Tiny(), 2)
    assert isinstance(seq, nn.Sequential)


def test_activation_ckpt_wrapper_accepts_flags_without_forwarding():
    seen: dict[str, object] = {}

    def module(*args: object, **kwargs: object) -> object:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "ok"

    wrapped = activation_ckpt_wrapper(module)
    assert wrapped(1, 2, act_ckpt_enable=True, use_reentrant=True, x=3) == "ok"
    assert seen["args"] == (1, 2)
    assert seen["kwargs"] == {"x": 3}
    assert "act_ckpt_enable" not in seen["kwargs"]
    assert "use_reentrant" not in seen["kwargs"]


def test_clone_output_wrapper_preserves_return():
    def f(x: int) -> int:
        return x + 1

    assert clone_output_wrapper(f)(4) == 5
