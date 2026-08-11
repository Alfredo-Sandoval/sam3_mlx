from __future__ import annotations

from typing import cast

import mlx.core as mx

from sam3_mlx.sam.tensor_protocols import ArrayModule


def _activation_name(activation: object) -> str:
    if isinstance(activation, type):
        return activation.__name__.lower()
    name: object = getattr(activation, "__name__", activation.__class__.__name__)
    if not isinstance(name, str):
        raise TypeError("activation names must be strings")
    return name.lower()


def addmm_act(activation: object, linear: ArrayModule, mat1: mx.array) -> mx.array:
    from mlx import nn

    y = linear(mat1)
    activation_name = _activation_name(activation)
    relu_value: object = getattr(nn, "relu")
    gelu_value: object = getattr(nn, "gelu")
    relu = cast(ArrayModule, relu_value)
    gelu = cast(ArrayModule, gelu_value)
    if activation is relu or activation_name == "relu":
        return relu(y)
    if activation is gelu or activation_name == "gelu":
        return gelu(y)
    raise ValueError(f"Unexpected activation {activation!r}")
