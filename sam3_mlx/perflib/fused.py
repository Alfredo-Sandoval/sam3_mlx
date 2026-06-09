from __future__ import annotations


def _activation_name(activation) -> str:
    if isinstance(activation, type):
        return activation.__name__.lower()
    return getattr(activation, "__name__", activation.__class__.__name__).lower()


def addmm_act(activation, linear, mat1):
    import mlx.nn as nn

    y = linear(mat1)
    activation_name = _activation_name(activation)
    if activation is nn.relu or activation_name == "relu":
        return nn.relu(y)
    if activation is nn.gelu or activation_name == "gelu":
        return nn.gelu(y)
    raise ValueError(f"Unexpected activation {activation!r}")
