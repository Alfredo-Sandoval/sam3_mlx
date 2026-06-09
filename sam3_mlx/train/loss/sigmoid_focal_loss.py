# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""MLX sigmoid focal-loss surface matching the official SAM3 API names.

The official file is a PyTorch autograd/Triton implementation. The MLX fork
keeps explicit MLX focal-loss helpers while exposing official Triton kernel
names as fail-fast shims.
"""

from __future__ import annotations

import mlx.core as mx

from sam3_mlx._unsupported import UPSTREAM_COMMIT, raise_unsupported

MLX_SIGMOID_FOCAL_LOSS_BASE_COMMIT = "dc33741d86020f34c73f9534deabff1007cdd886"

_UNSUPPORTED_TRITON_FOCAL_LOSS_MESSAGE = (
    "Official SAM3 sigmoid focal-loss Triton autograd kernel behavior is "
    "not implemented in sam3_mlx. The official implementation at commit "
    f"{UPSTREAM_COMMIT} depends on Torch autograd "
    "and Triton kernels. Use sigmoid_focal_loss or "
    "sigmoid_focal_loss_reduce for the explicit MLX implementation."
)


def _raise_triton_focal_loss_unsupported(feature: str) -> None:
    raise_unsupported(
        feature,
        reason="triton-kernel",
        alternative="sam3_mlx.train.loss.sigmoid_focal_loss.sigmoid_focal_loss",
        detail=_UNSUPPORTED_TRITON_FOCAL_LOSS_MESSAGE,
    )


def _as_float_array(value) -> mx.array:
    if isinstance(value, mx.array):
        return value.astype(mx.float32)
    return mx.array(value, dtype=mx.float32)


def _inner_focal_loss_fwd(*args, **kwargs):
    _raise_triton_focal_loss_unsupported("_inner_focal_loss_fwd")


def sigmoid_focal_loss_fwd_kernel(*args, **kwargs):
    _raise_triton_focal_loss_unsupported("sigmoid_focal_loss_fwd_kernel")


def sigmoid_focal_loss_fwd_kernel_reduce(*args, **kwargs):
    _raise_triton_focal_loss_unsupported("sigmoid_focal_loss_fwd_kernel_reduce")


def _inner_focal_loss_bwd(*args, **kwargs):
    _raise_triton_focal_loss_unsupported("_inner_focal_loss_bwd")


def sigmoid_focal_loss_bwd_kernel(*args, **kwargs):
    _raise_triton_focal_loss_unsupported("sigmoid_focal_loss_bwd_kernel")


def sigmoid_focal_loss_bwd_kernel_reduce(*args, **kwargs):
    _raise_triton_focal_loss_unsupported("sigmoid_focal_loss_bwd_kernel_reduce")


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2.0):
    """Elementwise sigmoid focal loss with the official alpha/gamma contract."""

    inputs = _as_float_array(inputs)
    targets = _as_float_array(targets)
    if inputs.shape != targets.shape:
        raise ValueError(
            f"inputs and targets must have the same shape, got {inputs.shape} "
            f"and {targets.shape}."
        )

    max_val = mx.maximum(inputs, mx.zeros_like(inputs))
    bce_loss = max_val - inputs * targets + mx.log(1 + mx.exp(-mx.abs(inputs)))
    prob = mx.sigmoid(inputs)
    inv_targets = 1 - targets
    p_t = prob * targets + (1 - prob) * inv_targets
    alpha_t = alpha * targets + (1 - alpha) * inv_targets
    return alpha_t * bce_loss * ((1 - p_t) ** gamma)


def sigmoid_focal_loss_reduce(inputs, targets, alpha: float = 0.25, gamma: float = 2.0):
    """Sum-reduced sigmoid focal loss, equivalent to the official reduced kernel."""

    return mx.sum(sigmoid_focal_loss(inputs, targets, alpha=alpha, gamma=gamma))


class SigmoidFocalLoss:
    """Fail-fast shim for official ``SigmoidFocalLoss`` Torch autograd calls."""

    BLOCK_SIZE = 256

    @staticmethod
    def apply(*args, **kwargs):
        _raise_triton_focal_loss_unsupported("SigmoidFocalLoss.apply")


class SigmoidFocalLossReduced:
    """Fail-fast shim for official reduced Torch autograd focal-loss calls."""

    BLOCK_SIZE = 256
    REDUCE_SIZE = 32

    @staticmethod
    def apply(*args, **kwargs):
        _raise_triton_focal_loss_unsupported("SigmoidFocalLossReduced.apply")


triton_sigmoid_focal_loss = SigmoidFocalLoss.apply
triton_sigmoid_focal_loss_reduce = SigmoidFocalLossReduced.apply


__all__ = [
    "MLX_SIGMOID_FOCAL_LOSS_BASE_COMMIT",
    "SigmoidFocalLoss",
    "SigmoidFocalLossReduced",
    "sigmoid_focal_loss",
    "sigmoid_focal_loss_bwd_kernel",
    "sigmoid_focal_loss_bwd_kernel_reduce",
    "sigmoid_focal_loss_fwd_kernel",
    "sigmoid_focal_loss_fwd_kernel_reduce",
    "sigmoid_focal_loss_reduce",
    "triton_sigmoid_focal_loss",
    "triton_sigmoid_focal_loss_reduce",
]
