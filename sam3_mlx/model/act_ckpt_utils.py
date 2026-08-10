from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TypeVar


_T = TypeVar("_T")
_R = TypeVar("_R")


def activation_ckpt_wrapper(module: Callable[..., _R]) -> Callable[..., _R]:
    """MLX-compatible wrapper for the official SAM3 activation checkpoint hook.

    MLX does not expose a direct equivalent of PyTorch activation checkpointing
    here, so the flag is accepted as an optimization hint and the callable runs
    normally.
    """

    @wraps(module)
    def act_ckpt_wrapper(
        *args: object,
        act_ckpt_enable: bool = True,
        use_reentrant: bool = False,
        **kwargs: object,
    ) -> _R:
        del act_ckpt_enable, use_reentrant
        return module(*args, **kwargs)

    return act_ckpt_wrapper


def clone_output_wrapper(f: Callable[..., _T]) -> Callable[..., _T]:
    """Torch output cloning is not needed for MLX; preserve the callable API."""

    @wraps(f)
    def wrapped(*args: object, **kwargs: object) -> _T:
        return f(*args, **kwargs)

    return wrapped
