from functools import wraps
from typing import Callable, TypeVar


T = TypeVar("T")


def activation_ckpt_wrapper(module: Callable) -> Callable:
    """MLX-compatible wrapper for the official SAM3 activation checkpoint hook.

    MLX does not expose a direct equivalent of PyTorch activation checkpointing
    here, so the flag is accepted as an optimization hint and the callable runs
    normally.
    """

    @wraps(module)
    def act_ckpt_wrapper(
        *args,
        act_ckpt_enable: bool = True,
        use_reentrant: bool = False,
        **kwargs,
    ):
        del act_ckpt_enable, use_reentrant
        return module(*args, **kwargs)

    return act_ckpt_wrapper


def clone_output_wrapper(f: Callable[..., T]) -> Callable[..., T]:
    """Torch Torch output cloning is not needed for MLX; preserve the callable API."""

    @wraps(f)
    def wrapped(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapped
