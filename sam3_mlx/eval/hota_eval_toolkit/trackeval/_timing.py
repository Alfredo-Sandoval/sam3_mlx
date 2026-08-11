"""Fail-fast timing decorator shim for HOTA TrackEval compatibility."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Never

from sam3_mlx._unsupported import raise_unsupported

DO_TIMING = False
DISPLAY_LESS_PROGRESS = False
timer_dict: dict[str, float] = {}
counter = 0


def time[**P, R](f: Callable[P, R]) -> Callable[P, Never]:
    @wraps(f)
    def wrap(*args: P.args, **kw: P.kwargs) -> Never:
        del args, kw
        raise_unsupported(
            "sam3_mlx.eval.hota_eval_toolkit.trackeval._timing.time",
            reason="eval-stack",
            detail=(
                "The official SAM3 behavior depends on the TrackEval/HOTA "
                "timing and evaluator execution area."
            ),
        )

    return wrap
