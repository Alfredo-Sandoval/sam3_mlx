"""Memory retry compatibility helpers.

The official helper retries upstream OOM paths. This MLX fork does not own
execution here, so the wrapper preserves the callable shape without retrying.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager


@contextmanager
def _ignore_torch_oom() -> Generator[None]:  # pyright: ignore[reportUnusedFunction]
    yield


def retry_if_backend_oom[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Return ``func`` unchanged; backend retry behavior is intentionally unported."""

    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **kwargs)

    return wrapped
