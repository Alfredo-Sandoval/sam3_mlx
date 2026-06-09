"""Memory retry compatibility helpers.

The official helper retries upstream OOM paths. This MLX fork does not own
execution here, so the wrapper preserves the callable shape without retrying.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def _ignore_torch_oom():
    yield


def retry_if_backend_oom(func):
    """Return ``func`` unchanged; backend retry behavior is intentionally unported."""

    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapped
