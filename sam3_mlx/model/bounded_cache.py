"""Small shared helpers for bounded in-process caches."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class BoundedLRUCache(Generic[K, V]):
    """Thread-safe insertion/access-ordered map with a fixed entry bound.

    The image runtime shares model modules across predictor sessions. Position,
    RoPE, and decoder-coordinate caches therefore need synchronized mutations:
    ``OrderedDict`` operations are individually small, but compound operations
    such as membership plus ``move_to_end`` are not a safe public concurrency
    contract. Values are created by callers outside this lock, so expensive MLX
    graph construction is never serialized unnecessarily.
    """

    def __init__(self, maxsize: object = 8) -> None:
        if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1:
            raise ValueError("maxsize must be a positive integer")
        self.maxsize = maxsize
        self._data: OrderedDict[K, V] = OrderedDict()
        self._lock = RLock()

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def get(self, key: K, default: V | None = None) -> V | None:
        with self._lock:
            if key not in self._data:
                return default
            self._data.move_to_end(key)
            return self._data[key]

    def __getitem__(self, key: K) -> V:
        with self._lock:
            value = self._data[key]
            self._data.move_to_end(key)
            return value

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def keys(self) -> tuple[K, ...]:
        """Return a stable snapshot rather than a live mutable dictionary view."""

        with self._lock:
            return tuple(self._data.keys())
