"""Small shared helpers for bounded in-process caches."""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class BoundedLRUCache(Generic[K, V]):
    """Insertion/access-ordered map with a fixed maximum entry count."""

    def __init__(self, maxsize: int = 8) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = int(maxsize)
        self._data: OrderedDict[K, V] = OrderedDict()

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: K, default: V | None = None) -> V | None:
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def __getitem__(self, key: K) -> V:
        value = self._data[key]
        self._data.move_to_end(key)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def keys(self):
        return self._data.keys()
