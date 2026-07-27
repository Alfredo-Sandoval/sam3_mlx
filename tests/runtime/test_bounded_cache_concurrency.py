from concurrent.futures import ThreadPoolExecutor

import pytest

from sam3_mlx.model.bounded_cache import BoundedLRUCache


def test_bounded_lru_cache_rejects_non_integer_capacity():
    for invalid in (True, False, 0, -1, 1.5, "8"):
        with pytest.raises(ValueError, match="positive integer"):
            BoundedLRUCache(invalid)


def test_bounded_lru_cache_keys_are_a_stable_snapshot():
    cache = BoundedLRUCache[str, int](maxsize=2)
    cache["a"] = 1
    snapshot = cache.keys()
    cache["b"] = 2

    assert snapshot == ("a",)
    assert cache.keys() == ("a", "b")


def test_bounded_lru_cache_remains_consistent_under_concurrent_access():
    # Keep capacity above the shared key cardinality so this test isolates
    # synchronization rather than intentionally racing against eviction.
    cache = BoundedLRUCache[int, int](maxsize=32)

    def exercise(worker: int) -> None:
        for step in range(2_000):
            key = (worker * 17 + step) % 31
            cache[key] = step
            assert cache.get(key) is not None
            if step % 97 == 0:
                snapshot = cache.keys()
                assert len(snapshot) == len(set(snapshot))

    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(exercise, range(24)))

    assert 0 < len(cache) <= cache.maxsize
    assert len(cache.keys()) == len(cache)
