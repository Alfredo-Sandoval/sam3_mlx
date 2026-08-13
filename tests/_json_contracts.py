"""Fail-closed validators for committed JSON test fixtures."""

from collections.abc import Mapping
from typing import cast


def require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise TypeError(f"{context} must use string keys")
    return cast(Mapping[str, object], mapping)


def require_object_mapping(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return cast(Mapping[object, object], value)


def require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return cast(list[object], value)


def require_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    return value


def require_real(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a real number")
    return float(value)
