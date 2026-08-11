# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Checkpoint helper surface for the MLX port.

The official checkpoint loading functions are PyTorch serialization utilities.
This module preserves the import paths and ports key-filtering helpers that are
plain Python, while torch checkpoint IO fails explicitly.
"""

from __future__ import annotations

import contextlib
import fnmatch
import logging
from collections.abc import Generator, Iterable, Mapping, Sequence
from typing import Never, Protocol, SupportsFloat, cast

import numpy as np

from sam3_mlx._unsupported import UPSTREAM_COMMIT, raise_unsupported


_UNSUPPORTED_CHECKPOINT_MESSAGE = (
    "SAM3 PyTorch checkpoint loading is not implemented in the MLX port yet. "
    "The official implementation at commit "
    f"{UPSTREAM_COMMIT} uses torch.load and "
    "torch.nn.Module.load_state_dict. Use an explicit MLX weight-loading path."
)


def _raise_checkpoint_unsupported(feature: str) -> Never:
    raise_unsupported(
        feature,
        reason="training-loop",
        alternative="sam3_mlx.model_builder.build_sam3_image_model",
        detail=_UNSUPPORTED_CHECKPOINT_MESSAGE,
    )


class _Summable(Protocol):
    def sum(self) -> object: ...


class _ItemValue(Protocol):
    def item(self) -> object: ...


class _StateDictGetter(Protocol):
    def __call__(self) -> object: ...


class _NamedParameters(Protocol):
    def __call__(self) -> Iterable[tuple[str, object]]: ...


class _LoadStateDict(Protocol):
    def __call__(self, state_dict: dict[str, object], *, strict: bool) -> object: ...


class CheckpointKernel(Protocol):
    def __call__(self, *, state_dict: dict[str, object]) -> dict[str, object]: ...


def _state_dict(model: object) -> dict[str, object]:
    getter = getattr(model, "state_dict", None)
    if not callable(getter):
        _raise_checkpoint_unsupported("model must expose state_dict")
    result = cast(_StateDictGetter, getter)()
    if not isinstance(result, dict):
        raise TypeError("model state_dict() must return a dict")
    return cast(dict[str, object], result)


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    items = list(cast(Sequence[object], value))
    if not all(isinstance(item, str) for item in items):
        raise TypeError(f"{name} must be a sequence of strings")
    return cast(list[str], items)


def unix_pattern_to_parameter_names(
    constraints: Sequence[str], all_parameter_names: Sequence[str]
) -> set[str]:
    """Select names matching any of the provided unix-style constraints."""

    parameter_names: list[set[str]] = []
    for param_name in constraints:
        matching_parameters = set(fnmatch.filter(all_parameter_names, param_name))
        if len(matching_parameters) <= 0:
            raise AssertionError(
                f"param_names {param_name} don't match any param in the given names."
            )
        parameter_names.append(matching_parameters)
    return parameter_names[0].union(*parameter_names[1:]) if parameter_names else set()


def filter_params_matching_unix_pattern[T](
    patterns: Sequence[str], state_dict: Mapping[str, T]
) -> dict[str, T]:
    """Keep only state-dict entries matching the provided unix patterns."""

    if len(patterns) == 0:
        return {}

    all_keys = list(state_dict.keys())
    included_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {key: state_dict[key] for key in included_keys}


def exclude_params_matching_unix_pattern[T](
    patterns: Sequence[str], state_dict: dict[str, T]
) -> dict[str, T]:
    """Remove state-dict entries matching the provided unix patterns."""

    if len(patterns) == 0:
        return state_dict

    all_keys = list(state_dict.keys())
    excluded_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {key: value for key, value in state_dict.items() if key not in excluded_keys}


def _to_scalar_sum(value: object) -> float:
    sum_method = getattr(value, "sum", None)
    summed = (
        cast(_Summable, value).sum()
        if callable(sum_method)
        else np.asarray(value).sum()
    )
    item_method = getattr(summed, "item", None)
    if callable(item_method):
        summed = cast(_ItemValue, summed).item()
    return float(cast(SupportsFloat, summed))


def _get_state_dict_summary(state_dict: Mapping[str, object]) -> np.ndarray:
    keys: list[str] = []
    trace: list[float] = []
    for key, value in state_dict.items():
        keys.append(key)
        trace.append(_to_scalar_sum(value))
    return np.array(trace)[np.argsort(keys)]


def assert_skipped_parameters_are_frozen(
    model: object, patterns: Sequence[str]
) -> None:
    """Verify that skipped parameters are frozen when the model exposes that API."""

    if not patterns:
        return
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        _raise_checkpoint_unsupported("assert_skipped_parameters_are_frozen")

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=_state_dict(model)
    )
    non_frozen_keys = {
        name
        for name, parameter in cast(_NamedParameters, named_parameters)()
        if name in frozen_state_dict and getattr(parameter, "requires_grad", False)
    }
    if non_frozen_keys:
        raise ValueError(
            "Parameters excluded with `skip_saving_parameters` should be frozen: "
            f"{non_frozen_keys}"
        )


@contextlib.contextmanager
def with_check_parameter_frozen(
    model: object, patterns: Sequence[str], disabled: bool = True
) -> Generator[None, None, None]:
    """Context manager checking that selected state-dict values stay unchanged."""

    if not patterns or disabled:
        yield
        return
    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=_state_dict(model)
    )
    summary_before = _get_state_dict_summary(frozen_state_dict)

    yield

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=_state_dict(model)
    )
    summary_after = _get_state_dict_summary(frozen_state_dict)

    if not np.allclose(summary_before, summary_after, atol=1e-6):
        raise ValueError(
            "The `model_weight_initializer` has initialized parameters frozen with "
            "`skip_saving_parameters`."
        )


class CkptExcludeKernel:
    """Remove keys from a state dict when they match a unix pattern."""

    def __init__(self, key_pattern: Sequence[str]):
        self.key_pattern = key_pattern

    def __call__[T](self, state_dict: dict[str, T]) -> dict[str, T]:
        if len(self.key_pattern) == 0:
            return state_dict
        exclude_keys = unix_pattern_to_parameter_names(
            self.key_pattern, list(state_dict.keys())
        )
        return {
            key: value for key, value in state_dict.items() if key not in exclude_keys
        }


def load_checkpoint(
    path_list: Sequence[str],
    pick_recursive_keys: Sequence[str] | None = None,
    map_location: str = "cpu",
) -> Never:
    _raise_checkpoint_unsupported("load_checkpoint")


def get_state_dict(
    checkpoint: object, ckpt_state_dict_keys: Sequence[str | int]
) -> object:
    pre_train_dict = checkpoint
    for index, key in enumerate(ckpt_state_dict_keys):
        available: object
        if isinstance(pre_train_dict, Mapping):
            mapping = cast(Mapping[object, object], pre_train_dict)
            key_exists = key in mapping
            available = mapping.keys()
        elif isinstance(pre_train_dict, Sequence) and not isinstance(
            pre_train_dict, (str, bytes)
        ):
            sequence = cast(Sequence[object], pre_train_dict)
            if isinstance(key, bool):
                raise TypeError("checkpoint sequence keys must not be booleans")
            key_exists = isinstance(key, int) and -len(sequence) <= key < len(sequence)
            available = f"sequence length {len(sequence)}"
        else:
            key_exists = False
            available = type(pre_train_dict).__name__
        if not key_exists:
            key_str = "".join(
                f"[{prior_key!r}]" for prior_key in ckpt_state_dict_keys[:index]
            )
            raise KeyError(
                f"{key!r} not found in checkpoint{key_str} with keys: {available}"
            )
        if isinstance(pre_train_dict, Mapping):
            pre_train_dict = cast(Mapping[object, object], pre_train_dict)[key]
        else:
            pre_train_dict = cast(Sequence[object], pre_train_dict)[cast(int, key)]
    return pre_train_dict


def load_checkpoint_and_apply_kernels(
    checkpoint_path: str,
    checkpoint_kernels: Sequence[CheckpointKernel] | None = None,
    ckpt_state_dict_keys: tuple[str, ...] = ("state_dict",),
    map_location: str = "cpu",
) -> Never:
    _raise_checkpoint_unsupported("load_checkpoint_and_apply_kernels")


def check_load_state_dict_errors(
    missing_keys: Sequence[str],
    unexpected_keys: Sequence[str],
    strict: bool,
    ignore_missing_keys: Sequence[str] | None = None,
    ignore_unexpected_keys: Sequence[str] | None = None,
) -> None:
    missing_keys = list(missing_keys)
    unexpected_keys = list(unexpected_keys)
    if ignore_missing_keys is not None and len(ignore_missing_keys) > 0:
        ignored_keys = unix_pattern_to_parameter_names(
            ignore_missing_keys, missing_keys
        )
        missing_keys = [key for key in missing_keys if key not in ignored_keys]

    if ignore_unexpected_keys is not None and len(ignore_unexpected_keys) > 0:
        ignored_unexpected_keys = unix_pattern_to_parameter_names(
            ignore_unexpected_keys, unexpected_keys
        )
        unexpected_keys = [
            key for key in unexpected_keys if key not in ignored_unexpected_keys
        ]

    err = "State key mismatch."
    if unexpected_keys:
        err += f" Unexpected keys: {unexpected_keys}."
    if missing_keys:
        err += f" Missing keys: {missing_keys}."

    if unexpected_keys or missing_keys:
        logging.warning(err)
        if unexpected_keys or strict:
            raise KeyError(err)


def load_state_dict_into_model(
    state_dict: dict[str, object],
    model: object,
    strict: bool = True,
    ignore_missing_keys: Sequence[str] | None = None,
    ignore_unexpected_keys: Sequence[str] | None = None,
    checkpoint_kernels: Sequence[CheckpointKernel] | None = None,
) -> object:
    """Load a state dict into a model only when the model exposes the API."""

    if checkpoint_kernels is not None:
        for fn in checkpoint_kernels:
            state_dict = fn(state_dict=state_dict)
    load_state_dict = getattr(model, "load_state_dict", None)
    if not callable(load_state_dict):
        _raise_checkpoint_unsupported("load_state_dict_into_model")
    result = cast(_LoadStateDict, load_state_dict)(state_dict, strict=False)
    if isinstance(result, tuple):
        result_tuple = cast(tuple[object, ...], result)
        if len(result_tuple) != 2:
            raise TypeError("load_state_dict result tuple must have two items")
        missing_value, unexpected_value = result_tuple
    else:
        missing_value = getattr(result, "missing_keys", [])
        unexpected_value = getattr(result, "unexpected_keys", [])

    missing_keys = _string_list(missing_value, "missing_keys")
    unexpected_keys = _string_list(unexpected_value, "unexpected_keys")

    check_load_state_dict_errors(
        missing_keys,
        unexpected_keys,
        strict=strict,
        ignore_missing_keys=ignore_missing_keys,
        ignore_unexpected_keys=ignore_unexpected_keys,
    )
    return model
