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
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from sam3_mlx._unsupported import UPSTREAM_COMMIT, raise_unsupported


_UNSUPPORTED_CHECKPOINT_MESSAGE = (
    "SAM3 PyTorch checkpoint loading is not implemented in the MLX port yet. "
    "The official implementation at commit "
    f"{UPSTREAM_COMMIT} uses torch.load and "
    "torch.nn.Module.load_state_dict. Use an explicit MLX weight-loading path."
)


def _raise_checkpoint_unsupported(feature: str) -> None:
    raise_unsupported(
        feature,
        reason="training-loop",
        alternative="sam3_mlx.model_builder.build_sam3_image_model",
        detail=_UNSUPPORTED_CHECKPOINT_MESSAGE,
    )


def unix_pattern_to_parameter_names(
    constraints: List[str], all_parameter_names: Sequence[str]
) -> Union[None, Set[str]]:
    """Select names matching any of the provided unix-style constraints."""

    parameter_names = []
    for param_name in constraints:
        matching_parameters = set(fnmatch.filter(all_parameter_names, param_name))
        if len(matching_parameters) <= 0:
            raise AssertionError(
                f"param_names {param_name} don't match any param in the given names."
            )
        parameter_names.append(matching_parameters)
    return set.union(*parameter_names) if parameter_names else set()


def filter_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Keep only state-dict entries matching the provided unix patterns."""

    if len(patterns) == 0:
        return {}

    all_keys = list(state_dict.keys())
    included_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {key: state_dict[key] for key in included_keys}


def exclude_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Remove state-dict entries matching the provided unix patterns."""

    if len(patterns) == 0:
        return state_dict

    all_keys = list(state_dict.keys())
    excluded_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {key: value for key, value in state_dict.items() if key not in excluded_keys}


def _to_scalar_sum(value: Any) -> float:
    summed = value.sum() if hasattr(value, "sum") else np.asarray(value).sum()
    if hasattr(summed, "item"):
        summed = summed.item()
    return float(summed)


def _get_state_dict_summary(state_dict: Dict[str, Any]):
    keys = []
    trace = []
    for key, value in state_dict.items():
        keys.append(key)
        trace.append(_to_scalar_sum(value))
    return np.array(trace)[np.argsort(keys)]


def assert_skipped_parameters_are_frozen(model, patterns: List[str]):
    """Verify that skipped parameters are frozen when the model exposes that API."""

    if not patterns:
        return
    if not hasattr(model, "state_dict") or not hasattr(model, "named_parameters"):
        _raise_checkpoint_unsupported("assert_skipped_parameters_are_frozen")

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    non_frozen_keys = {
        name
        for name, parameter in model.named_parameters()
        if name in frozen_state_dict and getattr(parameter, "requires_grad", False)
    }
    if non_frozen_keys:
        raise ValueError(
            "Parameters excluded with `skip_saving_parameters` should be frozen: "
            f"{non_frozen_keys}"
        )


@contextlib.contextmanager
def with_check_parameter_frozen(model, patterns: List[str], disabled: bool = True):
    """Context manager checking that selected state-dict values stay unchanged."""

    if not patterns or disabled:
        yield
        return
    if not hasattr(model, "state_dict"):
        _raise_checkpoint_unsupported("with_check_parameter_frozen")

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_before = _get_state_dict_summary(frozen_state_dict)

    yield

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_after = _get_state_dict_summary(frozen_state_dict)

    if not np.allclose(summary_before, summary_after, atol=1e-6):
        raise ValueError(
            "The `model_weight_initializer` has initialized parameters frozen with "
            "`skip_saving_parameters`."
        )


class CkptExcludeKernel:
    """Remove keys from a state dict when they match a unix pattern."""

    def __init__(self, key_pattern: List[str]):
        self.key_pattern = key_pattern

    def __call__(self, state_dict: Dict[str, Any]):
        if len(self.key_pattern) == 0:
            return state_dict
        exclude_keys = unix_pattern_to_parameter_names(
            self.key_pattern, list(state_dict.keys())
        )
        return {
            key: value for key, value in state_dict.items() if key not in exclude_keys
        }


def load_checkpoint(
    path_list: List[str],
    pick_recursive_keys: Optional[List[str]] = None,
    map_location: str = "cpu",
) -> Any:
    _raise_checkpoint_unsupported("load_checkpoint")


def get_state_dict(checkpoint, ckpt_state_dict_keys):
    pre_train_dict = checkpoint
    for index, key in enumerate(ckpt_state_dict_keys):
        key_exists = (
            isinstance(pre_train_dict, Mapping)
            and key in pre_train_dict
            or isinstance(pre_train_dict, Sequence)
            and not isinstance(pre_train_dict, (str, bytes))
            and isinstance(key, int)
            and key < len(pre_train_dict)
        )
        if not key_exists:
            key_str = "".join(
                f"[{prior_key!r}]" for prior_key in ckpt_state_dict_keys[:index]
            )
            available = (
                pre_train_dict.keys()
                if isinstance(pre_train_dict, Mapping)
                else f"sequence length {len(pre_train_dict)}"
                if isinstance(pre_train_dict, Sequence)
                else type(pre_train_dict).__name__
            )
            raise KeyError(
                f"{key!r} not found in checkpoint{key_str} with keys: {available}"
            )
        pre_train_dict = pre_train_dict[key]
    return pre_train_dict


def load_checkpoint_and_apply_kernels(
    checkpoint_path: str,
    checkpoint_kernels: List[Callable] = None,
    ckpt_state_dict_keys: Tuple[str] = ("state_dict",),
    map_location: str = "cpu",
):
    _raise_checkpoint_unsupported("load_checkpoint_and_apply_kernels")


def check_load_state_dict_errors(
    missing_keys,
    unexpected_keys,
    strict: bool,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
):
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
    state_dict: Dict[str, Any],
    model,
    strict: bool = True,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
    checkpoint_kernels: List[Callable] = None,
):
    """Load a state dict into a model only when the model exposes the API."""

    if checkpoint_kernels is not None:
        for fn in checkpoint_kernels:
            state_dict = fn(state_dict=state_dict)
    load_state_dict = getattr(model, "load_state_dict", None)
    if load_state_dict is None:
        _raise_checkpoint_unsupported("load_state_dict_into_model")
    result = load_state_dict(state_dict, strict=False)
    if isinstance(result, tuple):
        missing_keys, unexpected_keys = result
    else:
        missing_keys = getattr(result, "missing_keys", [])
        unexpected_keys = getattr(result, "unexpected_keys", [])

    check_load_state_dict_errors(
        missing_keys,
        unexpected_keys,
        strict=strict,
        ignore_missing_keys=ignore_missing_keys,
        ignore_unexpected_keys=ignore_unexpected_keys,
    )
    return model
