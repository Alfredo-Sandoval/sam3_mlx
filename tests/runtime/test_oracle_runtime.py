from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import ContextManager

import pytest

from scripts._oracle_runtime import (
    install_cpu_oracle_adapters,
    restore_construction_adapters,
    TensorClass,
    validate_case_specs,
)


class _TensorApi:
    def __init__(self) -> None:
        self.pin_memory: Callable[..., object] = self._pin_memory

    @staticmethod
    def _pin_memory(tensor: object, *args: object, **kwargs: object) -> object:
        del args, kwargs
        return ("pinned", tensor)


class _TorchRuntime:
    bfloat16: object = object()
    __version__: object = "test"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.zeros: Callable[..., object] = self._zeros
        self.arange: Callable[..., object] = self._arange
        self.Tensor: TensorClass = _TensorApi()
        self.autocast: Callable[..., ContextManager[None]] = self._autocast

    def _zeros(self, *args: object, **kwargs: object) -> object:
        del args
        self.calls.append(("zeros", kwargs))
        return kwargs

    def _arange(self, *args: object, **kwargs: object) -> object:
        del args
        self.calls.append(("arange", kwargs))
        return kwargs

    @staticmethod
    def _autocast(*args: object, **kwargs: object) -> ContextManager[None]:
        del args, kwargs
        return nullcontext()


def test_cpu_oracle_adapters_redirect_only_cuda_construction_devices() -> None:
    torch = _TorchRuntime()
    original_zeros = torch.zeros
    original_arange = torch.arange
    original_pin_memory = torch.Tensor.pin_memory

    originals = install_cpu_oracle_adapters(torch)
    torch.zeros(2, device="cuda:0")
    torch.arange(2, device="mps")

    assert torch.calls == [
        ("zeros", {"device": "cpu"}),
        ("arange", {"device": "mps"}),
    ]
    marker = object()
    assert torch.Tensor.pin_memory(marker) is marker

    restore_construction_adapters(torch, originals)
    assert torch.zeros is original_zeros
    assert torch.arange is original_arange
    assert originals.pin_memory is original_pin_memory


def test_oracle_case_validation_returns_precise_shape_and_rejects_bool_aliases() -> (
    None
):
    valid = [
        {
            "name": "positive_box",
            "resolution": 1008,
            "prompt": None,
            "geometric_prompts": [{"box": [1, 2.5, 3, 4], "label": True}],
        }
    ]

    assert validate_case_specs(valid) == valid

    invalid_label = [
        {
            **valid[0],
            "geometric_prompts": [{"box": [1, 2, 3, 4], "label": 1}],
        }
    ]
    with pytest.raises(ValueError, match="invalid box label"):
        validate_case_specs(invalid_label)

    with pytest.raises(ValueError, match="resolution must be an integer"):
        validate_case_specs([{**valid[0], "resolution": True}])
