from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sam3_mlx.perflib.compile import (
    clone_output_wrapper,
    compile_wrapper,
    recursive_clone,
    recursive_contiguous,
    shape_logging_wrapper,
)


FloatArray = npt.NDArray[np.float32]


@dataclass
class NestedTensor:
    tensors: object
    mask: object | None


def test_recursive_contiguous_preserves_tree_and_nested_tensor_contract():
    noncontiguous = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    value = {
        "items": [noncontiguous],
        "nested": NestedTensor(tensors=noncontiguous, mask=noncontiguous > 2),
    }

    transformed = recursive_contiguous(value)

    assert isinstance(transformed, dict)
    transformed_dict = cast(dict[str, object], transformed)
    items = cast(list[object], transformed_dict["items"])
    item = cast(FloatArray, items[0])
    assert item.flags.c_contiguous
    nested = transformed_dict["nested"]
    assert isinstance(nested, NestedTensor)
    assert cast(FloatArray, nested.tensors).flags.c_contiguous


def test_recursive_clone_and_clone_wrapper_return_independent_arrays():
    source = np.arange(4, dtype=np.float32)
    cloned = recursive_clone({"array": source})
    assert isinstance(cloned, dict)
    cloned_array = cast(FloatArray, cast(dict[str, object], cloned)["array"])
    assert np.array_equal(cloned_array, source)
    assert not np.shares_memory(cloned_array, source)

    def identity(value: FloatArray) -> FloatArray:
        return value

    wrapped = clone_output_wrapper(identity)
    wrapped_array = wrapped(source)
    assert np.array_equal(wrapped_array, source)
    assert not np.shares_memory(wrapped_array, source)


def test_compile_wrapper_normalizes_inputs_and_clones_cudagraph_mode_outputs():
    seen_contiguous: list[bool] = []

    def identity(value: FloatArray) -> FloatArray:
        seen_contiguous.append(value.flags.c_contiguous)
        return value

    source = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    wrapped = compile_wrapper(identity, mode="reduce-overhead")
    output = wrapped(source)

    assert seen_contiguous == [True]
    assert np.array_equal(output, source)
    assert not np.shares_memory(output, source)

    unmodeled = compile_wrapper(identity, mode="default")
    direct_output = unmodeled(source)
    assert direct_output.flags.c_contiguous


def test_shape_logging_wrapper_logs_each_shape_once_and_exposes_toggle(
    capsys: pytest.CaptureFixture[str],
):
    def add(left: FloatArray, *, right: FloatArray) -> FloatArray:
        return left + right

    wrapped = shape_logging_wrapper(add, keep_kwargs={"right"})
    left = np.ones((2, 3), dtype=np.float32)
    right = np.ones((2, 3), dtype=np.float32)

    wrapped(left, right=right)
    assert capsys.readouterr().out == ""

    wrapped.set_logging(True)
    assert wrapped.enable_logging is True
    wrapped(left, right=right)
    assert capsys.readouterr().out == ""

    wrapped(np.ones((1, 3), dtype=np.float32), right=right)
    output = capsys.readouterr().out
    assert "[ShapeLogger] New input shapes" in output
    assert "(1, 3)" in output

    wrapped.set_logging()
    assert wrapped.enable_logging is False
