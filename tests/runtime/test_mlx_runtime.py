import importlib.metadata
import numpy as np
import platform
import mlx.core as mx

from sam3_mlx.mlx_runtime import (
    evaluate_boundary,
    finite_all,
    runtime_info,
    shape_dtype,
    to_numpy,
)


def test_to_numpy_synchronizes_lazy_mlx_expression_and_preserves_values():
    value = mx.array([1.0, 2.0], dtype=mx.float32) + mx.array([3.0, 4.0])

    array = to_numpy(value)

    assert array.dtype == np.float32
    np.testing.assert_array_equal(array, np.array([4.0, 6.0], dtype=np.float32))


def test_to_numpy_dtype_and_copy_are_explicit_host_export_options():
    source = np.array([1, 2], dtype=np.int64)

    exported = to_numpy(source, dtype=np.float32, copy=True)
    exported[0] = 9.0

    assert exported.dtype == np.float32
    np.testing.assert_array_equal(source, np.array([1, 2], dtype=np.int64))


def test_shape_dtype_reports_synchronized_metadata():
    value = mx.zeros((2, 3), dtype=mx.float32) + 1.0

    assert shape_dtype(value) == {"shape": [2, 3], "dtype": "mlx.core.float32"}


def test_finite_all_detects_nonfinite_values_after_host_export():
    assert finite_all(mx.array([0.0, 1.0], dtype=mx.float32))
    assert not finite_all(mx.array([0.0, float("inf")], dtype=mx.float32))


def test_evaluate_boundary_accepts_multiple_arrays():
    left = mx.array([1.0], dtype=mx.float32) + 1.0
    right = mx.array([2.0], dtype=mx.float32) + 1.0

    evaluate_boundary(left, right)

    np.testing.assert_array_equal(np.asarray(left), np.array([2.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(right), np.array([3.0], dtype=np.float32))


def test_runtime_info_contains_artifact_metadata():
    info = runtime_info()

    assert info.python == platform.python_version()
    assert info.mlx_version == importlib.metadata.version("mlx")
    assert info.default_device == str(mx.default_device())
    assert isinstance(info.metal_available, bool)
    assert info.platform == platform.platform()
