"""CPU connected-components helpers with MLX-compatible return wrapping.

These functions intentionally materialize input masks to host NumPy arrays.
They are used for postprocess cleanup and should not be treated as MLX kernels.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, TypeAlias, TypeGuard, cast, overload

import numpy as np
import numpy.typing as npt

from sam3_mlx.mlx_runtime import to_numpy

if TYPE_CHECKING:
    import mlx.core as mx


NumpyArray = npt.NDArray[np.generic]
IntArray = npt.NDArray[np.int64]
if TYPE_CHECKING:
    ComponentArray: TypeAlias = IntArray | mx.array
else:
    ComponentArray: TypeAlias = object
ComponentResult: TypeAlias = tuple[ComponentArray, ComponentArray]


def _is_mlx_array(value: object) -> TypeGuard["mx.array"]:
    return type(value).__module__.startswith("mlx.")


def _from_numpy(value: IntArray, like: object) -> ComponentArray:
    if _is_mlx_array(like):
        import mlx.core as mx

        return mx.array(value)
    return value


def _to_host_connected_components_input(value: object) -> NumpyArray:
    """Synchronize and export masks at the explicit CPU connected-components boundary."""

    return cast(NumpyArray, to_numpy(value))


def _label_single(values: NumpyArray) -> tuple[IntArray, IntArray]:
    if values.ndim != 2:
        raise ValueError("connected_components_cpu_single expects a 2D array.")

    binary = values.astype(bool, copy=False)
    labels = cast(IntArray, np.zeros(binary.shape, dtype=np.int64))
    counts = cast(IntArray, np.zeros(binary.shape, dtype=np.int64))
    height, width = binary.shape
    next_label = 1
    neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    for y in range(height):
        for x in range(width):
            if not binary[y, x] or labels[y, x] != 0:
                continue
            queue = deque([(y, x)])
            labels[y, x] = next_label
            pixels: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and binary[ny, nx]
                        and labels[ny, nx] == 0
                    ):
                        labels[ny, nx] = next_label
                        queue.append((ny, nx))
            component_size = len(pixels)
            for py, px in pixels:
                counts[py, px] = component_size
            next_label += 1

    return labels, counts


def connected_components_cpu_single(values: object) -> ComponentResult:
    """Label one 2D mask on the host and return labels/counts like ``values``."""

    labels, counts = _label_single(_to_host_connected_components_input(values))
    return _from_numpy(labels, values), _from_numpy(counts, values)


@overload
def connected_components_cpu(
    input_tensor: NumpyArray,
) -> tuple[IntArray, IntArray]: ...


@overload
def connected_components_cpu(input_tensor: object) -> ComponentResult: ...


def connected_components_cpu(input_tensor: object) -> ComponentResult:
    """Label a batch of masks on the host and return labels/counts."""

    arr = _to_host_connected_components_input(input_tensor)
    out_shape = arr.shape
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim != 3:
        raise ValueError("Input tensor must be (B, H, W) or (B, 1, H, W).")

    labels_list: list[IntArray] = []
    counts_list: list[IntArray] = []
    for image in arr:
        labels, counts = _label_single(image)
        labels_list.append(labels)
        counts_list.append(counts)

    labels_tensor = cast(IntArray, np.stack(labels_list, axis=0).reshape(out_shape))
    counts_tensor = cast(IntArray, np.stack(counts_list, axis=0).reshape(out_shape))
    return _from_numpy(labels_tensor, input_tensor), _from_numpy(
        counts_tensor, input_tensor
    )


@overload
def connected_components(input_tensor: NumpyArray) -> tuple[IntArray, IntArray]: ...


@overload
def connected_components(input_tensor: object) -> ComponentResult: ...


def connected_components(input_tensor: object) -> ComponentResult:
    """MLX-compatible wrapper around host connected-components cleanup."""

    arr = _to_host_connected_components_input(input_tensor)
    if arr.ndim == 3:
        arr = arr[:, None]
    if arr.ndim != 4 or arr.shape[1] != 1:
        raise ValueError("Input tensor must be (B, H, W) or (B, 1, H, W).")
    labels, counts = connected_components_cpu(arr)
    return _from_numpy(np.asarray(labels), input_tensor), _from_numpy(
        np.asarray(counts), input_tensor
    )
