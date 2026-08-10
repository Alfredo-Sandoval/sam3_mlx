from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

import mlx.core as mx


Shape2D = tuple[int, int]
Shape4D = tuple[int, int, int, int]
BoxesList = list[mx.array] | tuple[mx.array, ...]
BoxesArg = mx.array | BoxesList


class _ArrayMethods(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> mx.Dtype: ...

    def astype(self, dtype: mx.Dtype) -> mx.array: ...

    def sum(
        self, axis: int | Sequence[int] | None = None, keepdims: bool = False
    ) -> mx.array: ...


def _array_methods(array: mx.array) -> _ArrayMethods:
    return cast(_ArrayMethods, array)


def _shape2(array: mx.array, *, name: str) -> Shape2D:
    shape = _array_methods(array).shape
    assert len(shape) == 2, f"`{name}` must be 2D."
    return shape


def _shape4(array: mx.array, *, name: str) -> Shape4D:
    shape = _array_methods(array).shape
    assert len(shape) == 4, f"`{name}` must be 4D."
    return shape


def _dtype(array: mx.array) -> mx.Dtype:
    return _array_methods(array).dtype


def _astype(array: mx.array, dtype: mx.Dtype) -> mx.array:
    return _array_methods(array).astype(dtype)


def _sum_last_two_axes(array: mx.array) -> mx.array:
    return _array_methods(array).sum(axis=(-1, -2))


# NB: all inputs are mx.array objects
def _bilinear_interpolate(
    input: mx.array,  # [N, C, H, W]
    roi_batch_ind: mx.array,  # [K]
    y: mx.array,  # [K, PH, IY]
    x: mx.array,  # [K, PW, IX]
    ymask: mx.array | None,  # [K, IY]
    xmask: mx.array | None,  # [K, IX]
) -> mx.array:
    _, channels, height, width = _shape4(input, name="input")
    input_dtype = _dtype(input)
    # deal with inverse element out of feature map boundary
    y = mx.clip(y, a_min=0, a_max=None)
    x = mx.clip(x, a_min=0, a_max=None)

    y_low = y.astype(mx.int32)
    x_low = x.astype(mx.int32)
    y_high = mx.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = mx.where(y_low >= height - 1, height - 1, y_low)
    y = mx.where(y_low >= height - 1, _astype(y, input_dtype), y)

    x_high = mx.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = mx.where(x_low >= width - 1, width - 1, x_low)
    x = mx.where(x_low >= width - 1, _astype(x, input_dtype), x)

    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx

    ly = y - y_low

    # Respect adaptive-sampling masks before indexing padded interpolation bins.
    def masked_index(
        y_indices: mx.array,  # [K, PH, IY]
        x_indices: mx.array,  # [K, PW, IX]
    ) -> mx.array:
        if ymask is not None:
            assert xmask is not None
            y_indices = mx.where(ymask[:, None, :], y_indices, 0)
            x_indices = mx.where(xmask[:, None, :], x_indices, 0)
        return input[
            roi_batch_ind[:, None, None, None, None, None],
            mx.arange(channels)[None, :, None, None, None, None],
            y_indices[:, None, :, None, :, None],  # prev [K, PH, IY]
            x_indices[:, None, None, :, None, :],  # prev [K, PW, IX]
        ]  # [K, C, PH, PW, IY, IX]

    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)

    # all ws preemptively [K, C, PH, PW, IY, IX]
    def outer_prod(y_weights: mx.array, x_weights: mx.array) -> mx.array:
        return (
            y_weights[:, None, :, None, :, None] * x_weights[:, None, None, :, None, :]
        )

    w1 = outer_prod(hy, hx)
    w2 = outer_prod(hy, lx)
    w3 = outer_prod(ly, hx)
    w4 = outer_prod(ly, lx)

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val


def convert_boxes_to_roi_format(boxes: BoxesList) -> mx.array:
    concat_boxes = mx.concat(list(boxes), axis=0)
    batch_ids: list[mx.array] = []
    for i, b in enumerate(boxes):
        batch_ids.append(mx.full(b[:, :1].shape, i))
    ids = mx.concat(batch_ids, axis=0)
    rois = mx.concat([ids, concat_boxes], axis=1)
    return rois


def check_roi_boxes_shape(boxes: BoxesArg) -> None:
    if isinstance(boxes, (list, tuple)):
        for _array in boxes:
            assert _shape2(_array, name="boxes element")[1] == 4, (
                "The shape of the array in the boxes list is not correct as List[mx.array of shape (L, 4)]"
            )
    else:
        assert _shape2(boxes, name="boxes")[1] == 5, (
            "The boxes array shape is not correct as mx.array of shape (K, 5)"
        )


def _roi_align(
    input: mx.array,
    rois: mx.array,
    spatial_scale: float,
    pooled_height: int,
    pooled_width: int,
    sampling_ratio: int,
    aligned: bool,
) -> mx.array:
    orig_dtype = _dtype(input)

    _, _, height, width = _shape4(input, name="input")
    assert _shape2(rois, name="rois")[1] == 5, "`rois` must have shape (K, 5)."

    ph = mx.arange(pooled_height)
    pw = mx.arange(pooled_width)

    # inputs: [N, C, H, W]
    # rois: [K, 5]

    roi_batch_ind = _astype(rois[:, 0], mx.int32)
    offset = 0.5 if aligned else 0.0
    roi_start_w = rois[:, 1] * spatial_scale - offset  # [K]
    roi_start_h = rois[:, 2] * spatial_scale - offset  # [K]
    roi_end_w = rois[:, 3] * spatial_scale - offset  # [K]
    roi_end_h = rois[:, 4] * spatial_scale - offset  # [K]

    roi_width = roi_end_w - roi_start_w  # [K]
    roi_height = roi_end_h - roi_start_h  # [K]
    if not aligned:
        roi_width = mx.clip(roi_width, a_min=1.0, a_max=None)
        roi_height = mx.clip(roi_height, a_min=1.0, a_max=None)

    bin_size_h = roi_height / pooled_height  # [K]
    bin_size_w = roi_width / pooled_width  # [K]

    exact_sampling = sampling_ratio > 0

    count: int | mx.array
    if exact_sampling:
        count = max(sampling_ratio * sampling_ratio, 1)
        iy = mx.arange(sampling_ratio)
        ix = mx.arange(sampling_ratio)
        ymask: mx.array | None = None
        xmask: mx.array | None = None
        bin_step_h = bin_size_h / sampling_ratio
        bin_step_w = bin_size_w / sampling_ratio
    else:
        roi_bin_grid_h = mx.ceil(roi_height / pooled_height)
        roi_bin_grid_w = mx.ceil(roi_width / pooled_width)
        count = mx.clip(roi_bin_grid_h * roi_bin_grid_w, a_min=1, a_max=None)
        iy = mx.arange(height)
        ix = mx.arange(width)
        ymask = iy[None, :] < roi_bin_grid_h[:, None]
        xmask = ix[None, :] < roi_bin_grid_w[:, None]
        bin_step_h = bin_size_h / roi_bin_grid_h
        bin_step_w = bin_size_w / roi_bin_grid_w

    def from_K(t: mx.array) -> mx.array:
        return t[:, None, None]

    y = (
        from_K(roi_start_h)
        + ph[None, :, None] * from_K(bin_size_h)
        + _astype(iy[None, None, :] + 0.5, orig_dtype) * from_K(bin_step_h)
    )  # [K, PH, IY]
    x = (
        from_K(roi_start_w)
        + pw[None, :, None] * from_K(bin_size_w)
        + _astype(ix[None, None, :] + 0.5, orig_dtype) * from_K(bin_step_w)
    )  # [K, PW, IX]
    val = _bilinear_interpolate(
        input, roi_batch_ind, y, x, ymask, xmask
    )  # [K, C, PH, PW, IY, IX]

    if not exact_sampling:
        assert ymask is not None
        assert xmask is not None
        val = mx.where(ymask[:, None, None, None, :, None], val, 0)
        val = mx.where(xmask[:, None, None, None, None, :], val, 0)

    output = _sum_last_two_axes(val)  # remove IY, IX ~> [K, C, PH, PW]
    if isinstance(count, mx.array):
        output /= count[:, None, None, None]
    else:
        output /= count

    output = output.astype(orig_dtype)

    return output


def roi_align(
    input: mx.array,
    boxes: BoxesArg,
    height: int,
    width: int,
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> mx.array:
    check_roi_boxes_shape(boxes)
    rois = (
        convert_boxes_to_roi_format(boxes)
        if isinstance(boxes, (list, tuple))
        else boxes
    )
    return _roi_align(
        input, rois, spatial_scale, height, width, sampling_ratio, aligned
    )
