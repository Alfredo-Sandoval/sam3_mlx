from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, cast

import mlx.core as mx


Shape4D = tuple[int, int, int, int]
LaunchShape = tuple[int, int, int]
KernelTemplate = tuple[str, mx.Dtype]


class _ArrayMetadata(Protocol):
    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> mx.Dtype: ...


class _MetalKernel(Protocol):
    def __call__(
        self,
        *,
        inputs: Sequence[mx.array],
        template: Sequence[KernelTemplate],
        output_shapes: Sequence[Sequence[int]],
        output_dtypes: Sequence[mx.Dtype],
        grid: LaunchShape,
        threadgroup: LaunchShape,
        init_value: int | float | None = None,
    ) -> Sequence[mx.array]: ...


class _MetalKernelFactory(Protocol):
    def __call__(
        self,
        *,
        name: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        source: str,
        header: str = "",
        ensure_row_contiguous: bool = True,
        atomic_outputs: bool = False,
    ) -> _MetalKernel: ...


class _GridSampleFunction(Protocol):
    def __call__(self, x: mx.array, grid: mx.array) -> mx.array: ...

    def vjp(
        self,
        callback: Callable[
            [tuple[mx.array, mx.array], mx.array, object],
            tuple[mx.array, mx.array],
        ],
    ) -> Callable[
        [tuple[mx.array, mx.array], mx.array, object],
        tuple[mx.array, mx.array],
    ]: ...


class _CustomFunctionFactory(Protocol):
    def __call__(
        self, function: Callable[[mx.array, mx.array], mx.array]
    ) -> _GridSampleFunction: ...


_metal_kernel = cast(_MetalKernelFactory, mx.fast.metal_kernel)
_custom_function = cast(_CustomFunctionFactory, mx.custom_function)
_METAL_KERNEL_CACHE: dict[str, _MetalKernel] = {}


def _array_metadata(array: mx.array) -> _ArrayMetadata:
    return cast(_ArrayMetadata, array)


def _shape4(array: mx.array, *, name: str) -> Shape4D:
    metadata = _array_metadata(array)
    assert metadata.ndim == 4, f"`{name}` must be 4D."
    shape = metadata.shape
    assert len(shape) == 4, f"`{name}` must be 4D."
    return shape


def _dtype(array: mx.array) -> mx.Dtype:
    return _array_metadata(array).dtype


def _cached_metal_kernel(
    cache_key: str,
    *,
    name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    source: str,
    header: str = "",
    ensure_row_contiguous: bool = True,
    atomic_outputs: bool = False,
) -> _MetalKernel:
    kernel = _METAL_KERNEL_CACHE.get(cache_key)
    if kernel is None:
        kernel = _metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
            header=header,
            ensure_row_contiguous=ensure_row_contiguous,
            atomic_outputs=atomic_outputs,
        )
        _METAL_KERNEL_CACHE[cache_key] = kernel
    return kernel


def _grid_sample_impl(x: mx.array, grid: mx.array) -> mx.array:
    """Grid sample that matches torch.nn.functional.grid_sample with default arguments."""

    batch, _, _, channels = _shape4(x, name="x")
    _, grid_height, grid_width, coords = _shape4(grid, name="grid")
    out_shape: Shape4D = (batch, grid_height, grid_width, channels)
    out_size = batch * grid_height * grid_width * channels

    assert coords == 2, "Last dim of `grid` must be size 2."

    source = """
        uint elem = thread_position_in_grid.x;
        int H = x_shape[1];
        int W = x_shape[2];
        int C = x_shape[3];
        int w_stride = C;
        int h_stride = W * w_stride;
        int b_stride = H * h_stride;
        int gH = grid_shape[1];
        int gW = grid_shape[2];
        uint grid_idx = elem / C * 2;
        float ix = ((grid[grid_idx] + 1) * W - 1) / 2;
        float iy = ((grid[grid_idx + 1] + 1) * H - 1) / 2;
        int ix_nw = floor(ix);
        int iy_nw = floor(iy);
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;
        T nw = (ix_se - ix)    * (iy_se - iy);
        T ne = (ix    - ix_sw) * (iy_sw - iy);
        T sw = (ix_ne - ix)    * (iy    - iy_ne);
        T se = (ix    - ix_nw) * (iy    - iy_nw);
        int batch_idx = elem / C / gH / gW * b_stride;
        int channel_idx = elem % C;
        int base_idx = batch_idx + channel_idx;
        bool nw_in_bounds = iy_nw >= 0 && iy_nw < H && ix_nw >= 0 && ix_nw < W;
        bool ne_in_bounds = iy_ne >= 0 && iy_ne < H && ix_ne >= 0 && ix_ne < W;
        bool sw_in_bounds = iy_sw >= 0 && iy_sw < H && ix_sw >= 0 && ix_sw < W;
        bool se_in_bounds = iy_se >= 0 && iy_se < H && ix_se >= 0 && ix_se < W;
        T I_nw = T(0);
        T I_ne = T(0);
        T I_sw = T(0);
        T I_se = T(0);
        if (nw_in_bounds) {
            I_nw = x[base_idx + iy_nw * h_stride + ix_nw * w_stride];
        }
        if (ne_in_bounds) {
            I_ne = x[base_idx + iy_ne * h_stride + ix_ne * w_stride];
        }
        if (sw_in_bounds) {
            I_sw = x[base_idx + iy_sw * h_stride + ix_sw * w_stride];
        }
        if (se_in_bounds) {
            I_se = x[base_idx + iy_se * h_stride + ix_se * w_stride];
        }
        out[elem] = nw * I_nw + ne * I_ne + sw * I_sw + se * I_se;
    """
    kernel = _cached_metal_kernel(
        "grid_sample",
        name="grid_sample",
        input_names=("x", "grid"),
        output_names=("out",),
        source=source,
    )
    x_dtype = _dtype(x)
    outputs = kernel(
        inputs=[x, grid],
        template=[("T", x_dtype)],
        output_shapes=[out_shape],
        output_dtypes=[x_dtype],
        grid=(out_size, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return outputs[0]


grid_sample = _custom_function(_grid_sample_impl)


@grid_sample.vjp
def grid_sample_vjp(
    primals: tuple[mx.array, mx.array], cotangent: mx.array, _: object
) -> tuple[mx.array, mx.array]:
    x, grid = primals
    batch, _, _, channels = _shape4(x, name="x")
    _, grid_height, grid_width, coords = _shape4(grid, name="grid")

    assert coords == 2, "Last dim of `grid` must be size 2."

    source = """
        uint elem = thread_position_in_grid.x;
        int H = x_shape[1];
        int W = x_shape[2];
        int C = x_shape[3];
        int C_padded = ceildiv(C, threads_per_simdgroup) * threads_per_simdgroup;
        int w_stride = C;
        int h_stride = W * w_stride;
        int b_stride = H * h_stride;
        int gH = grid_shape[1];
        int gW = grid_shape[2];
        uint grid_idx = elem / C_padded * 2;
        float ix = ((grid[grid_idx] + 1) * W - 1) / 2;
        float iy = ((grid[grid_idx + 1] + 1) * H - 1) / 2;
        int ix_nw = floor(ix);
        int iy_nw = floor(iy);
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;
        T nw = (ix_se - ix)    * (iy_se - iy);
        T ne = (ix    - ix_sw) * (iy_sw - iy);
        T sw = (ix_ne - ix)    * (iy    - iy_ne);
        T se = (ix    - ix_nw) * (iy    - iy_nw);
        int batch_idx = elem / C_padded / gH / gW * b_stride;
        int channel_idx = elem % C_padded;
        int base_idx = batch_idx + channel_idx;
        T gix = T(0);
        T giy = T(0);
        if (channel_idx < C) {
            int cot_index = elem / C_padded * C + channel_idx;
            T cot = cotangent[cot_index];
            if (iy_nw >= 0 && iy_nw <= H - 1 && ix_nw >= 0 && ix_nw <= W - 1) {
                int offset = base_idx + iy_nw * h_stride + ix_nw * w_stride;
                atomic_fetch_add_explicit(&x_grad[offset], nw * cot, memory_order_relaxed);
                T I_nw = x[offset];
                gix -= I_nw * (iy_se - iy) * cot;
                giy -= I_nw * (ix_se - ix) * cot;
            }
            if (iy_ne >= 0 && iy_ne <= H - 1 && ix_ne >= 0 && ix_ne <= W - 1) {
                int offset = base_idx + iy_ne * h_stride + ix_ne * w_stride;
                atomic_fetch_add_explicit(&x_grad[offset], ne * cot, memory_order_relaxed);
                T I_ne = x[offset];
                gix += I_ne * (iy_sw - iy) * cot;
                giy -= I_ne * (ix - ix_sw) * cot;
            }
            if (iy_sw >= 0 && iy_sw <= H - 1 && ix_sw >= 0 && ix_sw <= W - 1) {
                int offset = base_idx + iy_sw * h_stride + ix_sw * w_stride;
                atomic_fetch_add_explicit(&x_grad[offset], sw * cot, memory_order_relaxed);
                T I_sw = x[offset];
                gix -= I_sw * (iy - iy_ne) * cot;
                giy += I_sw * (ix_ne - ix) * cot;
            }
            if (iy_se >= 0 && iy_se <= H - 1 && ix_se >= 0 && ix_se <= W - 1) {
                int offset = base_idx + iy_se * h_stride + ix_se * w_stride;
                atomic_fetch_add_explicit(&x_grad[offset], se * cot, memory_order_relaxed);
                T I_se = x[offset];
                gix += I_se * (iy - iy_nw) * cot;
                giy += I_se * (ix - ix_nw) * cot;
            }
        }
        T gix_mult = W / 2;
        T giy_mult = H / 2;
        gix = simd_sum(gix);
        giy = simd_sum(giy);
        if (thread_index_in_simdgroup == 0) {
            atomic_fetch_add_explicit(&grid_grad[grid_idx], gix * gix_mult, memory_order_relaxed);
            atomic_fetch_add_explicit(&grid_grad[grid_idx + 1], giy * giy_mult, memory_order_relaxed);
        }
    """
    kernel = _cached_metal_kernel(
        "grid_sample_grad",
        name="grid_sample_grad",
        input_names=("x", "grid", "cotangent"),
        output_names=("x_grad", "grid_grad"),
        source=source,
        atomic_outputs=True,
    )
    # pad output channels to simd group size
    simdgroup_size = 32
    padded_channels = (channels + simdgroup_size - 1) // simdgroup_size * simdgroup_size
    grid_size = batch * grid_height * grid_width * padded_channels
    x_dtype = _dtype(x)
    outputs = kernel(
        inputs=[x, grid, cotangent],
        template=[("T", x_dtype)],
        output_shapes=[x.shape, grid.shape],
        output_dtypes=[x_dtype, x_dtype],
        grid=(grid_size, 1, 1),
        threadgroup=(256, 1, 1),
        init_value=0,
    )
    return outputs[0], outputs[1]
