import mlx.core as mx


@mx.custom_function
def grid_sample(x, grid):
    """Grid sample that matches torch.nn.functional.grid_sample with default arguments."""
    
    assert x.ndim == 4, "`x` must be 4D."
    assert grid.ndim == 4, "`grid` must be 4D."

    B, _, _, C = x.shape
    _, gN, gM, D = grid.shape
    out_shape = (B, gN, gM, C)

    assert D == 2, "Last dim of `grid` must be size 2."

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
        T I_nw = x[base_idx + iy_nw * h_stride + ix_nw * w_stride];
        T I_ne = x[base_idx + iy_ne * h_stride + ix_ne * w_stride];
        T I_sw = x[base_idx + iy_sw * h_stride + ix_sw * w_stride];
        T I_se = x[base_idx + iy_se * h_stride + ix_se * w_stride];
        I_nw = iy_nw >= 0 && iy_nw <= H - 1 && ix_nw >= 0 && ix_nw <= W - 1 ? I_nw : 0;
        I_ne = iy_ne >= 0 && iy_ne <= H - 1 && ix_ne >= 0 && ix_ne <= W - 1 ? I_ne : 0;
        I_sw = iy_sw >= 0 && iy_sw <= H - 1 && ix_sw >= 0 && ix_sw <= W - 1 ? I_sw : 0;
        I_se = iy_se >= 0 && iy_se <= H - 1 && ix_se >= 0 && ix_se <= W - 1 ? I_se : 0;
        out[elem] = nw * I_nw + ne * I_ne + sw * I_sw + se * I_se;
    """
    kernel = mx.fast.metal_kernel(
        name="grid_sample",
        input_names=["x", "grid"],
        output_names=["out"],
        source=source,
    )
    outputs = kernel(
        inputs=[x, grid],
        template=[("T", x.dtype)],
        output_shapes=[out_shape],
        output_dtypes=[x.dtype],
        grid=(mx.prod(mx.array(out_shape)), 1, 1),
        threadgroup=(256, 1, 1),
    )
    return outputs[0]


@grid_sample.vjp
def grid_sample_vjp(primals, cotangent, _):
    x, grid = primals
    B, _, _, C = x.shape
    _, gN, gM, D = grid.shape

    assert D == 2, "Last dim of `grid` must be size 2."

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
    kernel = mx.fast.metal_kernel(
        name="grid_sample_grad",
        input_names=["x", "grid", "cotangent"],
        output_names=["x_grad", "grid_grad"],
        source=source,
        atomic_outputs=True,
    )
    # pad output channels to simd group size
    simdgroup_size = 32
    C_padded = (C + simdgroup_size - 1) // simdgroup_size * simdgroup_size
    grid_size = B * gN * gM * C_padded
    outputs = kernel(
        inputs=[x, grid, cotangent],
        template=[("T", x.dtype)],
        output_shapes=[x.shape, grid.shape],
        output_dtypes=[x.dtype, x.dtype],
        grid=(grid_size, 1, 1),
        threadgroup=(256, 1, 1),
        init_value=0,
    )
    return outputs[0], outputs[1]


# mx.random.seed(7)
# n, m, gn, gm = 1024, 1024, 256, 256
# b, c = 8, 64
# x = mx.random.normal(shape=(b, n, m, c))
# grid = mx.random.uniform(-1.5, 1, shape=(b, gn, gm, 2))

# output = grid_sample(x, grid)

# cotangent = mx.random.normal(shape=output.shape)
# output, (x_grad, grid_grad) = mx.vjp(grid_sample, [x, grid], [cotangent])