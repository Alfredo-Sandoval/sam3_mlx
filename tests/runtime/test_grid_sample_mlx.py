import numpy as np
import mlx.core as mx

from sam3_mlx.model.grid_sample_mlx import grid_sample


def _grid_sample_numpy_bilinear_zero_nhwc(
    x: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Independent align_corners=False bilinear reference with zero padding."""
    batch, height, width, channels = x.shape
    grid_batch, grid_height, grid_width, coords = grid.shape
    assert grid_batch == batch
    assert coords == 2

    out = np.zeros((batch, grid_height, grid_width, channels), dtype=x.dtype)
    for b in range(batch):
        for gy in range(grid_height):
            for gx in range(grid_width):
                norm_x, norm_y = grid[b, gy, gx]
                src_x = ((norm_x + 1.0) * width - 1.0) / 2.0
                src_y = ((norm_y + 1.0) * height - 1.0) / 2.0
                x0 = int(np.floor(src_x))
                y0 = int(np.floor(src_y))

                for yy in (y0, y0 + 1):
                    if yy < 0 or yy >= height:
                        continue
                    y_weight = max(0.0, 1.0 - abs(src_y - yy))
                    for xx in (x0, x0 + 1):
                        if xx < 0 or xx >= width:
                            continue
                        x_weight = max(0.0, 1.0 - abs(src_x - xx))
                        out[b, gy, gx] += x[b, yy, xx] * x_weight * y_weight
    return out


def _mlx_to_numpy(array: mx.array) -> np.ndarray:
    mx.eval(array)
    return np.asarray(array)


def test_grid_sample_forward_matches_numpy_reference_at_normalized_border_edges():
    x_np = np.array(
        [
            [
                [[1.0], [2.0], [3.0]],
                [[4.0], [5.0], [6.0]],
                [[7.0], [8.0], [9.0]],
            ]
        ],
        dtype=np.float32,
    )
    grid_np = np.array(
        [
            [
                [[-1.0, -1.0], [1.0, -1.0]],
                [[-1.0, 1.0], [1.0, 1.0]],
            ]
        ],
        dtype=np.float32,
    )

    actual = grid_sample(mx.array(x_np), mx.array(grid_np))
    expected = _grid_sample_numpy_bilinear_zero_nhwc(x_np, grid_np)

    np.testing.assert_allclose(_mlx_to_numpy(actual), expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        expected,
        np.array([[[[0.25], [0.75]], [[1.75], [2.25]]]], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_grid_sample_forward_matches_numpy_reference_for_center_interpolation():
    x_np = np.array(
        [
            [
                [[0.0], [2.0]],
                [[4.0], [6.0]],
            ]
        ],
        dtype=np.float32,
    )
    grid_np = np.array([[[[0.0, 0.0]]]], dtype=np.float32)

    actual = grid_sample(mx.array(x_np), mx.array(grid_np))
    expected = _grid_sample_numpy_bilinear_zero_nhwc(x_np, grid_np)

    np.testing.assert_allclose(_mlx_to_numpy(actual), expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(expected, np.array([[[[3.0]]]], dtype=np.float32))


def test_grid_sample_forward_matches_numpy_reference_for_out_of_bounds_zero_padding():
    x_np = np.array(
        [
            [
                [[16.0], [2.0]],
                [[4.0], [8.0]],
            ]
        ],
        dtype=np.float32,
    )
    grid_np = np.array(
        [
            [
                [[-1.25, -1.25], [-2.0, -2.0]],
                [[1.25, 1.25], [2.0, 2.0]],
            ]
        ],
        dtype=np.float32,
    )

    actual = grid_sample(mx.array(x_np), mx.array(grid_np))
    expected = _grid_sample_numpy_bilinear_zero_nhwc(x_np, grid_np)

    np.testing.assert_allclose(_mlx_to_numpy(actual), expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        expected,
        np.array([[[[1.0], [0.0]], [[0.5], [0.0]]]], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_grid_sample_forward_matches_numpy_for_batched_non_square_multi_channel():
    x_np = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3) / 10.0
    grid_np = np.array(
        [
            [
                [[-0.75, -0.25], [0.0, 0.0], [0.8, 0.6]],
                [[-1.2, 0.9], [0.25, -0.5], [1.1, -1.1]],
            ],
            [
                [[-0.5, -0.5], [0.5, -0.5], [0.0, 0.8]],
                [[-0.2, 0.2], [0.9, 0.1], [-0.9, -0.9]],
            ],
        ],
        dtype=np.float32,
    )

    actual = grid_sample(mx.array(x_np), mx.array(grid_np))
    expected = _grid_sample_numpy_bilinear_zero_nhwc(x_np, grid_np)

    np.testing.assert_allclose(_mlx_to_numpy(actual), expected, rtol=0, atol=1e-5)


def test_grid_sample_forward_matches_numpy_for_simdgroup_channel_boundaries():
    grid_np = np.array(
        [[[[0.0, 0.0], [-0.4, 0.7]], [[1.0, -1.0], [-1.3, 1.2]]]],
        dtype=np.float32,
    )

    for channels in (32, 33):
        x_np = (
            np.arange(1 * 3 * 4 * channels, dtype=np.float32).reshape(1, 3, 4, channels)
            / 100.0
        )

        actual = grid_sample(mx.array(x_np), mx.array(grid_np))
        expected = _grid_sample_numpy_bilinear_zero_nhwc(x_np, grid_np)

        np.testing.assert_allclose(
            _mlx_to_numpy(actual),
            expected,
            rtol=0,
            atol=1e-5,
        )


def test_grid_sample_forward_accepts_transposed_non_contiguous_nhwc_input():
    nchw_np = np.arange(1 * 3 * 3 * 5, dtype=np.float32).reshape(1, 3, 3, 5)
    x_np = np.transpose(nchw_np, (0, 2, 3, 1))
    x = mx.array(nchw_np).transpose(0, 2, 3, 1)
    grid_np = np.array(
        [[[[0.0, 0.0], [0.4, -0.4]], [[-0.8, 0.8], [1.2, 0.0]]]],
        dtype=np.float32,
    )

    actual = grid_sample(x, mx.array(grid_np))
    expected = _grid_sample_numpy_bilinear_zero_nhwc(x_np, grid_np)

    np.testing.assert_allclose(_mlx_to_numpy(actual), expected, rtol=0, atol=1e-5)


def test_grid_sample_vjp_at_center_matches_hand_derived_bilinear_gradients():
    x_np = np.array(
        [
            [
                [[0.0], [1.0]],
                [[2.0], [3.0]],
            ]
        ],
        dtype=np.float32,
    )
    grid_np = np.array([[[[0.0, 0.0]]]], dtype=np.float32)
    x = mx.array(x_np)
    grid = mx.array(grid_np)
    cotangent = mx.ones((1, 1, 1, 1), dtype=mx.float32)

    outputs, vjps = mx.vjp(grid_sample, [x, grid], [cotangent])
    mx.eval(outputs[0], vjps[0], vjps[1])

    np.testing.assert_allclose(
        np.asarray(outputs[0]),
        np.array([[[[1.5]]]], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(vjps[0]),
        np.full((1, 2, 2, 1), 0.25, dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(vjps[1]),
        np.array([[[[1.0, 2.0]]]], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_grid_sample_vjp_handles_channel_padding_above_simdgroup_size():
    spatial = np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)
    channel_offsets = np.arange(33, dtype=np.float32).reshape(1, 1, 1, 33)
    x_np = spatial[..., None] + channel_offsets
    x = mx.array(x_np)
    grid = mx.array([[[[0.0, 0.0]]]], dtype=mx.float32)
    cotangent = mx.ones((1, 1, 1, 33), dtype=mx.float32)

    outputs, vjps = mx.vjp(grid_sample, [x, grid], [cotangent])
    mx.eval(outputs[0], vjps[0], vjps[1])

    expected_output = np.full((1, 1, 1, 33), 1.5, dtype=np.float32)
    expected_output += np.arange(33, dtype=np.float32).reshape(1, 1, 1, 33)
    np.testing.assert_allclose(
        np.asarray(outputs[0]),
        expected_output,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(vjps[0]),
        np.full((1, 2, 2, 33), 0.25, dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(vjps[1]),
        np.array([[[[33.0, 66.0]]]], dtype=np.float32),
        rtol=0,
        atol=1e-5,
    )
