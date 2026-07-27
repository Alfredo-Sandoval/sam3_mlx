import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.model.data_misc import interpolate


def _to_numpy(value):
    mx.eval(value)
    return np.asarray(value)


def test_interpolate_antialias_bilinear_matches_torch_downsample():
    torch = pytest.importorskip("torch")
    input_np = np.linspace(
        -2.0,
        3.0,
        num=2 * 3 * 5 * 4,
        dtype=np.float32,
    ).reshape(2, 3, 5, 4)

    expected = torch.nn.functional.interpolate(
        torch.from_numpy(input_np),
        size=(3, 2),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).numpy()
    observed = interpolate(
        mx.array(input_np, dtype=mx.float32),
        size=(3, 2),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    np.testing.assert_allclose(_to_numpy(observed), expected, rtol=0.0, atol=5e-7)


def test_interpolate_antialias_bilinear_rejects_singleton_downsample_grid():
    with pytest.raises(ValueError, match="non-singleton output grids"):
        interpolate(
            mx.zeros((1, 1, 4, 4), dtype=mx.float32),
            size=(2, 1),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )


def test_interpolate_bicubic_does_not_silently_become_linear():
    # A low-frequency ramp is similar under both kernels; a peaked pattern is not.
    coords = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    peak = np.exp(-8.0 * (xx * xx + yy * yy)).astype(np.float32)
    input_np = peak.reshape(1, 1, 8, 8)
    input_mx = mx.array(input_np, dtype=mx.float32)

    bilinear = _to_numpy(
        interpolate(input_mx, size=(16, 16), mode="bilinear", align_corners=False)
    )
    bicubic = _to_numpy(
        interpolate(input_mx, size=(16, 16), mode="bicubic", align_corners=False)
    )

    assert bilinear.shape == (1, 1, 16, 16)
    assert bicubic.shape == (1, 1, 16, 16)
    assert not np.allclose(bilinear, bicubic, rtol=0.0, atol=1e-5)


def test_interpolate_bicubic_matches_literal_upstream_reference():
    input_mx = mx.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=mx.float32)
    expected = np.array(
        [
            [
                [
                    [0.68359375, 1.015625, 1.5625, 1.89453125],
                    [1.34765625, 1.6796875, 2.2265625, 2.55859375],
                    [2.44140625, 2.7734375, 3.3203125, 3.65234375],
                    [3.10546875, 3.4375, 3.984375, 4.31640625],
                ]
            ]
        ],
        dtype=np.float32,
    )
    observed = interpolate(
        input_mx,
        size=(4, 4),
        mode="bicubic",
        align_corners=False,
    )
    np.testing.assert_allclose(_to_numpy(observed), expected, rtol=0.0, atol=1e-7)


def test_interpolate_empty_accepts_int_size_and_tuple_scale_factor():
    empty = mx.zeros((2, 3, 0, 5), dtype=mx.float32)

    by_int_size = interpolate(empty, size=8, mode="nearest")
    by_tuple_scale = interpolate(empty, scale_factor=(2.0, 3.0), mode="nearest")

    assert tuple(by_int_size.shape) == (2, 3, 8, 8)
    assert tuple(by_tuple_scale.shape) == (2, 3, 0, 15)
    assert by_int_size.dtype == mx.float32
    assert by_tuple_scale.dtype == mx.float32


def test_interpolate_empty_size_int_on_empty_channel_dim():
    empty_channels = mx.zeros((1, 0, 4, 5), dtype=mx.float32)
    out = interpolate(empty_channels, size=8, mode="nearest")
    assert tuple(out.shape) == (1, 0, 8, 8)


def test_interpolate_rejects_empty_batch_and_channel_dims():
    with pytest.raises(ValueError, match="both empty batch and channel"):
        interpolate(
            mx.zeros((0, 0, 4, 5), dtype=mx.float32),
            size=8,
            mode="nearest",
        )


def test_interpolate_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported interpolate mode"):
        interpolate(mx.zeros((1, 1, 2, 2), dtype=mx.float32), size=4, mode="lanczos")
