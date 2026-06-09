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
