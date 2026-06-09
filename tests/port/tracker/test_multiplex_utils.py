import importlib.util
import subprocess

import numpy as np
import mlx.core as mx
import pytest

from sam3_mlx.model.multiplex_utils import MultiplexState
from tests._paths import REPO_ROOT

OFFICIAL_SAM3_MULTIPLEX_UTILS_COMMIT = "2814fa619404a722d03e9a012e083e4f293a4e53"


def _to_numpy(value):
    mx.eval(value)
    return np.asarray(value)


def _load_official_multiplex_utils():
    torch = pytest.importorskip("torch")
    official_checkout = REPO_ROOT / "third_party" / "facebook-sam3"
    official_source = official_checkout / "sam3" / "model" / "multiplex_utils.py"
    if not official_source.exists():
        pytest.skip("official facebookresearch/sam3 checkout is not available")

    commit = subprocess.check_output(
        ["git", "-C", str(official_checkout), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert commit == OFFICIAL_SAM3_MULTIPLEX_UTILS_COMMIT

    spec = importlib.util.spec_from_file_location(
        "_official_sam3_multiplex_utils",
        official_source,
    )
    assert spec is not None and spec.loader is not None
    official_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official_module)
    return official_module, torch


def _copy_assignments(assignments):
    return [bucket.copy() for bucket in assignments]


def test_multiplex_state_mux_and_demux_use_index_permutation_for_mlx_inputs():
    state = MultiplexState(
        [[2, -1, 0], [-1, 1, -1]],
        allowed_bucket_capacity=2,
    )
    values = mx.array(
        [
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
        ],
        dtype=mx.float32,
    )

    muxed = state.mux(values)
    demuxed = state.demux(muxed)

    expected_muxed = np.array(
        [
            [[30.0, 31.0], [0.0, 0.0], [10.0, 11.0]],
            [[0.0, 0.0], [20.0, 21.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(_to_numpy(muxed), expected_muxed)
    np.testing.assert_array_equal(_to_numpy(demuxed), _to_numpy(values))


def test_multiplex_state_numpy_path_matches_mlx_path_and_preserves_padding_zeroes():
    state = MultiplexState(
        [[1, -1], [0, 2]],
        allowed_bucket_capacity=2,
    )
    values_np = np.arange(12, dtype=np.float32).reshape(3, 2, 2)

    muxed_np = state.mux(values_np)
    muxed_mlx = state.mux(mx.array(values_np))
    demuxed_np = state.demux(muxed_np)
    demuxed_mlx = state.demux(muxed_mlx)

    expected_muxed = np.array(
        [
            [values_np[1], np.zeros((2, 2), dtype=np.float32)],
            [values_np[0], values_np[2]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(muxed_np, expected_muxed)
    np.testing.assert_array_equal(_to_numpy(muxed_mlx), expected_muxed)
    np.testing.assert_array_equal(demuxed_np, values_np)
    np.testing.assert_array_equal(_to_numpy(demuxed_mlx), values_np)


def test_multiplex_state_empty_assignments_return_empty_demux_and_zero_mux_slots():
    state = MultiplexState(
        [[-1, -1], [-1, -1]],
        allowed_bucket_capacity=2,
    )
    values = mx.zeros((0, 3), dtype=mx.float32)

    muxed = state.mux(values)
    demuxed = state.demux(muxed)

    np.testing.assert_array_equal(
        _to_numpy(muxed), np.zeros((2, 2, 3), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        _to_numpy(demuxed), np.zeros((0, 3), dtype=np.float32)
    )


def test_multiplex_state_valid_mask_reflects_non_padding_slots():
    state = MultiplexState(
        [[2, -1, 0], [-1, 1, -1]],
        allowed_bucket_capacity=2,
    )

    np.testing.assert_array_equal(
        _to_numpy(state.get_valid_object_mask()),
        np.array([[True, False, True], [False, True, False]]),
    )


def test_multiplex_state_matches_official_torch_transition_matrices():
    official_module, torch = _load_official_multiplex_utils()
    assignments = [[2, -1, 0], [-1, 3, -1], [1, -1, -1]]
    values_np = np.linspace(-3.5, 4.0, num=4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)

    official_state = official_module.MultiplexState(
        _copy_assignments(assignments),
        device=torch.device("cpu"),
        dtype=torch.float32,
        allowed_bucket_capacity=2,
    )
    torch_values = torch.from_numpy(values_np)
    with torch.no_grad():
        expected_muxed = official_state.mux(torch_values).detach().numpy()
        expected_demuxed = (
            official_state.demux(torch.from_numpy(expected_muxed)).detach().numpy()
        )
        expected_valid_mask = official_state.get_valid_object_mask().detach().numpy()

    mlx_state = MultiplexState(
        _copy_assignments(assignments),
        dtype=mx.float32,
        allowed_bucket_capacity=2,
    )
    muxed = mlx_state.mux(mx.array(values_np, dtype=mx.float32))
    demuxed = mlx_state.demux(muxed)

    np.testing.assert_allclose(_to_numpy(muxed), expected_muxed, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(_to_numpy(demuxed), expected_demuxed, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(
        _to_numpy(mlx_state.get_valid_object_mask()), expected_valid_mask
    )


def test_multiplex_state_mutation_matches_official_torch_numeric_parity():
    official_module, torch = _load_official_multiplex_utils()
    assignments = [[0, 2], [1, -1]]
    object_ids = [100, 200, 300]

    official_state = official_module.MultiplexState(
        _copy_assignments(assignments),
        device=torch.device("cpu"),
        dtype=torch.float32,
        allowed_bucket_capacity=2,
        object_ids=object_ids.copy(),
    )
    mlx_state = MultiplexState(
        _copy_assignments(assignments),
        dtype=mx.float32,
        allowed_bucket_capacity=2,
        object_ids=object_ids.copy(),
    )

    assert official_state.remove_objects([1]) == mlx_state.remove_objects([1])
    official_state.add_objects([2], object_ids=[400], allow_new_buckets=True)
    mlx_state.add_objects([2], object_ids=[400], allow_new_buckets=True)

    assert mlx_state.assignments == official_state.assignments
    assert mlx_state.object_ids == official_state.object_ids == [100, 300, 400]

    values_np = np.array(
        [
            [[1.0, -1.0], [2.0, -2.0]],
            [[3.0, -3.0], [4.0, -4.0]],
            [[5.0, -5.0], [6.0, -6.0]],
        ],
        dtype=np.float32,
    )
    with torch.no_grad():
        expected_muxed = (
            official_state.mux(torch.from_numpy(values_np)).detach().numpy()
        )
        expected_demuxed = (
            official_state.demux(torch.from_numpy(expected_muxed)).detach().numpy()
        )

    muxed = mlx_state.mux(mx.array(values_np, dtype=mx.float32))
    demuxed = mlx_state.demux(muxed)

    np.testing.assert_allclose(_to_numpy(muxed), expected_muxed, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(_to_numpy(demuxed), expected_demuxed, rtol=0.0, atol=0.0)
