# pyright: reportPrivateUsage=false

from __future__ import annotations

import mlx.core as mx
import numpy as np
import numpy.typing as npt
import pytest

import sam3_mlx.perflib.associate_det_trk as association
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.perflib.connected_components import connected_components
from sam3_mlx.perflib.fused import addmm_act


IndexArray = npt.NDArray[np.intp]


def test_associate_det_trk_returns_typed_matches_and_new_detections(
    monkeypatch: pytest.MonkeyPatch,
):
    def diagonal_assignment(
        cost_matrix: npt.NDArray[np.float32],
    ) -> tuple[IndexArray, IndexArray]:
        size = min(cost_matrix.shape)
        indices = np.arange(size, dtype=np.intp)
        return indices, indices

    monkeypatch.setattr(association, "_linear_sum_assignment", diagonal_assignment)
    det_masks = np.array(
        [
            [[True, False], [False, False]],
            [[False, True], [False, False]],
        ]
    )
    track_masks = np.array(
        [
            [[True, False], [False, False]],
            [[False, False], [True, False]],
        ]
    )

    new_det, unmatched_trk, det_to_trk, matched_scores = association.associate_det_trk(
        det_masks,
        track_masks,
        det_scores=np.array([0.9, 0.8], dtype=np.float32),
    )

    assert new_det == [1]
    assert unmatched_trk == [1]
    assert det_to_trk == {0: [0]}
    np.testing.assert_allclose(matched_scores[0], [0.9, 0.9], atol=1e-6)


def test_associate_det_trk_empty_inputs_do_not_require_scipy():
    result = association.associate_det_trk(
        np.zeros((0, 2, 2), dtype=bool),
        np.zeros((1, 2, 2), dtype=bool),
    )

    assert result == ([], [], {}, {})


def test_connected_components_numpy_preserves_shape_and_eight_connectivity():
    masks = np.array(
        [[[[True, False], [False, True]]]],
        dtype=bool,
    )

    labels, counts = connected_components(masks)

    assert labels.shape == masks.shape
    assert counts.shape == masks.shape
    np.testing.assert_array_equal(counts[0, 0], [[2, 0], [0, 2]])
    assert labels[0, 0, 0, 0] == labels[0, 0, 1, 1] == 1


def test_addmm_act_dispatches_named_relu_and_gelu_without_changing_shape():
    def identity(x: mx.array) -> mx.array:
        return x

    def relu(x: mx.array) -> mx.array:
        return x

    def gelu(x: mx.array) -> mx.array:
        return x

    values = mx.array([[-1.0, 0.0, 1.0]], dtype=mx.float32)
    relu_output = addmm_act(relu, identity, values)
    gelu_output = addmm_act(gelu, identity, values)

    np.testing.assert_allclose(to_numpy(relu_output), [[0.0, 0.0, 1.0]])
    assert gelu_output.shape == values.shape
    assert np.isfinite(to_numpy(gelu_output)).all()

    with pytest.raises(ValueError, match="Unexpected activation"):
        addmm_act(identity, identity, values)
