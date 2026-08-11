import json

import numpy as np
import pytest

from sam3_mlx.parity_evidence import (
    compare_case,
    load_evidence_bundle,
    optimal_assignment,
    write_evidence_bundle,
)
from sam3_mlx.release_contract import COMPARISON_ALGORITHM


def test_hungarian_assignment_avoids_greedy_suboptimal_pairing():
    scores = np.array(
        [
            [0.90, 0.80],
            [0.85, 0.10],
        ]
    )

    assignment = optimal_assignment(scores)

    assert assignment == [(0, 1), (1, 0)]
    assert sum(scores[row, column] for row, column in assignment) == pytest.approx(1.65)


def test_hungarian_assignment_is_deterministic_for_ties():
    scores = np.ones((3, 3), dtype=np.float64)

    assert optimal_assignment(scores) == [(0, 0), (1, 1), (2, 2)]


def test_hungarian_assignment_rejects_invalid_matrices():
    with pytest.raises(ValueError, match="square"):
        optimal_assignment(np.ones((2, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        optimal_assignment(np.array([[1.0, np.nan], [0.0, 1.0]]))


def _outputs(order=(0, 1)):
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, :2, :2] = True
    masks[1, 2:, 2:] = True
    boxes = np.array([[0, 0, 2, 2], [2, 2, 4, 4]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    index = np.asarray(order)
    return {
        "masks": masks[index],
        "boxes": boxes[index],
        "scores": scores[index],
    }


def test_compare_case_matches_permuted_objects_by_mask_iou():
    case = compare_case(
        {
            "name": "synthetic",
            "resolution": 14,
            "prompt": "object",
            "geometric_prompts": [],
        },
        _outputs(),
        _outputs(order=(1, 0)),
    )

    assert case["status"] == "passed"
    assert case["mask_iou_min"] == 1.0
    assert case["mask_iou_mean"] == 1.0
    assert [
        (item["official_index"], item["mlx_index"]) for item in case["matches"]
    ] == [
        (0, 1),
        (1, 0),
    ]


def test_raw_evidence_bundle_round_trips_without_pickle(tmp_path):
    evidence_path = tmp_path / "evidence.npz"
    metadata = {
        "profile": "synthetic",
        "case_specs": [
            {
                "name": "synthetic",
                "resolution": 14,
                "prompt": "object",
                "geometric_prompts": [],
            }
        ],
    }
    write_evidence_bundle(
        evidence_path,
        metadata=metadata,
        official_outputs=[_outputs()],
        mlx_outputs=[_outputs(order=(1, 0))],
    )

    loaded_metadata, official, mlx = load_evidence_bundle(evidence_path)

    assert loaded_metadata["comparison_algorithm"] == COMPARISON_ALGORITHM
    assert loaded_metadata["case_count"] == 1
    assert loaded_metadata["profile"] == "synthetic"
    np.testing.assert_array_equal(official[0]["masks"], _outputs()["masks"])
    np.testing.assert_array_equal(mlx[0]["masks"], _outputs(order=(1, 0))["masks"])
    with np.load(evidence_path, allow_pickle=False) as archive:
        assert json.loads(str(archive["metadata_json"]))["case_count"] == 1


def test_evidence_bundle_requires_npz_suffix(tmp_path):
    with pytest.raises(ValueError, match="end in .npz"):
        write_evidence_bundle(
            tmp_path / "evidence.bin",
            metadata={},
            official_outputs=[],
            mlx_outputs=[],
        )
