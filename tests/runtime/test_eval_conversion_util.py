from __future__ import annotations

import json
from pathlib import Path

import pytest

from sam3_mlx.eval.conversion_util import (
    convert_ytbvis_to_cocovid_gt,
    convert_ytbvis_to_cocovid_pred,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_ytvis_ground_truth_conversion_preserves_frame_order(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    output_path = tmp_path / "nested" / "converted.json"
    source = {
        "categories": [{"id": 3, "name": "object"}],
        "videos": [
            {
                "id": 7,
                "file_names": ["clip/000.jpg", "clip/001.jpg"],
                "width": 12,
                "height": 8,
                "length": 2,
            }
        ],
        "annotations": [
            {
                "id": 4,
                "video_id": 7,
                "category_id": 3,
                "bboxes": [[1, 2, 3, 4], None],
                "areas": [12, None],
                "segmentations": [{"size": [8, 12], "counts": "abc"}, None],
                "iscrowd": 0,
            }
        ],
    }
    _write_json(source_path, source)

    converted = convert_ytbvis_to_cocovid_gt(source_path, output_path)

    assert converted["images"] == [
        {
            "id": 1,
            "video_id": 7,
            "file_name": "clip/000.jpg",
            "width": 12,
            "height": 8,
            "frame_index": 0,
            "frame_id": 0,
        },
        {
            "id": 2,
            "video_id": 7,
            "file_name": "clip/001.jpg",
            "width": 12,
            "height": 8,
            "frame_index": 1,
            "frame_id": 1,
        },
    ]
    assert converted["annotations"] == [
        {
            "id": 1,
            "video_id": 7,
            "image_id": 1,
            "track_id": 4,
            "category_id": 3,
            "bbox": [1, 2, 3, 4],
            "area": 12,
            "segmentation": {"size": [8, 12], "counts": "abc"},
            "iscrowd": 0,
        }
    ]
    assert json.loads(output_path.read_text(encoding="utf-8")) == converted


def test_ytvis_prediction_empty_optional_lists_use_frame_fallbacks(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.json"
    dataset_path = tmp_path / "dataset.json"
    output_path = tmp_path / "converted_predictions.json"
    _write_json(
        predictions_path,
        [
            {
                "video_id": 7,
                "category_id": 3,
                "score": 0.75,
                "bboxes": [[1, 2, 3, 4]],
                "segmentations": [],
                "areas": [],
            }
        ],
    )
    _write_json(
        dataset_path,
        {"images": [{"id": 11, "video_id": 7, "frame_index": 0}]},
    )

    convert_ytbvis_to_cocovid_pred(predictions_path, dataset_path, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        {
            "image_id": 11,
            "video_id": 7,
            "track_id": 1,
            "category_id": 3,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "area": 12.0,
            "iscrowd": 0,
            "score": 0.75,
        }
    ]


@pytest.mark.parametrize("field", ["video_id", "category_id"])
def test_ytvis_prediction_rejects_boolean_integer_fields(
    tmp_path: Path, field: str
) -> None:
    predictions_path = tmp_path / "predictions.json"
    dataset_path = tmp_path / "dataset.json"
    output_path = tmp_path / "converted_predictions.json"
    prediction: dict[str, object] = {
        "video_id": 7,
        "category_id": 3,
        "score": 0.75,
        "bboxes": [[1, 2, 3, 4]],
    }
    prediction[field] = True
    _write_json(predictions_path, [prediction])
    _write_json(
        dataset_path,
        {"images": [{"id": 11, "video_id": 7, "frame_index": 0}]},
    )

    with pytest.raises(TypeError, match=rf"prediction\.{field} must be an integer"):
        convert_ytbvis_to_cocovid_pred(predictions_path, dataset_path, output_path)
