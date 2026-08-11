from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.eval import coco_eval, coco_eval_offline
from sam3_mlx.eval.saco_veval_evaluators import YTVISPredFileEvaluator
from sam3_mlx.eval.ytvis_eval import YTVISResultsWriter


class _Postprocessor:
    def process_results(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        return [{"video_id": 7}]


class _Evaluator:
    def __init__(self) -> None:
        self.received_path: str | None = None

    def evaluate(self, pred_file: str) -> dict[str, float]:
        self.received_path = pred_file
        return {"metric": 0.5}


def test_coco_box_conversion_is_shared_and_preserves_order() -> None:
    boxes = [[5, 7, 13, 19], [1, 2, 4, 6]]

    converted = coco_eval.convert_to_xywh(boxes)

    assert coco_eval.convert_to_xywh is coco_eval_offline.convert_to_xywh
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, [[5, 7, 8, 12], [1, 2, 3, 4]])


def test_coco_evaluator_compatibility_surface_stays_fail_fast() -> None:
    with pytest.raises(Sam3MlxUnsupportedError, match="evaluation surface") as exc:
        coco_eval.CocoEvaluator()

    assert exc.value.feature == "eval.coco_eval.CocoEvaluator"
    assert exc.value.reason == "eval-stack"


def test_ytvis_results_writer_keeps_local_json_contract(tmp_path: Path) -> None:
    dump_file = tmp_path / "nested" / "predictions.json"
    evaluator = _Evaluator()
    writer = YTVISResultsWriter(
        str(dump_file), _Postprocessor(), pred_file_evaluators=[evaluator]
    )

    writer.update(ignored=True)
    metrics = writer.compute_synced()

    assert dump_file.read_text(encoding="utf-8") == '[{"video_id": 7}]'
    assert evaluator.received_path == str(dump_file)
    assert metrics == {"metric": 0.5}
    assert (
        Path(str(dump_file) + ".sam3_eval_metrics").read_text(encoding="utf-8")
        == '{"metric": 0.5}'
    )


def test_ytvis_prediction_file_evaluator_validates_iou_types() -> None:
    evaluator = YTVISPredFileEvaluator("ground_truth.json", iou_types=["bbox"])

    assert evaluator.iou_types == ["bbox"]
    with pytest.raises(Sam3MlxUnsupportedError, match="evaluation surface") as exc:
        evaluator.evaluate("predictions.json")
    assert exc.value.reason == "eval-stack"

    with pytest.raises(AssertionError, match="iou_types must be bbox or segm"):
        YTVISPredFileEvaluator("ground_truth.json", iou_types=["keypoints"])
