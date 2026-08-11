import numpy as np
import pytest
import mlx.core as mx
from mlx import nn
from typing import Callable, cast

from sam3_mlx._unsupported import UPSTREAM_COMMIT, Sam3MlxUnsupportedError
from sam3_mlx.model.sam3_video_base import (
    Sam3VideoBase,
    _associate_det_trk_compilable,
)


class _StubModule(nn.Module):
    pass


def _base(**kwargs: object) -> Sam3VideoBase:
    constructor = cast(Callable[..., Sam3VideoBase], Sam3VideoBase)
    return constructor(detector=_StubModule(), tracker=_StubModule(), **kwargs)


def test_sam3_video_base_constructor_ports_upstream_state_for_helper_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")

    base = _base(
        hotstart_delay=5,
        hotstart_unmatch_thresh=3,
        hotstart_dup_thresh=4,
        max_num_objects=17,
        recondition_every_nth_frame=6,
    )

    assert base.device == "mlx"
    assert base.rank == 1
    assert base.world_size == 4
    assert base.max_num_objects == 17
    assert base.num_obj_for_compile == 5
    assert base.hotstart_delay == 5
    assert base.hotstart_unmatch_thresh == 3
    assert base.hotstart_dup_thresh == 4
    assert base.recondition_every_nth_frame == 6


def test_sam3_video_base_constructor_preserves_hotstart_threshold_assertions():
    with pytest.raises(AssertionError):
        _base(
            hotstart_delay=2,
            hotstart_unmatch_thresh=3,
        )


def test_sam3_video_base_object_limit_helpers_match_upstream_ordering_contract():
    base = _base()

    kept = base.drop_new_det_with_obj_limit(
        new_det_fa_inds=np.array([0, 1, 2, 3]),
        det_scores_np=np.array([0.1, 0.9, 0.4, 0.8]),
        num_to_keep=2,
    )
    gpu_ids = base.assign_new_det_to_gpus(
        new_det_num=5,
        prev_workload_per_gpu=np.array([2, 0, 1]),
    )

    np.testing.assert_array_equal(kept, np.array([1, 3]))
    np.testing.assert_array_equal(gpu_ids, np.array([1, 1, 2, 0, 1]))


def test_sam3_video_base_prep_for_evaluator_builds_official_prediction_schema():
    base = _base()
    first_mask = np.zeros((1, 4, 5), dtype=bool)
    first_mask[0, 1, 2:4] = True
    tracking_res = {0: {2: first_mask}}
    scores_labels = {2: (mx.array(0.75, dtype=mx.float32), mx.array(7, dtype=mx.int64))}

    preds = base.prep_for_evaluator(
        video_frames=[
            np.zeros((4, 5, 3), dtype=np.uint8),
            np.zeros((4, 5, 3), dtype=np.uint8),
        ],
        tracking_res=tracking_res,
        scores_labels=scores_labels,
    )

    np.testing.assert_allclose(preds["scores"], np.array([0.75], dtype=np.float32))
    np.testing.assert_array_equal(preds["per_frame_scores"], preds["scores"])
    np.testing.assert_array_equal(preds["labels"], np.array([7], dtype=np.int64))
    np.testing.assert_array_equal(
        preds["boxes"],
        np.array([[[2, 1, 3, 1], [0, 0, 0, 0]]], dtype=np.float32),
    )
    assert len(preds["masks_rle"]) == 1
    assert len(preds["masks_rle"][0]) == 2
    assert preds["masks_rle"][0][0].get("area") == 2
    assert preds["masks_rle"][0][1].get("area") == 0


def test_sam3_video_base_forward_still_fails_with_canonical_boundary():
    base = _base()

    with pytest.raises(Sam3MlxUnsupportedError, match="tracker-memory") as exc_info:
        base.forward()

    assert exc_info.value.reason == "video-multiplex"
    assert exc_info.value.alternative is not None
    assert exc_info.value.alternative.endswith("Sam3VideoInference")


@pytest.mark.parametrize(
    "method_name",
    ["forward", "_process_hotstart", "update_masklet_confirmation_status"],
)
def test_sam3_video_base_unsupported_shims_accept_arbitrary_call_shapes(
    method_name: str,
):
    method = getattr(_base(), method_name)

    with pytest.raises(Sam3MlxUnsupportedError) as exc_info:
        method("unexpected", extra=object())

    error = exc_info.value
    assert error.feature.endswith(method_name)
    assert error.reason == "video-multiplex"
    assert error.detail
    assert error.upstream_commit == UPSTREAM_COMMIT


def test_associate_det_trk_rejects_o2o_matching_with_canonical_boundary():
    with pytest.raises(Sam3MlxUnsupportedError, match="Hungarian matching") as exc_info:
        _associate_det_trk_compilable(
            det_masks=np.zeros((0, 2, 2), dtype=np.float32),
            det_scores=np.zeros((0,), dtype=np.float32),
            det_keep=np.zeros((0,), dtype=bool),
            trk_masks=np.zeros((0, 2, 2), dtype=np.float32),
            new_det_thresh=0.5,
            iou_threshold_trk=0.5,
            iou_threshold=0.5,
            HIGH_CONF_THRESH=0.7,
            use_iom_recondition=False,
            o2o_matching_masklets_enable=True,
            iom_thresh_recondition=0.5,
            iou_thresh_recondition=0.5,
        )

    assert exc_info.value.reason == "video-multiplex"
    assert exc_info.value.alternative == "o2o_matching_masklets_enable=False"
