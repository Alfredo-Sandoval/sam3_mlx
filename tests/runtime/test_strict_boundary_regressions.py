# pyright: reportPrivateUsage=false

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.model.io_utils import load_video_frames_from_video_file
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.sam3_multiplex_base import (
    Sam3MultiplexBase,
    TrackerState,
    _is_bucket_state,
    _optional_int_attribute,
    _require_multiplex_controller,
    text_outputs_for_batch,
)
from sam3_mlx.model.sam3_multiplex_tracking import (
    _integer_value,
    _require_feature_cache,
)
from sam3_mlx.model.sam3_tracking_predictor import _positive_integer
from sam3_mlx.model.sam3_video_base import _dimension
from sam3_mlx.model.video_tracking_multiplex_demo import (
    _init_multiplex_demo_state,
)


class _RemovalController:
    allowed_bucket_capacity = 4
    training = False


class _RemovalTracker:
    is_multiplex = True
    multiplex_controller = _RemovalController()

    def __init__(self) -> None:
        self.remove_calls: list[int] = []

    def remove_object(
        self,
        inference_state: TrackerState,
        obj_id: int,
        strict: bool = False,
        need_output: bool = True,
    ) -> tuple[list[int], list[object]]:
        del strict, need_output
        self.remove_calls.append(obj_id)
        values = inference_state.get("obj_ids", [])
        if not isinstance(values, list):
            raise TypeError("test tracker state obj_ids must be an integer list")
        integer_values: list[int] = []
        for value in cast(list[object], values):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("test tracker state obj_ids must be an integer list")
            integer_values.append(value)
        remaining = [value for value in integer_values if value != obj_id]
        inference_state["obj_ids"] = remaining
        return remaining, []


class _RemovalDetector:
    is_multiplex = True
    running_in_prod = False


def test_tracker_remove_objects_sorts_set_before_sequential_fallback():
    tracker = _RemovalTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_RemovalDetector(),
        is_multiplex=True,
    )
    states: list[TrackerState] = [{"obj_ids": [2, 4, 7, 9]}]

    base._tracker_remove_objects(states, {7, 2, 4})

    assert states == [{"obj_ids": [9]}]
    assert tracker.remove_calls == [2, 4, 7]


def test_tracker_remove_objects_preserves_explicit_list_order():
    tracker = _RemovalTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_RemovalDetector(),
        is_multiplex=True,
    )
    states: list[TrackerState] = [{"obj_ids": [2, 4, 7, 9]}]

    base._tracker_remove_objects(states, [7, 2, 4])

    assert states == [{"obj_ids": [9]}]
    assert tracker.remove_calls == [7, 2, 4]


def test_torchcodec_video_loader_is_explicitly_unsupported_on_mlx():
    with pytest.raises(Sam3MlxUnsupportedError) as exc_info:
        load_video_frames_from_video_file(
            "unused.mp4",
            image_size=14,
            video_loader_type="torchcodec",
        )

    error = exc_info.value
    assert error.reason == "torchcodec"
    assert "TorchCodec/Torch-only path" in str(error)
    assert error.alternative == "video_loader_type='cv2' or an image folder"


def test_integer_normalizers_reject_boolean_quantities():
    with pytest.raises(TypeError, match="frame_idx must be an integer"):
        _integer_value(True, name="frame_idx")
    with pytest.raises(TypeError, match="frame dimensions must be integers"):
        _dimension(True)
    with pytest.raises(ValueError, match="num_frames must be a positive integer"):
        _positive_integer(True, name="num_frames")

    assert _integer_value(np.int64(3), name="frame_idx") == 3
    assert _positive_integer(np.int64(3), name="num_frames") == 3


def test_multiplex_state_boundaries_reject_boolean_counts_and_dimensions():
    with pytest.raises(ValueError, match="num_frames must be a positive integer"):
        _init_multiplex_demo_state(
            video_height=720,
            video_width=1280,
            num_frames=True,
        )
    with pytest.raises(
        ValueError, match="video_height and video_width must be positive integers"
    ):
        _init_multiplex_demo_state(
            video_height=True,
            video_width=1280,
            num_frames=3,
        )
    with pytest.raises(TypeError, match="input_mask_size must be an integer"):
        _optional_int_attribute(
            SimpleNamespace(input_mask_size=True), "input_mask_size"
        )
    with pytest.raises(TypeError, match="integer allowed_bucket_capacity"):
        _require_multiplex_controller(
            SimpleNamespace(
                multiplex_controller=SimpleNamespace(
                    allowed_bucket_capacity=True,
                    training=False,
                )
            )
        )

    assert not _is_bucket_state(SimpleNamespace(num_buckets=True))


def test_feature_cache_rejects_boolean_frame_keys():
    with pytest.raises(TypeError, match="string or integer keys"):
        _require_feature_cache({True: object()})


def test_image_matching_requires_populated_optional_target_at_transition():
    matching_model = SimpleNamespace(
        training=True,
        num_interactive_steps_val=0,
        _validate_interactive_steps_val=lambda: None,
    )

    with pytest.raises(ValueError, match="matching requires a populated find_target"):
        Sam3Image.forward_grounding(
            matching_model,
            backbone_out={},
            find_input=object(),
            find_target=None,
            geometric_prompt=object(),
        )


def test_empty_packed_output_does_not_treat_boolean_as_zero_count():
    false_count_state = {"multiplex_state": SimpleNamespace(total_valid_entries=False)}
    empty_count_state = {"multiplex_state": SimpleNamespace(total_valid_entries=0)}

    Sam3MultiplexBase._ensure_empty_packed_current_output(false_count_state, 3)
    Sam3MultiplexBase._ensure_empty_packed_current_output(empty_count_state, 3)

    assert "output_dict" not in false_count_state
    assert empty_count_state["output_dict"]["cond_frame_outputs"][3] == {
        "conditioning_objects": set()
    }


@pytest.mark.parametrize("text_batch", ["shoe", b"shoe", ["shoe", 1]])
def test_detector_text_batch_rejects_string_like_and_mixed_iterables(text_batch):
    with pytest.raises(TypeError, match="sequences of strings"):
        text_outputs_for_batch(object(), {}, text_batch, device="mlx")


def test_detector_text_batch_accepts_string_sequence_and_reuses_cache():
    class _TextBackbone:
        def __init__(self) -> None:
            self.calls = 0

        def forward_text(self, prompts, *, device):
            self.calls += 1
            assert prompts == ["shoe", "visual"]
            assert device == "mlx"
            return {"language_features": np.zeros((2, 1), dtype=np.float32)}

    backbone = _TextBackbone()
    detector = SimpleNamespace(backbone=backbone)
    feature_cache = {}

    first = text_outputs_for_batch(
        detector, feature_cache, ["shoe", "visual"], device="mlx"
    )
    second = text_outputs_for_batch(
        detector, feature_cache, ["shoe", "visual"], device="mlx"
    )

    assert second["language_features"] is first["language_features"]
    assert backbone.calls == 1
