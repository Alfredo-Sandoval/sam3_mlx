import mlx.core as mx
import numpy as np
import pytest
from PIL import Image as PILImage

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.train.data.collator import collate_fn_api
from sam3_mlx.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
    Object,
)
from sam3_mlx.train.transforms.basic import TargetDict, resize
from sam3_mlx.train.transforms.basic_for_api import ToTensorAPI


def test_resize_preserves_fractional_geometry_from_integer_mlx_targets():
    image = PILImage.new("RGB", (10, 10))
    target: TargetDict = {
        "boxes": mx.array([[0, 0, 9, 9]], dtype=mx.int32),
        "area": mx.array([81], dtype=mx.int32),
    }

    _, resized_target = resize(image, target, (5, 5))

    assert resized_target is not None
    boxes = resized_target["boxes"]
    area = resized_target["area"]
    assert isinstance(boxes, mx.array)
    assert isinstance(area, mx.array)
    np.testing.assert_array_equal(
        to_numpy(boxes),
        np.array([[0.0, 0.0, 4.5, 4.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        to_numpy(area),
        np.array([20.25], dtype=np.float32),
    )


def test_to_tensor_api_rejects_non_pil_non_mlx_image_data():
    image = Image(
        data=mx.zeros((3, 2, 2)),
        objects=[],
        size=(2, 2),
    )
    object.__setattr__(image, "data", np.zeros((2, 2, 3), dtype=np.uint8))
    datapoint = Datapoint(
        find_queries=[],
        images=[image],
    )

    with pytest.raises(TypeError, match="Unsupported image type"):
        ToTensorAPI()(datapoint)


def test_collator_converts_optional_prompt_builders_to_mlx_arrays() -> None:
    image = Image(
        data=mx.zeros((3, 4, 4), dtype=mx.float32),
        objects=[
            Object(
                bbox=mx.array([0.1, 0.2, 0.3, 0.4], dtype=mx.float32),
                area=0.12,
                object_id=0,
            )
        ],
        size=(4, 4),
    )
    query = FindQueryLoaded(
        query_text="target",
        image_id=0,
        object_ids_output=[0],
        is_exhaustive=True,
        input_bbox=mx.array([0.2, 0.3, 0.4, 0.5], dtype=mx.float32),
        input_bbox_label=mx.array([1], dtype=mx.int64),
        input_points=mx.array([[0.25, 0.75, 1.0]], dtype=mx.float32),
        inference_metadata=InferenceMetadata(
            coco_image_id=1,
            original_image_id=1,
            original_category_id=2,
            original_size=(4, 4),
            object_id=0,
            frame_index=0,
        ),
    )

    collated = collate_fn_api(
        [Datapoint(find_queries=[query], images=[image])], "batch"
    )
    stage = collated["batch"].find_inputs[0]

    assert isinstance(stage.input_boxes, mx.array)
    assert isinstance(stage.input_boxes_label, mx.array)
    assert isinstance(stage.input_boxes_mask, mx.array)
    assert isinstance(stage.input_points, mx.array)
    assert isinstance(stage.input_points_mask, mx.array)
    assert stage.input_boxes.shape == (1, 1, 4)
    assert stage.input_points.shape == (1, 1, 3)
