import mlx.core as mx
import numpy as np
import pytest
from PIL import Image as PILImage

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.train.data.sam3_image_dataset import Datapoint, Image
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
