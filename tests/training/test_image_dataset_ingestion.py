import json

import numpy as np
import pytest
from PIL import Image as PILImage

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.train.data.coco_json_loaders import COCO_FROM_JSON
from sam3_mlx.train.data.sam3_image_dataset import CustomCocoDetectionAPI
from sam3_mlx.train.transforms.segmentation import DecodeRle, InstanceToSemantic


def test_coco_json_ingestion_preserves_ids_geometry_and_compressed_rle(tmp_path):
    image_path = tmp_path / "sample.png"
    PILImage.new("RGB", (3, 2)).save(image_path)
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 17,
                        "file_name": image_path.name,
                        "width": 3,
                        "height": 2,
                    }
                ],
                "annotations": [
                    {
                        "id": 9,
                        "image_id": 17,
                        "category_id": 4,
                        "bbox": [0, 0, 1, 1],
                        "segmentation": {"counts": [1, 3, 2]},
                    }
                ],
                "categories": [{"id": 4, "name": "target"}],
            }
        ),
        encoding="utf-8",
    )

    dataset = CustomCocoDetectionAPI(
        root=str(tmp_path),
        annFile=str(annotation_path),
        load_segmentation=True,
        training=False,
    )
    datapoint = dataset[0]

    assert len(datapoint.images) == 1
    assert len(datapoint.images[0].objects) == 1
    obj = datapoint.images[0].objects[0]
    np.testing.assert_array_equal(
        to_numpy(obj.bbox), np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    )
    assert obj.area == pytest.approx(1 / 6)
    assert obj.segment == {"size": [2, 3], "counts": "132"}

    query = datapoint.find_queries[0]
    assert query.query_text == "target"
    assert query.object_ids_output == [0]
    assert query.inference_metadata is not None
    assert query.inference_metadata.coco_image_id == 17
    assert query.inference_metadata.original_image_id == 17
    assert query.inference_metadata.original_category_id == 4
    assert query.inference_metadata.original_size == (2, 3)

    semantic_datapoint = InstanceToSemantic(use_rle=True)(datapoint)
    assert semantic_datapoint.find_queries[0].semantic_target == {
        "size": [2, 3],
        "counts": "132",
    }
    decoded = DecodeRle()(semantic_datapoint)
    assert decoded.find_queries[0].semantic_target is not None
    np.testing.assert_array_equal(
        to_numpy(decoded.find_queries[0].semantic_target),
        np.array(
            [[False, True, False], [True, True, False]],
            dtype=bool,
        ),
    )


def test_coco_json_ingestion_rejects_non_integer_image_dimensions(tmp_path):
    annotation_path = tmp_path / "bad_annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "sample.png",
                        "width": "3",
                        "height": 2,
                    }
                ],
                "annotations": [],
                "categories": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="COCO image width must be an integer"):
        COCO_FROM_JSON(str(annotation_path))
