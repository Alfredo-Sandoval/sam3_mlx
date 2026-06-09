import numpy as np
import mlx.core as mx
from PIL import Image

from sam3_mlx.model.sam1_task_predictor import (
    SAM3InteractiveImageModel,
    SAM3InteractiveImagePredictor,
)


def _to_numpy(array):
    mx.eval(array)
    return np.asarray(array)


def test_interactive_predictor_allows_official_cleanup_default():
    model = SAM3InteractiveImageModel(image_size=16, backbone_stride=4)

    predictor = SAM3InteractiveImagePredictor(model)

    assert predictor._transforms.max_hole_area == 256.0
    assert predictor._transforms.max_sprinkle_area == 0.0
    assert predictor._bb_feat_sizes == [(16, 16), (8, 8), (4, 4)]


def test_interactive_predictor_feature_sizes_follow_model_resolution():
    model = SAM3InteractiveImageModel(image_size=16, backbone_stride=4)
    predictor = SAM3InteractiveImagePredictor(model)
    backbone_out = {
        "backbone_fpn": [
            mx.zeros((1, 32, 16, 16), dtype=mx.float32),
            mx.zeros((1, 64, 8, 8), dtype=mx.float32),
            mx.zeros((1, 256, 4, 4), dtype=mx.float32),
        ],
        "vision_pos_enc": [
            mx.zeros((1, 32, 16, 16), dtype=mx.float32),
            mx.zeros((1, 64, 8, 8), dtype=mx.float32),
            mx.zeros((1, 256, 4, 4), dtype=mx.float32),
        ],
    }

    predictor._set_features_from_backbone(backbone_out)

    assert predictor._features["high_res_feats"][0].shape == (1, 32, 16, 16)
    assert predictor._features["high_res_feats"][1].shape == (1, 64, 8, 8)
    assert predictor._features["image_embed"].shape == (1, 256, 4, 4)


def test_interactive_predictor_transform_preserves_normalized_rgb_values():
    model = SAM3InteractiveImageModel(image_size=1, backbone_stride=1)
    predictor = SAM3InteractiveImagePredictor(model)

    transformed = predictor._transforms(Image.new("RGB", (1, 1), (0, 128, 255)))

    expected = np.array([[[-1.0]], [[128.0 / 127.5 - 1.0]], [[1.0]]], dtype=np.float32)
    np.testing.assert_allclose(_to_numpy(transformed), expected, rtol=0, atol=1e-7)


def test_interactive_predictor_runs_point_prompt_with_synthetic_features():
    model = SAM3InteractiveImageModel(image_size=16, backbone_stride=4)
    predictor = SAM3InteractiveImagePredictor(model, max_hole_area=0.0)
    predictor._features = {
        "high_res_feats": [
            mx.zeros((1, 32, 16, 16), dtype=mx.float32),
            mx.zeros((1, 64, 8, 8), dtype=mx.float32),
        ],
        "image_embed": mx.zeros((1, 256, 4, 4), dtype=mx.float32),
    }
    predictor._is_image_set = True
    predictor._orig_hw = [(16, 16)]

    masks, ious, low_res = predictor.predict(
        point_coords=np.array([[8.0, 8.0]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        normalize_coords=True,
    )

    assert masks.shape == (3, 16, 16)
    assert masks.dtype == np.float32
    assert ious.shape == (3,)
    assert ious.dtype == np.float32
    assert low_res.shape == (3, 16, 16)
    assert low_res.dtype == np.float32


def test_interactive_postprocess_fills_small_holes_with_mlx_components():
    model = SAM3InteractiveImageModel(image_size=16, backbone_stride=4)
    predictor = SAM3InteractiveImagePredictor(
        model,
        mask_threshold=0.0,
        max_hole_area=1.0,
        max_sprinkle_area=0.0,
    )
    masks = mx.array(
        [
            [
                [
                    [1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ]
        ],
        dtype=mx.float32,
    )

    postprocessed = predictor._transforms.postprocess_masks(masks, (3, 3))

    assert _to_numpy(postprocessed)[0, 0, 1, 1] > predictor.mask_threshold


def test_interactive_postprocess_removes_small_sprinkles_with_mlx_components():
    model = SAM3InteractiveImageModel(image_size=16, backbone_stride=4)
    predictor = SAM3InteractiveImagePredictor(
        model,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=1.0,
    )
    masks = mx.array(
        [
            [
                [
                    [-1.0, -1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, -1.0],
                ]
            ]
        ],
        dtype=mx.float32,
    )

    postprocessed = predictor._transforms.postprocess_masks(masks, (3, 3))

    assert _to_numpy(postprocessed)[0, 0, 1, 1] < predictor.mask_threshold
