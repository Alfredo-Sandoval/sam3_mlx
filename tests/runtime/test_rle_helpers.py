import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.rle import (
    ann_to_rle,
    rle_area,
    rle_decode,
    rle_encode,
    rle_to_bbox,
)


def test_rle_encode_decode_area_and_bbox_match_coco_ordering():
    masks = np.array(
        [
            [[False, True, False], [True, True, False]],
            [[False, False, False], [True, True, True]],
        ],
        dtype=bool,
    )

    encoded = rle_encode(masks, return_areas=True)

    assert encoded == [
        {"size": [2, 3], "counts": "132", "area": 3},
        {"size": [2, 3], "counts": "111000", "area": 3},
    ]
    np.testing.assert_array_equal(rle_decode(encoded[0]), masks[0])
    assert rle_area(encoded[0]) == 3
    assert rle_to_bbox(encoded[0]) == [0.0, 0.0, 2.0, 2.0]


def test_rle_encode_accepts_mlx_masks_at_explicit_host_export_boundary():
    mask = mx.array(np.array([[[False, True, False], [True, True, False]]], dtype=bool))

    assert rle_encode(mask) == [{"size": [2, 3], "counts": "132"}]


def test_rle_helpers_reject_ambiguous_mask_and_annotation_shapes():
    with pytest.raises(TypeError, match="boolean dtype"):
        rle_encode(np.ones((1, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        rle_encode(np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="cover 3 pixels, expected 6"):
        ann_to_rle({"counts": [1, 2]}, {"height": 2, "width": 3})
