import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.train.masks_ops import (
    _decode_coco_rle,
    _normalize_rle_counts,
    ann_to_rle,
    compute_F_measure,
    rle_encode,
    robust_rle_encode,
)

UPSTREAM_TRAIN_MASKS_OPS_SOURCE_COMMIT = "2814fa619404a722d03e9a012e083e4f293a4e53"


def test_rle_encode_returns_pycocotools_compressed_counts_and_area():
    masks = np.array(
        [
            [[False, True, False], [True, True, False]],
            [[False, False, False], [True, True, True]],
        ],
        dtype=bool,
    )

    actual = rle_encode(masks, return_areas=True)

    assert actual == [
        {"size": [2, 3], "counts": "132", "area": 3},
        {"size": [2, 3], "counts": "111000", "area": 3},
    ]
    for encoded, expected_mask in zip(actual, masks, strict=True):
        decoded = _decode_coco_rle(encoded)
        np.testing.assert_array_equal(decoded, expected_mask)


def test_rle_encode_explicitly_exports_mlx_masks_to_host_rle_strings():
    mask = mx.array(
        np.array(
            [[[False, True, False], [True, True, False]]],
            dtype=bool,
        )
    )

    actual = rle_encode(mask)

    assert actual == [{"size": [2, 3], "counts": "132"}]
    assert isinstance(actual[0]["counts"], str)


def test_rle_encode_rejects_non_bool_or_non_rank3_masks():
    with pytest.raises(TypeError, match="boolean dtype"):
        rle_encode(np.ones((1, 2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="shape"):
        rle_encode(np.ones((2, 2), dtype=bool))


def test_decode_coco_rle_accepts_compressed_and_uncompressed_counts():
    expected = np.array(
        [[False, True, False], [True, True, False]],
        dtype=bool,
    )

    compressed = {"size": [2, 3], "counts": "132"}
    uncompressed = {"size": [2, 3], "counts": [1, 3, 2]}

    assert _normalize_rle_counts("132") == [1, 3, 2]
    np.testing.assert_array_equal(_decode_coco_rle(compressed), expected)
    np.testing.assert_array_equal(_decode_coco_rle(uncompressed), expected)


def test_ann_to_rle_compresses_uncompressed_counts_and_preserves_compressed():
    im_info = {"height": 2, "width": 3}

    from_uncompressed = ann_to_rle({"counts": [1, 3, 2]}, im_info)
    already_compressed = ann_to_rle({"size": [2, 3], "counts": "132"}, im_info)

    assert from_uncompressed == {"size": [2, 3], "counts": "132"}
    assert already_compressed == {"size": [2, 3], "counts": "132"}


def test_ann_to_rle_rejects_uncompressed_counts_with_wrong_size():
    with pytest.raises(ValueError, match="cover 3 pixels, expected 6"):
        ann_to_rle({"counts": [1, 2]}, {"height": 2, "width": 3})


def test_robust_rle_encode_uses_the_same_compressed_contract():
    mask = np.array(
        [[[False, True, False], [True, True, False]]],
        dtype=bool,
    )

    assert robust_rle_encode(mask) == [{"size": [2, 3], "counts": "132"}]


def test_compute_f_measure_accepts_compressed_rle_inputs():
    mask_rle = rle_encode(np.array([[[False, True], [False, False]]], dtype=bool))[0]

    assert compute_F_measure(mask_rle, mask_rle, mask_rle, mask_rle) == 1.0
