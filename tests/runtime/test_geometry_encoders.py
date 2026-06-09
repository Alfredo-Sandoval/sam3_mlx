import pytest
import mlx.core as mx

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.model.geometry_encoders import (  # noqa: E402
    Prompt,
    SequenceGeometryEncoder,
    concat_padded_sequences,
)


def test_concat_padded_sequences_matches_scatter_order_and_index():
    seq1 = mx.array(
        [
            [[10], [100]],
            [[11], [101]],
            [[12], [102]],
        ],
        dtype=mx.int64,
    )
    mask1 = mx.array(
        [
            [False, False, True],
            [False, True, True],
        ]
    )
    seq2 = mx.array(
        [
            [[20], [200]],
            [[21], [201]],
        ],
        dtype=mx.int64,
    )
    mask2 = mx.array(
        [
            [False, True],
            [False, False],
        ]
    )

    sequence, mask, index = concat_padded_sequences(
        seq1, mask1, seq2, mask2, return_index=True
    )

    assert sequence.squeeze(-1).tolist() == [
        [10, 100],
        [11, 200],
        [20, 201],
        [21, 0],
        [0, 0],
    ]
    assert mask.tolist() == [
        [False, False, False, True, True],
        [False, False, False, True, True],
    ]
    assert index.tolist() == [
        [2, 1],
        [3, 2],
    ]


def test_prompt_append_points_and_masks_follow_official_prompt_contract():
    prompt = Prompt(
        point_embeddings=mx.array(
            [
                [[1, 1]],
                [[9, 9]],
            ],
            dtype=mx.float32,
        ),
        point_mask=mx.array([[False, True]]),
        point_labels=mx.array([[1], [1]], dtype=mx.int64),
    )

    prompt.append_points(
        mx.array(
            [
                [[2, 2]],
                [[3, 3]],
            ],
            dtype=mx.float32,
        ),
        mx.array([[0], [1]], dtype=mx.int64),
        mx.array([[False, False]]),
    )

    assert prompt.point_embeddings.tolist() == [
        [[1.0, 1.0]],
        [[2.0, 2.0]],
        [[3.0, 3.0]],
        [[0.0, 0.0]],
    ]
    assert prompt.point_labels.tolist() == [[1], [0], [1], [0]]
    assert prompt.point_mask.tolist() == [[False, False, False, True]]

    null_prompt = Prompt()
    masks = mx.zeros((1, 2, 1, 4, 4), dtype=mx.float32)
    null_prompt.append_masks(masks)

    assert null_prompt.mask_embeddings.shape == (1, 2, 1, 4, 4)
    assert null_prompt.mask_labels.tolist() == [[1, 1]]
    assert null_prompt.mask_mask.tolist() == [[False], [False]]

    with pytest.raises(
        Sam3MlxUnsupportedError,
        match="Only one mask per prompt",
    ) as exc_info:
        null_prompt.append_masks(masks)
    assert exc_info.value.reason == "image-interactivity"
    assert exc_info.value.feature.endswith("Prompt.append_masks(multiple_masks=True)")


def test_prompt_clone_preserves_box_and_point_fields_independently():
    prompt = Prompt(
        box_embeddings=mx.array([[[0.5, 0.5, 0.25, 0.25]]], dtype=mx.float32),
        box_mask=mx.array([[False]]),
        box_labels=mx.array([[1]], dtype=mx.int64),
        point_embeddings=mx.array([[[0.25, 0.75]]], dtype=mx.float32),
        point_mask=mx.array([[False]]),
        point_labels=mx.array([[0]], dtype=mx.int64),
    )

    clone = prompt.clone()

    assert clone is not prompt
    assert clone.box_embeddings.tolist() == prompt.box_embeddings.tolist()
    assert clone.box_mask.tolist() == prompt.box_mask.tolist()
    assert clone.box_labels.tolist() == prompt.box_labels.tolist()
    assert clone.point_embeddings.tolist() == prompt.point_embeddings.tolist()
    assert clone.point_mask.tolist() == prompt.point_mask.tolist()
    assert clone.point_labels.tolist() == prompt.point_labels.tolist()

    clone.append_points(
        mx.array([[[0.5, 0.5]]], dtype=mx.float32),
        mx.array([[1]], dtype=mx.int64),
    )

    assert prompt.point_embeddings.shape == (1, 1, 2)
    assert clone.point_embeddings.shape == (2, 1, 2)


def test_sequence_geometry_encoder_boxes_pool_uses_mlx_channels_last_conv():
    encoder = SequenceGeometryEncoder(
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=False,
        points_pos_enc=False,
        boxes_direct_project=False,
        boxes_pool=True,
        boxes_pos_enc=False,
        d_model=4,
        pos_enc=None,
        num_layers=0,
        layer=None,
        add_cls=False,
        add_post_encode_proj=False,
    )
    boxes = mx.zeros((0, 1, 4), dtype=mx.float32)
    boxes_mask = mx.zeros((1, 0), dtype=mx.bool_)
    boxes_labels = mx.zeros((0, 1), dtype=mx.int64)
    img_feats = mx.zeros((1, 4, 8, 8), dtype=mx.float32)

    embeds, mask = encoder._encode_boxes(
        boxes=boxes,
        boxes_mask=boxes_mask,
        boxes_labels=boxes_labels,
        img_feats=img_feats,
    )

    assert embeds.shape == (0, 1, 4)
    assert mask.shape == (1, 0)
