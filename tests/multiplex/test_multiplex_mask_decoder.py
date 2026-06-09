import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn

from sam3_mlx.model.multiplex_mask_decoder import MultiplexMaskDecoder


class _RecordingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __call__(self, src, pos_src, tokens):
        self.calls.append(
            {
                "src_shape": src.shape,
                "pos_src_shape": pos_src.shape,
                "tokens_shape": tokens.shape,
            }
        )
        batch_size, channels, height, width = src.shape
        flat_src = src.reshape(batch_size, channels, height * width).transpose(0, 2, 1)
        return tokens, flat_src


def _inputs(batch_size=2, channels=8, height=2, width=2):
    values = mx.arange(
        batch_size * channels * height * width,
        dtype=mx.float32,
    ).reshape(batch_size, channels, height, width)
    image_embeddings = values / values.size
    image_pe = mx.zeros((1, channels, height, width), dtype=mx.float32)
    return image_embeddings, image_pe


def _to_numpy(value):
    mx.eval(value)
    return np.asarray(value)


def test_predict_masks_ports_default_multiplex_shapes_and_object_scores():
    transformer = _RecordingTransformer()
    decoder = MultiplexMaskDecoder(
        transformer_dim=8,
        transformer=transformer,
        multiplex_count=2,
        num_multimask_outputs=3,
    )
    image_embeddings, image_pe = _inputs()

    out = decoder.predict_masks(image_embeddings=image_embeddings, image_pe=image_pe)

    assert out["masks"].shape == (2, 2, 4, 8, 8)
    assert out["iou_pred"].shape == (2, 2, 4)
    assert out["mask_tokens_out"].shape == (2, 2, 4, 8)
    assert out["object_score_logits"].shape == (2, 2)
    np.testing.assert_array_equal(
        _to_numpy(out["object_score_logits"]), np.full((2, 2), 10.0)
    )
    assert transformer.calls[-1]["tokens_shape"] == (2, 10, 8)


def test_forward_selects_single_and_multimask_outputs():
    decoder = MultiplexMaskDecoder(
        transformer_dim=8,
        transformer=_RecordingTransformer(),
        multiplex_count=2,
        num_multimask_outputs=3,
        use_multimask_token_for_obj_ptr=True,
    )
    image_embeddings, image_pe = _inputs(batch_size=1)

    single = decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        multimask_output=False,
    )
    multi = decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        multimask_output=True,
    )

    assert single["masks"].shape == (1, 2, 1, 8, 8)
    assert single["iou_pred"].shape == (1, 2, 1)
    assert single["sam_tokens_out"].shape == (1, 2, 1, 8)
    assert multi["masks"].shape == (1, 2, 3, 8, 8)
    assert multi["iou_pred"].shape == (1, 2, 3)
    assert multi["sam_tokens_out"].shape == (1, 2, 3, 8)


def test_shared_mask_tokens_support_extra_per_object_embeddings():
    transformer = _RecordingTransformer()
    decoder = MultiplexMaskDecoder(
        transformer_dim=8,
        transformer=transformer,
        multiplex_count=2,
        num_multimask_outputs=2,
        multimask_outputs_only=True,
        decode_mask_with_shared_tokens=True,
        use_multimask_token_for_obj_ptr=True,
    )
    image_embeddings, image_pe = _inputs(batch_size=1)
    extra_embeddings = mx.ones((1, 2, 8), dtype=mx.float32)

    out = decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        multimask_output=True,
        extra_per_object_embeddings=extra_embeddings,
    )

    assert out["masks"].shape == (1, 2, 2, 8, 8)
    assert out["iou_pred"].shape == (1, 2, 2)
    assert out["sam_tokens_out"].shape == (1, 2, 1, 8)
    assert transformer.calls[-1]["tokens_shape"] == (1, 4, 8)


def test_high_res_decoder_requires_high_res_features():
    decoder = MultiplexMaskDecoder(
        transformer_dim=8,
        transformer=_RecordingTransformer(),
        multiplex_count=2,
        use_high_res_features=True,
    )
    image_embeddings, image_pe = _inputs(batch_size=1)

    with pytest.raises(ValueError, match="high_res_features"):
        decoder.predict_masks(image_embeddings=image_embeddings, image_pe=image_pe)
