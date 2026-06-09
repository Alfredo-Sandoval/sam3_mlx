import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

from sam3_mlx.convert import normalize_sam3_image_weight_layout
from sam3_mlx.model_builder import (
    build_tracker,
    _normalize_inst_interactive_weights,
    _normalize_sam3_image_weights,
    _normalize_sam31_multiplex_tracker_weights,
    _normalize_sam31_multiplex_weights,
    _normalize_tracker_checkpoint_weights,
)


class _WeightLeaf(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.weight = mx.zeros(shape)


class _TinySam31MultiplexTree(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = {
            "backbone": {
                "language_backbone": {
                    "encoder": {
                        "text_projection": mx.zeros((2, 1)),
                    },
                    "resizer": {
                        "weight": mx.zeros((2, 1)),
                        "bias": mx.zeros((2,)),
                    },
                },
                "vision_backbone": {
                    "trunk": {
                        "blocks": [
                            {"attn": {"qkv": _WeightLeaf((6, 2))}},
                        ],
                        "pos_embed": mx.zeros((1, 4, 2)),
                    },
                    "convs": [
                        {"conv_1x1": _WeightLeaf((2, 1, 1, 2))},
                    ],
                },
            },
            "transformer": {
                "decoder": {
                    "layers": [
                        {
                            "norm1": {
                                "weight": mx.zeros((2,)),
                                "bias": mx.zeros((2,)),
                            },
                            "norm2": {
                                "weight": mx.zeros((2,)),
                                "bias": mx.zeros((2,)),
                            },
                            "catext_norm": {
                                "weight": mx.zeros((2,)),
                                "bias": mx.zeros((2,)),
                            },
                            "norm3": {
                                "weight": mx.zeros((2,)),
                                "bias": mx.zeros((2,)),
                            },
                        }
                    ]
                }
            },
            "geometry_encoder": {
                "norm": {"weight": mx.zeros((2,)), "bias": mx.zeros((2,))},
                "img_pre_norm": {
                    "weight": mx.zeros((2,)),
                    "bias": mx.zeros((2,)),
                },
                "encode_norm": {
                    "weight": mx.zeros((2,)),
                    "bias": mx.zeros((2,)),
                },
            },
        }
        self.tracker = {
            "model": {
                "interactive_sam_prompt_encoder": {
                    "point_embeddings": [
                        _WeightLeaf((1, 2)),
                        _WeightLeaf((1, 2)),
                    ],
                },
                "sam_mask_decoder": {
                    "conv_s0": {"conv": _WeightLeaf((2, 1, 1, 2))},
                    "output_hypernetworks_mlps": [
                        {
                            "layers": [
                                _WeightLeaf((2, 2)),
                                _WeightLeaf((2, 2)),
                                _WeightLeaf((1, 2)),
                            ]
                        }
                    ],
                },
            }
        }


class _TinySam31TrackerTree(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_valid_embed = mx.zeros((2, 2))


def test_normalize_sam3_image_weight_layout_converts_sam2_neck_conv_transpose_to_hwio():
    # dconv branch contract: PyTorch ConvTranspose2d weight (O, I, H, W) -> MLX (I, H, W, O).
    # Expected literal is hand-computed, NOT a transpose call mirroring production.
    key = "backbone.vision_backbone.sam2_convs.0.dconv_2x2_0.weight"
    torch_layout = mx.array(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
        ],
        dtype=mx.float32,
    )
    assert torch_layout.shape == (2, 1, 2, 2)

    mlx_layout = normalize_sam3_image_weight_layout(key, torch_layout)

    np.testing.assert_array_equal(
        np.asarray(mlx_layout),
        np.array(
            [
                [
                    [[1.0, 5.0], [2.0, 6.0]],
                    [[3.0, 7.0], [4.0, 8.0]],
                ]
            ],
            dtype=np.float32,
        ),
    )


def test_normalize_sam3_image_weight_layout_converts_sam2_neck_conv2d_to_ohwi():
    # conv2d branch contract: PyTorch Conv2d weight (O, I, H, W) -> MLX (O, H, W, I).
    # Expected literal is hand-computed, NOT a transpose call mirroring production.
    key = "backbone.vision_backbone.sam2_convs.1.conv_3x3.weight"
    torch_layout = mx.array(
        [
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ]
        ],
        dtype=mx.float32,
    )
    assert torch_layout.shape == (1, 2, 3, 3)

    mlx_layout = normalize_sam3_image_weight_layout(key, torch_layout)

    np.testing.assert_array_equal(
        np.asarray(mlx_layout),
        np.array(
            [
                [
                    [[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]],
                    [[4.0, 13.0], [5.0, 14.0], [6.0, 15.0]],
                    [[7.0, 16.0], [8.0, 17.0], [9.0, 18.0]],
                ]
            ],
            dtype=np.float32,
        ),
    )


def test_normalize_sam3_image_weight_layout_leaves_non_conv_keys_untouched():
    # Boundary: keys outside both conv sets must pass through unchanged.
    key = "geometry_encoder.unrelated.weight"
    payload = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=mx.float32)

    out = normalize_sam3_image_weight_layout(key, payload)

    assert out is payload, "Pass-through path must not allocate a new array."


def test_normalize_sam3_image_weights_repairs_legacy_unprefixed_checkpoint():
    # Distinct content per channel distinguishes "ran the right transpose" from
    # "ran any transpose": a wrong axis order would land different scalars in
    # different positions, not just produce a wrong final shape.
    dconv_key = "backbone.vision_backbone.sam2_convs.1.dconv_2x2.weight"
    conv_key = "backbone.vision_backbone.sam2_convs.2.conv_1x1.weight"
    dconv_input = mx.array(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
        ],
        dtype=mx.float32,
    )
    conv_input = mx.array(
        [[[[10.0]], [[20.0]], [[30.0]]]],
        dtype=mx.float32,
    )
    payload = {dconv_key: dconv_input, conv_key: conv_input}

    normalized = _normalize_sam3_image_weights(payload, include_tracker=False)

    np.testing.assert_array_equal(
        np.asarray(normalized[dconv_key]),
        np.array(
            [
                [
                    [[1.0, 5.0], [2.0, 6.0]],
                    [[3.0, 7.0], [4.0, 8.0]],
                ]
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(normalized[conv_key]),
        np.array([[[[10.0, 20.0, 30.0]]]], dtype=np.float32),
    )


def test_normalize_sam3_image_weights_strips_detector_prefix_and_keeps_layout():
    # Verifies (1) prefix stripping, (2) tracker keys dropped when include_tracker=False,
    # (3) the layout transform still fires on the stripped key. Expected output is
    # hand-computed, not produced by re-running the production transpose.
    conv_key = "backbone.vision_backbone.sam2_convs.3.conv_3x3.weight"
    torch_layout = mx.array(
        [
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ]
        ],
        dtype=mx.float32,
    )
    payload = {
        "detector." + conv_key: torch_layout,
        "tracker.prompt_encoder.weight": mx.ones((2, 2)),
    }

    normalized = _normalize_sam3_image_weights(payload, include_tracker=False)

    assert sorted(normalized) == [conv_key]
    np.testing.assert_array_equal(
        np.asarray(normalized[conv_key]),
        np.array(
            [
                [
                    [[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]],
                    [[4.0, 13.0], [5.0, 14.0], [6.0, 15.0]],
                    [[7.0, 16.0], [8.0, 17.0], [9.0, 18.0]],
                ]
            ],
            dtype=np.float32,
        ),
    )


def test_normalize_sam3_image_weights_maps_tracker_when_requested():
    # 1-D inputs do not trigger any layout transpose; this pins the *key renaming*
    # contract specifically (detector prefix stripped, tracker keys remapped).
    payload = {
        "sam3_model.some_detector_weight": mx.array([1.0], dtype=mx.float32),
        "sam2_predictor.sam_prompt_encoder.no_mask_embed.weight": mx.array(
            [2.0],
            dtype=mx.float32,
        ),
    }

    normalized = _normalize_sam3_image_weights(payload, include_tracker=True)

    assert sorted(normalized) == [
        "inst_interactive_predictor.model.sam_prompt_encoder.no_mask_embed.weight",
        "some_detector_weight",
    ]
    assert normalized["some_detector_weight"].tolist() == [1.0]
    assert normalized[
        "inst_interactive_predictor.model.sam_prompt_encoder.no_mask_embed.weight"
    ].tolist() == [2.0]


def test_normalize_sam3_image_weights_rejects_tracker_only_official_payload():
    payload = {"tracker.prompt_encoder.weight": mx.ones((2, 2))}

    with pytest.raises(
        ValueError,
        match="official prefixes but no detector weights",
    ):
        _normalize_sam3_image_weights(payload, include_tracker=True)


def test_normalize_inst_interactive_weights_splits_transformers_point_embeddings():
    payload = {
        "tracker_model.prompt_encoder.point_embed.weight": mx.arange(12)
        .reshape(4, 3)
        .astype(mx.float32)
    }

    normalized = _normalize_inst_interactive_weights(payload)

    assert sorted(normalized) == [
        "inst_interactive_predictor.model.sam_prompt_encoder.point_embeddings.0.weight",
        "inst_interactive_predictor.model.sam_prompt_encoder.point_embeddings.1.weight",
        "inst_interactive_predictor.model.sam_prompt_encoder.point_embeddings.2.weight",
        "inst_interactive_predictor.model.sam_prompt_encoder.point_embeddings.3.weight",
    ]
    np.testing.assert_array_equal(
        np.asarray(
            normalized[
                "inst_interactive_predictor.model.sam_prompt_encoder."
                "point_embeddings.2.weight"
            ]
        ),
        np.array([[6.0, 7.0, 8.0]], dtype=np.float32),
    )


def test_normalize_inst_interactive_weights_maps_transformers_aliases():
    payload = {
        "tracker_model.no_memory_embedding": mx.ones((1, 1, 2)),
        "tracker_model.prompt_encoder.shared_embedding.positional_embedding": mx.ones(
            (2, 1)
        ),
        "tracker_model.prompt_encoder.mask_embed.layer_norm1.weight": mx.ones((4,)),
        "tracker_model.mask_decoder.conv_s0.bias": mx.ones((32,)),
        "tracker_model.mask_decoder.iou_prediction_head.proj_in.weight": mx.ones(
            (2, 2)
        ),
        "tracker_model.mask_decoder.iou_prediction_head.layers.0.bias": mx.ones((2,)),
        "tracker_model.mask_decoder.iou_prediction_head.proj_out.bias": mx.ones((1,)),
        "tracker_model.mask_decoder.output_hypernetworks_mlps.3.proj_out.weight": mx.ones(
            (1, 2)
        ),
        "tracker_model.mask_decoder.transformer.layers.0.self_attn.o_proj.bias": mx.ones(
            (2,)
        ),
        "tracker_model.mask_decoder.transformer.layers.0.layer_norm3.weight": mx.ones(
            (2,)
        ),
        "tracker_model.mask_decoder.transformer.layers.0.mlp.proj_in.bias": mx.ones(
            (4,)
        ),
    }

    normalized = _normalize_inst_interactive_weights(payload)

    assert "inst_interactive_predictor.model.no_mem_embed" in normalized
    assert (
        "inst_interactive_predictor.model.sam_prompt_encoder.pe_layer."
        "positional_encoding_gaussian_matrix" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_prompt_encoder."
        "mask_downscaling.1.weight" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder.conv_s0.conv.bias"
        in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder."
        "iou_prediction_head.layers.0.weight" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder."
        "iou_prediction_head.layers.1.bias" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder."
        "iou_prediction_head.layers.2.bias" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder."
        "output_hypernetworks_mlps.3.layers.2.weight" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder.transformer."
        "layers.0.self_attn.out_proj.bias" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder.transformer."
        "layers.0.norm3.weight" in normalized
    )
    assert (
        "inst_interactive_predictor.model.sam_mask_decoder.transformer."
        "layers.0.mlp.lin1.bias" in normalized
    )


def test_normalize_inst_interactive_weights_converts_official_conv2d_layout():
    payload = {
        "tracker_model.prompt_encoder.mask_embed.conv1.weight": mx.arange(16)
        .reshape(4, 1, 2, 2)
        .astype(mx.float32)
    }

    normalized = _normalize_inst_interactive_weights(payload)

    key = (
        "inst_interactive_predictor.model.sam_prompt_encoder."
        "mask_downscaling.0.conv.weight"
    )
    assert tuple(normalized[key].shape) == (4, 2, 2, 1)
    np.testing.assert_array_equal(
        np.asarray(normalized[key]),
        np.array(
            [
                [[[0.0], [1.0]], [[2.0], [3.0]]],
                [[[4.0], [5.0]], [[6.0], [7.0]]],
                [[[8.0], [9.0]], [[10.0], [11.0]]],
                [[[12.0], [13.0]], [[14.0], [15.0]]],
            ],
            dtype=np.float32,
        ),
    )


def test_normalize_inst_interactive_weights_maps_sam2_predictor_prompt_conv_alias():
    payload = {
        "sam2_predictor.sam_prompt_encoder.mask_downscaling.0.weight": mx.arange(16)
        .reshape(4, 1, 2, 2)
        .astype(mx.float32)
    }

    normalized = _normalize_inst_interactive_weights(payload)

    key = (
        "inst_interactive_predictor.model.sam_prompt_encoder."
        "mask_downscaling.0.conv.weight"
    )
    assert sorted(normalized) == [key]
    np.testing.assert_array_equal(
        np.asarray(normalized[key]),
        np.array(
            [
                [[[0.0], [1.0]], [[2.0], [3.0]]],
                [[[4.0], [5.0]], [[6.0], [7.0]]],
                [[[8.0], [9.0]], [[10.0], [11.0]]],
                [[[12.0], [13.0]], [[14.0], [15.0]]],
            ],
            dtype=np.float32,
        ),
    )


def test_normalize_inst_interactive_weights_converts_official_convtranspose_layout():
    payload = {
        "tracker_model.mask_decoder.upscale_conv1.weight": mx.arange(256 * 64 * 2 * 2)
        .reshape(256, 64, 2, 2)
        .astype(mx.float32)
    }

    normalized = _normalize_inst_interactive_weights(payload)

    key = (
        "inst_interactive_predictor.model.sam_mask_decoder."
        "output_upscaling.0.conv.weight"
    )
    converted = normalized[key]
    assert tuple(converted.shape) == (64, 2, 2, 256)
    assert (
        converted[3, 1, 0, 5].item()
        == payload["tracker_model.mask_decoder.upscale_conv1.weight"][5, 3, 1, 0].item()
    )


def test_normalize_inst_interactive_weights_maps_tracker_decoder_upscale_alias():
    payload = {
        "tracker.sam_mask_decoder.output_upscaling.0.weight": mx.arange(
            256 * 64 * 2 * 2
        )
        .reshape(256, 64, 2, 2)
        .astype(mx.float32)
    }

    normalized = _normalize_inst_interactive_weights(payload)

    key = (
        "inst_interactive_predictor.model.sam_mask_decoder."
        "output_upscaling.0.conv.weight"
    )
    assert sorted(normalized) == [key]
    converted = normalized[key]
    assert tuple(converted.shape) == (64, 2, 2, 256)
    assert (
        converted[3, 1, 0, 5].item()
        == payload["tracker.sam_mask_decoder.output_upscaling.0.weight"][
            5, 3, 1, 0
        ].item()
    )


def test_normalize_tracker_checkpoint_weights_maps_core_tracker_aliases():
    model = build_tracker(apply_temporal_disambiguation=False, with_backbone=False)
    payload = {
        "tracker_model.no_memory_embedding": mx.ones((1, 1, 256), dtype=mx.float32),
        "tracker_model.memory_temporal_positional_encoding": mx.ones(
            (7, 1, 1, 64),
            dtype=mx.float32,
        ),
        "tracker_model.mask_downsample.bias": mx.array([3.0], dtype=mx.float32),
        "tracker_model.mask_downsample.weight": mx.arange(16)
        .reshape(1, 1, 4, 4)
        .astype(mx.float32),
        "tracker_model.prompt_encoder.point_embed.weight": mx.arange(4 * 256)
        .reshape(4, 256)
        .astype(mx.float32),
        "tracker_model.memory_encoder.projection.weight": mx.arange(64 * 256)
        .reshape(64, 256, 1, 1)
        .astype(mx.float32),
        "tracker_model.memory_attention.layers.0.self_attn.o_proj.bias": mx.ones(
            (256,),
            dtype=mx.float32,
        ),
        "tracker_model.mask_decoder.upscale_conv1.weight": mx.arange(256 * 64 * 2 * 2)
        .reshape(256, 64, 2, 2)
        .astype(mx.float32),
    }

    normalized = _normalize_tracker_checkpoint_weights(payload, model)

    assert normalized["no_mem_embed"].shape == (1, 1, 256)
    assert normalized["maskmem_tpos_enc"].shape == (7, 1, 1, 64)
    assert normalized["mask_downsample.bias"].shape == (1,)
    mask_downsample = normalized["mask_downsample.weight"]
    assert tuple(mask_downsample.shape) == (1, 4, 4, 1)
    assert (
        mask_downsample[0, 2, 3, 0].item()
        == payload["tracker_model.mask_downsample.weight"][0, 0, 2, 3].item()
    )
    assert "sam_prompt_encoder.point_embeddings.3.weight" in normalized
    assert "transformer.encoder.layers.0.self_attn.out_proj.bias" in normalized
    projection = normalized["maskmem_backbone.out_proj.conv.weight"]
    assert tuple(projection.shape) == (64, 1, 1, 256)
    assert (
        projection[3, 0, 0, 5].item()
        == payload["tracker_model.memory_encoder.projection.weight"][3, 5, 0, 0].item()
    )
    upscaled = normalized["sam_mask_decoder.output_upscaling.0.conv.weight"]
    assert tuple(upscaled.shape) == (64, 2, 2, 256)
    assert (
        upscaled[3, 1, 0, 5].item()
        == payload["tracker_model.mask_decoder.upscale_conv1.weight"][5, 3, 1, 0].item()
    )


def test_normalize_sam31_multiplex_weights_maps_detector_and_tracker_aliases():
    model = _TinySam31MultiplexTree()
    q = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    k = mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float32)
    v = mx.array([[9.0, 10.0], [11.0, 12.0]], dtype=mx.float32)
    pos = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=mx.float32)
    text_projection = mx.array([[13.0, 14.0]], dtype=mx.float32)
    text_resizer = mx.array([[15.0], [16.0]], dtype=mx.float32)
    text_resizer_bias = mx.array([17.0, 18.0], dtype=mx.float32)
    neck = mx.arange(4).reshape(2, 1, 1, 2).astype(mx.float32)
    decoder_self_norm = mx.array([31.0, 32.0], dtype=mx.float32)
    decoder_vision_norm = mx.array([33.0, 34.0], dtype=mx.float32)
    geometry_prompt_norm = mx.array([35.0, 36.0], dtype=mx.float32)
    geometry_vision_norm = mx.array([37.0, 38.0], dtype=mx.float32)
    point_embed = mx.arange(8).reshape(4, 2).astype(mx.float32)
    conv_s0 = mx.ones((2, 1, 1, 2), dtype=mx.float32)
    proj_out = mx.array([[21.0, 22.0]], dtype=mx.float32)
    payload = {
        "detector_model.vision_encoder.backbone.layers.0.attention.q_proj.weight": q,
        "detector_model.vision_encoder.backbone.layers.0.attention.k_proj.weight": k,
        "detector_model.vision_encoder.backbone.layers.0.attention.v_proj.weight": v,
        "detector_model.vision_encoder.backbone.embeddings.position_embeddings": pos,
        "detector_model.text_encoder.text_projection.weight": text_projection,
        "detector_model.text_projection.weight": text_resizer,
        "detector_model.text_projection.bias": text_resizer_bias,
        "detector_model.vision_encoder.neck.convs.0.proj1.weight": neck,
        "detector_model.detr_decoder.layers.0.self_attn_layer_norm.weight": (
            decoder_self_norm
        ),
        "detector_model.detr_decoder.layers.0.vision_cross_attn_layer_norm.weight": (
            decoder_vision_norm
        ),
        "detector_model.geometry_encoder.prompt_layer_norm.weight": (
            geometry_prompt_norm
        ),
        "detector_model.geometry_encoder.vision_layer_norm.weight": (
            geometry_vision_norm
        ),
        "tracker_model.interactive_sam_prompt_encoder.point_embed.weight": point_embed,
        "tracker_model.sam_mask_decoder.conv_s0.weight": conv_s0,
        "tracker_model.sam_mask_decoder.output_hypernetworks_mlps.0.proj_out.weight": proj_out,
    }

    normalized = _normalize_sam31_multiplex_weights(payload, model)

    np.testing.assert_array_equal(
        np.asarray(
            normalized[
                "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight"
            ]
        ),
        np.asarray(mx.concat([q, k, v], axis=0)),
    )
    np.testing.assert_array_equal(
        np.asarray(normalized["detector.backbone.vision_backbone.trunk.pos_embed"]),
        np.array(
            [[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(
            normalized["detector.backbone.language_backbone.encoder.text_projection"]
        ),
        np.array([[13.0], [14.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(normalized["detector.backbone.language_backbone.resizer.weight"]),
        np.asarray(text_resizer),
    )
    np.testing.assert_array_equal(
        np.asarray(normalized["detector.backbone.language_backbone.resizer.bias"]),
        np.asarray(text_resizer_bias),
    )
    np.testing.assert_array_equal(
        np.asarray(
            normalized["detector.backbone.vision_backbone.convs.0.conv_1x1.weight"]
        ),
        np.asarray(neck),
    )
    np.testing.assert_array_equal(
        np.asarray(normalized["detector.transformer.decoder.layers.0.norm1.weight"]),
        np.asarray(decoder_self_norm),
    )
    np.testing.assert_array_equal(
        np.asarray(normalized["detector.transformer.decoder.layers.0.norm2.weight"]),
        np.asarray(decoder_vision_norm),
    )
    np.testing.assert_array_equal(
        np.asarray(normalized["detector.geometry_encoder.img_pre_norm.weight"]),
        np.asarray(geometry_prompt_norm),
    )
    np.testing.assert_array_equal(
        np.asarray(normalized["detector.geometry_encoder.norm.weight"]),
        np.asarray(geometry_vision_norm),
    )
    np.testing.assert_array_equal(
        np.asarray(
            normalized[
                "tracker.model.interactive_sam_prompt_encoder.point_embeddings.1.weight"
            ]
        ),
        np.array([[2.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(normalized["tracker.model.sam_mask_decoder.conv_s0.conv.weight"]),
        np.asarray(conv_s0),
    )
    np.testing.assert_array_equal(
        np.asarray(
            normalized[
                "tracker.model.sam_mask_decoder.output_hypernetworks_mlps.0."
                "layers.2.weight"
            ]
        ),
        np.asarray(proj_out),
    )


def test_normalize_sam31_multiplex_tracker_weights_maps_direct_model_prefix():
    model = _TinySam31TrackerTree()
    payload = {
        "tracker_model.output_valid_embed": mx.array(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=mx.float32,
        )
    }

    normalized = _normalize_sam31_multiplex_tracker_weights(payload, model)

    assert sorted(normalized) == ["output_valid_embed"]
    np.testing.assert_array_equal(
        np.asarray(normalized["output_valid_embed"]),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
