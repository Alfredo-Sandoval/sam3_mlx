"""Checkpoint translation, validation, and loading for SAM3 MLX."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from sam3_mlx.convert import (
    PYTORCH_REPO,
    PYTORCH_REVISION,
    SAM31_REPO,
    SAM31_REVISION,
    normalize_sam3_image_weight_layout,
)
from sam3_mlx.model.sam3_image import Sam3Image


@dataclass(frozen=True)
class Sam3CheckpointShapeMismatch:
    key: str
    model_shape: tuple[int, ...]
    checkpoint_shape: tuple[int, ...]


@dataclass(frozen=True)
class Sam3CheckpointLoadReport:
    """Checkpoint audit produced before loading compatible weights."""

    loaded: tuple[str, ...]
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    shape_mismatched: tuple[Sam3CheckpointShapeMismatch, ...]


def _unwrap_checkpoint_payload(payload):
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        return payload["model"]
    return payload


def _normalize_sam3_image_weights(payload, include_tracker: bool):
    """Normalize official SAM3 checkpoint keys to the local image model names."""
    ckpt = _unwrap_checkpoint_payload(payload)
    if not isinstance(ckpt, Mapping):
        raise ValueError("SAM3 checkpoint payload must be a mapping of weight names.")

    ckpt = dict(ckpt)
    if any(
        key.startswith("sam3_model.") or key.startswith("sam2_predictor.")
        for key in ckpt
    ):
        remapped = {}
        for key, value in ckpt.items():
            if key.startswith("sam3_model."):
                key = "detector." + key[len("sam3_model.") :]
            elif key.startswith("sam2_predictor."):
                key = "tracker." + key[len("sam2_predictor.") :]
            remapped[key] = value
        ckpt = remapped

    has_official_prefix = any(
        key.startswith("detector.") or key.startswith("tracker.") for key in ckpt
    )
    if not has_official_prefix:
        if any(key.startswith("detector_model.") for key in ckpt):
            raise ValueError(
                "Transformers-style SAM3 detector_model checkpoints are not yet "
                "mapped into the sam3_mlx image model. Use the "
                "mlx-community/sam3-image checkpoint for image weights and pass "
                "tracker_model weights with interactive_checkpoint_path."
            )
        return {
            key: normalize_sam3_image_weight_layout(key, value)
            for key, value in ckpt.items()
        }

    image_weights = {
        key[len("detector.") :]: value
        for key, value in ckpt.items()
        if key.startswith("detector.")
    }
    if not image_weights:
        raise ValueError(
            "SAM3 checkpoint had official prefixes but no detector weights for the "
            "image model."
        )
    if include_tracker:
        image_weights.update(_normalize_inst_interactive_weights(ckpt))
    return {
        key: normalize_sam3_image_weight_layout(key, value)
        for key, value in image_weights.items()
    }


_INTERACTIVE_PREFIX = "inst_interactive_predictor.model."

_INTERACTIVE_CONV2D_TARGET_SHAPES = {
    "inst_interactive_predictor.model.sam_prompt_encoder.mask_downscaling.0.conv.weight": (
        4,
        2,
        2,
        1,
    ),
    "inst_interactive_predictor.model.sam_prompt_encoder.mask_downscaling.3.conv.weight": (
        16,
        2,
        2,
        4,
    ),
    "inst_interactive_predictor.model.sam_prompt_encoder.mask_downscaling.6.conv.weight": (
        256,
        1,
        1,
        16,
    ),
    "inst_interactive_predictor.model.sam_mask_decoder.conv_s0.conv.weight": (
        32,
        1,
        1,
        256,
    ),
    "inst_interactive_predictor.model.sam_mask_decoder.conv_s1.conv.weight": (
        64,
        1,
        1,
        256,
    ),
}

_INTERACTIVE_CONVTRANSPOSE2D_TARGET_SHAPES = {
    "inst_interactive_predictor.model.sam_mask_decoder.output_upscaling.0.conv.weight": (
        64,
        2,
        2,
        256,
    ),
    "inst_interactive_predictor.model.sam_mask_decoder.output_upscaling.3.conv.weight": (
        32,
        2,
        2,
        64,
    ),
}


def _normalize_inst_interactive_weight_layout(key: str, value):
    """Map SAM3 interactive conv kernels into MLX's channels-last layout."""

    target_shape = _INTERACTIVE_CONV2D_TARGET_SHAPES.get(key)
    if target_shape is not None and len(value.shape) == 4:
        if tuple(value.shape) == target_shape:
            return value
        torch_shape = (
            target_shape[0],
            target_shape[3],
            target_shape[1],
            target_shape[2],
        )
        if tuple(value.shape) == torch_shape:
            return value.transpose(0, 2, 3, 1)

    target_shape = _INTERACTIVE_CONVTRANSPOSE2D_TARGET_SHAPES.get(key)
    if target_shape is not None and len(value.shape) == 4:
        if tuple(value.shape) == target_shape:
            return value
        torch_shape = (
            target_shape[3],
            target_shape[0],
            target_shape[1],
            target_shape[2],
        )
        if tuple(value.shape) == torch_shape:
            return value.transpose(1, 2, 3, 0)

    return value


def _map_tracker_inner_key(inner: str) -> str:
    """Map official tracker/SAM1 interactive keys into local module names."""

    if inner.startswith("sam_prompt_encoder.mask_downscaling."):
        for layer in ("0", "3", "6"):
            stem = f"sam_prompt_encoder.mask_downscaling.{layer}."
            if inner.startswith(stem) and not inner.startswith(stem + "conv."):
                suffix = inner[len(stem) :]
                if suffix in {"weight", "bias"}:
                    return stem + "conv." + suffix

    if inner.startswith("sam_mask_decoder."):
        decoder_inner = inner[len("sam_mask_decoder.") :]
        for stem in ("conv_s0", "conv_s1"):
            prefix = f"{stem}."
            if decoder_inner.startswith(prefix) and not decoder_inner.startswith(
                f"{stem}.conv."
            ):
                suffix = decoder_inner[len(prefix) :]
                if suffix in {"weight", "bias"}:
                    return f"sam_mask_decoder.{stem}.conv.{suffix}"
        for layer in ("0", "3"):
            stem = f"output_upscaling.{layer}."
            if decoder_inner.startswith(stem) and not decoder_inner.startswith(
                f"output_upscaling.{layer}.conv."
            ):
                suffix = decoder_inner[len(stem) :]
                if suffix in {"weight", "bias"}:
                    return f"sam_mask_decoder.output_upscaling.{layer}.conv.{suffix}"

    return inner


def _map_tracker_model_key(key: str) -> str | None:
    inner = key[len("tracker_model.") :]

    if inner == "no_memory_embedding":
        return _INTERACTIVE_PREFIX + "no_mem_embed"

    if inner == "prompt_encoder.shared_embedding.positional_embedding":
        return (
            _INTERACTIVE_PREFIX
            + "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"
        )
    if inner == "prompt_encoder.point_embed.weight":
        return None
    if inner.startswith("prompt_encoder."):
        prompt_inner = inner[len("prompt_encoder.") :]
        prompt_aliases = {
            "not_a_point_embed.": "not_a_point_embed.",
            "no_mask_embed.": "no_mask_embed.",
            "mask_embed.conv1.": "mask_downscaling.0.conv.",
            "mask_embed.layer_norm1.": "mask_downscaling.1.",
            "mask_embed.conv2.": "mask_downscaling.3.conv.",
            "mask_embed.layer_norm2.": "mask_downscaling.4.",
            "mask_embed.conv3.": "mask_downscaling.6.conv.",
        }
        for source, target in prompt_aliases.items():
            if prompt_inner.startswith(source):
                return (
                    _INTERACTIVE_PREFIX
                    + "sam_prompt_encoder."
                    + target
                    + prompt_inner[len(source) :]
                )
        return None

    if not inner.startswith("mask_decoder."):
        return None

    decoder_inner = inner[len("mask_decoder.") :]
    decoder_aliases = {
        "conv_s0.": "conv_s0.conv.",
        "conv_s1.": "conv_s1.conv.",
        "upscale_conv1.": "output_upscaling.0.conv.",
        "upscale_layer_norm.": "output_upscaling.1.",
        "upscale_conv2.": "output_upscaling.3.conv.",
    }
    for source, target in decoder_aliases.items():
        if decoder_inner.startswith(source):
            return (
                _INTERACTIVE_PREFIX
                + "sam_mask_decoder."
                + target
                + decoder_inner[len(source) :]
            )

    for head in ("iou_prediction_head", "pred_obj_score_head"):
        head_prefix = f"{head}."
        if decoder_inner.startswith(head_prefix):
            rest = decoder_inner[len(head_prefix) :]
            if rest.startswith("proj_in."):
                rest = "layers.0." + rest[len("proj_in.") :]
            elif rest.startswith("layers.0."):
                rest = "layers.1." + rest[len("layers.0.") :]
            elif rest.startswith("proj_out."):
                rest = "layers.2." + rest[len("proj_out.") :]
            return _INTERACTIVE_PREFIX + "sam_mask_decoder." + head_prefix + rest

    hyper_prefix = "output_hypernetworks_mlps."
    if decoder_inner.startswith(hyper_prefix):
        parts = decoder_inner.split(".", 2)
        if len(parts) == 3:
            prefix = ".".join(parts[:2]) + "."
            rest = parts[2]
            if rest.startswith("proj_in."):
                rest = "layers.0." + rest[len("proj_in.") :]
            elif rest.startswith("layers.0."):
                rest = "layers.1." + rest[len("layers.0.") :]
            elif rest.startswith("proj_out."):
                rest = "layers.2." + rest[len("proj_out.") :]
            return _INTERACTIVE_PREFIX + "sam_mask_decoder." + prefix + rest

    if decoder_inner.startswith("transformer."):
        transformer_inner = decoder_inner
        transformer_inner = transformer_inner.replace(".o_proj.", ".out_proj.")
        transformer_inner = transformer_inner.replace(
            ".layer_norm_final_attn.",
            ".norm_final_attn.",
        )
        for index in range(1, 5):
            transformer_inner = transformer_inner.replace(
                f".layer_norm{index}.",
                f".norm{index}.",
            )
        transformer_inner = transformer_inner.replace(".mlp.proj_in.", ".mlp.lin1.")
        transformer_inner = transformer_inner.replace(".mlp.proj_out.", ".mlp.lin2.")
        return _INTERACTIVE_PREFIX + "sam_mask_decoder." + transformer_inner

    return _INTERACTIVE_PREFIX + "sam_mask_decoder." + decoder_inner


def _normalize_inst_interactive_weights(payload):
    """Normalize SAM3/SAM2 interactive predictor keys into local image-model keys."""

    ckpt = _unwrap_checkpoint_payload(payload)
    if not isinstance(ckpt, Mapping):
        raise ValueError(
            "SAM3 interactive checkpoint payload must be a mapping of weight names."
        )

    weights = {}

    point_embed = ckpt.get("tracker_model.prompt_encoder.point_embed.weight")
    if point_embed is not None:
        for index in range(min(4, int(point_embed.shape[0]))):
            key = (
                _INTERACTIVE_PREFIX
                + f"sam_prompt_encoder.point_embeddings.{index}.weight"
            )
            weights[key] = point_embed[index : index + 1]

    for key, value in ckpt.items():
        target_key = None
        if key.startswith(_INTERACTIVE_PREFIX):
            target_key = key
        elif key.startswith("sam2_predictor."):
            inner = key[len("sam2_predictor.") :]
            target_key = _INTERACTIVE_PREFIX + _map_tracker_inner_key(inner)
        elif key.startswith("tracker."):
            inner = key[len("tracker.") :]
            target_key = _INTERACTIVE_PREFIX + _map_tracker_inner_key(inner)
        elif key.startswith("tracker_model."):
            target_key = _map_tracker_model_key(key)

        if target_key is None:
            continue
        weights[target_key] = _normalize_inst_interactive_weight_layout(
            target_key,
            value,
        )

    return weights


def _map_tracker_mlp_alias(inner: str) -> str:
    if inner.startswith("proj_in."):
        return "layers.0." + inner[len("proj_in.") :]
    if inner.startswith("layers.0."):
        return "layers.1." + inner[len("layers.0.") :]
    if inner.startswith("proj_out."):
        return "layers.2." + inner[len("proj_out.") :]
    return inner


def _map_tracker_mask_decoder_inner(inner: str) -> str:
    decoder_aliases = {
        "conv_s0.": "conv_s0.conv.",
        "conv_s1.": "conv_s1.conv.",
        "upscale_conv1.": "output_upscaling.0.conv.",
        "upscale_layer_norm.": "output_upscaling.1.",
        "upscale_conv2.": "output_upscaling.3.conv.",
    }
    for source, target in decoder_aliases.items():
        if inner.startswith(source):
            return "sam_mask_decoder." + target + inner[len(source) :]

    for head in ("iou_prediction_head", "pred_obj_score_head"):
        head_prefix = f"{head}."
        if inner.startswith(head_prefix):
            rest = _map_tracker_mlp_alias(inner[len(head_prefix) :])
            return "sam_mask_decoder." + head_prefix + rest

    hyper_prefix = "output_hypernetworks_mlps."
    if inner.startswith(hyper_prefix):
        parts = inner.split(".", 2)
        if len(parts) == 3:
            prefix = ".".join(parts[:2]) + "."
            rest = _map_tracker_mlp_alias(parts[2])
            return "sam_mask_decoder." + prefix + rest

    if inner.startswith("transformer."):
        transformer_inner = inner
        transformer_inner = transformer_inner.replace(".o_proj.", ".out_proj.")
        transformer_inner = transformer_inner.replace(
            ".layer_norm_final_attn.",
            ".norm_final_attn.",
        )
        for index in range(1, 5):
            transformer_inner = transformer_inner.replace(
                f".layer_norm{index}.",
                f".norm{index}.",
            )
        transformer_inner = transformer_inner.replace(".mlp.proj_in.", ".mlp.lin1.")
        transformer_inner = transformer_inner.replace(".mlp.proj_out.", ".mlp.lin2.")
        return "sam_mask_decoder." + transformer_inner

    return "sam_mask_decoder." + inner


def _map_tracker_prompt_encoder_inner(inner: str) -> str | None:
    if inner == "shared_embedding.positional_embedding":
        return "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"
    if inner == "point_embed.weight":
        return None

    prompt_aliases = {
        "not_a_point_embed.": "not_a_point_embed.",
        "no_mask_embed.": "no_mask_embed.",
        "mask_embed.conv1.": "mask_downscaling.0.conv.",
        "mask_embed.layer_norm1.": "mask_downscaling.1.",
        "mask_embed.conv2.": "mask_downscaling.3.conv.",
        "mask_embed.layer_norm2.": "mask_downscaling.4.",
        "mask_embed.conv3.": "mask_downscaling.6.conv.",
    }
    for source, target in prompt_aliases.items():
        if inner.startswith(source):
            return "sam_prompt_encoder." + target + inner[len(source) :]
    return None


def _map_tracker_memory_attention_inner(inner: str) -> str:
    if inner.startswith("layer_norm."):
        return "transformer.encoder.norm." + inner[len("layer_norm.") :]
    mapped = inner.replace(".o_proj.", ".out_proj.")
    for index in range(1, 4):
        mapped = mapped.replace(f".layer_norm{index}.", f".norm{index}.")
    return "transformer.encoder." + mapped


def _map_tracker_memory_encoder_inner(inner: str) -> str | None:
    if inner.startswith("feature_projection."):
        return (
            "maskmem_backbone.pix_feat_proj.conv." + inner[len("feature_projection.") :]
        )
    if inner.startswith("projection."):
        return "maskmem_backbone.out_proj.conv." + inner[len("projection.") :]
    if inner.startswith("mask_downsampler.final_conv."):
        return (
            "maskmem_backbone.mask_downsampler.encoder.12.conv."
            + inner[len("mask_downsampler.final_conv.") :]
        )
    if inner.startswith("mask_downsampler.layers."):
        parts = inner.split(".")
        if len(parts) >= 5:
            layer = int(parts[2])
            kind = parts[3]
            suffix = ".".join(parts[4:])
            conv_index = (0, 3, 6, 9)[layer]
            norm_index = (1, 4, 7, 10)[layer]
            if kind == "conv":
                return f"maskmem_backbone.mask_downsampler.encoder.{conv_index}.conv.{suffix}"
            if kind == "layer_norm":
                return (
                    f"maskmem_backbone.mask_downsampler.encoder.{norm_index}.{suffix}"
                )
    if inner.startswith("memory_fuser.layers."):
        mapped = "maskmem_backbone.fuser." + inner[len("memory_fuser.") :]
        mapped = mapped.replace(".depthwise_conv.", ".dwconv.conv.")
        mapped = mapped.replace(".layer_norm.", ".norm.")
        mapped = mapped.replace(".pointwise_conv1.", ".pwconv1.")
        mapped = mapped.replace(".pointwise_conv2.", ".pwconv2.")
        if mapped.endswith(".scale"):
            mapped = mapped[: -len(".scale")] + ".gamma"
        return mapped
    return None


def _map_tracker_model_checkpoint_key(key: str) -> str | None:
    inner = key[len("tracker_model.") :]
    simple_aliases = {
        "mask_downsample.bias": "mask_downsample.bias",
        "mask_downsample.weight": "mask_downsample.weight",
        "memory_temporal_positional_encoding": "maskmem_tpos_enc",
        "no_memory_embedding": "no_mem_embed",
        "no_memory_positional_encoding": "no_mem_pos_enc",
        "no_object_pointer": "no_obj_ptr",
        "occlusion_spatial_embedding_parameter": "no_obj_embed_spatial",
    }
    target = simple_aliases.get(inner)
    if target is not None:
        return target

    if inner.startswith("prompt_encoder."):
        return _map_tracker_prompt_encoder_inner(inner[len("prompt_encoder.") :])
    if inner.startswith("mask_decoder."):
        return _map_tracker_mask_decoder_inner(inner[len("mask_decoder.") :])
    if inner.startswith("memory_attention."):
        return _map_tracker_memory_attention_inner(inner[len("memory_attention.") :])
    if inner.startswith("memory_encoder."):
        return _map_tracker_memory_encoder_inner(inner[len("memory_encoder.") :])
    if inner.startswith("object_pointer_proj."):
        rest = _map_tracker_mlp_alias(inner[len("object_pointer_proj.") :])
        return "obj_ptr_proj." + rest
    if inner.startswith("temporal_positional_encoding_projection_layer."):
        return (
            "obj_ptr_tpos_proj."
            + inner[len("temporal_positional_encoding_projection_layer.") :]
        )
    return None


def _normalize_tracker_weight_to_shape(key: str, value, target_shape):
    if not isinstance(value, mx.array):
        raise TypeError(
            f"Expected checkpoint value for {key!r} to be an MLX array, "
            f"got {type(value).__name__}."
        )
    if tuple(value.shape) == tuple(target_shape):
        return value
    if len(value.shape) == 4:
        for perm in ((0, 2, 3, 1), (1, 2, 3, 0)):
            converted = value.transpose(*perm)
            if tuple(converted.shape) == tuple(target_shape):
                return converted
    return value


def _flatten_model_weights(model) -> dict[str, mx.array]:
    """Flatten an MLX parameter tree into the string-keyed mapping we require."""

    flattened = tree_flatten(model.parameters(), destination={})
    if not isinstance(flattened, dict):
        raise TypeError("MLX tree_flatten(destination={}) must return a mapping.")
    model_weights: dict[str, mx.array] = {}
    for key, value in flattened.items():
        if not isinstance(key, str) or not isinstance(value, mx.array):
            raise TypeError(
                "Flattened model parameters must contain string keys and MLX arrays."
            )
        model_weights[key] = value
    return model_weights


def _normalize_tracker_checkpoint_weights(payload, model):
    """Normalize official tracker checkpoint aliases into local tracker keys."""
    ckpt = _unwrap_checkpoint_payload(payload)
    if not isinstance(ckpt, Mapping):
        raise ValueError("SAM3 tracker checkpoint payload must be a mapping.")

    model_weights = _flatten_model_weights(model)
    weights = {}

    point_embed = ckpt.get("tracker_model.prompt_encoder.point_embed.weight")
    if point_embed is not None:
        for index in range(min(4, int(point_embed.shape[0]))):
            target_key = f"sam_prompt_encoder.point_embeddings.{index}.weight"
            if target_key in model_weights:
                weights[target_key] = _normalize_tracker_weight_to_shape(
                    target_key,
                    point_embed[index : index + 1],
                    model_weights[target_key].shape,
                )

    for key, value in ckpt.items():
        target_key = None
        if key.startswith("tracker_model."):
            target_key = _map_tracker_model_checkpoint_key(key)
        elif key.startswith("tracker."):
            target_key = key[len("tracker.") :]
        elif key.startswith("sam2_predictor."):
            target_key = key[len("sam2_predictor.") :]
        elif key in model_weights:
            target_key = key

        if target_key is None or target_key not in model_weights:
            continue
        weights[target_key] = _normalize_tracker_weight_to_shape(
            target_key,
            value,
            model_weights[target_key].shape,
        )

    return weights


def _map_sam31_projection_alias(name: str) -> str:
    aliases = {
        "q_proj": "query_proj",
        "k_proj": "key_proj",
        "v_proj": "value_proj",
        "o_proj": "out_proj",
    }
    return aliases.get(name, name)


def _map_sam31_mlp_layer_alias(inner: str) -> str:
    if inner.startswith("proj_in."):
        return "layers.0." + inner[len("proj_in.") :]
    if inner.startswith("layers.0."):
        return "layers.1." + inner[len("layers.0.") :]
    if inner.startswith("proj_out."):
        return "layers.2." + inner[len("proj_out.") :]
    return inner


def _normalize_sam31_weight_to_shape(key: str, value, target_shape):
    """Normalize a mapped SAM 3.1 checkpoint value to a local MLX parameter."""

    if not isinstance(value, mx.array):
        raise TypeError(
            f"Expected checkpoint value for {key!r} to be an MLX array, "
            f"got {type(value).__name__}."
        )
    if tuple(value.shape) == tuple(target_shape):
        return value
    if (
        key.endswith("text_projection")
        and len(value.shape) == 2
        and tuple(value.T.shape) == tuple(target_shape)
    ):
        return value.T
    if (
        key.endswith("pos_embed")
        and len(value.shape) == 3
        and len(target_shape) == 3
        and value.shape[0] == target_shape[0]
        and value.shape[1] + 1 == target_shape[1]
        and value.shape[2] == target_shape[2]
    ):
        cls_slot = mx.zeros((value.shape[0], 1, value.shape[2]), dtype=value.dtype)
        return mx.concat([cls_slot, value], axis=1)
    if len(value.shape) == 4:
        for perm in ((0, 2, 3, 1), (1, 2, 3, 0)):
            converted = value.transpose(*perm)
            if tuple(converted.shape) == tuple(target_shape):
                return converted
    return value


def _map_sam31_neck_inner_key(inner: str) -> str | None:
    match = re.fullmatch(
        r"(convs|interactive_convs|propagation_convs)\.(\d+)\."
        r"(proj1|proj2|scale_layers\.[02])\.(weight|bias)",
        inner,
    )
    if match is None:
        return inner

    head, layer, module, suffix = match.groups()
    if module == "proj1":
        target_module = "conv_1x1"
    elif module == "proj2":
        target_module = "conv_3x3"
    elif layer == "0" and module == "scale_layers.0":
        target_module = "dconv_2x2_0"
    elif layer == "0" and module == "scale_layers.2":
        target_module = "dconv_2x2_1"
    elif layer == "1" and module == "scale_layers.0":
        target_module = "dconv_2x2"
    else:
        return None
    return f"{head}.{layer}.{target_module}.{suffix}"


def _map_sam31_detector_model_key(
    key: str,
    value,
    qkv_groups: dict[str, dict[str, mx.array]],
) -> tuple[str, mx.array] | None:
    if key.startswith(
        "detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection."
    ):
        suffix = key.rsplit(".", 1)[1]
        return (
            "detector.backbone.vision_backbone.trunk.patch_embed.proj." + suffix,
            value,
        )
    if key == "detector_model.vision_encoder.backbone.embeddings.position_embeddings":
        return "detector.backbone.vision_backbone.trunk.pos_embed", value
    if key.startswith("detector_model.vision_encoder.backbone.layer_norm."):
        suffix = key.rsplit(".", 1)[1]
        return "detector.backbone.vision_backbone.trunk.ln_pre." + suffix, value

    match = re.fullmatch(
        r"detector_model\.vision_encoder\.backbone\.layers\.(\d+)\."
        r"attention\.(q_proj|k_proj|v_proj)\.(weight|bias)",
        key,
    )
    if match is not None:
        layer, projection, suffix = match.groups()
        target = (
            f"detector.backbone.vision_backbone.trunk.blocks.{layer}.attn.qkv.{suffix}"
        )
        qkv_groups.setdefault(target, {})[projection[0]] = value
        return None

    match = re.fullmatch(
        r"detector_model\.vision_encoder\.backbone\.layers\.(\d+)\."
        r"attention\.o_proj\.(weight|bias)",
        key,
    )
    if match is not None:
        layer, suffix = match.groups()
        return (
            "detector.backbone.vision_backbone.trunk.blocks."
            f"{layer}.attn.proj.{suffix}",
            value,
        )

    match = re.fullmatch(
        r"detector_model\.vision_encoder\.backbone\.layers\.(\d+)\."
        r"(layer_norm1|layer_norm2)\.(weight|bias)",
        key,
    )
    if match is not None:
        layer, norm, suffix = match.groups()
        target_norm = {"layer_norm1": "norm1", "layer_norm2": "norm2"}[norm]
        return (
            "detector.backbone.vision_backbone.trunk.blocks."
            f"{layer}.{target_norm}.{suffix}",
            value,
        )

    match = re.fullmatch(
        r"detector_model\.vision_encoder\.backbone\.layers\.(\d+)\."
        r"mlp\.(fc1|fc2)\.(weight|bias)",
        key,
    )
    if match is not None:
        layer, fc, suffix = match.groups()
        return (
            f"detector.backbone.vision_backbone.trunk.blocks.{layer}.mlp.{fc}.{suffix}",
            value,
        )

    if key.startswith("detector_model.vision_encoder.neck."):
        inner = key[len("detector_model.vision_encoder.neck.") :]
        mapped = _map_sam31_neck_inner_key(inner)
        if mapped is None:
            return None
        return "detector.backbone.vision_backbone." + mapped, value

    if (
        key
        == "detector_model.text_encoder.text_model.embeddings.position_embedding.weight"
    ):
        return "detector.backbone.language_backbone.encoder.positional_embedding", value
    if key.startswith(
        "detector_model.text_encoder.text_model.embeddings.token_embedding."
    ):
        suffix = key.rsplit(".", 1)[1]
        return (
            "detector.backbone.language_backbone.encoder.token_embedding." + suffix,
            value,
        )
    if key.startswith("detector_model.text_encoder.text_model.final_layer_norm."):
        suffix = key.rsplit(".", 1)[1]
        return "detector.backbone.language_backbone.encoder.ln_final." + suffix, value
    if key == "detector_model.text_encoder.text_projection.weight":
        return "detector.backbone.language_backbone.encoder.text_projection", value
    if key.startswith("detector_model.text_projection."):
        suffix = key[len("detector_model.text_projection.") :]
        if suffix in {"weight", "bias"}:
            return "detector.backbone.language_backbone.resizer." + suffix, value

    match = re.fullmatch(
        r"detector_model\.text_encoder\.text_model\.encoder\.layers\.(\d+)\."
        r"self_attn\.(q_proj|k_proj|v_proj|out_proj)\.(weight|bias)",
        key,
    )
    if match is not None:
        layer, projection, suffix = match.groups()
        return (
            "detector.backbone.language_backbone.encoder.transformer."
            f"resblocks.{layer}.attn."
            f"{_map_sam31_projection_alias(projection)}.{suffix}",
            value,
        )

    match = re.fullmatch(
        r"detector_model\.text_encoder\.text_model\.encoder\.layers\.(\d+)\."
        r"layer_norm([12])\.(weight|bias)",
        key,
    )
    if match is not None:
        layer, norm_index, suffix = match.groups()
        return (
            "detector.backbone.language_backbone.encoder.transformer."
            f"resblocks.{layer}.ln_{norm_index}.{suffix}",
            value,
        )

    match = re.fullmatch(
        r"detector_model\.text_encoder\.text_model\.encoder\.layers\.(\d+)\."
        r"mlp\.(fc1|fc2)\.(weight|bias)",
        key,
    )
    if match is not None:
        layer, fc, suffix = match.groups()
        target_fc = {"fc1": "c_fc", "fc2": "c_proj"}[fc]
        return (
            "detector.backbone.language_backbone.encoder.transformer."
            f"resblocks.{layer}.mlp.{target_fc}.{suffix}",
            value,
        )

    match = re.fullmatch(
        r"detector_model\.detr_encoder\.layers\.(\d+)\."
        r"(self_attn|cross_attn)\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)",
        key,
    )
    if match is not None:
        layer, attention, projection, suffix = match.groups()
        target_attention = {
            "self_attn": "self_attn",
            "cross_attn": "cross_attn_image",
        }[attention]
        return (
            "detector.transformer.encoder.layers."
            f"{layer}.{target_attention}."
            f"{_map_sam31_projection_alias(projection)}.{suffix}",
            value,
        )

    match = re.fullmatch(
        r"detector_model\.detr_encoder\.layers\.(\d+)\."
        r"layer_norm([123])\.(weight|bias)",
        key,
    )
    if match is not None:
        layer, norm_index, suffix = match.groups()
        return (
            f"detector.transformer.encoder.layers.{layer}.norm{norm_index}.{suffix}",
            value,
        )

    match = re.fullmatch(
        r"detector_model\.detr_encoder\.layers\.(\d+)\."
        r"mlp\.(fc1|fc2)\.(weight|bias)",
        key,
    )
    if match is not None:
        layer, fc, suffix = match.groups()
        target_fc = {"fc1": "linear1", "fc2": "linear2"}[fc]
        return (
            f"detector.transformer.encoder.layers.{layer}.{target_fc}.{suffix}",
            value,
        )

    match = re.fullmatch(
        r"detector_model\.detr_decoder\.layers\.(\d+)\."
        r"(self_attn|text_cross_attn|vision_cross_attn)\."
        r"(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)",
        key,
    )
    if match is not None:
        layer, attention, projection, suffix = match.groups()
        target_attention = {
            "self_attn": "self_attn",
            "text_cross_attn": "ca_text",
            "vision_cross_attn": "cross_attn",
        }[attention]
        return (
            "detector.transformer.decoder.layers."
            f"{layer}.{target_attention}."
            f"{_map_sam31_projection_alias(projection)}.{suffix}",
            value,
        )

    match = re.fullmatch(
        r"detector_model\.detr_decoder\.layers\.(\d+)\."
        r"(self_attn_layer_norm|text_cross_attn_layer_norm|"
        r"vision_cross_attn_layer_norm|mlp_layer_norm)\.(weight|bias)",
        key,
    )
    if match is not None:
        layer, norm, suffix = match.groups()
        target_norm = {
            "self_attn_layer_norm": "norm1",
            "text_cross_attn_layer_norm": "catext_norm",
            "vision_cross_attn_layer_norm": "norm2",
            "mlp_layer_norm": "norm3",
        }[norm]
        return (
            f"detector.transformer.decoder.layers.{layer}.{target_norm}.{suffix}",
            value,
        )

    match = re.fullmatch(
        r"detector_model\.detr_decoder\.layers\.(\d+)\."
        r"mlp\.(fc1|fc2)\.(weight|bias)",
        key,
    )
    if match is not None:
        layer, fc, suffix = match.groups()
        target_fc = {"fc1": "linear1", "fc2": "linear2"}[fc]
        return (
            f"detector.transformer.decoder.layers.{layer}.{target_fc}.{suffix}",
            value,
        )

    if key.startswith("detector_model.detr_decoder."):
        inner = key[len("detector_model.detr_decoder.") :]
        decoder_aliases = {
            "box_head.layer1.": "bbox_embed.layers.0.",
            "box_head.layer2.": "bbox_embed.layers.1.",
            "box_head.layer3.": "bbox_embed.layers.2.",
            "box_rpb_embed_x.layer1.": "boxRPB_embed_x.layers.0.",
            "box_rpb_embed_x.layer2.": "boxRPB_embed_x.layers.1.",
            "box_rpb_embed_y.layer1.": "boxRPB_embed_y.layers.0.",
            "box_rpb_embed_y.layer2.": "boxRPB_embed_y.layers.1.",
            "presence_head.layer1.": "presence_token_head.layers.0.",
            "presence_head.layer2.": "presence_token_head.layers.1.",
            "presence_head.layer3.": "presence_token_head.layers.2.",
            "ref_point_head.layer1.": "ref_point_head.layers.0.",
            "ref_point_head.layer2.": "ref_point_head.layers.1.",
            "output_layer_norm.": "norm.",
            "presence_layer_norm.": "presence_token_out_norm.",
        }
        for source, target in decoder_aliases.items():
            if inner.startswith(source):
                return (
                    "detector.transformer.decoder." + target + inner[len(source) :],
                    value,
                )
        if inner in {
            "presence_token.weight",
            "query_embed.weight",
            "reference_points.weight",
        }:
            return "detector.transformer.decoder." + inner, value

    if key.startswith("detector_model.geometry_encoder."):
        inner = key[len("detector_model.geometry_encoder.") :]
        if inner.startswith("layers."):
            inner = "encode." + inner[len("layers.") :]
            inner = inner.replace(".cross_attn.", ".cross_attn_image.")
            inner = inner.replace(".layer_norm1.", ".norm1.")
            inner = inner.replace(".layer_norm2.", ".norm2.")
            inner = inner.replace(".layer_norm3.", ".norm3.")
            inner = inner.replace(".mlp.fc1.", ".linear1.")
            inner = inner.replace(".mlp.fc2.", ".linear2.")
            inner = inner.replace(".o_proj.", ".out_proj.")
            inner = inner.replace(".q_proj.", ".query_proj.")
            inner = inner.replace(".k_proj.", ".key_proj.")
            inner = inner.replace(".v_proj.", ".value_proj.")
            return "detector.geometry_encoder." + inner, value
        geometry_aliases = {
            "output_layer_norm.": "encode_norm.",
            "prompt_layer_norm.": "img_pre_norm.",
            "vision_layer_norm.": "norm.",
        }
        for source, target in geometry_aliases.items():
            if inner.startswith(source):
                return "detector.geometry_encoder." + target + inner[
                    len(source) :
                ], value
        return "detector.geometry_encoder." + inner, value

    if key.startswith("detector_model.mask_decoder."):
        inner = key[len("detector_model.mask_decoder.") :]
        mask_aliases = {
            "semantic_projection.": "semantic_seg_head.",
            "instance_projection.": "instance_seg_head.",
            "prompt_cross_attn_norm.": "cross_attn_norm.",
            "prompt_cross_attn.": "cross_attend_prompt.",
            "mask_embedder.": "mask_predictor.mask_embed.",
        }
        for source, target in mask_aliases.items():
            if inner.startswith(source):
                inner = target + inner[len(source) :]
                break
        inner = inner.replace(".o_proj.", ".out_proj.")
        inner = inner.replace(".q_proj.", ".query_proj.")
        inner = inner.replace(".k_proj.", ".key_proj.")
        inner = inner.replace(".v_proj.", ".value_proj.")
        return "detector.segmentation_head." + inner, value

    if key.startswith("detector_model.dot_product_scoring."):
        inner = key[len("detector_model.dot_product_scoring.") :]
        scoring_aliases = {
            "query_proj.": "hs_proj.",
            "text_proj.": "prompt_proj.",
            "text_mlp.layer1.": "prompt_mlp.layers.0.",
            "text_mlp.layer2.": "prompt_mlp.layers.1.",
            "text_mlp_out_norm.": "prompt_mlp.out_norm.",
        }
        for source, target in scoring_aliases.items():
            if inner.startswith(source):
                return "detector.dot_prod_scoring." + target + inner[
                    len(source) :
                ], value

    return None


def _map_sam31_multiplex_prompt_encoder_inner(inner: str) -> str | None:
    if inner == "shared_embedding.positional_embedding":
        return "pe_layer.positional_encoding_gaussian_matrix"

    prompt_aliases = {
        "not_a_point_embed.": "not_a_point_embed.",
        "no_mask_embed.": "no_mask_embed.",
        "mask_embed.conv1.": "mask_downscaling.0.conv.",
        "mask_embed.layer_norm1.": "mask_downscaling.1.",
        "mask_embed.conv2.": "mask_downscaling.3.conv.",
        "mask_embed.layer_norm2.": "mask_downscaling.4.",
        "mask_embed.conv3.": "mask_downscaling.6.conv.",
    }
    for source, target in prompt_aliases.items():
        if inner.startswith(source):
            return target + inner[len(source) :]
    return None


def _map_sam31_multiplex_mask_decoder_inner(inner: str) -> str:
    decoder_aliases = {
        "conv_s0.": "conv_s0.conv.",
        "conv_s1.": "conv_s1.conv.",
        "upscale_conv1.": "output_upscaling.0.conv.",
        "upscale_layer_norm.": "output_upscaling.1.",
        "upscale_conv2.": "output_upscaling.3.conv.",
    }
    for source, target in decoder_aliases.items():
        if inner.startswith(source):
            inner = target + inner[len(source) :]
            break

    if inner.startswith("transformer."):
        inner = inner.replace(".o_proj.", ".out_proj.")
        inner = inner.replace(".layer_norm_final_attn.", ".norm_final_attn.")
        inner = inner.replace(".mlp.proj_in.", ".mlp.lin1.")
        inner = inner.replace(".mlp.proj_out.", ".mlp.lin2.")
        for index in range(1, 5):
            inner = inner.replace(f".layer_norm{index}.", f".norm{index}.")

    for head in ("iou_prediction_head.", "pred_obj_score_head."):
        if inner.startswith(head):
            return head + _map_sam31_mlp_layer_alias(inner[len(head) :])

    hyper_prefix = "output_hypernetworks_mlps."
    if inner.startswith(hyper_prefix):
        parts = inner.split(".", 2)
        if len(parts) == 3:
            return ".".join(parts[:2]) + "." + _map_sam31_mlp_layer_alias(parts[2])

    return inner


def _map_sam31_multiplex_tracker_inner_key(inner: str) -> str | None:
    simple_aliases = {
        "memory_temporal_positional_encoding": "maskmem_tpos_enc",
        "no_memory_embedding": "no_mem_embed",
        "no_memory_positional_encoding": "no_mem_pos_enc",
        "no_object_pointer": "no_obj_ptr",
        "occlusion_spatial_embedding_parameter": "no_obj_embed_spatial",
        "no_obj_embed_spatial": "no_obj_embed_spatial",
        "interactivity_no_mem_embed": "interactivity_no_mem_embed",
        "output_valid_embed": "output_valid_embed",
        "output_invalid_embed": "output_invalid_embed",
    }
    target = simple_aliases.get(inner)
    if target is not None:
        return target

    if inner == "image_pe_layer.positional_embedding":
        return "image_pe_layer.positional_encoding_gaussian_matrix"

    if inner.startswith("no_obj_ptr_linear."):
        return inner

    if inner.startswith("interactive_mask_downsample."):
        return (
            "interactive_mask_downsample.conv."
            + inner[len("interactive_mask_downsample.") :]
        )

    if inner.startswith("interactive_sam_prompt_encoder."):
        prompt_inner = inner[len("interactive_sam_prompt_encoder.") :]
        if prompt_inner == "point_embed.weight":
            return None
        target = _map_sam31_multiplex_prompt_encoder_inner(prompt_inner)
        if target is None:
            return None
        return "interactive_sam_prompt_encoder." + target

    if inner.startswith("memory_attention."):
        return "transformer.encoder." + inner[len("memory_attention.") :]

    if inner.startswith("memory_encoder."):
        memory_inner = inner[len("memory_encoder.") :]
        if memory_inner.startswith("feature_projection."):
            return (
                "maskmem_backbone.pix_feat_proj.conv."
                + memory_inner[len("feature_projection.") :]
            )
        if memory_inner.startswith("mask_downsampler.final_conv."):
            return (
                "maskmem_backbone.mask_downsampler.encoder.12.conv."
                + memory_inner[len("mask_downsampler.final_conv.") :]
            )
        if memory_inner.startswith("mask_downsampler.layers."):
            parts = memory_inner.split(".")
            if len(parts) >= 5:
                layer = int(parts[2])
                kind = parts[3]
                suffix = ".".join(parts[4:])
                conv_index = (0, 3, 6, 9)[layer]
                norm_index = (1, 4, 7, 10)[layer]
                if kind == "conv":
                    return (
                        "maskmem_backbone.mask_downsampler.encoder."
                        f"{conv_index}.conv.{suffix}"
                    )
                if kind == "layer_norm":
                    return (
                        "maskmem_backbone.mask_downsampler.encoder."
                        f"{norm_index}.{suffix}"
                    )
        if memory_inner.startswith("memory_fuser.layers."):
            mapped = "maskmem_backbone.fuser." + memory_inner[len("memory_fuser.") :]
            mapped = mapped.replace(".depthwise_conv.", ".dwconv.conv.")
            mapped = mapped.replace(".layer_norm.", ".norm.")
            mapped = mapped.replace(".pointwise_conv1.", ".pwconv1.")
            mapped = mapped.replace(".pointwise_conv2.", ".pwconv2.")
            if mapped.endswith(".scale"):
                mapped = mapped[: -len(".scale")] + ".gamma"
            return mapped

    for decoder_name in ("sam_mask_decoder", "interactive_sam_mask_decoder"):
        decoder_prefix = decoder_name + "."
        if inner.startswith(decoder_prefix):
            decoder_inner = inner[len(decoder_prefix) :]
            return (
                decoder_name
                + "."
                + _map_sam31_multiplex_mask_decoder_inner(decoder_inner)
            )

    for mlp_name in ("obj_ptr_proj", "interactive_obj_ptr_proj"):
        mlp_prefix = mlp_name + "."
        if inner.startswith(mlp_prefix):
            return mlp_name + "." + _map_sam31_mlp_layer_alias(inner[len(mlp_prefix) :])

    if inner.startswith("temporal_positional_encoding_projection_layer."):
        return (
            "obj_ptr_tpos_proj."
            + inner[len("temporal_positional_encoding_projection_layer.") :]
        )

    return None


def _normalize_sam31_multiplex_tracker_weights(
    payload,
    model,
    *,
    prefix: str = "",
) -> dict[str, mx.array]:
    ckpt = _unwrap_checkpoint_payload(payload)
    if not isinstance(ckpt, Mapping):
        raise ValueError("SAM 3.1 multiplex checkpoint payload must be a mapping.")

    model_weights = _flatten_model_weights(model)
    weights: dict[str, mx.array] = {}

    point_embed = ckpt.get(
        "tracker_model.interactive_sam_prompt_encoder.point_embed.weight"
    )
    if point_embed is not None:
        for index in range(min(4, int(point_embed.shape[0]))):
            target_key = (
                prefix
                + "interactive_sam_prompt_encoder."
                + f"point_embeddings.{index}.weight"
            )
            if target_key in model_weights:
                weights[target_key] = _normalize_sam31_weight_to_shape(
                    target_key,
                    point_embed[index : index + 1],
                    model_weights[target_key].shape,
                )

    for key, value in ckpt.items():
        if not key.startswith("tracker_model."):
            continue
        inner = key[len("tracker_model.") :]
        target_inner = _map_sam31_multiplex_tracker_inner_key(inner)
        if target_inner is None:
            continue
        target_key = prefix + target_inner
        if target_key not in model_weights:
            continue
        weights[target_key] = _normalize_sam31_weight_to_shape(
            target_key,
            value,
            model_weights[target_key].shape,
        )

    return weights


def _normalize_sam31_multiplex_weights(payload, model) -> dict[str, mx.array]:
    """Normalize SAM 3.1 multiplex checkpoint keys into the local predictor tree."""

    ckpt = _unwrap_checkpoint_payload(payload)
    if not isinstance(ckpt, Mapping):
        raise ValueError("SAM 3.1 multiplex checkpoint payload must be a mapping.")

    model_weights = _flatten_model_weights(model)
    weights: dict[str, mx.array] = {}
    qkv_groups: dict[str, dict[str, mx.array]] = {}

    def add_weight(target_key: str, value) -> None:
        if target_key not in model_weights:
            return
        weights[target_key] = _normalize_sam31_weight_to_shape(
            target_key,
            value,
            model_weights[target_key].shape,
        )

    for key, value in ckpt.items():
        if key.startswith("detector_model."):
            mapped = _map_sam31_detector_model_key(key, value, qkv_groups)
            if mapped is not None:
                add_weight(*mapped)

    for target_key, parts in qkv_groups.items():
        if {"q", "k", "v"} <= set(parts):
            add_weight(
                target_key,
                mx.concat([parts["q"], parts["k"], parts["v"]], axis=0),
            )

    tracker_weights = _normalize_sam31_multiplex_tracker_weights(
        ckpt,
        model,
        prefix="tracker.model.",
    )
    weights.update(tracker_weights)
    return weights


def _shape_tuple(value) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)


def _audit_sam3_image_checkpoint_load(
    model,
    weights: Mapping[str, mx.array],
) -> Sam3CheckpointLoadReport:
    """Report compatible, missing, extra, and shape-mismatched checkpoint keys."""

    model_weights = _flatten_model_weights(model)
    model_keys = set(model_weights)
    checkpoint_keys = set(weights)
    loaded = []
    shape_mismatched = []

    for key in sorted(model_keys & checkpoint_keys):
        checkpoint_value = weights[key]
        if not isinstance(checkpoint_value, mx.array):
            raise ValueError(
                "Expected checkpoint value for "
                f"{key!r} to be an MLX array, got {type(checkpoint_value).__name__}."
            )
        model_shape = _shape_tuple(model_weights[key])
        checkpoint_shape = _shape_tuple(checkpoint_value)
        if checkpoint_shape == model_shape:
            loaded.append(key)
        else:
            shape_mismatched.append(
                Sam3CheckpointShapeMismatch(
                    key=key,
                    model_shape=model_shape,
                    checkpoint_shape=checkpoint_shape,
                )
            )

    return Sam3CheckpointLoadReport(
        loaded=tuple(loaded),
        missing=tuple(sorted(model_keys - checkpoint_keys)),
        extra=tuple(sorted(checkpoint_keys - model_keys)),
        shape_mismatched=tuple(shape_mismatched),
    )


def _format_checkpoint_shape_mismatches(
    report: Sam3CheckpointLoadReport,
    *,
    limit: int = 10,
) -> str:
    mismatches = report.shape_mismatched
    shown = [
        f"{mismatch.key}: model {mismatch.model_shape}, "
        f"checkpoint {mismatch.checkpoint_shape}"
        for mismatch in mismatches[:limit]
    ]
    if len(mismatches) > limit:
        shown.append(f"... and {len(mismatches) - limit} more")
    return "; ".join(shown)


def _validate_checkpoint_component_coverage(
    model,
    report: Sam3CheckpointLoadReport,
    checkpoint_path: Path | str,
) -> None:
    if isinstance(model, Sam3Image):
        loaded_image = tuple(
            key
            for key in report.loaded
            if not key.startswith("inst_interactive_predictor.")
        )
        if not loaded_image:
            raise ValueError(
                "SAM3 image checkpoint did not load any image-model weights: "
                f"{checkpoint_path}. Use an MLX image checkpoint such as "
                "checkpoints/mlx-community/sam3-image/model.safetensors for "
                "checkpoint_path."
            )

    if getattr(model, "inst_interactive_predictor", None) is not None:
        missing_interactive = tuple(
            key
            for key in report.missing
            if key.startswith("inst_interactive_predictor.")
        )
        if missing_interactive:
            loaded_interactive = tuple(
                key
                for key in report.loaded
                if key.startswith("inst_interactive_predictor.")
            )
            example = missing_interactive[0]
            raise ValueError(
                "SAM1-style interactive prediction was requested, but checkpoint "
                f"{checkpoint_path} does not fully cover the interactive predictor: "
                f"loaded_interactive={len(loaded_interactive)}, "
                f"missing_interactive={len(missing_interactive)}. "
                f"First missing key: {example}. Use an MLX checkpoint with mapped "
                "interactive predictor weights, or build with "
                "enable_inst_interactivity=False for text-prompt image segmentation."
            )

    missing_required = [
        key for key in report.missing if not _is_allowed_missing_model_key(key)
    ]
    if missing_required:
        example = missing_required[0]
        raise ValueError(
            "SAM3 checkpoint did not cover required model weights: "
            f"{checkpoint_path}. loaded={len(report.loaded)}, "
            f"missing_required={len(missing_required)}. "
            f"First missing key: {example}."
        )


def _is_allowed_missing_model_key(key: str) -> bool:
    """Return whether a model value is deterministically rebuilt at runtime."""

    if "freqs_cis" in key:
        return True
    if ".position_encoding.cache." in key:
        return True
    if key == "attn_mask" or key.endswith("language_backbone.encoder.attn_mask"):
        return True
    return False


def _is_allowed_missing_tracker_key(key: str) -> bool:
    if _is_allowed_missing_model_key(key):
        return True
    if key.startswith("backbone."):
        return True
    return False


def _validate_tracker_checkpoint_coverage(
    report: Sam3CheckpointLoadReport,
    checkpoint_path: Path | str,
) -> None:
    if not report.loaded:
        raise ValueError(
            f"SAM3 tracker checkpoint did not load any weights: {checkpoint_path}."
        )
    missing_required = [
        key for key in report.missing if not _is_allowed_missing_tracker_key(key)
    ]
    if missing_required:
        example = missing_required[0]
        raise ValueError(
            "SAM3 tracker checkpoint did not cover required tracker weights: "
            f"loaded={len(report.loaded)}, missing_required={len(missing_required)}. "
            f"First missing key: {example}."
        )


def _validate_sam31_multiplex_checkpoint_coverage(
    report: Sam3CheckpointLoadReport,
    checkpoint_path: Path | str,
) -> None:
    loaded_detector = [key for key in report.loaded if key.startswith("detector.")]
    loaded_tracker = [key for key in report.loaded if key.startswith("tracker.model.")]
    if not loaded_detector or not loaded_tracker:
        raise ValueError(
            "SAM 3.1 multiplex checkpoint did not load both detector and tracker "
            f"weights: {checkpoint_path}. "
            f"loaded_detector={len(loaded_detector)}, "
            f"loaded_tracker={len(loaded_tracker)}."
        )
    missing_text_resizer = [
        key
        for key in (
            "detector.backbone.language_backbone.resizer.bias",
            "detector.backbone.language_backbone.resizer.weight",
        )
        if key in report.missing
    ]
    if missing_text_resizer:
        example = missing_text_resizer[0]
        raise ValueError(
            "SAM 3.1 multiplex checkpoint did not cover required VE text "
            f"resizer weights: {checkpoint_path}. First missing key: {example}."
        )

    missing_required = [
        key for key in report.missing if not _is_allowed_missing_model_key(key)
    ]
    if missing_required:
        example = missing_required[0]
        raise ValueError(
            "SAM 3.1 multiplex checkpoint did not cover required model weights: "
            f"{checkpoint_path}. loaded={len(report.loaded)}, "
            f"missing_required={len(missing_required)}. "
            f"First missing key: {example}."
        )


def _load_multiplex_tracker_checkpoint(
    model,
    checkpoint_path,
    *,
    strict_state_dict_loading: bool = True,
):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix in {".pt", ".pth"}:
        raise ValueError(
            "Official PyTorch SAM 3.1 multiplex checkpoints must be converted "
            "before MLX loading. Pass an MLX .safetensors/.npz checkpoint."
        )
    payload = mx.load(str(checkpoint_path))
    weights = _normalize_sam31_multiplex_tracker_weights(payload, model)
    if not weights:
        raise ValueError(
            f"No SAM 3.1 multiplex tracker weights found in checkpoint: {checkpoint_path}"
        )
    report = _audit_sam3_image_checkpoint_load(model, weights)
    if report.shape_mismatched:
        mismatch_details = _format_checkpoint_shape_mismatches(report)
        raise ValueError(
            "SAM 3.1 multiplex tracker checkpoint has shape-mismatched weights "
            "and was not loaded: "
            f"loaded={len(report.loaded)}, missing={len(report.missing)}, "
            f"extra={len(report.extra)}, "
            f"shape_mismatched={len(report.shape_mismatched)}. "
            f"{mismatch_details}"
        )
    if strict_state_dict_loading:
        _validate_tracker_checkpoint_coverage(report, checkpoint_path)
    elif not report.loaded:
        raise ValueError(
            "SAM 3.1 multiplex tracker checkpoint did not load any weights: "
            f"{checkpoint_path}."
        )
    model.load_weights([(key, weights[key]) for key in report.loaded], strict=False)
    mx.eval(model.parameters())
    return report


def _load_multiplex_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix in {".pt", ".pth"}:
        raise ValueError(
            "Official PyTorch SAM 3.1 multiplex checkpoints must be converted "
            "before MLX loading. Pass an MLX .safetensors/.npz checkpoint."
        )
    payload = mx.load(str(checkpoint_path))
    weights = _normalize_sam31_multiplex_weights(payload, model)
    if not weights:
        raise ValueError(
            f"No SAM 3.1 multiplex weights found in checkpoint: {checkpoint_path}"
        )
    report = _audit_sam3_image_checkpoint_load(model, weights)
    if report.shape_mismatched:
        mismatch_details = _format_checkpoint_shape_mismatches(report)
        raise ValueError(
            "SAM 3.1 multiplex checkpoint has shape-mismatched weights and was "
            "not loaded: "
            f"loaded={len(report.loaded)}, missing={len(report.missing)}, "
            f"extra={len(report.extra)}, "
            f"shape_mismatched={len(report.shape_mismatched)}. "
            f"{mismatch_details}"
        )
    _validate_sam31_multiplex_checkpoint_coverage(report, checkpoint_path)
    model.load_weights([(key, weights[key]) for key in report.loaded], strict=False)
    mx.eval(model.parameters())
    return report


def _load_tracker_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix in {".pt", ".pth"}:
        raise ValueError(
            "Official PyTorch SAM3 tracker checkpoints must be converted before "
            "MLX loading. Pass an MLX .safetensors/.npz checkpoint."
        )
    payload = mx.load(str(checkpoint_path))
    weights = _normalize_tracker_checkpoint_weights(payload, model)
    if not weights:
        raise ValueError(f"No tracker weights found in checkpoint: {checkpoint_path}")
    report = _audit_sam3_image_checkpoint_load(model, weights)
    if report.shape_mismatched:
        mismatch_details = _format_checkpoint_shape_mismatches(report)
        raise ValueError(
            "SAM3 tracker checkpoint has shape-mismatched weights and was not "
            f"loaded: loaded={len(report.loaded)}, missing={len(report.missing)}, "
            f"extra={len(report.extra)}, "
            f"shape_mismatched={len(report.shape_mismatched)}. "
            f"{mismatch_details}"
        )
    _validate_tracker_checkpoint_coverage(report, checkpoint_path)
    model.load_weights([(key, weights[key]) for key in report.loaded], strict=False)
    mx.eval(model.parameters())
    return report


def _load_checkpoint(model, checkpoint_path, *, interactive_checkpoint_path=None):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix in {".pt", ".pth"}:
        raise ValueError(
            "Official PyTorch SAM3 checkpoints must be converted before MLX loading. "
            "Use build_sam3_image_model(convert_from_pytorch=True, ...) or "
            "sam3_mlx.convert.download_and_convert."
        )
    payload = mx.load(str(checkpoint_path))
    weights = _normalize_sam3_image_weights(
        payload,
        include_tracker=getattr(model, "inst_interactive_predictor", None) is not None,
    )
    checkpoint_label: Path | str = checkpoint_path
    if interactive_checkpoint_path is not None:
        if getattr(model, "inst_interactive_predictor", None) is None:
            raise ValueError(
                "interactive_checkpoint_path requires enable_inst_interactivity=True."
            )
        interactive_checkpoint_path = Path(interactive_checkpoint_path)
        if interactive_checkpoint_path.suffix in {".pt", ".pth"}:
            raise ValueError(
                "Official PyTorch interactive checkpoints must be converted before "
                "MLX loading. Pass an MLX .safetensors/.npz checkpoint."
            )
        interactive_payload = mx.load(str(interactive_checkpoint_path))
        interactive_weights = _normalize_inst_interactive_weights(interactive_payload)
        if not interactive_weights:
            raise ValueError(
                "No SAM1-style interactive predictor weights were found in "
                f"interactive_checkpoint_path: {interactive_checkpoint_path}"
            )
        weights.update(interactive_weights)
        checkpoint_label = (
            f"{checkpoint_path} + interactive {interactive_checkpoint_path}"
        )
    if not weights:
        raise ValueError(f"No weights found in checkpoint: {checkpoint_path}")
    report = _audit_sam3_image_checkpoint_load(model, weights)
    if report.shape_mismatched:
        mismatch_details = _format_checkpoint_shape_mismatches(report)
        raise ValueError(
            "SAM3 checkpoint has shape-mismatched weights and was not loaded: "
            f"loaded={len(report.loaded)}, missing={len(report.missing)}, "
            f"extra={len(report.extra)}, "
            f"shape_mismatched={len(report.shape_mismatched)}. "
            f"{mismatch_details}"
        )
    _validate_checkpoint_component_coverage(model, report, checkpoint_label)
    model.load_weights([(key, weights[key]) for key in report.loaded], strict=False)
    mx.eval(model.parameters())
    return report


def download_ckpt_from_hf(version="sam3"):
    """Download an official PyTorch checkpoint for conversion/parity work."""
    if version == "sam3.1":
        repo_id = SAM31_REPO
        ckpt_name = "sam3.1_multiplex.pt"
        revision = SAM31_REVISION
    elif version == "sam3":
        repo_id = PYTORCH_REPO
        ckpt_name = "sam3.pt"
        revision = PYTORCH_REVISION
    else:
        raise ValueError(f"Unknown version: {version!r}. Use 'sam3' or 'sam3.1'.")

    from huggingface_hub import hf_hub_download

    _ = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        revision=revision,
    )
    return hf_hub_download(
        repo_id=repo_id,
        filename=ckpt_name,
        revision=revision,
    )
