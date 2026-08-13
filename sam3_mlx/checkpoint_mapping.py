"""Checkpoint normalization and key mapping for SAM 3 model variants."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TypeAlias, cast

import mlx.core as mx
import mlx.utils as mlx_utils

import sam3_mlx.convert as sam3_convert

CheckpointSource: TypeAlias = Mapping[str, object]
NormalizedWeights: TypeAlias = dict[str, mx.array]


class _HasParameters(Protocol):
    def parameters(self) -> object: ...


class _TreeFlatten(Protocol):
    def __call__(
        self,
        tree: object,
        prefix: str = "",
        is_leaf: Callable[..., object] | None = None,
        destination: NormalizedWeights | None = None,
    ) -> NormalizedWeights: ...


class _NormalizeImageWeightLayout(Protocol):
    def __call__(self, key: str, value: mx.array) -> mx.array: ...


class _Transpose4D(Protocol):
    def __call__(
        self,
        axis0: int,
        axis1: int,
        axis2: int,
        axis3: int,
        /,
        stream: object | None = None,
    ) -> mx.array: ...


_tree_flatten = cast(_TreeFlatten, getattr(mlx_utils, "tree_flatten"))
_normalize_sam3_image_weight_layout = cast(
    _NormalizeImageWeightLayout,
    getattr(sam3_convert, "normalize_sam3_image_weight_layout"),
)


def _flatten_model_parameters(model: _HasParameters) -> NormalizedWeights:
    return _tree_flatten(model.parameters(), destination={})


def _require_checkpoint_mapping(payload: object, *, message: str) -> CheckpointSource:
    ckpt = _unwrap_checkpoint_payload(payload)
    if not isinstance(ckpt, Mapping):
        raise ValueError(message)
    raw_ckpt = cast(Mapping[object, object], ckpt)
    normalized: dict[str, object] = {}
    for key, value in raw_ckpt.items():
        if not isinstance(key, str):
            raise ValueError(message)
        normalized[key] = value
    return normalized


def _require_mx_array(value: object, *, key: str) -> mx.array:
    if not isinstance(value, mx.array):
        raise TypeError(
            f"Expected checkpoint value for {key!r} to be an MLX array, "
            f"got {type(value).__name__}."
        )
    return value


def _transpose_4d(value: mx.array, axes: tuple[int, int, int, int]) -> mx.array:
    transpose = cast(_Transpose4D, getattr(value, "transpose"))
    return transpose(*axes)


def _unwrap_checkpoint_payload(payload: object) -> object:
    if isinstance(payload, Mapping):
        source = cast(Mapping[object, object], payload)
        model_payload = source.get("model")
        if isinstance(model_payload, Mapping):
            return cast(Mapping[object, object], model_payload)
        return source
    return payload


def _normalize_sam3_image_weights(
    payload: object,
    include_tracker: bool,
) -> NormalizedWeights:
    """Normalize official SAM3 checkpoint keys to the local image model names."""
    ckpt = _require_checkpoint_mapping(
        payload,
        message="SAM3 checkpoint payload must be a mapping of weight names.",
    )

    ckpt_dict = dict(ckpt)
    if any(
        key.startswith("sam3_model.") or key.startswith("sam2_predictor.")
        for key in ckpt_dict
    ):
        remapped: dict[str, object] = {}
        for key, value in ckpt_dict.items():
            if key.startswith("sam3_model."):
                key = "detector." + key[len("sam3_model.") :]
            elif key.startswith("sam2_predictor."):
                key = "tracker." + key[len("sam2_predictor.") :]
            remapped[key] = value
        ckpt_dict = remapped

    has_official_prefix = any(
        key.startswith("detector.") or key.startswith("tracker.") for key in ckpt_dict
    )
    if not has_official_prefix:
        if any(key.startswith("detector_model.") for key in ckpt_dict):
            raise ValueError(
                "Transformers-style SAM3 detector_model checkpoints are not yet "
                "mapped into the sam3_mlx image model. Use the "
                "mlx-community/sam3-image checkpoint for image weights and pass "
                "tracker_model weights with interactive_checkpoint_path."
            )
        return {
            key: _normalize_sam3_image_weight_layout(
                key,
                _require_mx_array(value, key=key),
            )
            for key, value in ckpt_dict.items()
        }

    image_weights = {
        key[len("detector.") :]: value
        for key, value in ckpt_dict.items()
        if key.startswith("detector.")
    }
    if not image_weights:
        raise ValueError(
            "SAM3 checkpoint had official prefixes but no detector weights for the "
            "image model."
        )
    if include_tracker:
        image_weights.update(_normalize_inst_interactive_weights(ckpt_dict))
    return {
        key: _normalize_sam3_image_weight_layout(
            key,
            _require_mx_array(value, key=key),
        )
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


def _normalize_inst_interactive_weight_layout(key: str, value: mx.array) -> mx.array:
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
            return _transpose_4d(value, (0, 2, 3, 1))

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
            return _transpose_4d(value, (1, 2, 3, 0))

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


def _normalize_inst_interactive_weights(payload: object) -> NormalizedWeights:
    """Normalize SAM3/SAM2 interactive predictor keys into local image-model keys."""

    ckpt = _require_checkpoint_mapping(
        payload,
        message="SAM3 interactive checkpoint payload must be a mapping of weight names.",
    )

    weights: NormalizedWeights = {}

    point_embed = ckpt.get("tracker_model.prompt_encoder.point_embed.weight")
    if isinstance(point_embed, mx.array):
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
            _require_mx_array(value, key=target_key),
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


def _normalize_tracker_weight_to_shape(
    key: str,
    value: object,
    target_shape: Sequence[int],
) -> mx.array:
    value = _require_mx_array(value, key=key)
    if tuple(value.shape) == tuple(target_shape):
        return value
    if len(value.shape) == 4:
        for perm in ((0, 2, 3, 1), (1, 2, 3, 0)):
            converted = _transpose_4d(value, perm)
            if tuple(converted.shape) == tuple(target_shape):
                return converted
    return value


def _normalize_tracker_checkpoint_weights(
    payload: object,
    model: _HasParameters,
) -> NormalizedWeights:
    """Normalize official tracker checkpoint aliases into local tracker keys."""
    ckpt = _require_checkpoint_mapping(
        payload,
        message="SAM3 tracker checkpoint payload must be a mapping.",
    )

    model_weights = _flatten_model_parameters(model)
    weights: NormalizedWeights = {}

    point_embed = ckpt.get("tracker_model.prompt_encoder.point_embed.weight")
    if isinstance(point_embed, mx.array):
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


def _normalize_sam31_weight_to_shape(
    key: str,
    value: object,
    target_shape: Sequence[int],
) -> mx.array:
    """Normalize a mapped SAM 3.1 checkpoint value to a local MLX parameter."""

    value = _require_mx_array(value, key=key)
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
            converted = _transpose_4d(value, perm)
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
    value: mx.array,
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
    payload: object,
    model: _HasParameters,
    *,
    prefix: str = "",
) -> NormalizedWeights:
    ckpt = _require_checkpoint_mapping(
        payload,
        message="SAM 3.1 multiplex checkpoint payload must be a mapping.",
    )

    model_weights = _flatten_model_parameters(model)
    weights: NormalizedWeights = {}

    point_embed = ckpt.get(
        "tracker_model.interactive_sam_prompt_encoder.point_embed.weight"
    )
    if isinstance(point_embed, mx.array):
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


def _normalize_sam31_multiplex_weights(
    payload: object,
    model: _HasParameters,
) -> NormalizedWeights:
    """Normalize SAM 3.1 multiplex checkpoint keys into the local predictor tree."""

    ckpt = _require_checkpoint_mapping(
        payload,
        message="SAM 3.1 multiplex checkpoint payload must be a mapping.",
    )

    model_weights = _flatten_model_parameters(model)
    weights: NormalizedWeights = {}
    qkv_groups: dict[str, dict[str, mx.array]] = {}

    def add_weight(target_key: str, value: mx.array) -> None:
        if target_key not in model_weights:
            return
        weights[target_key] = _normalize_sam31_weight_to_shape(
            target_key,
            value,
            model_weights[target_key].shape,
        )

    for key, value in ckpt.items():
        if key.startswith("detector_model."):
            mapped = _map_sam31_detector_model_key(
                key,
                _require_mx_array(value, key=key),
                qkv_groups,
            )
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


# Public cross-module entry points; private helpers remain implementation details.
HasParameters = _HasParameters
flatten_model_parameters = _flatten_model_parameters
normalize_inst_interactive_weights = _normalize_inst_interactive_weights
normalize_sam31_multiplex_tracker_weights = _normalize_sam31_multiplex_tracker_weights
normalize_sam31_multiplex_weights = _normalize_sam31_multiplex_weights
normalize_sam3_image_weights = _normalize_sam3_image_weights
normalize_tracker_checkpoint_weights = _normalize_tracker_checkpoint_weights
require_mx_array = _require_mx_array
