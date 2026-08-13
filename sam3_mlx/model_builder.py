from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    Never,
    NotRequired,
    Protocol,
    TypedDict,
    TypeAlias,
    TypeVar,
    cast,
)

import mlx.core as mx
from mlx import nn

from sam3_mlx._device import is_mlx_runtime_device
from sam3_mlx._unsupported import raise_unsupported
import sam3_mlx.convert as sam3_convert
from sam3_mlx.checkpoint_mapping import (
    HasParameters as _HasParameters,
    flatten_model_parameters as _flatten_model_parameters,
    normalize_inst_interactive_weights as _normalize_inst_interactive_weights,
    normalize_sam31_multiplex_tracker_weights as _normalize_sam31_multiplex_tracker_weights,
    normalize_sam31_multiplex_weights as _normalize_sam31_multiplex_weights,
    normalize_sam3_image_weights as _normalize_sam3_image_weights,
    normalize_tracker_checkpoint_weights as _normalize_tracker_checkpoint_weights,
    require_mx_array as _require_mx_array,
)
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3_mlx.model.sam1_task_predictor import (
    SAM3InteractiveImageModel,
    SAM3InteractiveImagePredictor,
)
from sam3_mlx.model.text_encoder_ve import VETextEncoder
from sam3_mlx.model.tokenizer_ve import SimpleTokenizer
from sam3_mlx.model.vitdet import ViT
from sam3_mlx.model.position_encoding import PositionEmbeddingSine
from sam3_mlx.model.necks import Sam3DualViTDetNeck
from sam3_mlx.model.necks import Sam3TriViTDetNeck
from sam3_mlx.model.vl_combiner import (
    SAM3VLBackbone,
    SAM3VLBackboneTri,
    TriHeadVisionOnly,
)
from sam3_mlx.model.geometry_encoders import SequenceGeometryEncoder
from sam3_mlx.model.maskformer_segmentation import (
    CrossAttention,
    PixelDecoder,
    UniversalSegmentationHead,
)
from sam3_mlx.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3_mlx.model.decoder import (
    DecoupledTransformerDecoderLayerv2,
    SimpleRoPEAttention,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderDecoupledCrossAttention,
    TransformerEncoderCrossAttention,
)
from sam3_mlx.model.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from sam3_mlx.model.multiplex_utils import MultiplexController
from sam3_mlx.sam.transformer import RoPEAttention
from sam3_mlx.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)

if TYPE_CHECKING:
    from sam3_mlx.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3_mlx.model.sam3_multiplex_tracking import (
        Sam3MultiplexTrackingWithInteractivity,
    )
    from sam3_mlx.model.sam3_multiplex_video_predictor import (
        Sam3MultiplexVideoPredictor,
    )
    from sam3_mlx.model.sam3_video_inference import (
        ProcessorFactory,
        Sam3VideoInferenceWithInstanceInteractivity,
    )
    from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor
    from sam3_mlx.model.video_tracking_multiplex_demo import (
        Sam3VideoTrackingMultiplexDemo,
    )


ComputeDevice: TypeAlias = Literal["mlx"] | None
PathLikeStr: TypeAlias = str | os.PathLike[str]
CompileMode: TypeAlias = str | bool | None
ImageStats: TypeAlias = tuple[float, float, float]
_ModelT = TypeVar("_ModelT", bound="_SupportsEval")


class _SupportsEval(Protocol):
    def eval(self) -> object: ...


class _CheckpointLoadable(_HasParameters, Protocol):
    def load_weights(
        self,
        file_or_weights: str | list[tuple[str, mx.array]],
        strict: bool = True,
    ) -> object: ...


class _MlxEval(Protocol):
    def __call__(self, *values: object) -> None: ...


class _MlxLoad(Protocol):
    def __call__(self, file: PathLikeStr, /) -> object: ...


class _HfHubDownload(Protocol):
    def __call__(self, repo_id: str, filename: str, **kwargs: object) -> str: ...


class _VideoPredictorFactory(Protocol):
    def __call__(
        self,
        *model_args: object,
        **model_kwargs: object,
    ) -> Sam3VideoPredictor: ...


class _CheckpointProvenance(TypedDict):
    status: str
    repo: str | None
    revision: str | None
    output_sha256: str | None
    checkpoint_path: NotRequired[str]


class _MultiplexPredictorKwargs(TypedDict, total=False):
    session_expiration_sec: int
    default_output_prob_thresh: float
    score_threshold_detection: float
    image_only_det_thresh: float
    suppress_det_close_to_boundary: bool
    strict_state_dict_loading: bool


_mx_eval = cast(_MlxEval, getattr(mx, "eval"))
_mx_load = cast(_MlxLoad, getattr(mx, "load"))


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


def _has_inst_interactivity(model: object) -> bool:
    return getattr(model, "inst_interactive_predictor", None) is not None


def _setup_tf32() -> None:
    """Official TF32 setup hook; no-op for the Apple Silicon MLX port."""

    return None


_setup_tf32()


def _default_bpe_path() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "assets",
        "bpe_simple_vocab_16e6.txt.gz",
    )


def _raise_builder_unsupported(
    feature: str,
    *,
    reason: str,
    detail: str,
    alternative: str | None = None,
) -> Never:
    raise_unsupported(
        feature,
        reason=reason,
        alternative=alternative,
        detail=detail,
    )


def _raise_compile_unsupported(feature: str) -> Never:
    _raise_builder_unsupported(
        feature,
        reason="torch-compile",
        detail="torch.compile is not part of the sam3_mlx runtime.",
        alternative="compile=False",
    )


def _normalize_mlx_api_device(device: ComputeDevice) -> str:
    if is_mlx_runtime_device(device):
        return "mlx"
    _raise_builder_unsupported(
        f"sam3_mlx.model_builder.device={device!r}",
        reason="unsupported-device",
        detail=(
            "sam3_mlx only runs on the explicit MLX runtime. Non-MLX "
            "device strings are not accepted as aliases."
        ),
        alternative="device='mlx'",
    )


def _validate_mlx_device(device: ComputeDevice) -> None:
    _normalize_mlx_api_device(device)


def _validate_sam3_video_runtime_options(
    feature_prefix: str,
    *,
    compile: bool,
    device: ComputeDevice,
    has_presence_token: bool,
    geo_encoder_use_img_cross_attn: bool,
    strict_state_dict_loading: bool,
    apply_temporal_disambiguation: bool,
) -> None:
    if compile:
        _raise_compile_unsupported(f"{feature_prefix}(compile=True)")
    _validate_mlx_device(device)
    if not has_presence_token:
        _raise_builder_unsupported(
            f"{feature_prefix}(has_presence_token=False)",
            reason="video-multiplex",
            detail="The current MLX video model keeps the official presence-token path.",
            alternative="has_presence_token=True",
        )
    if not geo_encoder_use_img_cross_attn:
        _raise_builder_unsupported(
            f"{feature_prefix}(geo_encoder_use_img_cross_attn=False)",
            reason="video-multiplex",
            detail="The MLX port has not implemented the alternate geometry encoder video path.",
            alternative="geo_encoder_use_img_cross_attn=True",
        )
    if not strict_state_dict_loading:
        _raise_builder_unsupported(
            f"{feature_prefix}(strict_state_dict_loading=False)",
            reason="video-multiplex",
            detail="Checkpoint loading strictness is not configurable for the MLX video slice.",
            alternative="strict_state_dict_loading=True",
        )
    if not apply_temporal_disambiguation:
        _raise_builder_unsupported(
            f"{feature_prefix}(apply_temporal_disambiguation=False)",
            reason="video-multiplex",
            detail="This changes tracker behavior, and the tracker is not ported to MLX yet.",
            alternative="apply_temporal_disambiguation=True",
        )


def _setup_device_and_mode(
    model: _ModelT,
    device: ComputeDevice,
    eval_mode: bool,
) -> _ModelT:
    """Setup the explicit MLX device contract and evaluation mode."""

    _validate_mlx_device(device)
    if eval_mode:
        model.eval()
    return model


def _create_position_encoding(
    precompute_resolution: int | None = None,
) -> PositionEmbeddingSine:
    """Create a PositionEmbeddingSine block (used by the backbone and geometry encoder)."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(compile_mode: str | None = None) -> ViT:
    """Create the ViT backbone."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )


def _create_vit_neck(
    position_encoding: PositionEmbeddingSine,
    vit_backbone: ViT,
    enable_inst_interactivity: bool = False,
) -> Sam3DualViTDetNeck:
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(
    vit_neck: Sam3DualViTDetNeck,
    text_encoder: VETextEncoder,
    *,
    compile_visual: bool = False,
) -> SAM3VLBackbone:
    """Create visual-language backbone."""
    return SAM3VLBackbone(
        visual=vit_neck,
        text=text_encoder,
        compile_visual=compile_visual,
        scalp=1,
    )


def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create the transformer encoder."""

    def encoder_layer() -> TransformerEncoderLayer:
        return TransformerEncoderLayer(
            activation="relu",
            d_model=256,
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=True,
            pos_enc_at_cross_attn_keys=False,
            pos_enc_at_cross_attn_queries=False,
            pre_norm=True,
            self_attention=MultiheadAttention(
                num_heads=8,
                dims=256,
            ),
            cross_attention=MultiheadAttention(
                num_heads=8,
                dims=256,
            ),
        )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder() -> TransformerDecoder:
    """Create the transformer decoder."""

    def decoder_layer() -> TransformerDecoderLayer:
        return TransformerDecoderLayer(
            activation="relu",
            d_model=256,
            dim_feedforward=2048,
            dropout=0.1,
            cross_attention=MultiheadAttention(
                num_heads=8,
                dims=256,
            ),
            n_heads=8,
            use_text_cross_attention=True,
        )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder


def _create_dot_product_scoring() -> DotProductScoring:
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head() -> UniversalSegmentationHead:
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dims=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        cross_attend_prompt=cast(CrossAttention, cross_attend_prompt),
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def _create_geometry_encoder() -> SequenceGeometryEncoder:
    geo_pos_enc = _create_position_encoding()

    def geo_layer() -> TransformerEncoderLayer:
        return TransformerEncoderLayer(
            activation="relu",
            d_model=256,
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            pre_norm=True,
            self_attention=MultiheadAttention(
                num_heads=8,
                dims=256,
            ),
            pos_enc_at_cross_attn_queries=False,
            pos_enc_at_cross_attn_keys=True,
            cross_attention=MultiheadAttention(
                num_heads=8,
                dims=256,
            ),
        )

    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_inst_interactive_predictor() -> SAM3InteractiveImagePredictor:
    interactive_model = SAM3InteractiveImageModel(
        image_size=1008,
        backbone_stride=14,
        hidden_dim=256,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
    )
    return SAM3InteractiveImagePredictor(
        interactive_model,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
    )


def _create_sam3_model(
    backbone: SAM3VLBackbone,
    transformer: TransformerWrapper,
    input_geometry_encoder: SequenceGeometryEncoder,
    segmentation_head: UniversalSegmentationHead | None,
    dot_prod_scoring: DotProductScoring,
    inst_interactive_predictor: SAM3InteractiveImagePredictor | None = None,
) -> Sam3Image:
    return Sam3Image(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=input_geometry_encoder,
        segmentation_head=segmentation_head,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=dot_prod_scoring,
        use_instance_query=False,
        multimask_output=True,
        inst_interactive_predictor=inst_interactive_predictor,
    )


def create_tracker_maskmem_backbone() -> SimpleMaskEncoder:
    """Create the SAM3 Tracker memory encoder (SimpleMaskEncoder)."""
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )
    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)
    return SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )


def create_tracker_transformer() -> TransformerWrapper:
    """Create the SAM3 Tracker memory-attention transformer (encoder-only)."""
    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )
    return TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )


def build_tracker(
    apply_temporal_disambiguation: bool,
    with_backbone: bool = False,
    compile_mode: CompileMode = None,
    checkpoint_path: PathLikeStr | None = None,
) -> Sam3TrackerPredictor:
    """Build the SAM3 SAM2-style tracker predictor."""
    if compile_mode not in (None, False):
        _raise_compile_unsupported("sam3_mlx.model_builder.build_tracker(compile_mode)")
    if checkpoint_path is not None and with_backbone:
        _raise_builder_unsupported(
            "sam3_mlx.model_builder.build_tracker(checkpoint_path, with_backbone=True)",
            reason="video-tracker",
            detail=(
                "Tracker-model checkpoint keys are mapped, but detector/tracker-neck "
                "backbone checkpoint keys are not mapped to the MLX tracker backbone yet."
            ),
            alternative="build_tracker(..., with_backbone=False, checkpoint_path=...)",
        )

    maskmem_backbone = create_tracker_maskmem_backbone()
    transformer = create_tracker_transformer()
    backbone = None
    if with_backbone:
        vision_backbone = _create_vision_backbone(
            compile_mode=None,
            enable_inst_interactivity=True,
        )
        backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)

    model = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multimask_output_in_sam=True,
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        always_start_from_first_ann_frame=False,
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
        use_memory_selection=apply_temporal_disambiguation,
    )
    if checkpoint_path is not None:
        _load_tracker_checkpoint(model, checkpoint_path)
    return model


def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer, d_model=256, width=1024, heads=16, layers=24
    )


def _create_vision_backbone(
    compile_mode: str | None = None,
    enable_inst_interactivity: bool = True,
) -> Sam3DualViTDetNeck:
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    vit_backbone = _create_vit_backbone(compile_mode=compile_mode)

    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    return vit_neck


def _create_sam3_transformer(
    has_presence_token: bool = True,
) -> TransformerWrapper:
    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _shape_tuple(value: mx.array) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)


def _audit_sam3_image_checkpoint_load(
    model: _HasParameters,
    weights: Mapping[str, mx.array],
) -> Sam3CheckpointLoadReport:
    """Report compatible, missing, extra, and shape-mismatched checkpoint keys."""

    model_weights = _flatten_model_parameters(model)
    model_keys = set(model_weights)
    checkpoint_keys = set(weights)
    loaded: list[str] = []
    shape_mismatched: list[Sam3CheckpointShapeMismatch] = []

    for key in sorted(model_keys & checkpoint_keys):
        checkpoint_value = _require_mx_array(weights[key], key=key)
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
    model: object,
    report: Sam3CheckpointLoadReport,
    checkpoint_path: Path | str,
) -> None:
    unexpected_source = tuple(
        key
        for key in report.extra
        if not key.startswith("backbone.vision_backbone.sam2_convs.")
    )
    if unexpected_source:
        example = unexpected_source[0]
        raise ValueError(
            "SAM3 checkpoint contains source weights without a reviewed model "
            f"mapping: {checkpoint_path}. unexpected={len(unexpected_source)}. "
            f"First unexpected key: {example}."
        )
    missing_required = tuple(
        key for key in report.missing if not _is_generated_checkpoint_key(key)
    )
    if missing_required:
        example = missing_required[0]
        raise ValueError(
            "SAM3 checkpoint did not cover all required model weights: "
            f"{checkpoint_path}. loaded={len(report.loaded)}, "
            f"missing_required={len(missing_required)}. "
            f"First missing key: {example}. Pass "
            "strict_checkpoint_loading=False only for explicit development-time "
            "partial loading."
        )

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

    if not _has_inst_interactivity(model):
        return

    missing_interactive = tuple(
        key for key in report.missing if key.startswith("inst_interactive_predictor.")
    )
    if not missing_interactive:
        return

    loaded_interactive = tuple(
        key for key in report.loaded if key.startswith("inst_interactive_predictor.")
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


def _is_generated_checkpoint_key(key: str) -> bool:
    """Return whether a model value is deterministically generated at construction."""
    if key == "backbone.language_backbone.encoder.attn_mask":
        return True
    return "freqs_cis" in key or ".position_encoding.cache." in key


def _is_allowed_missing_tracker_key(key: str) -> bool:
    if _is_generated_checkpoint_key(key):
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
        key for key in report.missing if not _is_generated_checkpoint_key(key)
    ]
    if missing_required:
        example = missing_required[0]
        raise ValueError(
            "SAM 3.1 multiplex checkpoint did not cover all required model "
            f"weights: {checkpoint_path}. loaded={len(report.loaded)}, "
            f"missing_required={len(missing_required)}. First missing key: {example}."
        )


def _load_multiplex_tracker_checkpoint(
    model: _CheckpointLoadable,
    checkpoint_path: PathLikeStr,
    *,
    strict: bool = True,
) -> Sam3CheckpointLoadReport:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix in {".pt", ".pth"}:
        raise ValueError(
            "Official PyTorch SAM 3.1 multiplex checkpoints must be converted "
            "before MLX loading. Pass an MLX .safetensors/.npz checkpoint."
        )
    payload = _mx_load(str(checkpoint_path))
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
    if strict:
        _validate_tracker_checkpoint_coverage(report, checkpoint_path)
    elif not report.loaded:
        raise ValueError(
            f"SAM 3.1 multiplex tracker checkpoint did not load any weights: "
            f"{checkpoint_path}."
        )
    model.load_weights([(key, weights[key]) for key in report.loaded], strict=False)
    _mx_eval(model.parameters())
    setattr(model, "checkpoint_load_report", report)
    return report


def _load_multiplex_checkpoint(
    model: _CheckpointLoadable,
    checkpoint_path: PathLikeStr,
    *,
    strict: bool = True,
) -> Sam3CheckpointLoadReport:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix in {".pt", ".pth"}:
        raise ValueError(
            "Official PyTorch SAM 3.1 multiplex checkpoints must be converted "
            "before MLX loading. Pass an MLX .safetensors/.npz checkpoint."
        )
    payload = _mx_load(str(checkpoint_path))
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
    if strict:
        _validate_sam31_multiplex_checkpoint_coverage(report, checkpoint_path)
    elif not report.loaded:
        raise ValueError(
            f"SAM 3.1 multiplex checkpoint did not load any weights: {checkpoint_path}."
        )
    model.load_weights([(key, weights[key]) for key in report.loaded], strict=False)
    _mx_eval(model.parameters())
    setattr(model, "checkpoint_load_report", report)
    return report


def _load_tracker_checkpoint(
    model: _CheckpointLoadable,
    checkpoint_path: PathLikeStr,
) -> Sam3CheckpointLoadReport:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix in {".pt", ".pth"}:
        raise ValueError(
            "Official PyTorch SAM3 tracker checkpoints must be converted before "
            "MLX loading. Pass an MLX .safetensors/.npz checkpoint."
        )
    payload = _mx_load(str(checkpoint_path))
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
    _mx_eval(model.parameters())
    setattr(model, "checkpoint_load_report", report)
    return report


def _load_checkpoint(
    model: _CheckpointLoadable,
    checkpoint_path: PathLikeStr,
    *,
    interactive_checkpoint_path: PathLikeStr | None = None,
    strict: bool = True,
) -> Sam3CheckpointLoadReport:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix in {".pt", ".pth"}:
        raise ValueError(
            "Official PyTorch SAM3 checkpoints must be converted before MLX loading. "
            "Use build_sam3_image_model(convert_from_pytorch=True, ...) or "
            "sam3_mlx.convert.download_and_convert."
        )
    payload = _mx_load(str(checkpoint_path))
    weights = _normalize_sam3_image_weights(
        payload,
        include_tracker=_has_inst_interactivity(model),
    )
    checkpoint_label: Path | str = checkpoint_path
    if interactive_checkpoint_path is not None:
        if not _has_inst_interactivity(model):
            raise ValueError(
                "interactive_checkpoint_path requires enable_inst_interactivity=True."
            )
        interactive_checkpoint_path = Path(interactive_checkpoint_path)
        if interactive_checkpoint_path.suffix in {".pt", ".pth"}:
            raise ValueError(
                "Official PyTorch interactive checkpoints must be converted before "
                "MLX loading. Pass an MLX .safetensors/.npz checkpoint."
            )
        interactive_payload = _mx_load(str(interactive_checkpoint_path))
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
    if strict:
        _validate_checkpoint_component_coverage(model, report, checkpoint_label)
    elif not report.loaded:
        raise ValueError(
            f"SAM3 checkpoint did not load any weights: {checkpoint_label}."
        )
    model.load_weights([(key, weights[key]) for key in report.loaded], strict=False)
    _mx_eval(model.parameters())
    setattr(model, "checkpoint_load_report", report)
    return report


def download_ckpt_from_hf(version: str = "sam3") -> str:
    """Download an official PyTorch checkpoint for conversion/parity work."""
    if version == "sam3.1":
        repo_id = "facebook/sam3.1"
        ckpt_name = "sam3.1_multiplex.pt"
    elif version == "sam3":
        repo_id = "facebook/sam3"
        ckpt_name = "sam3.pt"
    else:
        raise ValueError(f"Unknown version: {version!r}. Use 'sam3' or 'sam3.1'.")

    import huggingface_hub

    download = cast(_HfHubDownload, getattr(huggingface_hub, "hf_hub_download"))
    _ = download(repo_id=repo_id, filename="config.json")
    return download(repo_id=repo_id, filename=ckpt_name)


def build_sam3_image_model(
    bpe_path: PathLikeStr | None = None,
    device: ComputeDevice = "mlx",
    eval_mode: bool = True,
    checkpoint_path: PathLikeStr | None = None,
    load_from_HF: bool = True,
    enable_segmentation: bool = True,
    enable_inst_interactivity: bool = False,
    compile: bool = False,
    hf_repo: str = sam3_convert.DEFAULT_MLX_CHECKPOINT.repo,
    hf_revision: str = sam3_convert.DEFAULT_MLX_CHECKPOINT.revision,
    local_weights_dir: str | None = None,
    convert_from_pytorch: bool = False,
    interactive_checkpoint_path: PathLikeStr | None = None,
    strict_checkpoint_loading: bool = True,
    conversion_source_revision: str | None = None,
    expected_output_sha256: str | None = None,
    verify_hub_provenance: bool = True,
) -> Sam3Image:
    _validate_mlx_device(device)
    if checkpoint_path is None and convert_from_pytorch and not load_from_HF:
        raise ValueError("convert_from_pytorch=True requires load_from_HF=True.")
    if convert_from_pytorch and not conversion_source_revision:
        raise ValueError(
            "convert_from_pytorch=True requires conversion_source_revision to be "
            "an immutable Hugging Face commit."
        )
    if bpe_path is None:
        bpe_path = _default_bpe_path()
    bpe_path = os.fspath(bpe_path)

    vision_encoder = _create_vision_backbone(
        compile_mode=None,
        enable_inst_interactivity=enable_inst_interactivity,
    )

    text_encoder = _create_text_encoder(bpe_path)

    backbone = _create_vl_backbone(
        vision_encoder,
        text_encoder,
        compile_visual=compile,
    )

    transformer = _create_sam3_transformer()

    dot_product_scoring = _create_dot_product_scoring()

    segmentation_head = _create_segmentation_head() if enable_segmentation else None

    input_geometry_encoder = _create_geometry_encoder()
    inst_interactive_predictor = (
        _create_inst_interactive_predictor() if enable_inst_interactivity else None
    )

    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring=dot_product_scoring,
        inst_interactive_predictor=inst_interactive_predictor,
    )

    provenance: _CheckpointProvenance = {
        "status": "unloaded",
        "repo": None,
        "revision": None,
        "output_sha256": None,
    }

    if checkpoint_path is None and load_from_HF:
        if convert_from_pytorch:
            source_revision = conversion_source_revision
            if source_revision is None:
                raise ValueError(
                    "convert_from_pytorch=True requires conversion_source_revision to be "
                    "an immutable Hugging Face commit."
                )
            checkpoint_path = sam3_convert.download_and_convert(
                hf_repo="facebook/sam3",
                mlx_path=local_weights_dir or "sam3-mod-weights",
                source_revision=source_revision,
            )
            provenance = {
                "status": "converted-from-pytorch",
                "repo": "facebook/sam3",
                "revision": source_revision,
                "output_sha256": None,
            }
        else:
            if (
                expected_output_sha256 is None
                and hf_repo == sam3_convert.DEFAULT_MLX_CHECKPOINT.repo
            ):
                if hf_revision == sam3_convert.DEFAULT_MLX_CHECKPOINT.revision:
                    expected_output_sha256 = (
                        sam3_convert.DEFAULT_MLX_CHECKPOINT.output_sha256
                    )
            checkpoint_path = sam3_convert.load_from_hub(
                hf_repo=hf_repo,
                local_dir=local_weights_dir,
                revision=hf_revision,
                expected_output_sha256=expected_output_sha256,
                expected_architecture=sam3_convert.DEFAULT_MLX_CHECKPOINT.architecture,
                verify_provenance=verify_hub_provenance,
            )
            provenance = {
                "status": (
                    "package-pinned" if verify_hub_provenance else "hub-unverified"
                ),
                "repo": hf_repo,
                "revision": hf_revision,
                "output_sha256": expected_output_sha256,
            }
    elif checkpoint_path is not None:
        provenance = {
            "status": "user-supplied-unverified",
            "repo": None,
            "revision": None,
            "output_sha256": None,
            "checkpoint_path": str(checkpoint_path),
        }

    if checkpoint_path is not None:
        _load_checkpoint(
            model,
            f"{checkpoint_path}",
            interactive_checkpoint_path=interactive_checkpoint_path,
            strict=strict_checkpoint_loading,
        )

    model.checkpoint_provenance = provenance
    return _setup_device_and_mode(model, device, eval_mode)


def build_sam3_video_predictor(
    *model_args: object,
    gpus_to_use: object | None = None,
    **model_kwargs: object,
) -> Sam3VideoPredictor:
    if gpus_to_use is not None:
        _raise_builder_unsupported(
            "sam3_mlx.model_builder.build_sam3_video_predictor(gpus_to_use)",
            reason="video-multi-gpu",
            detail="gpus_to_use is not supported by the MLX runtime.",
            alternative="gpus_to_use=None",
        )
    from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor

    predictor_cls = cast(_VideoPredictorFactory, Sam3VideoPredictor)
    return predictor_cls(*model_args, **model_kwargs)


def build_sam3_video_model(
    checkpoint_path: PathLikeStr | None = None,
    load_from_HF: bool = True,
    bpe_path: PathLikeStr | None = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device: ComputeDevice = "mlx",
    compile: bool = False,
    image_model: Sam3Image | None = None,
    image_size: int = 1008,
    image_mean: ImageStats = (0.5, 0.5, 0.5),
    image_std: ImageStats = (0.5, 0.5, 0.5),
    confidence_threshold: float = 0.5,
    hf_repo: str = sam3_convert.MLX_COMMUNITY_REPO,
    local_weights_dir: str | None = None,
    convert_from_pytorch: bool = False,
    enable_segmentation: bool = True,
    processor_factory: ProcessorFactory | None = None,
    frame_feature_cache_size: int = 4,
    conversion_source_revision: str | None = None,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    _validate_sam3_video_runtime_options(
        "sam3_mlx.model_builder.build_sam3_video_model",
        compile=compile,
        device=device,
        has_presence_token=has_presence_token,
        geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
        strict_state_dict_loading=strict_state_dict_loading,
        apply_temporal_disambiguation=apply_temporal_disambiguation,
    )
    if image_model is None:
        image_model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            eval_mode=True,
            checkpoint_path=checkpoint_path,
            load_from_HF=load_from_HF,
            hf_repo=hf_repo,
            local_weights_dir=local_weights_dir,
            convert_from_pytorch=convert_from_pytorch,
            enable_segmentation=enable_segmentation,
            enable_inst_interactivity=False,
            compile=compile,
            strict_checkpoint_loading=strict_state_dict_loading,
            conversion_source_revision=conversion_source_revision,
        )

    from sam3_mlx.model.sam3_video_inference import (
        Sam3VideoInferenceWithInstanceInteractivity,
    )

    model = Sam3VideoInferenceWithInstanceInteractivity(
        image_model=image_model,
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
        compile_model=compile,
        confidence_threshold=confidence_threshold,
        processor_factory=processor_factory,
        frame_feature_cache_size=frame_feature_cache_size,
    )
    return _setup_device_and_mode(model, device, eval_mode=True)


def _create_multiplex_maskmem_backbone(
    multiplex_count: int = 16,
) -> SimpleMaskEncoder:
    """Create the multiplex memory encoder with per-object mask channels."""
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3,
        stride=2,
        padding=1,
        interpol_size=[1152, 1152],
        multiplex_count=multiplex_count,
        starting_out_chan=4,
        input_channel_multiplier=2,
    )
    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)
    return SimpleMaskEncoder(
        out_dim=256,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )


def _create_multiplex_transformer(
    use_fa3: bool = False,
    use_rope_real: bool = False,
) -> TransformerWrapper:
    """Create the multiplex decoupled memory-attention transformer."""
    self_attention_rope = SimpleRoPEAttention(
        d_model=256,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    cross_attention_rope = SimpleRoPEAttention(
        d_model=256,
        num_heads=8,
        dropout_p=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    encoder_layer = DecoupledTransformerDecoderLayerv2(
        activation="gelu",
        d_model=256,
        num_heads=8,
        dropout=0.1,
        dim_feedforward=2048,
        pos_enc_at_attn=False,
        pre_norm=True,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention_rope=self_attention_rope,
        cross_attention_rope=cross_attention_rope,
    )
    encoder = TransformerEncoderDecoupledCrossAttention(
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        use_image_in_output=False,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
        batch_first=True,
    )
    return TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )


def _create_multiplex_tri_backbone(
    compile_mode: str | None = None,
    use_fa3: bool = False,
    use_rope_real: bool = False,
) -> Sam3TriViTDetNeck:
    """Create the tri-head vision backbone used by the multiplex model."""
    del use_fa3, use_rope_real
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    vit_backbone = _create_vit_backbone(compile_mode=compile_mode)
    return Sam3TriViTDetNeck(
        trunk=vit_backbone,
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0],
    )


def build_sam3_multiplex_video_model(
    checkpoint_path: PathLikeStr | None = None,
    load_from_HF: bool = True,
    multiplex_count: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = False,
    strict_state_dict_loading: bool = True,
    device: ComputeDevice = "mlx",
    compile: bool = False,
) -> Sam3VideoTrackingMultiplexDemo:
    _validate_mlx_device(device)
    if compile:
        _raise_compile_unsupported(
            "sam3_mlx.model_builder.build_sam3_multiplex_video_model(compile=True)"
        )
    if load_from_HF:
        _raise_builder_unsupported(
            "sam3_mlx.model_builder.build_sam3_multiplex_video_model(load_from_HF=True)",
            reason="video-multiplex",
            detail=(
                "Automatic SAM 3.1 multiplex checkpoint download/conversion is "
                "not wired into the MLX runtime yet."
            ),
            alternative="checkpoint_path=<local MLX checkpoint>, load_from_HF=False",
        )

    maskmem_backbone = _create_multiplex_maskmem_backbone(
        multiplex_count=multiplex_count
    )
    transformer = _create_multiplex_transformer(
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    tri_neck = _create_multiplex_tri_backbone(
        compile_mode=None,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    backbone = TriHeadVisionOnly(
        visual=tri_neck,
        n_features=256,
        scalp=0,
    )

    from sam3_mlx.model.video_tracking_multiplex_demo import (
        Sam3VideoTrackingMultiplexDemo,
    )

    model = Sam3VideoTrackingMultiplexDemo(
        backbone=backbone,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multiplex_controller=MultiplexController(
            multiplex_count=multiplex_count,
            eval_multiplex_count=multiplex_count,
        ),
        image_size=1008,
        backbone_stride=14,
        num_maskmem=7,
        use_high_res_features_in_sam=True,
        use_obj_ptrs_in_encoder=True,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_mlp_for_obj_ptr_proj=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        use_no_obj_ptr=True,
        use_linear_no_obj_ptr=True,
        no_obj_embed_spatial=True,
        sincos_tpos_enc=True,
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_multimask_token_for_obj_ptr=True,
        num_multimask_outputs=3,
        apply_sigmoid_to_mask_logits_for_mem_enc=True,
        sigmoid_scale_for_mem_enc=2.0,
        sigmoid_bias_for_mem_enc=-1.0,
        non_overlap_masks_for_mem_enc=False,
        add_output_suppression_embeddings=True,
        add_object_conditional_embeddings=False,
        condition_as_mask_input=True,
        condition_as_mask_input_fg=1.0,
        condition_as_mask_input_bg=0.0,
        use_maskmem_tpos_v2=True,
        save_image_features=True,
        randomness_fix=True,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        iou_prediction_use_sigmoid=False,
        forward_backbone_per_frame_for_eval=True,
        offload_output_to_cpu_for_eval=False,
        trim_past_non_cond_mem_for_eval=False,
        max_cond_frames_in_attn=4,
        is_dynamic_model=True,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        compile_all_components=False,
        use_memory_selection=False,
    )
    if checkpoint_path is not None:
        _load_multiplex_tracker_checkpoint(
            model,
            checkpoint_path,
            strict=strict_state_dict_loading,
        )
    return _setup_device_and_mode(model, device, eval_mode=True)


def _build_multiplex_detector_for_predictor(
    *,
    bpe_path: PathLikeStr,
    use_fa3: bool,
    use_rope_real: bool,
) -> Sam3MultiplexDetector:
    """Build the text-grounded detector used by the SAM 3.1 predictor wrapper."""
    bpe_path = os.fspath(bpe_path)
    tri_neck = _create_multiplex_tri_backbone(
        compile_mode=None,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackboneTri(scalp=0, visual=tri_neck, text=text_encoder)
    transformer = _create_sam3_transformer()
    segmentation_head = _create_segmentation_head()
    geometry_encoder = _create_geometry_encoder()
    dot_prod_scoring = _create_dot_product_scoring()

    from sam3_mlx.model.sam3_multiplex_detector import Sam3MultiplexDetector

    return Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        input_geometry_encoder=geometry_encoder,
        use_dot_prod_scoring=True,
        dot_prod_scoring=dot_prod_scoring,
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )


def _build_checkpoint_free_multiplex_predictor_model(
    *,
    bpe_path: PathLikeStr,
    max_num_objects: int,
    multiplex_count: int,
    use_fa3: bool,
    use_rope_real: bool,
    compile_model: bool,
    score_threshold_detection: float = 0.4,
    image_only_det_thresh: float = 0.5,
    suppress_det_close_to_boundary: bool = True,
) -> Sam3MultiplexTrackingWithInteractivity:
    """Assemble the checkpoint-free MLX version of the official SAM 3.1 stack."""
    tracker_model = build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        multiplex_count=multiplex_count,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
        strict_state_dict_loading=False,
        device="mlx",
        compile=False,
    )
    setattr(tracker_model, "backbone", None)

    from sam3_mlx.model.sam3_multiplex_base import Sam3MultiplexPredictorWrapper
    from sam3_mlx.model.sam3_multiplex_tracking import (
        Sam3MultiplexTrackingWithInteractivity,
    )

    tracker = Sam3MultiplexPredictorWrapper(
        model=tracker_model,
        per_obj_inference=False,
        fill_hole_area=0,
        is_multiplex=True,
        is_multiplex_dynamic=True,
    )
    detector = _build_multiplex_detector_for_predictor(
        bpe_path=bpe_path,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )

    demo_model = Sam3MultiplexTrackingWithInteractivity(
        tracker=tracker,
        detector=detector,
        score_threshold_detection=score_threshold_detection,
        image_only_det_thresh=image_only_det_thresh,
        det_nms_thresh=0.1,
        det_nms_use_iom=True,
        assoc_iou_thresh=0.1,
        new_det_thresh=0.65,
        hotstart_delay=15,
        hotstart_unmatch_thresh=8,
        hotstart_dup_thresh=8,
        suppress_unmatched_only_within_hotstart=False,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        suppress_det_close_to_boundary=suppress_det_close_to_boundary,
        fill_hole_area=0,
        recondition_every_nth_frame=16,
        use_iom_recondition=True,
        iom_thresh_recondition=0.5,
        masklet_confirmation_enable=True,
        reconstruction_bbox_iou_thresh=-1,
        reconstruction_bbox_det_score=0.8,
        max_num_objects=max_num_objects,
        postprocess_batch_size=16,
        use_batched_grounding=True,
        batched_grounding_batch_size=16,
        max_num_kboxes=0,
        sprinkle_removal_area=0,
        is_multiplex=True,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=compile_model,
    )
    demo_model.eval()
    return demo_model


def build_sam3_multiplex_video_predictor(
    checkpoint_path: PathLikeStr | None = None,
    bpe_path: PathLikeStr | None = None,
    max_num_objects: int = 16,
    multiplex_count: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = True,
    compile: bool = False,
    warm_up: bool = False,
    session_expiration_sec: int = 1200,
    default_output_prob_thresh: float = 0.5,
    async_loading_frames: bool = False,
    load_from_HF: bool = True,
    score_threshold_detection: float = 0.4,
    image_only_det_thresh: float = 0.5,
    suppress_det_close_to_boundary: bool = True,
    strict_state_dict_loading: bool = True,
) -> Sam3MultiplexVideoPredictor:
    if load_from_HF:
        _raise_builder_unsupported(
            "sam3_mlx.model_builder.build_sam3_multiplex_video_predictor(load_from_HF=True)",
            reason="video-multiplex",
            detail=(
                "Automatic SAM 3.1 multiplex checkpoint download/conversion is "
                "not wired into the MLX runtime yet."
            ),
            alternative="checkpoint_path=<local MLX checkpoint>, load_from_HF=False",
        )
    if compile:
        _raise_compile_unsupported(
            "sam3_mlx.model_builder.build_sam3_multiplex_video_predictor(compile=True)"
        )
    if bpe_path is None:
        bpe_path = _default_bpe_path()
    bpe_path = os.fspath(bpe_path)

    model = _build_checkpoint_free_multiplex_predictor_model(
        bpe_path=bpe_path,
        max_num_objects=max_num_objects,
        multiplex_count=multiplex_count,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
        compile_model=False,
        score_threshold_detection=score_threshold_detection,
        image_only_det_thresh=image_only_det_thresh,
        suppress_det_close_to_boundary=suppress_det_close_to_boundary,
    )
    if checkpoint_path is not None:
        _load_multiplex_checkpoint(
            model,
            checkpoint_path,
            strict=strict_state_dict_loading,
        )

    from sam3_mlx.model.sam3_multiplex_video_predictor import (
        Sam3MultiplexVideoPredictor,
    )

    return Sam3MultiplexVideoPredictor(
        model=model,
        session_expiration_sec=session_expiration_sec,
        default_output_prob_thresh=default_output_prob_thresh,
        async_loading_frames=async_loading_frames,
        warm_up=warm_up,
    )


def build_sam3_predictor(
    checkpoint_path: PathLikeStr | None = None,
    bpe_path: PathLikeStr | None = None,
    version: str = "sam3",
    compile: bool = False,
    warm_up: bool = False,
    max_num_objects: int = 16,
    multiplex_count: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = True,
    async_loading_frames: bool = False,
    load_from_HF: bool = True,
    **kwargs: object,
) -> Sam3VideoPredictor | Sam3MultiplexVideoPredictor:
    if version == "sam3.1":
        return build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            max_num_objects=max_num_objects,
            multiplex_count=multiplex_count,
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
            compile=compile,
            warm_up=warm_up,
            async_loading_frames=async_loading_frames,
            load_from_HF=load_from_HF,
            **cast(_MultiplexPredictorKwargs, kwargs),
        )
    if version == "sam3":
        return build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            compile=compile,
            async_loading_frames=async_loading_frames,
            load_from_HF=load_from_HF,
            **kwargs,
        )
    raise ValueError(f"Unknown version: {version!r}. Use 'sam3' or 'sam3.1'.")
