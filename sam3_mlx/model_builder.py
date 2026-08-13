from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
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
import mlx.utils as mlx_utils
from mlx import nn

from sam3_mlx._device import is_mlx_runtime_device
from sam3_mlx._unsupported import raise_unsupported
import sam3_mlx.convert as sam3_convert
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
CheckpointSource: TypeAlias = Mapping[str, object]
NormalizedWeights: TypeAlias = dict[str, mx.array]

_ModelT = TypeVar("_ModelT", bound="_SupportsEval")


class _SupportsEval(Protocol):
    def eval(self) -> object: ...


class _HasParameters(Protocol):
    def parameters(self) -> object: ...


class _CheckpointLoadable(_HasParameters, Protocol):
    def load_weights(
        self,
        file_or_weights: str | list[tuple[str, mx.array]],
        strict: bool = True,
    ) -> object: ...


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


class _MlxEval(Protocol):
    def __call__(self, *values: object) -> None: ...


class _MlxLoad(Protocol):
    def __call__(self, file: PathLikeStr, /) -> object: ...


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


_tree_flatten = cast(_TreeFlatten, getattr(mlx_utils, "tree_flatten"))
_normalize_sam3_image_weight_layout = cast(
    _NormalizeImageWeightLayout,
    getattr(sam3_convert, "normalize_sam3_image_weight_layout"),
)
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


def _has_inst_interactivity(model: object) -> bool:
    return getattr(model, "inst_interactive_predictor", None) is not None


def _transpose_4d(value: mx.array, axes: tuple[int, int, int, int]) -> mx.array:
    transpose = cast(_Transpose4D, getattr(value, "transpose"))
    return transpose(*axes)


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
) -> SAM3VLBackbone:
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


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
    if compile:
        _raise_compile_unsupported(
            "sam3_mlx.model_builder.build_sam3_image_model(compile=True)"
        )
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

    backbone = _create_vl_backbone(vision_encoder, text_encoder)

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
