from __future__ import annotations

import os
from typing import NoReturn

import mlx.nn as nn

from sam3_mlx._device import is_mlx_runtime_device
from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.checkpoint import (
    Sam3CheckpointLoadReport as Sam3CheckpointLoadReport,
)
from sam3_mlx.checkpoint import (
    Sam3CheckpointShapeMismatch as Sam3CheckpointShapeMismatch,
)
from sam3_mlx.checkpoint import (
    _audit_sam3_image_checkpoint_load as _audit_sam3_image_checkpoint_load,
)
from sam3_mlx.checkpoint import (
    _load_checkpoint,
    _load_tracker_checkpoint,
)
from sam3_mlx.checkpoint import (
    _load_multiplex_checkpoint as _load_multiplex_checkpoint,
)
from sam3_mlx.checkpoint import (
    _load_multiplex_tracker_checkpoint as _load_multiplex_tracker_checkpoint,
)
from sam3_mlx.checkpoint import (
    _normalize_inst_interactive_weights as _normalize_inst_interactive_weights,
)
from sam3_mlx.checkpoint import (
    _normalize_sam3_image_weights as _normalize_sam3_image_weights,
)
from sam3_mlx.checkpoint import (
    _normalize_sam31_multiplex_tracker_weights as _normalize_sam31_multiplex_tracker_weights,
)
from sam3_mlx.checkpoint import (
    _normalize_sam31_multiplex_weights as _normalize_sam31_multiplex_weights,
)
from sam3_mlx.checkpoint import (
    _normalize_tracker_checkpoint_weights as _normalize_tracker_checkpoint_weights,
)
from sam3_mlx.checkpoint import (
    download_ckpt_from_hf as download_ckpt_from_hf,
)
from sam3_mlx.convert import (
    MLX_COMMUNITY_REPO,
    PYTORCH_REPO,
    download_and_convert,
    load_from_hub,
)
from sam3_mlx.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
)
from sam3_mlx.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3_mlx.model.geometry_encoders import SequenceGeometryEncoder
from sam3_mlx.model.maskformer_segmentation import (
    PixelDecoder,
    UniversalSegmentationHead,
)
from sam3_mlx.model.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from sam3_mlx.model.model_misc import (
    MLP,
    DotProductScoring,
    TransformerWrapper,
)
from sam3_mlx.model.model_misc import (
    MultiheadAttentionWrapper as MultiheadAttention,
)
from sam3_mlx.model.necks import Sam3DualViTDetNeck
from sam3_mlx.model.position_encoding import PositionEmbeddingSine
from sam3_mlx.model.sam1_task_predictor import (
    SAM3InteractiveImageModel,
    SAM3InteractiveImagePredictor,
)
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3_mlx.model.text_encoder_ve import VETextEncoder
from sam3_mlx.model.tokenizer_ve import SimpleTokenizer
from sam3_mlx.model.vitdet import ViT
from sam3_mlx.model.vl_combiner import (
    SAM3VLBackbone,
)
from sam3_mlx.sam.transformer import RoPEAttention


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
) -> NoReturn:
    raise_unsupported(
        feature,
        reason=reason,
        alternative=alternative,
        detail=detail,
    )


def _raise_compile_unsupported(feature: str) -> NoReturn:
    _raise_builder_unsupported(
        feature,
        reason="torch-compile",
        detail="torch.compile is not part of the sam3_mlx runtime.",
        alternative="compile=False",
    )


def _normalize_mlx_api_device(device) -> str:
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


def _validate_mlx_device(device) -> None:
    _normalize_mlx_api_device(device)


def _validate_sam3_video_runtime_options(
    feature_prefix: str,
    *,
    compile: bool,
    device,
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


def _setup_device_and_mode(model, device, eval_mode):
    """Setup the explicit MLX device contract and evaluation mode."""

    _validate_mlx_device(device)
    if eval_mode and hasattr(model, "eval"):
        model.eval()
    return model


def _create_position_encoding(precompute_resolution=None):
    """Create a PositionEmbeddingSine block (used by the backbone and geometry encoder)."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(compile_mode=None):
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


def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False):
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(vit_neck, text_encoder):
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create the transformer encoder."""

    def encoder_layer():
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

    def decoder_layer():
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


def _create_dot_product_scoring():
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


def _create_segmentation_head():
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
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def _create_geometry_encoder():
    geo_pos_enc = _create_position_encoding()

    def geo_layer():
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


def _create_inst_interactive_predictor():
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
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
    inst_interactive_predictor=None,
):
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": inst_interactive_predictor,
    }

    model = Sam3Image(**common_params)
    return model


def _unsupported_tracker_builder(feature: str):
    _raise_builder_unsupported(
        f"sam3_mlx.model_builder.{feature}",
        reason="video-multiplex",
        detail=(
            "This builder depends on the official Torch-only tracker or multiplex "
            "runtime. The current MLX port exposes the image model and selected-frame "
            "video API slice."
        ),
        alternative="build_sam3_predictor(version='sam3')",
    )


def _create_tracker_maskmem_backbone():
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


def _create_tracker_transformer():
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
    compile_mode=None,
    checkpoint_path=None,
):
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

    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()
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
    compile_mode=None, enable_inst_interactivity=True
) -> Sam3DualViTDetNeck:
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    vit_backbone = _create_vit_backbone(compile_mode=compile_mode)

    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    return vit_neck


def _create_sam3_transformer(has_presence_token: bool = True):
    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def build_sam3_image_model(
    bpe_path=None,
    device="mlx",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
    hf_repo=MLX_COMMUNITY_REPO,
    local_weights_dir=None,
    convert_from_pytorch=False,
    interactive_checkpoint_path=None,
):
    if compile:
        _raise_compile_unsupported(
            "sam3_mlx.model_builder.build_sam3_image_model(compile=True)"
        )
    _validate_mlx_device(device)
    if checkpoint_path is None and convert_from_pytorch and not load_from_HF:
        raise ValueError("convert_from_pytorch=True requires load_from_HF=True.")
    if interactive_checkpoint_path is not None:
        if not enable_inst_interactivity:
            raise ValueError(
                "interactive_checkpoint_path requires enable_inst_interactivity=True."
            )
        if checkpoint_path is None and not load_from_HF:
            raise ValueError(
                "interactive_checkpoint_path requires base image weights via "
                "checkpoint_path or load_from_HF=True."
            )
    if bpe_path is None:
        bpe_path = _default_bpe_path()

    vision_encoder = _create_vision_backbone(
        compile_mode=compile, enable_inst_interactivity=enable_inst_interactivity
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

    if checkpoint_path is None and load_from_HF:
        if convert_from_pytorch:
            checkpoint_path = download_and_convert(
                hf_repo=PYTORCH_REPO,
                mlx_path=local_weights_dir or "sam3-mod-weights",
            )
        else:
            checkpoint_path = load_from_hub(
                hf_repo=hf_repo,
                local_dir=local_weights_dir,
            )

    if checkpoint_path is not None:
        _load_checkpoint(
            model,
            f"{checkpoint_path}",
            interactive_checkpoint_path=interactive_checkpoint_path,
        )

    return _setup_device_and_mode(model, device, eval_mode)


def build_sam3_video_predictor(
    *model_args,
    gpus_to_use=None,
    **model_kwargs,
):
    if gpus_to_use is not None:
        _raise_builder_unsupported(
            "sam3_mlx.model_builder.build_sam3_video_predictor(gpus_to_use)",
            reason="video-multi-gpu",
            detail="gpus_to_use is not supported by the MLX runtime.",
            alternative="gpus_to_use=None",
        )
    from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor

    return Sam3VideoPredictor(*model_args, **model_kwargs)


def build_sam3_video_model(
    checkpoint_path: str | None = None,
    load_from_HF=True,
    bpe_path: str | None = None,
    has_presence_token=True,
    geo_encoder_use_img_cross_attn=True,
    strict_state_dict_loading=True,
    apply_temporal_disambiguation=True,
    device="mlx",
    compile=False,
    image_model=None,
    image_size=1008,
    image_mean=(0.5, 0.5, 0.5),
    image_std=(0.5, 0.5, 0.5),
    confidence_threshold=0.5,
    hf_repo=MLX_COMMUNITY_REPO,
    local_weights_dir=None,
    convert_from_pytorch=False,
    enable_segmentation=True,
    processor_factory=None,
):
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
    )
    return _setup_device_and_mode(model, device, eval_mode=True)


from sam3_mlx.multiplex_builder import (  # noqa: E402 - late import breaks a builder cycle
    _build_checkpoint_free_multiplex_predictor_model as _build_checkpoint_free_multiplex_predictor_model,
)
from sam3_mlx.multiplex_builder import (  # noqa: E402 - late import breaks a builder cycle
    _build_multiplex_detector_for_predictor as _build_multiplex_detector_for_predictor,
)
from sam3_mlx.multiplex_builder import (  # noqa: E402 - late import breaks a builder cycle
    _create_multiplex_maskmem_backbone as _create_multiplex_maskmem_backbone,
)
from sam3_mlx.multiplex_builder import (  # noqa: E402 - late import breaks a builder cycle
    _create_multiplex_transformer as _create_multiplex_transformer,
)
from sam3_mlx.multiplex_builder import (  # noqa: E402 - late import breaks a builder cycle
    _create_multiplex_tri_backbone as _create_multiplex_tri_backbone,
)
from sam3_mlx.multiplex_builder import (  # noqa: E402 - late import breaks a builder cycle
    build_sam3_multiplex_video_model as build_sam3_multiplex_video_model,
)
from sam3_mlx.multiplex_builder import (  # noqa: E402 - late import breaks a builder cycle
    build_sam3_multiplex_video_predictor as build_sam3_multiplex_video_predictor,
)


def build_sam3_predictor(
    checkpoint_path=None,
    bpe_path=None,
    version="sam3",
    compile=False,
    warm_up=False,
    max_num_objects=16,
    multiplex_count=16,
    use_fa3=False,
    use_rope_real=True,
    async_loading_frames=False,
    load_from_HF=True,
    **kwargs,
):
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
            **kwargs,
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
