"""SAM 3.1 multiplex model assembly for the MLX runtime."""

from __future__ import annotations

from sam3_mlx.checkpoint import (
    _load_multiplex_checkpoint,
    _load_multiplex_tracker_checkpoint,
)
from sam3_mlx.model.decoder import (
    DecoupledTransformerDecoderLayerv2,
    SimpleRoPEAttention,
    TransformerEncoderDecoupledCrossAttention,
)
from sam3_mlx.model.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from sam3_mlx.model.model_misc import TransformerWrapper
from sam3_mlx.model.multiplex_utils import MultiplexController
from sam3_mlx.model.necks import Sam3TriViTDetNeck
from sam3_mlx.model.position_encoding import PositionEmbeddingSine
from sam3_mlx.model.vl_combiner import SAM3VLBackboneTri, TriHeadVisionOnly
from sam3_mlx.model_builder import (
    _create_dot_product_scoring,
    _create_geometry_encoder,
    _create_position_encoding,
    _create_sam3_transformer,
    _create_segmentation_head,
    _create_text_encoder,
    _create_vit_backbone,
    _default_bpe_path,
    _raise_builder_unsupported,
    _raise_compile_unsupported,
    _setup_device_and_mode,
    _validate_mlx_device,
)


def _create_multiplex_maskmem_backbone(multiplex_count: int = 16):
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


def _create_multiplex_transformer(use_fa3: bool = False, use_rope_real: bool = False):
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
    compile_mode=None,
    use_fa3: bool = False,
    use_rope_real: bool = False,
):
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
    checkpoint_path: str | None = None,
    load_from_HF=True,
    multiplex_count: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = False,
    strict_state_dict_loading: bool = True,
    device="mlx",
    compile=False,
):
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
            strict_state_dict_loading=strict_state_dict_loading,
        )
    return _setup_device_and_mode(model, device, eval_mode=True)


def _build_multiplex_detector_for_predictor(
    *,
    bpe_path: str,
    use_fa3: bool,
    use_rope_real: bool,
):
    """Build the text-grounded detector used by the SAM 3.1 predictor wrapper."""
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
    bpe_path: str,
    max_num_objects: int,
    multiplex_count: int,
    use_fa3: bool,
    use_rope_real: bool,
    compile_model: bool,
    score_threshold_detection: float = 0.4,
    image_only_det_thresh: float = 0.5,
    suppress_det_close_to_boundary: bool = True,
):
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
    tracker_model.backbone = None

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
    checkpoint_path: str | None = None,
    bpe_path: str | None = None,
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
):
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
        _load_multiplex_checkpoint(model, checkpoint_path)

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
