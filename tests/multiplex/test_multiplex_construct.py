import pytest
from mlx.utils import tree_flatten

from sam3_mlx import Sam3MlxUnsupportedError
from sam3_mlx.model.multiplex_utils import UnsupportedMultiplexRuntimeError
from sam3_mlx.model.necks import Sam3TriViTDetNeck
from sam3_mlx.model.video_tracking_multiplex_demo import (
    Sam3VideoTrackingMultiplexDemo,
)
from sam3_mlx.model_builder import (
    _create_multiplex_maskmem_backbone,
    _create_multiplex_transformer,
    _create_multiplex_tri_backbone,
    build_sam3_multiplex_video_model,
)


def test_multiplex_component_builders_construct_official_shapes():
    maskmem = _create_multiplex_maskmem_backbone(multiplex_count=4)
    transformer = _create_multiplex_transformer(use_fa3=False, use_rope_real=False)
    tri_neck = _create_multiplex_tri_backbone()

    assert maskmem.mask_downsampler.multiplex_count == 8
    assert maskmem.out_proj.__class__.__name__ == "Identity"
    assert transformer.decoder is None
    assert transformer.d_model == 256
    assert transformer.encoder.num_layers == 4
    assert transformer.encoder.use_image_in_output is False
    assert isinstance(tri_neck, Sam3TriViTDetNeck)


def test_multiplex_video_model_builder_constructs_checkpoint_free_mlx_shell():
    model = build_sam3_multiplex_video_model(
        load_from_HF=False,
        checkpoint_path=None,
        multiplex_count=4,
        use_fa3=False,
        use_rope_real=False,
        device="mlx",
    )

    assert isinstance(model, Sam3VideoTrackingMultiplexDemo)
    assert model.hidden_dim == 256
    assert model.mem_dim == 256
    assert model.multiplex_count == 4
    assert model.num_feature_levels == 3
    assert model.low_res_mask_size == 1008 // 14 * 4
    assert model.sam_mask_decoder.multiplex_count == 4
    assert model.sam_mask_decoder.num_multimask_outputs == 3
    assert model.interactive_sam_mask_decoder.pred_obj_scores is True
    assert model.use_obj_ptrs_in_encoder is True
    assert model.obj_ptr_tpos_proj.weight.shape == (256, 256)
    params = dict(tree_flatten(model.parameters()))
    assert params["maskmem_tpos_enc"].shape == (7, 1, 1, 256)
    assert params["output_valid_embed"].shape == (4, 256)
    assert params["output_invalid_embed"].shape == (4, 256)


def test_multiplex_demo_init_state_builds_cached_feature_state():
    model = build_sam3_multiplex_video_model(
        load_from_HF=False,
        checkpoint_path=None,
        multiplex_count=4,
        use_fa3=False,
        use_rope_real=False,
        device="mlx",
    )
    cached_features = {1: ("image", {"features": "cached"})}

    state = model.init_state(
        video_height=720,
        video_width=1280,
        num_frames=3,
        cached_features=cached_features,
    )

    assert state["video_height"] == 720
    assert state["video_width"] == 1280
    assert state["num_frames"] == 3
    assert state["images"] is None
    assert state["cached_features"] is cached_features
    assert state["device"] == "mlx"
    assert state["storage_device"] == "mlx"
    assert state["output_dict"] == {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    assert state["consolidated_frame_inds"] == {
        "cond_frame_outputs": set(),
        "non_cond_frame_outputs": set(),
    }
    assert state["multiplex_state"] is None
    assert state["user_refined_frames_per_obj"] == {}
    assert state["tracking_has_started"] is False


def test_multiplex_demo_init_state_loads_dummy_video_resource():
    model = build_sam3_multiplex_video_model(
        load_from_HF=False,
        checkpoint_path=None,
        multiplex_count=4,
        use_fa3=False,
        use_rope_real=False,
        device="mlx",
    )

    state = model.init_state(video_path="<load-zero-video-2>")

    assert state["video_height"] == 480
    assert state["video_width"] == 640
    assert state["num_frames"] == 2
    assert tuple(state["images"].shape) == (
        2,
        3,
        model.image_size,
        model.image_size,
    )
    assert state["cached_features"] == {}
    assert state["output_dict"] == {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }


def test_multiplex_demo_init_state_validates_inputs_and_offload_boundary():
    model = build_sam3_multiplex_video_model(
        load_from_HF=False,
        checkpoint_path=None,
        multiplex_count=4,
        use_fa3=False,
        use_rope_real=False,
        device="mlx",
    )

    with pytest.raises(TypeError, match="cached_features must be a dict"):
        model.init_state(
            video_height=720,
            video_width=1280,
            num_frames=3,
            cached_features=[],
        )

    with pytest.raises(ValueError, match="num_frames must be a positive integer"):
        model.init_state(video_height=720, video_width=1280, num_frames=0)

    with pytest.raises(
        ValueError,
        match="video_height, video_width, and num_frames are required",
    ):
        model.init_state()

    with pytest.raises(
        UnsupportedMultiplexRuntimeError,
        match="Sam3VideoTrackingMultiplexDemo.init_state\\(offload\\)",
    ):
        model.init_state(
            video_height=720,
            video_width=1280,
            num_frames=3,
            offload_state_to_cpu=True,
        )

    with pytest.raises(
        UnsupportedMultiplexRuntimeError,
        match="VideoTrackingMultiplexDemo.init_state\\(async_loading_frames\\)",
    ):
        model.init_state(
            video_path="<load-zero-video-2>",
            async_loading_frames=True,
        )


def test_multiplex_model_builder_keeps_hf_download_fail_fast():
    with pytest.raises(Sam3MlxUnsupportedError) as exc_info:
        build_sam3_multiplex_video_model(
            load_from_HF=True,
            checkpoint_path=None,
            use_fa3=False,
            device="mlx",
        )

    assert exc_info.value.reason == "video-multiplex"
    assert "load_from_HF=True" in exc_info.value.feature


def test_multiplex_model_builder_rejects_fa3_for_mlx():
    with pytest.raises(Sam3MlxUnsupportedError) as exc_info:
        build_sam3_multiplex_video_model(
            load_from_HF=False,
            checkpoint_path=None,
            use_fa3=True,
            device="mlx",
        )

    assert exc_info.value.reason == "flash-attn-3"


def test_multiplex_constructed_runtime_still_fails_fast():
    model = build_sam3_multiplex_video_model(
        load_from_HF=False,
        checkpoint_path=None,
        use_fa3=False,
        device="mlx",
    )

    with pytest.raises(UnsupportedMultiplexRuntimeError):
        model.forward()
