from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.convert import MLX_COMMUNITY_REPO
from sam3_mlx.model.sam3_base_predictor import Sam3BasePredictor


class Sam3VideoPredictor(Sam3BasePredictor):
    """MLX video predictor wrapper matching the official SAM3 request API.

    The predictor owns session/request behavior. The model object underneath it
    owns frame loading, prompting, and propagation semantics.
    """

    def __init__(
        self,
        checkpoint_path=None,
        bpe_path=None,
        has_presence_token=True,
        geo_encoder_use_img_cross_attn=True,
        strict_state_dict_loading=True,
        async_loading_frames: bool = False,
        video_loader_type: str = "cv2",
        apply_temporal_disambiguation: bool = True,
        compile: bool = False,
        *,
        image_model=None,
        video_model=None,
        model=None,
        resolution: int = 1008,
        confidence_threshold: float = 0.5,
        device="mlx",
        load_from_HF=True,
        hf_repo=MLX_COMMUNITY_REPO,
        local_weights_dir=None,
        convert_from_pytorch=False,
        enable_segmentation=True,
        processor_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__()
        self.async_loading_frames = async_loading_frames
        self.video_loader_type = video_loader_type
        self.default_output_prob_thresh = confidence_threshold
        if model is not None:
            if image_model is not None:
                raise ValueError("Use only one of model= or image_model=.")
            image_model = model
        from sam3_mlx.model_builder import (
            _validate_sam3_video_runtime_options,
            build_sam3_video_model,
        )

        _validate_sam3_video_runtime_options(
            "sam3_mlx.model.sam3_video_predictor.Sam3VideoPredictor",
            compile=compile,
            device=device,
            has_presence_token=has_presence_token,
            geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
            strict_state_dict_loading=strict_state_dict_loading,
            apply_temporal_disambiguation=apply_temporal_disambiguation,
        )
        if checkpoint_path is not None and (
            image_model is not None or video_model is not None
        ):
            raise ValueError(
                "checkpoint_path cannot be used with image_model=, model=, "
                "or video_model=."
            )
        if video_model is not None:
            self.model = video_model
            return

        if image_model is None:
            self.model = build_sam3_video_model(
                checkpoint_path=checkpoint_path,
                load_from_HF=load_from_HF,
                bpe_path=bpe_path,
                has_presence_token=has_presence_token,
                geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
                strict_state_dict_loading=strict_state_dict_loading,
                apply_temporal_disambiguation=apply_temporal_disambiguation,
                device=device,
                compile=compile,
                image_size=resolution,
                confidence_threshold=confidence_threshold,
                hf_repo=hf_repo,
                local_weights_dir=local_weights_dir,
                convert_from_pytorch=convert_from_pytorch,
                enable_segmentation=enable_segmentation,
                processor_factory=processor_factory,
            )
            return

        self.model = build_sam3_video_model(
            has_presence_token=has_presence_token,
            geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
            strict_state_dict_loading=strict_state_dict_loading,
            apply_temporal_disambiguation=apply_temporal_disambiguation,
            device=device,
            compile=compile,
            image_model=image_model,
            image_size=resolution,
            confidence_threshold=confidence_threshold,
            processor_factory=processor_factory,
        )

    def remove_object(
        self,
        session_id: str,
        frame_idx: int = 0,
        obj_id: int = 0,
        is_user_action: bool = True,
    ) -> dict[str, bool]:
        session = self._get_session(session_id)
        self._extend_expiration_time(session)
        self.model.remove_object(
            inference_state=session["state"],
            obj_id=obj_id,
            frame_idx=frame_idx,
            is_user_action=is_user_action,
        )
        return {"is_success": True}

    def _get_session_stats(self) -> str:
        live_session_strs = []
        for sid, session in self._all_inference_states.items():
            num_frames = session["state"]["num_frames"]
            live_session_strs.append(f"'{sid}' ({num_frames} frames)")
        return f"live sessions: [{', '.join(live_session_strs)}], runtime: MLX"

    def _get_torch_and_gpu_properties(self) -> str:
        return "runtime: MLX; torch/non-MLX properties are not used by sam3_mlx"


class Sam3VideoPredictorMultiGPU(Sam3VideoPredictor):
    """Official SAM3 multi-GPU predictor name reserved as an unsupported shim."""

    def __init__(self, *model_args, gpus_to_use=None, **model_kwargs) -> None:
        del model_args, gpus_to_use, model_kwargs
        raise_unsupported(
            "sam3_mlx.model.sam3_video_predictor.Sam3VideoPredictorMultiGPU",
            reason="video-multi-gpu",
            detail=(
                "The official SAM3 class depends on the Torch-only multi-GPU video "
                "predictor stack, including multiprocessing and torch.distributed/NCCL."
            ),
            alternative="Sam3VideoPredictor",
        )
