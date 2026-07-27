from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.convert import MLX_COMMUNITY_REPO
from sam3_mlx.model.lifecycle_predictor import LifecycleSafeSam3BasePredictor


class Sam3VideoPredictor(LifecycleSafeSam3BasePredictor):
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
        conversion_source_revision=None,
        enable_segmentation=True,
        processor_factory: Callable[..., Any] | None = None,
        frame_feature_cache_size: int = 4,
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
                conversion_source_revision=conversion_source_revision,
                enable_segmentation=enable_segmentation,
                processor_factory=processor_factory,
                frame_feature_cache_size=frame_feature_cache_size,
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
            frame_feature_cache_size=frame_feature_cache_size,
        )

    def start_session(
        self,
        resource_path,
        session_id: str | None = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict[str, str]:
        # The selected-frame image-folder path is intentionally bounded and lazy.
        # The legacy async folder loader eagerly retains every decoded frame, so
        # accepting this combination would silently violate the release memory
        # contract. Other resources retain their existing loader-specific guards.
        if (
            self.async_loading_frames
            and isinstance(resource_path, (str, Path))
            and Path(resource_path).is_dir()
        ):
            raise_unsupported(
                "Sam3VideoPredictor.start_session(async image-folder loading)",
                reason="port-gap",
                detail=(
                    "Selected-frame image-folder sessions use a bounded on-demand "
                    "host cache. The legacy async preloader retains every frame and "
                    "would violate that bounded-memory contract."
                ),
                alternative="Construct the predictor with async_loading_frames=False.",
            )
        return super().start_session(
            resource_path=resource_path,
            session_id=session_id,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )

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
