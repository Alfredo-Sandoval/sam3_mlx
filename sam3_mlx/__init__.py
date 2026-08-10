# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version as _package_version
from os import PathLike
from typing import TYPE_CHECKING, Literal, Protocol, cast
import warnings

if TYPE_CHECKING:
    from sam3_mlx._unsupported import Sam3MlxUnsupportedError
    from sam3_mlx.model.sam3_image import Sam3Image
    from sam3_mlx.model.sam3_multiplex_video_predictor import Sam3MultiplexVideoPredictor
    from sam3_mlx.model.sam3_tracking_predictor import Sam3TrackerPredictor
    from sam3_mlx.model.sam3_video_inference import (
        Sam3VideoInferenceWithInstanceInteractivity,
    )
    from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor


class _BuildTrackerFn(Protocol):
    def __call__(
        self,
        apply_temporal_disambiguation: bool,
        with_backbone: bool = False,
        compile_mode: str | bool | None = None,
        checkpoint_path: str | PathLike[str] | None = None,
    ) -> Sam3TrackerPredictor: ...


class _BuildSam3ImageModelFn(Protocol):
    def __call__(
        self,
        bpe_path: str | PathLike[str] | None = None,
        device: str = "mlx",
        eval_mode: bool = True,
        checkpoint_path: str | PathLike[str] | None = None,
        load_from_HF: bool = True,
        enable_segmentation: bool = True,
        enable_inst_interactivity: bool = False,
        compile: bool = False,
        hf_repo: str = "",
        hf_revision: str = "",
        local_weights_dir: str | PathLike[str] | None = None,
        convert_from_pytorch: bool = False,
        interactive_checkpoint_path: str | PathLike[str] | None = None,
        strict_checkpoint_loading: bool = True,
        conversion_source_revision: str | None = None,
        expected_output_sha256: str | None = None,
        verify_hub_provenance: bool = True,
    ) -> Sam3Image: ...


class _BuildSam3PredictorFn(Protocol):
    def __call__(
        self,
        checkpoint_path: str | PathLike[str] | None = None,
        bpe_path: str | PathLike[str] | None = None,
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
    ) -> Sam3MultiplexVideoPredictor | Sam3VideoPredictor: ...


class _BuildSam3VideoModelFn(Protocol):
    def __call__(
        self,
        checkpoint_path: str | None = None,
        load_from_HF: bool = True,
        bpe_path: str | None = None,
        has_presence_token: bool = True,
        geo_encoder_use_img_cross_attn: bool = True,
        strict_state_dict_loading: bool = True,
        apply_temporal_disambiguation: bool = True,
        device: str = "mlx",
        compile: bool = False,
        image_model: object | None = None,
        image_size: int = 1008,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        confidence_threshold: float = 0.5,
        hf_repo: str = "",
        local_weights_dir: str | PathLike[str] | None = None,
        convert_from_pytorch: bool = False,
        enable_segmentation: bool = True,
        processor_factory: object | None = None,
        frame_feature_cache_size: int = 4,
        conversion_source_revision: str | None = None,
    ) -> Sam3VideoInferenceWithInstanceInteractivity: ...


class _BuildSam3VideoPredictorFn(Protocol):
    def __call__(
        self,
        *model_args: object,
        gpus_to_use: object | None = None,
        **model_kwargs: object,
    ) -> Sam3VideoPredictor: ...


class _DownloadCheckpointFn(Protocol):
    def __call__(self, version: Literal["sam3", "sam3.1"] = "sam3") -> str: ...


try:
    __version__ = _package_version("sam3_mlx")
except PackageNotFoundError:
    try:
        __version__ = _package_version("sam3-mlx")
    except PackageNotFoundError:
        __version__ = "0+unknown"

__all__ = [
    "Sam3MlxUnsupportedError",
    "build_tracker",
    "build_sam3_image_model",
    "build_sam3_predictor",
    "build_sam3_video_model",
    "build_sam3_video_predictor",
    "download_ckpt_from_hf",
]

_EXPERIMENTAL_EXPORTS = {
    "build_sam3_multiplex_video_model",
    "build_sam3_multiplex_video_predictor",
}


def _load_model_builder_attr(name: str) -> object:
    return getattr(importlib.import_module("sam3_mlx.model_builder"), name)


def build_tracker(
    apply_temporal_disambiguation: bool,
    with_backbone: bool = False,
    compile_mode: str | bool | None = None,
    checkpoint_path: str | PathLike[str] | None = None,
) -> Sam3TrackerPredictor:
    build = cast(_BuildTrackerFn, _load_model_builder_attr("build_tracker"))
    return build(
        apply_temporal_disambiguation=apply_temporal_disambiguation,
        with_backbone=with_backbone,
        compile_mode=compile_mode,
        checkpoint_path=checkpoint_path,
    )


def build_sam3_image_model(
    bpe_path: str | PathLike[str] | None = None,
    device: str = "mlx",
    eval_mode: bool = True,
    checkpoint_path: str | PathLike[str] | None = None,
    load_from_HF: bool = True,
    enable_segmentation: bool = True,
    enable_inst_interactivity: bool = False,
    compile: bool = False,
    hf_repo: str = "",
    hf_revision: str = "",
    local_weights_dir: str | PathLike[str] | None = None,
    convert_from_pytorch: bool = False,
    interactive_checkpoint_path: str | PathLike[str] | None = None,
    strict_checkpoint_loading: bool = True,
    conversion_source_revision: str | None = None,
    expected_output_sha256: str | None = None,
    verify_hub_provenance: bool = True,
) -> Sam3Image:
    build = cast(
        _BuildSam3ImageModelFn,
        _load_model_builder_attr("build_sam3_image_model"),
    )
    return build(
        bpe_path=bpe_path,
        device=device,
        eval_mode=eval_mode,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_HF,
        enable_segmentation=enable_segmentation,
        enable_inst_interactivity=enable_inst_interactivity,
        compile=compile,
        hf_repo=hf_repo,
        hf_revision=hf_revision,
        local_weights_dir=local_weights_dir,
        convert_from_pytorch=convert_from_pytorch,
        interactive_checkpoint_path=interactive_checkpoint_path,
        strict_checkpoint_loading=strict_checkpoint_loading,
        conversion_source_revision=conversion_source_revision,
        expected_output_sha256=expected_output_sha256,
        verify_hub_provenance=verify_hub_provenance,
    )


def build_sam3_predictor(
    checkpoint_path: str | PathLike[str] | None = None,
    bpe_path: str | PathLike[str] | None = None,
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
) -> Sam3MultiplexVideoPredictor | Sam3VideoPredictor:
    build = cast(_BuildSam3PredictorFn, _load_model_builder_attr("build_sam3_predictor"))
    return build(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        version=version,
        compile=compile,
        warm_up=warm_up,
        max_num_objects=max_num_objects,
        multiplex_count=multiplex_count,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
        async_loading_frames=async_loading_frames,
        load_from_HF=load_from_HF,
        **kwargs,
    )


def build_sam3_video_model(
    checkpoint_path: str | None = None,
    load_from_HF: bool = True,
    bpe_path: str | None = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device: str = "mlx",
    compile: bool = False,
    image_model: object | None = None,
    image_size: int = 1008,
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    confidence_threshold: float = 0.5,
    hf_repo: str = "",
    local_weights_dir: str | PathLike[str] | None = None,
    convert_from_pytorch: bool = False,
    enable_segmentation: bool = True,
    processor_factory: object | None = None,
    frame_feature_cache_size: int = 4,
    conversion_source_revision: str | None = None,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    build = cast(
        _BuildSam3VideoModelFn,
        _load_model_builder_attr("build_sam3_video_model"),
    )
    return build(
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_HF,
        bpe_path=bpe_path,
        has_presence_token=has_presence_token,
        geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
        strict_state_dict_loading=strict_state_dict_loading,
        apply_temporal_disambiguation=apply_temporal_disambiguation,
        device=device,
        compile=compile,
        image_model=image_model,
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
        confidence_threshold=confidence_threshold,
        hf_repo=hf_repo,
        local_weights_dir=local_weights_dir,
        convert_from_pytorch=convert_from_pytorch,
        enable_segmentation=enable_segmentation,
        processor_factory=processor_factory,
        frame_feature_cache_size=frame_feature_cache_size,
        conversion_source_revision=conversion_source_revision,
    )


def build_sam3_video_predictor(
    *model_args: object,
    gpus_to_use: object | None = None,
    **model_kwargs: object,
) -> Sam3VideoPredictor:
    build = cast(
        _BuildSam3VideoPredictorFn,
        _load_model_builder_attr("build_sam3_video_predictor"),
    )
    return build(*model_args, gpus_to_use=gpus_to_use, **model_kwargs)


def download_ckpt_from_hf(version: Literal["sam3", "sam3.1"] = "sam3") -> str:
    download = cast(_DownloadCheckpointFn, _load_model_builder_attr("download_ckpt_from_hf"))
    return download(version=version)


def __getattr__(name: str) -> object:
    if name == "Sam3MlxUnsupportedError":
        from sam3_mlx._unsupported import Sam3MlxUnsupportedError as unsupported_error

        return unsupported_error
    if name in _EXPERIMENTAL_EXPORTS:
        warnings.warn(
            f"sam3_mlx.{name} is experimental and not part of the stable 0.1.x "
            "API. Import it from sam3_mlx.experimental instead. Full SAM 3.1 "
            "multiplex/temporal support is reserved for 0.2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from sam3_mlx import experimental as _experimental

        return getattr(_experimental, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
