# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

__version__ = "0.1.0"

__all__ = [
    "Sam3MlxUnsupportedError",
    "build_tracker",
    "build_sam3_image_model",
    "build_sam3_multiplex_video_model",
    "build_sam3_multiplex_video_predictor",
    "build_sam3_predictor",
    "build_sam3_video_model",
    "build_sam3_video_predictor",
    "download_ckpt_from_hf",
]


def __getattr__(name: str):
    if name == "Sam3MlxUnsupportedError":
        from sam3_mlx._unsupported import Sam3MlxUnsupportedError

        return Sam3MlxUnsupportedError
    if name == "build_tracker":
        from sam3_mlx.model_builder import build_tracker

        return build_tracker
    if name == "build_sam3_image_model":
        from sam3_mlx.model_builder import build_sam3_image_model

        return build_sam3_image_model
    if name == "build_sam3_multiplex_video_model":
        from sam3_mlx.model_builder import build_sam3_multiplex_video_model

        return build_sam3_multiplex_video_model
    if name == "build_sam3_multiplex_video_predictor":
        from sam3_mlx.model_builder import build_sam3_multiplex_video_predictor

        return build_sam3_multiplex_video_predictor
    if name == "build_sam3_predictor":
        from sam3_mlx.model_builder import build_sam3_predictor

        return build_sam3_predictor
    if name == "build_sam3_video_model":
        from sam3_mlx.model_builder import build_sam3_video_model

        return build_sam3_video_model
    if name == "build_sam3_video_predictor":
        from sam3_mlx.model_builder import build_sam3_video_predictor

        return build_sam3_video_predictor
    if name == "download_ckpt_from_hf":
        from sam3_mlx.model_builder import download_ckpt_from_hf

        return download_ckpt_from_hf
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
