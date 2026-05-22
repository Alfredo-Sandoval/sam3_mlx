# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

__version__ = "0.1.0"

__all__ = ["build_sam3_image_model"]


def __getattr__(name: str):
    if name == "build_sam3_image_model":
        from sam3_mlx.model_builder import build_sam3_image_model

        return build_sam3_image_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
