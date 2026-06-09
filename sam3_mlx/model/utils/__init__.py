# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Model utility helpers ported from the official SAM3 ``sam3/model/utils``."""

__all__ = [
    "AsyncVideoFrameLoader",
    "SAM2Transforms",
    "copy_data_to_device",
    "load_video_frames",
    "load_video_frames_from_jpg_images",
    "load_video_frames_from_video_file",
]


def __getattr__(name: str):
    if name == "copy_data_to_device":
        from sam3_mlx.model.utils.misc import copy_data_to_device

        return copy_data_to_device
    if name == "SAM2Transforms":
        from sam3_mlx.model.utils.sam1_utils import SAM2Transforms

        return SAM2Transforms
    if name in {
        "AsyncVideoFrameLoader",
        "load_video_frames",
        "load_video_frames_from_jpg_images",
        "load_video_frames_from_video_file",
    }:
        from sam3_mlx.model.utils.sam2_utils import (
            AsyncVideoFrameLoader,
            load_video_frames,
            load_video_frames_from_jpg_images,
            load_video_frames_from_video_file,
        )

        return {
            "AsyncVideoFrameLoader": AsyncVideoFrameLoader,
            "load_video_frames": load_video_frames,
            "load_video_frames_from_jpg_images": load_video_frames_from_jpg_images,
            "load_video_frames_from_video_file": load_video_frames_from_video_file,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
