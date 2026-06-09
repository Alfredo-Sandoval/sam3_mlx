# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""Fail-fast video dataset surface for the MLX train-data port."""

from sam3_mlx.train._unsupported import raise_unsupported


SEED = 42


class VideoGroundingDataset:
    def __init__(self, *args, **kwargs):
        raise_unsupported("VideoGroundingDataset")
