"""Fail-fast ROIAlign compatibility surface."""

from __future__ import annotations

from typing import Never

from sam3_mlx.agent._unsupported import raise_unsupported


class ROIAlign:
    def __init__(
        self,
        output_size: int | tuple[int, int],
        spatial_scale: float,
        sampling_ratio: int,
        aligned: bool = True,
    ) -> None:
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input: object, rois: object) -> Never:
        del input, rois
        raise_unsupported("agent.helpers.roi_align.ROIAlign.forward")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(output_size={self.output_size}, "
            f"spatial_scale={self.spatial_scale}, sampling_ratio={self.sampling_ratio}, "
            f"aligned={self.aligned})"
        )
