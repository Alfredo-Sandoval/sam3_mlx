"""Fail-fast ROIAlign compatibility surface."""

from __future__ import annotations

from sam3_mlx.agent._unsupported import raise_unsupported


class ROIAlign:
    def __init__(
        self,
        output_size,
        spatial_scale,
        sampling_ratio,
        aligned=True,
    ):
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input, rois):
        raise_unsupported("agent.helpers.roi_align.ROIAlign.forward")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(output_size={self.output_size}, "
            f"spatial_scale={self.spatial_scale}, sampling_ratio={self.sampling_ratio}, "
            f"aligned={self.aligned})"
        )
