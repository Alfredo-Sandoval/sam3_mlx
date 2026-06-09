import os

from sam3_mlx.perflib.masks_ops import mask_iom, mask_iou, masks_to_boxes

is_enabled = os.getenv("USE_PERFLIB", "1") == "1"

__all__ = ["is_enabled", "mask_iou", "mask_iom", "masks_to_boxes"]
