"""MLX loss helpers ported from official SAM3 training utilities."""

from sam3_mlx.train.loss.loss_fns import (
    CORE_LOSS_KEY,
    MLX_LOSS_FNS_BASE_COMMIT,
    Boxes,
    Det2TrkAssoc,
    IABCEMdetr,
    LossWithWeights,
    Masks,
    SemanticSegCriterion,
    TrackingByDetectionAssoc,
    accuracy,
    dice_loss,
    instance_masks_to_semantic_masks,
    iou_loss,
    segment_miou,
)
from sam3_mlx.train.loss.mask_sampling import (
    calculate_uncertainty,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from sam3_mlx.train.loss.sam3_loss import (
    MLX_SAM3_LOSS_BASE_COMMIT,
    DummyLoss,
    Sam3LossWrapper,
)
from sam3_mlx.train.loss.sigmoid_focal_loss import (
    SigmoidFocalLoss,
    SigmoidFocalLossReduced,
    sigmoid_focal_loss,
    sigmoid_focal_loss_reduce,
    triton_sigmoid_focal_loss,
    triton_sigmoid_focal_loss_reduce,
)

__all__ = [
    "CORE_LOSS_KEY",
    "MLX_LOSS_FNS_BASE_COMMIT",
    "Boxes",
    "Det2TrkAssoc",
    "DummyLoss",
    "IABCEMdetr",
    "LossWithWeights",
    "Masks",
    "MLX_SAM3_LOSS_BASE_COMMIT",
    "SemanticSegCriterion",
    "Sam3LossWrapper",
    "SigmoidFocalLoss",
    "SigmoidFocalLossReduced",
    "TrackingByDetectionAssoc",
    "accuracy",
    "calculate_uncertainty",
    "dice_loss",
    "get_uncertain_point_coords_with_randomness",
    "instance_masks_to_semantic_masks",
    "iou_loss",
    "point_sample",
    "segment_miou",
    "sigmoid_focal_loss",
    "sigmoid_focal_loss_reduce",
    "triton_sigmoid_focal_loss",
    "triton_sigmoid_focal_loss_reduce",
]
