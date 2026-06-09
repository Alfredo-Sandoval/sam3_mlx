# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""

from typing import Tuple

import mlx.core as mx


def unbind(x: mx.array, dim):
    if dim < 0:
        dim += x.ndim
    if dim < 0 or dim >= x.ndim:
        raise ValueError(f"dim must be in range [-{x.ndim}, {x.ndim}), got {dim}.")
    if dim == x.ndim - 1:
        return [x[..., index] for index in range(x.shape[-1])]
    perm = list(range(x.ndim))
    perm.insert(0, perm.pop(dim))
    return [t for t in x.transpose(perm)]


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = unbind(x, -1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return mx.stack(b, axis=-1)


def box_cxcywh_to_xywh(x):
    x_c, y_c, w, h = unbind(x, -1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (w), (h)]
    return mx.stack(b, axis=-1)


def box_xywh_to_xyxy(x):
    x, y, w, h = unbind(x, -1)
    b = [(x), (y), (x + w), (y + h)]
    return mx.stack(b, axis=-1)


def box_xywh_to_cxcywh(x):
    x, y, w, h = unbind(x, -1)
    b = [(x + 0.5 * w), (y + 0.5 * h), (w), (h)]
    return mx.stack(b, axis=-1)


def box_xyxy_to_xywh(x):
    x, y, X, Y = unbind(x, -1)
    b = [(x), (y), (X - x), (Y - y)]
    return mx.stack(b, axis=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = unbind(x, -1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return mx.stack(b, axis=-1)


def box_area(boxes):
    """
    Batched version of box area. Boxes should be in [x0, y0, x1, y1] format.

    Inputs:
    - boxes: mx.array of shape (..., 4)

    Returns:
    - areas: mx.array of shape (...,)
    """
    x0, y0, x1, y1 = unbind(boxes, -1)
    return (x1 - x0) * (y1 - y0)


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns an mx.array of shape [N, 4], with the boxes in xyxy format
    """
    # `masks` can be bool or float; we treat non-zero as foreground.
    if masks.size == 0:
        return mx.zeros((0, 4), dtype=mx.float32)

    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (N, H, W), got {masks.shape}.")

    h, w = masks.shape[-2:]
    masks_bool = masks.astype(mx.bool_)
    has_x = mx.any(masks_bool, axis=1)
    has_y = mx.any(masks_bool, axis=2)
    has_any = mx.any(has_x, axis=1)

    x = mx.arange(w, dtype=mx.float32)[None, :]
    y = mx.arange(h, dtype=mx.float32)[None, :]
    x_min = mx.min(mx.where(has_x, x, float(w)), axis=1)
    y_min = mx.min(mx.where(has_y, y, float(h)), axis=1)
    x_max = mx.max(mx.where(has_x, x + 1.0, 0.0), axis=1)
    y_max = mx.max(mx.where(has_y, y + 1.0, 0.0), axis=1)

    boxes = mx.stack([x_min, y_min, x_max, y_max], axis=1)
    return boxes * has_any.astype(mx.float32)[:, None]


def box_iou(boxes1, boxes2):
    """
    Batched version of box_iou. Boxes should be in [x0, y0, x1, y1] format.

    Inputs:
    - boxes1: mx.array of shape (..., N, 4)
    - boxes2: mx.array of shape (..., M, 4)

    Returns:
    - iou, union: mx.arrays of shape (..., N, M)
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # boxes1: (..., N, 4) -> (..., N, 1, 2)
    # boxes2: (..., M, 4) -> (..., 1, M, 2)
    lt = mx.maximum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = mx.minimum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])

    wh = mx.clip((rb - lt), a_min=0, a_max=None)  # (..., N, M, 2)
    inter = wh[..., 0] * wh[..., 1]  # (..., N, M)

    union = area1[..., None] + area2[..., None, :] - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Batched version of Generalized IoU from https://giou.stanford.edu/

    Boxes should be in [x0, y0, x1, y1] format

    Inputs:
    - boxes1: mx.array of shape (..., N, 4)
    - boxes2: mx.array of shape (..., M, 4)

    Returns:
    - giou: mx.array of shape (..., N, M)
    """
    iou, union = box_iou(boxes1, boxes2)

    # boxes1: (..., N, 4) -> (..., N, 1, 2)
    # boxes2: (..., M, 4) -> (..., 1, M, 2)
    lt = mx.minimum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = mx.maximum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])

    wh = mx.clip((rb - lt), a_min=0, a_max=None)  # (..., N, M, 2)
    area = wh[..., 0] * wh[..., 1]  # (..., N, M)

    return iou - (area - union) / area


def fast_diag_generalized_box_iou(boxes1, boxes2):
    assert len(boxes1) == len(boxes2)
    box1_xy = boxes1[:, 2:]
    box1_XY = boxes1[:, :2]
    box2_xy = boxes2[:, 2:]
    box2_XY = boxes2[:, :2]
    # assert (box1_xy >= box1_XY).all()
    # assert (box2_xy >= box2_XY).all()
    area1 = mx.prod((box1_xy - box1_XY), axis=-1)
    area2 = mx.prod((box2_xy - box2_XY), axis=-1)

    lt = mx.maximum(box1_XY, box2_XY)  # [N,2]
    lt2 = mx.minimum(box1_XY, box2_XY)
    rb = mx.minimum(box1_xy, box2_xy)  # [N,2]
    rb2 = mx.maximum(box1_xy, box2_xy)

    inter = mx.prod(mx.clip((rb - lt), a_min=0, a_max=None), axis=-1)
    tot_area = mx.prod(mx.clip((rb2 - lt2), a_min=0, a_max=None), axis=-1)

    union = area1 + area2 - inter

    iou = inter / union

    return iou - (tot_area - union) / tot_area


def fast_diag_box_iou(boxes1, boxes2):
    assert len(boxes1) == len(boxes2)
    box1_xy = boxes1[:, 2:]
    box1_XY = boxes1[:, :2]
    box2_xy = boxes2[:, 2:]
    box2_XY = boxes2[:, :2]
    # assert (box1_xy >= box1_XY).all()
    # assert (box2_xy >= box2_XY).all()
    area1 = mx.prod((box1_xy - box1_XY), axis=-1)
    area2 = mx.prod((box2_xy - box2_XY), axis=-1)

    lt = mx.maximum(box1_XY, box2_XY)  # [N,2]
    rb = mx.minimum(box1_xy, box2_xy)  # [N,2]

    inter = mx.prod(mx.clip((rb - lt), a_min=0, a_max=None), axis=-1)

    union = area1 + area2 - inter

    iou = inter / union

    return iou


def box_xywh_inter_union(
    boxes1: mx.array, boxes2: mx.array
) -> Tuple[mx.array, mx.array]:
    # Assumes boxes in xywh format.
    if boxes1.shape[-1] != 4 or boxes2.shape[-1] != 4:
        raise ValueError("boxes1 and boxes2 must have last dimension 4.")
    boxes1 = box_xywh_to_xyxy(boxes1)
    boxes2 = box_xywh_to_xyxy(boxes2)
    box1_tl_xy = boxes1[..., :2]
    box1_br_xy = boxes1[..., 2:]
    box2_tl_xy = boxes2[..., :2]
    box2_br_xy = boxes2[..., 2:]
    area1 = mx.prod(mx.clip(box1_br_xy - box1_tl_xy, a_min=0, a_max=None), axis=-1)
    area2 = mx.prod(mx.clip(box2_br_xy - box2_tl_xy, a_min=0, a_max=None), axis=-1)

    tl = mx.maximum(box1_tl_xy, box2_tl_xy)
    br = mx.minimum(box1_br_xy, box2_br_xy)

    inter = mx.prod(mx.clip((br - tl), a_min=0, a_max=None), axis=-1)
    union = area1 + area2 - inter

    return inter, union
