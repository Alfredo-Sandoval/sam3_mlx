"""Postprocessor compatibility classes for the MLX port."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from sam3_mlx.eval._unsupported import raise_unsupported
from sam3_mlx.model.data_misc import interpolate
from sam3_mlx.train.masks_ops import rle_encode

MLX_EVAL_POSTPROCESSORS_BASE_COMMIT = "e30678519ff456845aafc13b705fe0ea0a3db028"


def _is_mlx_array(value) -> bool:
    return isinstance(value, mx.array)


def _to_numpy(value, dtype=None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    else:
        if _is_mlx_array(value):
            mx.eval(value)
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _to_output_array(value: np.ndarray, *, to_cpu: bool):
    value = np.asarray(value)
    return value if to_cpu else mx.array(value)


def _sigmoid_np(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return np.where(
        value >= 0,
        1 / (1 + np.exp(-value)),
        np.exp(value) / (1 + np.exp(value)),
    )


def _item(value):
    array = _to_numpy(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    raise ValueError(f"Expected scalar value, got shape {array.shape}.")


def _box_cxcywh_to_xyxy_np(boxes: np.ndarray) -> np.ndarray:
    x_c, y_c, w, h = np.moveaxis(np.asarray(boxes, dtype=np.float32), -1, 0)
    return np.stack(
        [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h],
        axis=-1,
    )


def _as_size_tuple(size) -> tuple[int, int]:
    size_np = _to_numpy(size, dtype=np.int64).reshape(-1)
    if size_np.size != 2:
        raise ValueError(
            f"Expected image size with 2 entries, got shape {size_np.shape}."
        )
    return int(size_np[0]), int(size_np[1])


def _resize_mask_block(mask_block: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if mask_block.ndim == 3:
        mask_block = mask_block[:, None, :, :]
        squeeze_channel = True
    elif mask_block.ndim == 4:
        squeeze_channel = False
    else:
        raise ValueError(
            f"Expected mask block with rank 3 or 4, got {mask_block.shape}."
        )

    masks_mx = mx.array(mask_block, dtype=mx.float32)
    resized = (
        mx.sigmoid(
            interpolate(masks_mx, size=size, mode="bilinear", align_corners=False)
        )
        > 0.5
    )
    mx.eval(resized)
    resized_np = np.asarray(resized)
    if squeeze_channel:
        resized_np = resized_np[:, 0]
    return resized_np.astype(bool, copy=False)


def _concat_values(left, right):
    if left is None:
        return right
    if right is None:
        return left
    if isinstance(left, list):
        return left + list(right)
    return np.concatenate([_to_numpy(left), _to_numpy(right)], axis=0)


def _take_value(value, indices: np.ndarray):
    if value is None:
        return None
    if isinstance(value, list):
        return [value[int(index)] for index in indices.tolist()]
    return _to_numpy(value)[indices]


class PostProcessNullOp:
    def __init__(self, **kwargs):
        pass

    def forward(self, input):
        return None

    def process_results(self, **kwargs):
        return kwargs["find_stages"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class PostProcessImage:
    """Convert SAM3 image outputs into the official evaluator result shape."""

    def __init__(
        self,
        max_dets_per_img: int,
        iou_type="bbox",
        to_cpu: bool = True,
        use_original_ids: bool = False,
        use_original_sizes_box: bool = False,
        use_original_sizes_mask: bool = False,
        convert_mask_to_rle: bool = False,
        always_interpolate_masks_on_gpu: bool = True,
        use_presence: bool = True,
        detection_threshold: float = -1.0,
    ) -> None:
        self.max_dets_per_img = max_dets_per_img
        self.iou_type = iou_type
        self.to_cpu = to_cpu
        self.use_original_ids = use_original_ids
        self.use_original_sizes_box = use_original_sizes_box
        self.use_original_sizes_mask = use_original_sizes_mask
        self.convert_mask_to_rle = convert_mask_to_rle
        self.always_interpolate_masks_on_gpu = always_interpolate_masks_on_gpu
        self.use_presence = use_presence
        self.detection_threshold = detection_threshold

    def forward(
        self,
        outputs,
        target_sizes_boxes,
        target_sizes_masks,
        forced_labels=None,
        consistent=False,
        ret_tensordict: bool = False,
    ):
        if ret_tensordict:
            raise_unsupported("eval.postprocessors.PostProcessImage.ret_tensordict")

        out_bbox = outputs["pred_boxes"] if "pred_boxes" in outputs else None
        out_logits = _to_numpy(outputs["pred_logits"], dtype=np.float32)
        pred_masks = outputs["pred_masks"] if self.iou_type == "segm" else None

        out_probs = _sigmoid_np(out_logits)
        if self.use_presence:
            presence = _sigmoid_np(outputs["presence_logit_dec"])
            if presence.ndim == 1:
                presence = presence[:, None]
            out_probs = out_probs * presence[:, :, None]

        target_sizes_boxes_np = _to_numpy(target_sizes_boxes, dtype=np.int64)
        target_sizes_masks_np = _to_numpy(target_sizes_masks, dtype=np.int64)
        if target_sizes_boxes_np.ndim != 2 or target_sizes_boxes_np.shape[1] != 2:
            raise AssertionError("target_sizes_boxes must have shape (B, 2).")
        if target_sizes_masks_np.ndim != 2 or target_sizes_masks_np.shape[1] != 2:
            raise AssertionError("target_sizes_masks must have shape (B, 2).")

        boxes, scores, labels, keep = self._process_boxes_and_labels(
            target_sizes_boxes_np,
            forced_labels,
            out_bbox,
            out_probs,
        )
        out_masks = self._process_masks(
            target_sizes_masks_np,
            pred_masks,
            consistent=consistent,
            keep=keep,
        )

        if boxes is None:
            if out_masks is None:
                raise AssertionError("PostProcessImage requires boxes or masks.")
            batch_size = len(out_masks)
            boxes = [None] * batch_size
            scores = [None] * batch_size
            labels = [None] * batch_size

        results = {
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
        }
        if out_masks is not None:
            results["masks_rle" if self.convert_mask_to_rle else "masks"] = out_masks

        return [
            dict(zip(results.keys(), per_image_values))
            for per_image_values in zip(*results.values())
        ]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _process_masks(self, target_sizes, pred_masks, consistent=True, keep=None):
        if pred_masks is None:
            return None

        pred_masks_np = _to_numpy(pred_masks, dtype=np.float32)
        if pred_masks_np.ndim != 4:
            raise AssertionError("pred_masks must have shape (B, Q, H, W).")

        if consistent:
            if keep is not None:
                raise_unsupported(
                    "eval.postprocessors.PostProcessImage.keep_consistent_masks"
                )
            unique_sizes = np.unique(_to_numpy(target_sizes, dtype=np.int64), axis=0)
            if unique_sizes.shape[0] != 1:
                raise AssertionError(
                    "consistent=True requires equal target mask sizes."
                )
            out_masks = _resize_mask_block(
                pred_masks_np, _as_size_tuple(unique_sizes[0])
            )
            if self.convert_mask_to_rle:
                return [rle_encode(out_masks[i]) for i in range(out_masks.shape[0])]
            return [
                _to_output_array(out_masks[i], to_cpu=self.to_cpu)
                for i in range(out_masks.shape[0])
            ]

        out_masks = []
        if keep is not None and len(keep) != len(pred_masks_np):
            raise AssertionError("keep and pred_masks batch dimensions must match.")
        for batch_index, mask in enumerate(pred_masks_np):
            if keep is not None:
                mask = mask[_to_numpy(keep[batch_index], dtype=bool)]
            resized = _resize_mask_block(
                mask, _as_size_tuple(target_sizes[batch_index])
            )
            if self.convert_mask_to_rle:
                out_masks.append(rle_encode(resized))
            else:
                out_masks.append(_to_output_array(resized, to_cpu=self.to_cpu))
        return out_masks

    def _process_boxes_and_labels(
        self, target_sizes, forced_labels, out_bbox, out_probs
    ):
        if out_bbox is None:
            return None, None, None, None
        out_bbox_np = _to_numpy(out_bbox, dtype=np.float32)
        out_probs_np = _to_numpy(out_probs, dtype=np.float32)
        if len(out_probs_np) != len(target_sizes):
            raise AssertionError(
                "prediction and target-size batch dimensions mismatch."
            )

        labels = np.argmax(out_probs_np, axis=-1).astype(np.int64)
        scores = np.take_along_axis(out_probs_np, labels[..., None], axis=-1).squeeze(
            -1
        )
        if forced_labels is None:
            labels = np.ones_like(labels, dtype=np.int64)
        else:
            forced = _to_numpy(forced_labels, dtype=np.int64).reshape(-1, 1)
            labels = np.broadcast_to(forced, labels.shape).astype(np.int64, copy=False)

        boxes = _box_cxcywh_to_xyxy_np(out_bbox_np)
        img_h = target_sizes[:, 0].astype(np.float32)
        img_w = target_sizes[:, 1].astype(np.float32)
        scale = np.stack([img_w, img_h, img_w, img_h], axis=1)
        boxes = boxes * scale[:, None, :]

        keep = None
        if self.detection_threshold > 0:
            keep = scores > self.detection_threshold
            boxes = [box[keep_i] for box, keep_i in zip(boxes, keep)]
            scores = [score[keep_i] for score, keep_i in zip(scores, keep)]
            labels = [label[keep_i] for label, keep_i in zip(labels, keep)]
        else:
            boxes = [box for box in boxes]
            scores = [score for score in scores]
            labels = [label for label in labels]

        boxes = [_to_output_array(box, to_cpu=self.to_cpu) for box in boxes]
        scores = [_to_output_array(score, to_cpu=self.to_cpu) for score in scores]
        labels = [_to_output_array(label, to_cpu=self.to_cpu) for label in labels]
        return boxes, scores, labels, keep

    def process_results(self, find_stages, find_metadatas, **kwargs):
        del kwargs
        if getattr(find_stages, "loss_stages", None) is not None:
            find_metadatas = [find_metadatas[i] for i in find_stages.loss_stages]
        if len(find_stages) != len(find_metadatas):
            raise AssertionError("find_stages and find_metadatas length mismatch.")

        results = {}
        for outputs, meta in zip(find_stages, find_metadatas):
            original_size = _to_numpy(meta.original_size, dtype=np.int64)
            unit_size = np.ones_like(original_size, dtype=np.int64)
            img_size_for_boxes = (
                original_size if self.use_original_sizes_box else unit_size
            )
            img_size_for_masks = (
                original_size if self.use_original_sizes_mask else unit_size
            )
            detection_results = self(
                outputs,
                img_size_for_boxes,
                img_size_for_masks,
                forced_labels=(
                    meta.original_category_id if self.use_original_ids else None
                ),
            )
            ids = (
                meta.original_image_id if self.use_original_ids else meta.coco_image_id
            )
            ids_np = _to_numpy(ids).reshape(-1)
            if len(detection_results) != len(ids_np):
                raise AssertionError("detection result and metadata lengths mismatch.")

            for img_id, result in zip(ids_np, detection_results):
                img_id = _item(img_id)
                if img_id not in results:
                    results[img_id] = result
                    continue
                if set(results[img_id].keys()) != set(result.keys()):
                    raise AssertionError("result keys mismatch for duplicate image id.")
                for key, value in result.items():
                    results[img_id][key] = _concat_values(results[img_id][key], value)

        for img_id, result in results.items():
            del img_id
            scores = result.get("scores")
            if (
                self.max_dets_per_img > 0
                and scores is not None
                and len(scores) > self.max_dets_per_img
            ):
                topk = np.argsort(_to_numpy(scores))[::-1][: self.max_dets_per_img]
                for key, value in list(result.items()):
                    result[key] = _take_value(value, topk)

        return results


class PostProcessAPIVideo(PostProcessImage):
    def __init__(
        self,
        *args,
        to_cpu: bool = True,
        convert_mask_to_rle: bool = False,
        always_interpolate_masks_on_gpu: bool = True,
        prob_thresh: float = 0.5,
        use_presence: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            convert_mask_to_rle=False,
            always_interpolate_masks_on_gpu=always_interpolate_masks_on_gpu,
            use_presence=use_presence,
            **kwargs,
        )
        self.EXPECTED_KEYS = ["pred_logits", "pred_boxes", "pred_masks"]
        self.convert_mask_to_rle_for_video = convert_mask_to_rle
        self.to_cpu_for_video = to_cpu
        self.prob_thresh = prob_thresh

    def process_results(self, find_stages, find_metadatas, **kwargs):
        raise_unsupported("eval.postprocessors.PostProcessAPIVideo.process_results")


class PostProcessTracking(PostProcessImage):
    def __init__(
        self,
        max_dets_per_img: int,
        iou_type="bbox",
        force_single_mask: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(max_dets_per_img=max_dets_per_img, iou_type=iou_type, **kwargs)
        self.force_single_mask = force_single_mask

    def process_results(self, find_stages, find_metadatas, **kwargs):
        raise_unsupported("eval.postprocessors.PostProcessTracking.process_results")


class PostProcessCounting:
    """Small NumPy-compatible counting postprocessor."""

    def __init__(
        self,
        use_original_ids: bool = False,
        threshold: float = 0.5,
        use_presence: bool = False,
    ) -> None:
        self.use_original_ids = use_original_ids
        self.threshold = threshold
        self.use_presence = use_presence

    def forward(self, outputs, target_sizes):
        del target_sizes
        logits = _to_numpy(outputs["pred_logits"], dtype=np.float32).squeeze(-1)
        scores = _sigmoid_np(logits)
        if self.use_presence:
            presence = _sigmoid_np(outputs["presence_logit_dec"])
            if presence.ndim == 1:
                presence = presence[:, None]
            scores = scores * presence
        counts = (scores > self.threshold).sum(axis=1)
        return [{"count": int(count)} for count in counts]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def process_results(self, find_stages, find_metadatas, **kwargs):
        del kwargs
        if len(find_stages) != len(find_metadatas):
            raise AssertionError("find_stages and find_metadatas length mismatch.")
        results = {}
        for outputs, meta in zip(find_stages, find_metadatas):
            detection_results = self(outputs, meta.original_size)
            ids = (
                meta.original_image_id if self.use_original_ids else meta.coco_image_id
            )
            ids_np = _to_numpy(ids).reshape(-1)
            if len(detection_results) != len(ids_np):
                raise AssertionError("count result and metadata lengths mismatch.")
            for img_id, result in zip(ids_np, detection_results):
                results[_item(img_id)] = result
        return results


__all__ = [
    "MLX_EVAL_POSTPROCESSORS_BASE_COMMIT",
    "PostProcessAPIVideo",
    "PostProcessCounting",
    "PostProcessImage",
    "PostProcessNullOp",
    "PostProcessTracking",
]
