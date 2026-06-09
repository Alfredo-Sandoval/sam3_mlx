from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL.Image import Image

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.mlx_runtime import to_numpy as _to_numpy
from sam3_mlx.model.utils.sam1_utils import SAM2Transforms
from sam3_mlx.sam.mask_decoder import MaskDecoder
from sam3_mlx.sam.prompt_encoder import PromptEncoder
from sam3_mlx.sam.transformer import TwoWayTransformer


def _raise_sam1_unsupported(
    feature: str, *, reason: str, detail: str, alternative=None
):
    raise_unsupported(
        feature,
        reason=reason,
        detail=detail,
        alternative=alternative,
    )


class SAM3InteractiveImageModel(nn.Module):
    """Tracker-shaped SAM1 helper stack for image interactivity.

    The official predictor talks to a tracker object, but its image prompt path
    only needs SAM prompt/decoder heads plus flattened SAM2-neck features. This
    MLX module owns that non-video subset and intentionally leaves memory/tracker
    behavior out of scope.
    """

    def __init__(
        self,
        backbone=None,
        image_size: int = 1008,
        backbone_stride: int = 14,
        hidden_dim: int = 256,
        sam_mask_decoder_extra_args: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.image_size = int(image_size)
        self.backbone_stride = int(backbone_stride)
        self.hidden_dim = int(hidden_dim)
        self.num_feature_levels = 3
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.no_mem_embed = 0.02 * mx.random.normal((1, 1, self.hidden_dim))
        self._build_sam_heads()

    @property
    def device(self):
        return "mlx"

    def _build_sam_heads(self) -> None:
        sam_image_embedding_size = self.image_size // self.backbone_stride
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.hidden_dim,
            image_embedding_size=(sam_image_embedding_size, sam_image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.hidden_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.hidden_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            **(self.sam_mask_decoder_extra_args or {}),
        )

    def forward_image(self, img_batch):
        if self.backbone is None:
            _raise_sam1_unsupported(
                "sam3_mlx.model.sam1_task_predictor.SAM3InteractiveImageModel.forward_image",
                reason="image-interactivity",
                detail=(
                    "SAM3InteractiveImageModel.forward_image requires an attached "
                    "SAM3 backbone."
                ),
                alternative="Sam3Processor.set_image() plus Sam3Image.predict_inst()",
            )
        backbone_out = self.backbone.forward_image(img_batch)
        sam2_backbone_out = backbone_out.get("sam2_backbone_out")
        if sam2_backbone_out is None:
            _raise_sam1_unsupported(
                "sam3_mlx.model.sam1_task_predictor.SAM3InteractiveImageModel.forward_image(sam2_backbone_out)",
                reason="image-interactivity",
                detail=(
                    "Interactive prediction requires a backbone built with "
                    "enable_inst_interactivity=True so sam2_backbone_out is available."
                ),
                alternative="build_sam3_image_model(enable_inst_interactivity=True)",
            )
        return self.precompute_high_res_features(sam2_backbone_out)

    def precompute_high_res_features(self, backbone_out):
        backbone_out = backbone_out.copy()
        backbone_out["backbone_fpn"] = list(backbone_out["backbone_fpn"])
        backbone_out["vision_pos_enc"] = list(backbone_out["vision_pos_enc"])
        if len(backbone_out["backbone_fpn"]) < 2:
            raise ValueError("Expected at least two high-resolution feature levels.")
        backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        backbone_out = backbone_out.copy()
        if "backbone_fpn" not in backbone_out or "vision_pos_enc" not in backbone_out:
            raise KeyError("backbone_out must contain backbone_fpn and vision_pos_enc.")
        if len(backbone_out["backbone_fpn"]) != len(backbone_out["vision_pos_enc"]):
            raise AssertionError("backbone_fpn and vision_pos_enc length mismatch.")
        if len(backbone_out["backbone_fpn"]) < self.num_feature_levels:
            raise AssertionError(
                f"Expected at least {self.num_feature_levels} feature levels."
            )

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [
            x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 0, 1)
            for x in feature_maps
        ]
        vision_pos_embeds = [
            x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 0, 1)
            for x in vision_pos_embeds
        ]
        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes


class SAM3InteractiveImagePredictor(nn.Module):
    def __init__(
        self,
        sam_model: SAM3InteractiveImageModel,
        mask_threshold=0.0,
        max_hole_area=256.0,
        max_sprinkle_area=0.0,
        **kwargs,
    ) -> None:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected predictor keyword(s): {unexpected}")
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
        self.mask_threshold = mask_threshold
        image_embedding_size = self.model.image_size // self.model.backbone_stride
        self._bb_feat_sizes = [
            (image_embedding_size * 4, image_embedding_size * 4),
            (image_embedding_size * 2, image_embedding_size * 2),
            (image_embedding_size, image_embedding_size),
        ]

    def set_image(self, image: Union[np.ndarray, Image]) -> None:
        self.reset_predictor()
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise ValueError("Image must be a NumPy array or PIL image.")

        input_image = self._transforms(image)[None, ...]
        if len(input_image.shape) != 4 or input_image.shape[1] != 3:
            raise AssertionError(
                f"input_image must be of size 1x3xHxW, got {input_image.shape}"
            )
        logging.info("Computing image embeddings for the provided image...")
        self._set_features_from_backbone(self.model.forward_image(input_image))
        logging.info("Image embeddings computed.")

    def set_image_batch(self, image_list: List[np.ndarray]) -> None:
        self.reset_predictor()
        if not isinstance(image_list, list):
            raise AssertionError("image_list must be a list")
        self._orig_hw = []
        for image in image_list:
            if not isinstance(image, np.ndarray):
                raise AssertionError(
                    "Images are expected to be NumPy arrays in RGB HWC format."
                )
            self._orig_hw.append(image.shape[:2])
        img_batch = self._transforms.forward_batch(image_list)
        if len(img_batch.shape) != 4 or img_batch.shape[1] != 3:
            raise AssertionError(
                f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
            )
        logging.info("Computing image embeddings for the provided images...")
        self._set_features_from_backbone(self.model.forward_image(img_batch))
        self._is_batch = True
        logging.info("Image embeddings computed.")

    def _set_features_from_backbone(self, backbone_out) -> None:
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        batch_size = vision_feats[-1].shape[1]
        feats = [
            feat.transpose(1, 2, 0).reshape(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        if not self._is_batch:
            raise AssertionError("This function should only be used in batched mode")
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before prediction."
            )
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        for img_idx in range(num_images):
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            all_masks.append(_to_numpy(masks.squeeze(0).astype(mx.float32)))
            all_ious.append(_to_numpy(iou_predictions.squeeze(0).astype(mx.float32)))
            all_low_res_masks.append(
                _to_numpy(low_res_masks.squeeze(0).astype(mx.float32))
            )
        return all_masks, all_ious, all_low_res_masks

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords,
            point_labels,
            box,
            mask_input,
            normalize_coords,
        )
        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )
        return (
            _to_numpy(masks.squeeze(0).astype(mx.float32)),
            _to_numpy(iou_predictions.squeeze(0).astype(mx.float32)),
            _to_numpy(low_res_masks.squeeze(0).astype(mx.float32)),
        )

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            if point_labels is None:
                raise AssertionError(
                    "point_labels must be supplied if point_coords is supplied."
                )
            point_coords = mx.array(point_coords, dtype=mx.float32)
            unnorm_coords = self._transforms.transform_coords(
                point_coords,
                normalize=normalize_coords,
                orig_hw=self._orig_hw[img_idx],
            )
            labels = mx.array(point_labels, dtype=mx.int32)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords = unnorm_coords[None, ...]
                labels = labels[None, ...]
        if box is not None:
            box = mx.array(box, dtype=mx.float32)
            unnorm_box = self._transforms.transform_boxes(
                box,
                normalize=normalize_coords,
                orig_hw=self._orig_hw[img_idx],
            )
        if mask_logits is not None:
            mask_input = mx.array(mask_logits, dtype=mx.float32)
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    def _predict(
        self,
        point_coords: Optional[mx.array],
        point_labels: Optional[mx.array],
        boxes: Optional[mx.array] = None,
        mask_input: Optional[mx.array] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        concat_points = (
            (point_coords, point_labels) if point_coords is not None else None
        )
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = mx.broadcast_to(
                mx.array([[2, 3]], dtype=mx.int32),
                (box_coords.shape[0], 2),
            )
            if concat_points is not None:
                concat_coords = mx.concat([box_coords, concat_points[0]], axis=1)
                concat_labels = mx.concat([box_labels, concat_points[1]], axis=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )
        batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
        high_res_features = [
            feat_level[img_idx][None] for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx][None],
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        low_res_masks = mx.clip(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold
        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> mx.array:
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        if self._features is None:
            raise AssertionError("Features must exist if an image has been set.")
        return self._features["image_embed"]

    @property
    def device(self):
        device = self.model.device
        if device not in (None, "mlx"):
            _raise_sam1_unsupported(
                f"sam3_mlx.model.sam1_task_predictor.SAM3InteractiveImagePredictor.device={device!r}",
                reason="unsupported-device",
                detail="The MLX interactive predictor only supports the explicit MLX runtime.",
                alternative="device='mlx'",
            )
        return "mlx"

    def reset_predictor(self) -> None:
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
