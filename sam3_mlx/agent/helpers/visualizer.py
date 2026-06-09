"""Dependency-light visualization helpers compatible with the official names."""

from __future__ import annotations

from enum import Enum

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from sam3_mlx.agent.helpers.color_map import colormap
from sam3_mlx.agent.helpers.rle import rle_decode, rle_to_bbox


class ColorMode(Enum):
    IMAGE = 0
    SEGMENTATION = 1
    IMAGE_BW = 2


class GenericMask:
    def __init__(self, mask_or_polygons, height: int, width: int):
        self._mask = None
        self._polygons = None
        self._has_holes = False
        self.height = int(height)
        self.width = int(width)
        if isinstance(mask_or_polygons, dict):
            self._mask = rle_decode(mask_or_polygons)
        elif isinstance(mask_or_polygons, list):
            self._polygons = mask_or_polygons
        else:
            mask = np.asarray(mask_or_polygons, dtype=bool)
            if mask.shape != (self.height, self.width):
                raise ValueError(
                    f"Mask shape {mask.shape} does not match {(self.height, self.width)}"
                )
            self._mask = mask

    @property
    def mask(self):
        if self._mask is None:
            from .masks import polygons_to_bitmask

            self._mask = polygons_to_bitmask(
                [np.asarray(p) for p in self._polygons], self.height, self.width
            )
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons = []
        return self._polygons

    @property
    def has_holes(self):
        return self._has_holes

    def mask_to_polygons(self, mask):
        return [], False

    def polygons_to_mask(self, polygons):
        from .masks import polygons_to_bitmask

        return polygons_to_bitmask(
            [np.asarray(p) for p in polygons], self.height, self.width
        )

    def area(self):
        return int(self.mask.sum())

    def bbox(self):
        ys, xs = np.where(self.mask)
        if xs.size == 0 or ys.size == 0:
            return [0, 0, 0, 0]
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


class _PanopticPrediction:
    def __init__(self, panoptic_seg, segments_info, metadata=None):
        self.panoptic_seg = np.asarray(panoptic_seg)
        self.segments_info = segments_info
        self.metadata = metadata

    def non_empty_mask(self):
        return self.panoptic_seg >= 0

    def semantic_masks(self):
        for segment in self.segments_info:
            yield segment, self.panoptic_seg == segment.get("id")

    def instance_masks(self):
        yield from self.semantic_masks()


def _create_text_labels(classes, scores, class_names, is_crowd=None):
    labels = []
    for i, class_id in enumerate(classes):
        label = class_names[class_id] if class_names is not None else str(class_id)
        if scores is not None:
            label = f"{label} {scores[i]:.0%}"
        if is_crowd is not None and is_crowd[i]:
            label = f"{label}|crowd"
        labels.append(label)
    return labels


class VisImage:
    def __init__(self, img, scale: float = 1.0):
        self.img = np.asarray(img).astype(np.uint8)
        self.scale = scale

    def _setup_figure(self, img):
        return None

    def reset_image(self, img):
        self.img = np.asarray(img).astype(np.uint8)
        return self

    def save(self, filepath):
        Image.fromarray(self.img).save(filepath)

    def get_image(self):
        return self.img


def _as_rgb_tuple(color) -> tuple[int, int, int]:
    if color is None:
        return (0, 255, 0)
    if isinstance(color, str):
        return ImageColor.getrgb(color)
    arr = np.asarray(color, dtype=float)
    if arr.max() <= 1:
        arr = arr * 255
    return tuple(int(np.clip(v, 0, 255)) for v in arr[:3])


class Visualizer:
    def __init__(
        self,
        img_rgb,
        metadata=None,
        scale: float = 1.0,
        instance_mode=ColorMode.IMAGE,
        font_size_multiplier: float = 1.0,
        boarder_width_multiplier: float = 0,
    ):
        self.img = np.asarray(img_rgb).astype(np.uint8)
        self.metadata = metadata
        self.scale = scale
        self.output = VisImage(self.img.copy(), scale=scale)
        self.font_size_multiplier = font_size_multiplier
        self.boarder_width_multiplier = boarder_width_multiplier

    def draw_instance_predictions(self, predictions):
        return self

    def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8):
        return self

    def draw_panoptic_seg(
        self, panoptic_seg, segments_info, area_threshold=None, alpha=0.7
    ):
        return self

    def draw_dataset_dict(self, dic):
        return self

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5,
        label_mode="1",
        binary_masks=None,
    ):
        image = Image.fromarray(self.output.img).convert("RGBA")
        draw = ImageDraw.Draw(image)
        boxes_arr = np.asarray(boxes, dtype=float) if boxes is not None else None
        if binary_masks is None and masks is not None:
            binary_masks = [
                rle_decode(mask)
                if isinstance(mask, dict)
                else np.asarray(mask, dtype=bool)
                for mask in masks
            ]
        colors = assigned_colors
        if colors is None:
            base = colormap(rgb=True, maximum=255)
            count = (
                len(binary_masks)
                if binary_masks is not None
                else (0 if boxes_arr is None else len(boxes_arr))
            )
            colors = [base[i % len(base)] for i in range(count)]

        if binary_masks is not None:
            for idx, mask in enumerate(binary_masks):
                color = _as_rgb_tuple(colors[idx] if idx < len(colors) else None)
                mask = np.asarray(mask, dtype=bool)
                overlay = Image.new("RGBA", image.size, color + (0,))
                alpha_mask = Image.fromarray(
                    (mask.astype(np.uint8) * int(255 * alpha)), mode="L"
                )
                overlay.putalpha(alpha_mask)
                image = Image.alpha_composite(image, overlay)
                draw = ImageDraw.Draw(image)
                if boxes_arr is None:
                    bbox = rle_to_bbox({"counts": [], "size": mask.shape})
                    ys, xs = np.where(mask)
                    if xs.size and ys.size:
                        bbox = [
                            xs.min(),
                            ys.min(),
                            xs.max() - xs.min() + 1,
                            ys.max() - ys.min() + 1,
                        ]
                    xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    draw.rectangle(xyxy, outline=color, width=2)

        if boxes_arr is not None:
            for idx, box in enumerate(boxes_arr):
                color = _as_rgb_tuple(colors[idx] if idx < len(colors) else None)
                if len(box) == 4:
                    x, y, w, h = box
                    if max(box) <= 1.0:
                        x, w = x * image.width, w * image.width
                        y, h = y * image.height, h * image.height
                    xyxy = [x, y, x + w, y + h]
                    draw.rectangle(
                        xyxy,
                        outline=color,
                        width=max(1, int(2 + self.boarder_width_multiplier)),
                    )
                    if labels:
                        draw.text((x, max(0, y - 12)), str(labels[idx]), fill=color)
                    elif label_mode:
                        draw.text((x, max(0, y - 12)), str(idx + 1), fill=color)

        self.output.img = np.asarray(image.convert("RGB"))
        return self

    def overlay_rotated_instances(self, boxes=None, labels=None, assigned_colors=None):
        return self.overlay_instances(
            boxes=boxes, labels=labels, assigned_colors=assigned_colors
        )

    def draw_and_connect_keypoints(self, keypoints):
        return self

    def mask_dims_from_binary(self, mask, anchor_point=None):
        ys, xs = np.where(np.asarray(mask, dtype=bool))
        if xs.size == 0 or ys.size == 0:
            return 0, 0
        return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)

    def reposition_label(self, label_pos, dimensions, text_dims, padding=5):
        return label_pos

    def locate_label_position(self, mask, label_text, font, padding=5):
        ys, xs = np.where(np.asarray(mask, dtype=bool))
        if xs.size == 0 or ys.size == 0:
            return 0, 0
        return int(xs.min()), int(ys.min())

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="left",
        rotation=0,
    ):
        image = Image.fromarray(self.output.img).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.text(
            tuple(position),
            str(text),
            fill=_as_rgb_tuple(color),
            font=ImageFont.load_default(),
        )
        self.output.img = np.asarray(image)
        return self

    def draw_box(
        self, box_coord, edge_color="g", line_style="-", alpha=1.0, line_width=2
    ):
        image = Image.fromarray(self.output.img).convert("RGB")
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = box_coord
        draw.rectangle(
            [x0, y0, x1, y1], outline=_as_rgb_tuple(edge_color), width=int(line_width)
        )
        self.output.img = np.asarray(image)
        return self

    def draw_rotated_box_with_label(self, rotated_box, edge_color="g", label=None):
        return self

    def draw_circle(self, circle_coord, color, radius=3):
        image = Image.fromarray(self.output.img).convert("RGB")
        draw = ImageDraw.Draw(image)
        x, y = circle_coord
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius], fill=_as_rgb_tuple(color)
        )
        self.output.img = np.asarray(image)
        return self

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        image = Image.fromarray(self.output.img).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.line(
            list(zip(x_data, y_data)),
            fill=_as_rgb_tuple(color),
            width=int(linewidth or 1),
        )
        self.output.img = np.asarray(image)
        return self

    def draw_binary_mask(
        self,
        binary_mask,
        color=None,
        *,
        edge_color=None,
        text=None,
        alpha=0.5,
        area_threshold=0,
    ):
        return self.overlay_instances(
            binary_masks=[binary_mask], assigned_colors=[color], alpha=alpha
        )

    def draw_binary_mask_with_number(
        self, binary_mask, text, color=None, *, alpha=0.5, area_threshold=0
    ):
        return self.draw_binary_mask(
            binary_mask,
            color=color,
            text=text,
            alpha=alpha,
            area_threshold=area_threshold,
        )

    def draw_soft_mask(self, soft_mask, color=None, *, alpha=0.5):
        return self.draw_binary_mask(
            np.asarray(soft_mask) > 0.5, color=color, alpha=alpha
        )

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        image = Image.fromarray(self.output.img).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.polygon(
            [tuple(point) for point in np.asarray(segment).reshape(-1, 2)],
            fill=_as_rgb_tuple(color) + (int(alpha * 255),),
        )
        self.output.img = np.asarray(
            Image.alpha_composite(image, overlay).convert("RGB")
        )
        return self

    def _jitter(self, color):
        return np.asarray(color)

    def _create_grayscale_image(self, mask=None):
        gray = np.mean(self.img, axis=2).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=2)

    def _change_color_brightness(self, color, brightness_factor):
        return np.clip(np.asarray(color, dtype=float) * brightness_factor, 0, 1)

    def _convert_boxes(self, boxes):
        return boxes

    def _convert_masks(self, masks):
        return masks

    def _draw_number_in_box(self, box, num, color):
        return self

    def number_to_string(self, number):
        return str(number)

    def _draw_number_in_mask(self, binary_mask, number, color):
        return self

    def _draw_text_in_mask(self, binary_mask, text, color):
        return self

    def _convert_keypoints(self, keypoints):
        return keypoints

    def get_output(self):
        return self.output
