"""YouTube-VIS to COCO-video JSON conversion helpers."""

from __future__ import annotations

import json
import os
from collections import defaultdict


def convert_ytbvis_to_cocovid_gt(ann_json, save_path=None):
    """Convert YouTube-VIS ground truth JSON to a COCO-video-style dict."""
    VIS = {
        "info": {},
        "images": [],
        "videos": [],
        "tracks": [],
        "annotations": [],
        "categories": [],
        "licenses": [],
    }
    with open(ann_json, encoding="utf-8") as f:
        official_anns = json.load(f)
    VIS["categories"] = official_anns["categories"]
    records = dict(img_id=1, ann_id=1)
    vid_to_anns = defaultdict(list)
    for ann in official_anns["annotations"]:
        vid_to_anns[ann["video_id"]].append(ann)
    VIS["tracks"] = [
        {
            "id": ann["id"],
            "category_id": ann["category_id"],
            "video_id": ann["video_id"],
        }
        for ann in official_anns["annotations"]
    ]

    for video_info in official_anns["videos"]:
        video = {
            "id": video_info["id"],
            "name": os.path.dirname(video_info["file_names"][0]),
            "width": video_info["width"],
            "height": video_info["height"],
            "length": video_info["length"],
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [],
        }
        VIS["videos"].append(video)
        for frame_idx, file_name in enumerate(video_info["file_names"]):
            image = {
                "id": records["img_id"],
                "video_id": video_info["id"],
                "file_name": file_name,
                "width": video_info["width"],
                "height": video_info["height"],
                "frame_index": frame_idx,
                "frame_id": frame_idx,
            }
            VIS["images"].append(image)
            for ann in vid_to_anns.get(video_info["id"], []):
                bbox = ann["bboxes"][frame_idx]
                if bbox is None:
                    continue
                VIS["annotations"].append(
                    {
                        "id": records["ann_id"],
                        "video_id": video_info["id"],
                        "image_id": records["img_id"],
                        "track_id": ann["id"],
                        "category_id": ann["category_id"],
                        "bbox": bbox,
                        "area": ann["areas"][frame_idx],
                        "segmentation": ann["segmentations"][frame_idx],
                        "iscrowd": ann["iscrowd"],
                    }
                )
                records["ann_id"] += 1
            records["img_id"] += 1

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(VIS, f)
    return VIS


def convert_ytbvis_to_cocovid_pred(
    youtubevis_pred_path: str, converted_dataset_path: str, output_path: str
) -> None:
    """Convert YouTube-VIS predictions to COCO-style frame annotations."""
    with open(youtubevis_pred_path, encoding="utf-8") as f:
        ytv_predictions = json.load(f)
    with open(converted_dataset_path, encoding="utf-8") as f:
        coco_dataset = json.load(f)

    image_id_map = {
        (img["video_id"], img["frame_index"]): img["id"]
        for img in coco_dataset["images"]
    }
    coco_annotations = []
    track_id_counter = 1

    for pred in ytv_predictions:
        segmentations = pred.get("segmentations") or [None] * len(pred["bboxes"])
        areas = pred.get("areas") or [None] * len(pred["bboxes"])
        track_id = track_id_counter
        track_id_counter += 1
        for frame_idx, (bbox, segmentation, area_from_pred) in enumerate(
            zip(pred["bboxes"], segmentations, areas)
        ):
            if bbox is None or all(x == 0 for x in bbox):
                continue
            image_id = image_id_map.get((pred["video_id"], frame_idx))
            if image_id is None:
                raise RuntimeError(
                    f"prediction video_id={pred['video_id']}, frame_idx={frame_idx} "
                    "does not match converted COCO images"
                )
            x, y, w, h = bbox
            annotation = {
                "image_id": int(image_id),
                "video_id": pred["video_id"],
                "track_id": track_id,
                "category_id": pred["category_id"],
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(
                    area_from_pred
                    if area_from_pred is not None and area_from_pred > 0
                    else w * h
                ),
                "iscrowd": 0,
                "score": float(pred["score"]),
            }
            if segmentation is not None:
                annotation["segmentation"] = segmentation
            coco_annotations.append(annotation)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_annotations, f)
