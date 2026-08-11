"""YouTube-VIS to COCO-video JSON conversion helpers."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import cast


type JsonObject = dict[str, object]


def _json_object(value: object, name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object.")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise TypeError(f"{name} must use string keys.")
    return cast(JsonObject, mapping)


def _json_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list.")
    return cast(list[object], value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}.")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric, not {type(value).__name__}.")
    return float(value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _load_json(path: str | Path) -> object:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def convert_ytbvis_to_cocovid_gt(
    ann_json: str | Path, save_path: str | Path | None = None
) -> JsonObject:
    """Convert YouTube-VIS ground truth JSON to a COCO-video-style dict."""

    official = _json_object(_load_json(ann_json), "YouTube-VIS annotations")
    annotations = [
        _json_object(value, f"annotations[{index}]")
        for index, value in enumerate(
            _json_list(official["annotations"], "annotations")
        )
    ]
    video_records = [
        _json_object(value, f"videos[{index}]")
        for index, value in enumerate(_json_list(official["videos"], "videos"))
    ]

    images: list[object] = []
    videos: list[object] = []
    tracks: list[object] = []
    frame_annotations: list[object] = []
    records = {"img_id": 1, "ann_id": 1}
    vid_to_anns: defaultdict[int, list[JsonObject]] = defaultdict(list)
    for annotation in annotations:
        video_id = _integer(annotation["video_id"], "annotation.video_id")
        vid_to_anns[video_id].append(annotation)
        tracks.append(
            {
                "id": annotation["id"],
                "category_id": annotation["category_id"],
                "video_id": annotation["video_id"],
            }
        )

    for video_info in video_records:
        video_id = _integer(video_info["id"], "video.id")
        file_names = _json_list(video_info["file_names"], "video.file_names")
        if not file_names:
            raise ValueError("video.file_names must be non-empty.")
        first_file_name = _string(file_names[0], "video.file_names[0]")
        videos.append(
            {
                "id": video_info["id"],
                "name": os.path.dirname(first_file_name),
                "width": video_info["width"],
                "height": video_info["height"],
                "length": video_info["length"],
                "neg_category_ids": [],
                "not_exhaustive_category_ids": [],
            }
        )
        for frame_idx, file_name_value in enumerate(file_names):
            file_name = _string(file_name_value, f"video.file_names[{frame_idx}]")
            images.append(
                {
                    "id": records["img_id"],
                    "video_id": video_info["id"],
                    "file_name": file_name,
                    "width": video_info["width"],
                    "height": video_info["height"],
                    "frame_index": frame_idx,
                    "frame_id": frame_idx,
                }
            )
            for annotation in vid_to_anns.get(video_id, []):
                bboxes = _json_list(annotation["bboxes"], "annotation.bboxes")
                bbox = bboxes[frame_idx]
                if bbox is None:
                    continue
                areas = _json_list(annotation["areas"], "annotation.areas")
                segmentations = _json_list(
                    annotation["segmentations"], "annotation.segmentations"
                )
                frame_annotations.append(
                    {
                        "id": records["ann_id"],
                        "video_id": video_info["id"],
                        "image_id": records["img_id"],
                        "track_id": annotation["id"],
                        "category_id": annotation["category_id"],
                        "bbox": bbox,
                        "area": areas[frame_idx],
                        "segmentation": segmentations[frame_idx],
                        "iscrowd": annotation["iscrowd"],
                    }
                )
                records["ann_id"] += 1
            records["img_id"] += 1

    converted: JsonObject = {
        "info": {},
        "images": images,
        "videos": videos,
        "tracks": tracks,
        "annotations": frame_annotations,
        "categories": _json_list(official["categories"], "categories"),
        "licenses": [],
    }
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as handle:
            json.dump(converted, handle)
    return converted


def convert_ytbvis_to_cocovid_pred(
    youtubevis_pred_path: str | Path,
    converted_dataset_path: str | Path,
    output_path: str | Path,
) -> None:
    """Convert YouTube-VIS predictions to COCO-style frame annotations."""

    predictions = [
        _json_object(value, f"predictions[{index}]")
        for index, value in enumerate(
            _json_list(_load_json(youtubevis_pred_path), "predictions")
        )
    ]
    dataset = _json_object(_load_json(converted_dataset_path), "COCO dataset")
    image_id_map: dict[tuple[int, int], int] = {}
    for index, value in enumerate(_json_list(dataset["images"], "dataset.images")):
        image = _json_object(value, f"dataset.images[{index}]")
        image_id_map[
            (
                _integer(image["video_id"], "image.video_id"),
                _integer(image["frame_index"], "image.frame_index"),
            )
        ] = _integer(image["id"], "image.id")

    coco_annotations: list[object] = []
    for track_id, prediction in enumerate(predictions, start=1):
        bboxes = _json_list(prediction["bboxes"], "prediction.bboxes")
        segmentations_value = prediction.get("segmentations")
        segmentations = (
            [None] * len(bboxes)
            if segmentations_value is None or segmentations_value == []
            else _json_list(segmentations_value, "prediction.segmentations")
        )
        areas_value = prediction.get("areas")
        areas = (
            [None] * len(bboxes)
            if areas_value is None or areas_value == []
            else _json_list(areas_value, "prediction.areas")
        )
        video_id = _integer(prediction["video_id"], "prediction.video_id")
        for frame_idx, (bbox_value, segmentation, area_value) in enumerate(
            zip(bboxes, segmentations, areas)
        ):
            if bbox_value is None:
                continue
            bbox = [
                _number(item, "bbox coordinate")
                for item in _json_list(bbox_value, "bbox")
            ]
            if len(bbox) != 4:
                raise ValueError("prediction bbox must contain four coordinates.")
            if all(value == 0 for value in bbox):
                continue
            image_id = image_id_map.get((video_id, frame_idx))
            if image_id is None:
                raise RuntimeError(
                    f"prediction video_id={video_id}, frame_idx={frame_idx} "
                    "does not match converted COCO images"
                )
            x, y, width, height = bbox
            supplied_area = (
                None if area_value is None else _number(area_value, "prediction area")
            )
            area = (
                supplied_area
                if supplied_area is not None and supplied_area > 0
                else width * height
            )
            annotation: JsonObject = {
                "image_id": image_id,
                "video_id": video_id,
                "track_id": track_id,
                "category_id": _integer(
                    prediction["category_id"], "prediction.category_id"
                ),
                "bbox": [x, y, width, height],
                "area": area,
                "iscrowd": 0,
                "score": _number(prediction["score"], "prediction.score"),
            }
            if segmentation is not None:
                annotation["segmentation"] = segmentation
            coco_annotations.append(annotation)

    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(coco_annotations, handle)
