"""Self-contained COCO JSON re-indexing function."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _is_coco_json(data: Dict[str, Any]) -> bool:
    return isinstance(data, dict) and any(
        key in data for key in ("images", "annotations", "categories")
    )


def _check_zero_indexed(data: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    annotations_zero = any(
        ann.get("id", -1) == 0 for ann in data.get("annotations", [])
    )
    images_zero = any(img.get("id", -1) == 0 for img in data.get("images", []))
    categories_zero = any(cat.get("id", -1) == 0 for cat in data.get("categories", []))
    return annotations_zero, images_zero, categories_zero


def _reindex_coco_data(data: Dict[str, Any]) -> Dict[str, Any]:
    modified_data = json.loads(json.dumps(data))
    annotations_zero, images_zero, categories_zero = _check_zero_indexed(data)
    image_id_mapping = {}
    category_id_mapping = {}

    if images_zero:
        for img in modified_data.get("images", []):
            old_id = img["id"]
            image_id_mapping[old_id] = old_id + 1
            img["id"] = old_id + 1

    if categories_zero:
        for cat in modified_data.get("categories", []):
            old_id = cat["id"]
            category_id_mapping[old_id] = old_id + 1
            cat["id"] = old_id + 1

    for ann in modified_data.get("annotations", []):
        if annotations_zero:
            ann["id"] = ann["id"] + 1
        if images_zero and ann.get("image_id") in image_id_mapping:
            ann["image_id"] = image_id_mapping[ann["image_id"]]
        if categories_zero and ann.get("category_id") in category_id_mapping:
            ann["category_id"] = category_id_mapping[ann["category_id"]]

    return modified_data


def reindex_coco_to_temp(input_json_path: str) -> Optional[str]:
    """Convert 0-indexed COCO JSON IDs to 1-indexed IDs in a temp file."""
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input file not found: {input_json_path}")

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not _is_coco_json(data):
        raise ValueError(
            f"File does not appear to be in COCO format: {input_json_path}"
        )

    if any(_check_zero_indexed(data)):
        data = _reindex_coco_data(data)

    input_path = Path(input_json_path)
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(
        temp_dir, f"{input_path.stem}_1_indexed{input_path.suffix}"
    )
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return temp_path


def test_reindex_function():
    """Compatibility smoke function retained from upstream."""
    test_data = {
        "images": [{"id": 0, "width": 640, "height": 480, "file_name": "test.jpg"}],
        "categories": [{"id": 0, "name": "object"}],
        "annotations": [{"id": 0, "image_id": 0, "category_id": 0}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_data, f)
        test_file_path = f.name
    try:
        return reindex_coco_to_temp(test_file_path)
    finally:
        os.unlink(test_file_path)
