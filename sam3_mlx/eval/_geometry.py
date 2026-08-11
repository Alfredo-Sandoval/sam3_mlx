"""Shared NumPy geometry helpers for evaluation adapters."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import cast


def convert_to_xywh(boxes: object) -> npt.NDArray[np.float32]:
    """Convert boxes from ``XYXY`` to ``XYWH`` without changing their order."""
    box_array = np.asarray(cast(npt.ArrayLike, boxes), dtype=np.float32)
    xmin, ymin, xmax, ymax = np.moveaxis(box_array, -1, 0)
    return np.stack((xmin, ymin, xmax - xmin, ymax - ymin), axis=-1)
