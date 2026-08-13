"""Pure temporal-index planning shared by tracker memory paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def previous_memory_frame_index(
    *,
    frame_idx: int,
    temporal_distance: int,
    stride: int,
    track_in_reverse: bool,
    selected_indices: Sequence[int] | None = None,
) -> int | None:
    """Resolve one prior memory slot without reading or mutating tracker state."""
    if temporal_distance < 1:
        raise ValueError("temporal_distance must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    if selected_indices is not None:
        if temporal_distance > len(selected_indices):
            return None
        return selected_indices[-temporal_distance]
    if temporal_distance == 1:
        direction = 1 if track_in_reverse else -1
        return frame_idx + direction
    if track_in_reverse:
        aligned = -(-(frame_idx + 2) // stride) * stride
        return aligned + (temporal_distance - 2) * stride
    aligned = ((frame_idx - 2) // stride) * stride
    return aligned - (temporal_distance - 2) * stride


@dataclass(frozen=True)
class MemoryTrimPlan:
    expired_frame_idx: int
    far_history_frame_idx: int | None


def memory_trim_plan(
    *,
    frame_idx: int,
    stride: int,
    num_maskmem: int,
    use_memory_selection: bool,
    max_object_pointers: int,
) -> MemoryTrimPlan:
    """Identify mutable tracker entries to trim, without touching their state."""
    if stride < 1:
        raise ValueError("stride must be positive")
    if num_maskmem < 0:
        raise ValueError("num_maskmem must be non-negative")
    if max_object_pointers < 0:
        raise ValueError("max_object_pointers must be non-negative")
    return MemoryTrimPlan(
        expired_frame_idx=frame_idx - stride * num_maskmem,
        far_history_frame_idx=(
            frame_idx - 20 * max_object_pointers if use_memory_selection else None
        ),
    )
