from __future__ import annotations

import logging
import math
from typing import Any, NoReturn, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sam3_mlx._unsupported import UPSTREAM_COMMIT


# Special values for object tracking.
_PADDING_NUM = -1
_REMOVED_NUM = -1116

logger = logging.getLogger(__name__)


class UnsupportedMultiplexRuntimeError(NotImplementedError):
    """Raised for official SAM3 multiplex paths that still require Torch-only."""


def raise_unsupported_multiplex_runtime(component: str) -> NoReturn:
    raise UnsupportedMultiplexRuntimeError(
        f"{component} is part of the official SAM3 multiplex Torch-only video "
        "runtime and is not implemented in this MLX slice. The MLX port "
        "currently provides pure multiplex helpers and fail-fast API shells; "
        "port the model path explicitly before using this component. "
        f"Upstream oracle commit: {UPSTREAM_COMMIT}."
    )


def _is_mlx_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("mlx.")


class MultiplexState:
    """
    Records object-to-bucket assignments and converts arrays between data space
    and multiplex space.

    The upstream implementation stores Torch transition matrices. This MLX port
    stores compact gather indices for the same partial permutation and returns
    the same array family passed to mux/demux.
    """

    def __init__(
        self,
        assignments: list[list[int]],
        device: Any = None,
        dtype: Any = mx.float32,
        allowed_bucket_capacity: int = 1,
        *,
        object_ids: Optional[list[int]] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.allowed_bucket_capacity = allowed_bucket_capacity
        self._initialize_assignments(assignments, object_ids=object_ids)

    def _initialize_assignments(
        self, assignments: list[list[int]], *, object_ids: Optional[list[int]] = None
    ) -> None:
        self.assignments = assignments
        self.num_buckets = len(self.assignments)
        if self.num_buckets == 0:
            logger.error("No buckets found in the state")
            raise ValueError("No buckets found in the state")

        self.multiplex_count = len(self.assignments[0])
        assert all(
            len(bucket) == self.multiplex_count for bucket in self.assignments
        ), "all buckets must have the same multiplex_count"

        self.total_valid_entries = sum(
            sum(1 for value in bucket if value >= 0) for bucket in self.assignments
        )
        self.total_non_padding_entries = sum(
            sum(1 for value in bucket if value != _PADDING_NUM)
            for bucket in self.assignments
        )

        self.object_ids = object_ids
        if self.object_ids is not None:
            assert len(self.object_ids) == self.total_valid_entries, (
                "object_ids should map 1:1 to the valid entries"
            )

        all_object_idxs: set[int] = set()
        for bucket in self.assignments:
            valid_entries_in_bucket = sum(
                1 for value in bucket if value != _PADDING_NUM
            )
            assert valid_entries_in_bucket <= self.allowed_bucket_capacity, (
                f"{valid_entries_in_bucket=} > {self.allowed_bucket_capacity=}"
            )
            for obj_idx in bucket:
                if obj_idx >= 0:
                    assert obj_idx < self.total_non_padding_entries, (
                        f"object ID {obj_idx} >= {self.total_non_padding_entries}"
                    )
                    assert obj_idx not in all_object_idxs, "object IDs must be unique"
                    all_object_idxs.add(obj_idx)

        self._precompute_transition_matrices(self.device, self.dtype)

    @property
    def available_slots(self) -> int:
        return (
            self.num_buckets * self.allowed_bucket_capacity
            - self.total_non_padding_entries
        )

    def find_next_batch_of_available_indices(
        self,
        num_objects: int,
        *,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ) -> list[int]:
        del prefer_new_buckets
        assert num_objects > 0, f"{num_objects=} must be positive"
        if not allow_new_buckets:
            assert self.available_slots >= num_objects, (
                f"not enough available slots {self.available_slots} < {num_objects}"
            )
        return list(
            range(self.total_valid_entries, self.total_valid_entries + num_objects)
        )

    def add_objects(
        self,
        object_indices: list[int],
        *,
        object_ids: Optional[list[int]] = None,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ) -> None:
        if len(object_indices) == 0:
            return

        object_indices = object_indices.copy()
        assert (object_ids is None) == (self.object_ids is None), (
            "object_ids must either be always given or always omitted"
        )
        if object_ids is not None:
            assert len(object_ids) == len(object_indices), (
                "object_ids must have the same length as object_indices"
            )
            object_ids = object_ids.copy()

        num_new_objects = len(object_indices)
        assert object_indices == sorted(object_indices), "object_indices must be sorted"
        object_indices.reverse()
        if object_ids is not None:
            object_ids.reverse()

        if prefer_new_buckets:
            assert allow_new_buckets, "prefer_new_buckets requires allow_new_buckets"

        slots_filled = 0
        buckets_created = 0

        def _pop_next() -> int:
            idx = object_indices.pop()
            if object_ids is not None and self.object_ids is not None:
                self.object_ids.append(object_ids.pop())
            return idx

        if not prefer_new_buckets:
            for bucket in self.assignments:
                for slot_idx in range(self.allowed_bucket_capacity):
                    if bucket[slot_idx] == _PADDING_NUM:
                        bucket[slot_idx] = _pop_next()
                        slots_filled += 1
                        if len(object_indices) == 0:
                            break
                if len(object_indices) == 0:
                    break

        if len(object_indices) > 0 and not allow_new_buckets:
            raise ValueError(
                "Cannot place objects "
                f"{list(reversed(object_indices))} without creating new buckets"
            )

        while len(object_indices) > 0:
            new_bucket = [_PADDING_NUM] * self.multiplex_count
            for slot_idx in range(self.allowed_bucket_capacity):
                if len(object_indices) == 0:
                    break
                new_bucket[slot_idx] = _pop_next()
            self.assignments.append(new_bucket)
            buckets_created += 1

        original_num_entries = self.total_valid_entries
        self._initialize_assignments(self.assignments, object_ids=self.object_ids)
        assert self.total_valid_entries == original_num_entries + num_new_objects, (
            f"{self.total_valid_entries=} != {original_num_entries=} + "
            f"{num_new_objects=}"
        )

        logger.info(
            "Filled %s slots and created %s new buckets",
            slots_filled,
            buckets_created,
        )

    def remove_objects(
        self, object_indices: list[int], strict: bool = True
    ) -> list[int]:
        object_indices = object_indices.copy()

        for bucket_idx, bucket in enumerate(self.assignments):
            for slot_idx, obj_id in enumerate(bucket):
                if obj_id in object_indices:
                    self.assignments[bucket_idx][slot_idx] = _REMOVED_NUM
                    object_indices.remove(obj_id)

        if strict:
            assert len(object_indices) == 0, (
                f"Failed to remove objects: {object_indices}"
            )

        buckets_to_remove = []
        buckets_to_keep = []
        for bucket_idx, bucket in enumerate(self.assignments):
            all_removed = all(
                obj_id in (_PADDING_NUM, _REMOVED_NUM) for obj_id in bucket
            )
            if all_removed:
                buckets_to_remove.append(bucket_idx)
            else:
                buckets_to_keep.append(bucket_idx)

        for bucket_idx in reversed(buckets_to_remove):
            del self.assignments[bucket_idx]

        if len(buckets_to_keep) == 0:
            self.assignments = None
            if self.object_ids is not None:
                self.object_ids = []
            return buckets_to_keep

        all_positive_ids = {
            obj_id for bucket in self.assignments for obj_id in bucket if obj_id >= 0
        }
        sorted_ids = sorted(all_positive_ids)
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}

        for bucket in self.assignments:
            for slot_idx, obj_id in enumerate(bucket):
                if obj_id >= 0:
                    bucket[slot_idx] = id_mapping[obj_id]

        if self.object_ids is not None:
            new_object_ids = [None] * len(sorted_ids)
            for old_idx, new_idx in id_mapping.items():
                new_object_ids[new_idx] = self.object_ids[old_idx]
            assert not any(obj_id is None for obj_id in new_object_ids)
            self.object_ids = new_object_ids

        self._initialize_assignments(self.assignments, object_ids=self.object_ids)
        return buckets_to_keep

    def _precompute_transition_matrices(self, device: Any, dtype: Any) -> None:
        del device
        del dtype
        mux_indices = np.zeros(
            self.num_buckets * self.multiplex_count,
            dtype=np.int64,
        )
        mux_valid = np.zeros(
            self.num_buckets * self.multiplex_count,
            dtype=bool,
        )
        demux_indices = np.zeros(self.total_valid_entries, dtype=np.int64)

        for bucket_idx in range(self.num_buckets):
            for slot_idx in range(self.multiplex_count):
                flat_bucket_idx = bucket_idx * self.multiplex_count + slot_idx
                object_idx = self.assignments[bucket_idx][slot_idx]
                if object_idx >= 0:
                    mux_indices[flat_bucket_idx] = object_idx
                    mux_valid[flat_bucket_idx] = True
                    demux_indices[object_idx] = flat_bucket_idx

        self.mux_indices_np = mux_indices
        self.mux_valid_np = mux_valid
        self.demux_indices_np = demux_indices
        self.mux_indices = mx.array(mux_indices, dtype=mx.int64)
        self.mux_valid = mx.array(mux_valid)
        self.demux_indices = mx.array(demux_indices, dtype=mx.int64)

    def mux(self, x: Any) -> Any:
        """
        Convert data space `(total_valid_entries, ...)` to multiplex space
        `(num_buckets, multiplex_count, ...)`.
        """
        num_valid_entries = x.shape[0]
        assert num_valid_entries == self.total_valid_entries, (
            f"{num_valid_entries=} != {self.total_valid_entries=}"
        )
        output_shape = (self.num_buckets, self.multiplex_count) + x.shape[1:]

        if num_valid_entries == 0:
            if _is_mlx_array(x):
                return mx.zeros(output_shape, dtype=x.dtype)
            return np.zeros(output_shape, dtype=np.asarray(x).dtype)

        if _is_mlx_array(x):
            gathered = mx.take(x, self.mux_indices, axis=0)
            mask_shape = self.mux_valid.shape + (1,) * (len(x.shape) - 1)
            return mx.where(
                self.mux_valid.reshape(mask_shape),
                gathered,
                mx.zeros_like(gathered),
            ).reshape(output_shape)

        x_np = np.asarray(x)
        gathered_np = x_np[self.mux_indices_np]
        return np.where(
            self.mux_valid_np.reshape(self.mux_valid_np.shape + (1,) * (x_np.ndim - 1)),
            gathered_np,
            np.zeros_like(gathered_np),
        ).reshape(output_shape)

    def demux(self, x: Any) -> Any:
        """
        Convert multiplex space `(num_buckets, multiplex_count, ...)` back to
        data space `(total_valid_entries, ...)`.
        """
        num_buckets, multiplex_count = x.shape[:2]
        assert num_buckets == self.num_buckets, f"{num_buckets=} != {self.num_buckets=}"
        assert multiplex_count == self.multiplex_count, (
            f"{multiplex_count=} != {self.multiplex_count=}"
        )
        output_shape = (self.total_valid_entries,) + x.shape[2:]

        if self.total_valid_entries == 0:
            if _is_mlx_array(x):
                return mx.zeros(output_shape, dtype=x.dtype)
            return np.zeros(output_shape, dtype=np.asarray(x).dtype)

        if _is_mlx_array(x):
            return mx.take(
                x.reshape(num_buckets * multiplex_count, *x.shape[2:]),
                self.demux_indices,
                axis=0,
            ).reshape(output_shape)

        x_np = np.asarray(x)
        return x_np.reshape(num_buckets * multiplex_count, *x_np.shape[2:])[
            self.demux_indices_np
        ].reshape(output_shape)

    def get_valid_object_mask(self) -> Any:
        return self.mux_valid.reshape(self.num_buckets, self.multiplex_count)

    def get_all_valid_object_idx(self) -> set[int]:
        return {
            obj_idx for bucket in self.assignments for obj_idx in bucket if obj_idx >= 0
        }


class MultiplexController(nn.Module):
    def __init__(
        self,
        multiplex_count: int,
        full_shuffle: bool = False,
        eval_multiplex_count: int = -1,
    ):
        super().__init__()
        self.multiplex_count = multiplex_count
        self.full_shuffle = full_shuffle
        self.eval_multiplex_count = (
            multiplex_count if eval_multiplex_count < 0 else eval_multiplex_count
        )
        assert self.multiplex_count >= 1

    @property
    def allowed_bucket_capacity(self) -> int:
        if getattr(self, "training", True):
            return self.multiplex_count
        return self.eval_multiplex_count

    def get_state(
        self,
        num_valid_entries: int,
        device: Any = None,
        dtype: Any = mx.float32,
        random: bool = True,
        *,
        object_ids: Optional[list[int]] = None,
    ) -> MultiplexState:
        allowed_bucket_capacity = self.allowed_bucket_capacity
        true_bucket_capacity = self.multiplex_count
        num_buckets = math.ceil(num_valid_entries / allowed_bucket_capacity)

        if self.full_shuffle:
            ids = np.concatenate(
                [
                    np.arange(num_valid_entries, dtype=np.int64),
                    np.full(
                        num_buckets * true_bucket_capacity - num_valid_entries,
                        _PADDING_NUM,
                        dtype=np.int64,
                    ),
                ],
                axis=0,
            )
            if random:
                ids = ids[np.random.permutation(ids.shape[0])]
            assignments = [
                ids[
                    bucket_idx * true_bucket_capacity : (bucket_idx + 1)
                    * true_bucket_capacity
                ].tolist()
                for bucket_idx in range(num_buckets)
            ]
        else:
            if random:
                ids = np.random.permutation(num_valid_entries).astype(np.int64)
            else:
                ids = np.arange(num_valid_entries, dtype=np.int64)

            total_elements = num_buckets * allowed_bucket_capacity
            if ids.shape[0] < total_elements:
                ids = np.concatenate(
                    [
                        ids,
                        np.full(
                            total_elements - ids.shape[0],
                            _PADDING_NUM,
                            dtype=np.int64,
                        ),
                    ]
                )

            assignments = []
            for bucket_idx in range(num_buckets):
                bucket = ids[
                    bucket_idx * allowed_bucket_capacity : (bucket_idx + 1)
                    * allowed_bucket_capacity
                ].tolist()
                bucket += [_PADDING_NUM] * (
                    true_bucket_capacity - allowed_bucket_capacity
                )
                assignments.append(bucket)

        return MultiplexState(
            assignments,
            device=device,
            dtype=dtype,
            allowed_bucket_capacity=allowed_bucket_capacity,
            object_ids=object_ids,
        )
