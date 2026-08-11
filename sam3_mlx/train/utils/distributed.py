# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Single-process-safe distributed utility surface for the MLX port."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Never

from sam3_mlx._unsupported import UPSTREAM_COMMIT, raise_unsupported


_ACCELERATOR_DEVICE_INDEX: int = 0
_CPU_DEVICE_INDEX = -1
_PRIMARY_RANK = 0

_UNSUPPORTED_DISTRIBUTED_MESSAGE = (
    "SAM3 distributed training is not implemented in the MLX port yet. The "
    "official implementation at commit "
    f"{UPSTREAM_COMMIT} depends on torch.distributed, NCCL, "
    "and PyTorch DDP. Add an explicit MLX/distributed design before enabling it."
)


def _raise_distributed_unsupported(feature: str) -> Never:
    raise_unsupported(
        feature,
        reason="torch-distributed",
        detail=_UNSUPPORTED_DISTRIBUTED_MESSAGE,
    )


@functools.lru_cache()
def get_global_gloo_group() -> Never:
    """Fail explicitly because the MLX port has no Gloo process group."""

    _raise_distributed_unsupported("_get_global_gloo_group")


def is_main_process() -> bool:
    """Return true if the current process is the main one."""

    return get_rank() == 0


def all_gather_via_filesys[T](
    data: T,
    filesys_save_dir: object = None,
    gather_to_rank_0_only: bool = False,
) -> list[T]:
    """Single-process no-op equivalent of upstream filesystem gather."""

    return [data]


def all_gather[T](
    data: T,
    force_cpu: bool = False,
    force_filesys: bool = False,
    filesys_save_dir: object = None,
) -> list[T]:
    """Single-process no-op equivalent of upstream object gather."""

    return [data]


def convert_to_distributed_tensor(tensor: object) -> Never:
    _raise_distributed_unsupported("convert_to_distributed_tensor")


def convert_to_normal_tensor[T](tensor: T, orig_device: str) -> T:
    return tensor


def is_distributed_training_run() -> bool:
    return False


def is_primary() -> bool:
    return get_rank() == _PRIMARY_RANK


def all_reduce_mean[T](tensor: T) -> T:
    return tensor


def all_reduce_sum[T](tensor: T) -> T:
    return tensor


def all_reduce_min[T](tensor: T) -> T:
    return tensor


def all_reduce_max[T](tensor: T) -> T:
    return tensor


def all_reduce_op[T, R](
    tensor: T,
    op: object,
    after_op_func: Callable[[T], R] | None = None,
) -> T | R:
    if after_op_func is not None:
        return after_op_func(tensor)
    return tensor


def gather_tensors_from_all[T](tensor: T) -> list[T]:
    return [tensor]


def gather_from_all[T](tensor: T) -> T:
    return tensor


def broadcast[T](tensor: T, src: int = 0) -> T:
    return tensor


def barrier() -> None:
    return None


def get_world_size() -> int:
    return 1


def get_rank() -> int:
    return 0


def get_primary_rank() -> int:
    return _PRIMARY_RANK


def set_accelerator_device_index(idx: int) -> None:
    _raise_distributed_unsupported("set_accelerator_device_index")


def set_cpu_device() -> None:
    _raise_distributed_unsupported("set_cpu_device")


def get_accelerator_device_index() -> int:
    return _ACCELERATOR_DEVICE_INDEX


def init_distributed_data_parallel_model(
    model: object,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = True,
    bucket_cap_mb: int = 25,
) -> Never:
    _raise_distributed_unsupported("init_distributed_data_parallel_model")


def broadcast_object[T](obj: T, src: int = _PRIMARY_RANK, use_disk: bool = True) -> T:
    return obj


def all_gather_tensor[T](tensor: T, world_size: int | None = None) -> list[T]:
    return [tensor]


def all_gather_batch[T](tensors: list[T]) -> list[T]:
    return tensors


class GatherLayer:
    @staticmethod
    def apply(*args: object, **kwargs: object) -> Never:
        _raise_distributed_unsupported("GatherLayer")


def all_gather_batch_with_grad[T](tensors: T) -> T:
    return tensors


def unwrap_ddp_if_wrapped[T](model: T) -> T:
    return model


def create_new_process_group(group_size: int) -> Never:
    _raise_distributed_unsupported("create_new_process_group")


def is_dist_avail_and_initialized() -> bool:
    return False


def gather_to_rank_0_via_filesys[T](
    data: T, filesys_save_dir: object = None
) -> list[T]:
    return [data]
