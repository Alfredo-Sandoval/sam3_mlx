# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Single-process-safe distributed utility surface for the MLX port."""

from __future__ import annotations

import functools
from typing import Any, Callable, List, Tuple

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


def _raise_distributed_unsupported(feature: str) -> None:
    raise_unsupported(
        feature,
        reason="torch-distributed",
        detail=_UNSUPPORTED_DISTRIBUTED_MESSAGE,
    )


@functools.lru_cache()
def _get_global_gloo_group():
    _raise_distributed_unsupported("_get_global_gloo_group")


def is_main_process():
    """Return true if the current process is the main one."""

    return get_rank() == 0


def all_gather_via_filesys(data, filesys_save_dir=None, gather_to_rank_0_only=False):
    """Single-process no-op equivalent of upstream filesystem gather."""

    return [data]


def all_gather(data, force_cpu=False, force_filesys=False, filesys_save_dir=None):
    """Single-process no-op equivalent of upstream object gather."""

    return [data]


def convert_to_distributed_tensor(tensor) -> Tuple[Any, str]:
    _raise_distributed_unsupported("convert_to_distributed_tensor")


def convert_to_normal_tensor(tensor, orig_device: str):
    return tensor


def is_distributed_training_run() -> bool:
    return False


def is_primary() -> bool:
    return get_rank() == _PRIMARY_RANK


def all_reduce_mean(tensor):
    return tensor


def all_reduce_sum(tensor):
    return tensor


def all_reduce_min(tensor):
    return tensor


def all_reduce_max(tensor):
    return tensor


def all_reduce_op(
    tensor,
    op,
    after_op_func: Callable[[Any], Any] = None,
):
    if after_op_func is not None:
        return after_op_func(tensor)
    return tensor


def gather_tensors_from_all(tensor) -> List[Any]:
    return [tensor]


def gather_from_all(tensor):
    return tensor


def broadcast(tensor, src: int = 0):
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
    model,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = True,
    bucket_cap_mb: int = 25,
):
    _raise_distributed_unsupported("init_distributed_data_parallel_model")


def broadcast_object(obj: Any, src: int = _PRIMARY_RANK, use_disk: bool = True) -> Any:
    return obj


def all_gather_tensor(tensor, world_size=None):
    return [tensor]


def all_gather_batch(tensors: List[Any]):
    return tensors


class GatherLayer:
    @staticmethod
    def apply(*args, **kwargs):
        _raise_distributed_unsupported("GatherLayer")


def all_gather_batch_with_grad(tensors):
    return tensors


def unwrap_ddp_if_wrapped(model):
    return model


def create_new_process_group(group_size):
    _raise_distributed_unsupported("create_new_process_group")


def is_dist_avail_and_initialized():
    return False


def gather_to_rank_0_via_filesys(data, filesys_save_dir=None):
    return [data]
