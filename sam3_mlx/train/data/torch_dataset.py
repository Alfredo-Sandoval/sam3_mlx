# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""Pure-Python DataLoader compatibility surface for the MLX data port."""

from __future__ import annotations

import random
from typing import Callable, Iterable, Optional

from sam3_mlx._unsupported import raise_unsupported

MLX_TORCH_DATASET_BASE_COMMIT = "13ec0366cb85f7a025a9a36af94fa9eb9599b9d9"


class _PythonBatchLoader:
    def __init__(
        self,
        dataset,
        indices,
        batch_size: int,
        drop_last: bool,
        collate_fn: Optional[Callable],
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn

    def __iter__(self):
        batch = []
        for index in self.indices:
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch) if self.collate_fn is not None else batch
                batch = []
        if batch and not self.drop_last:
            yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def __len__(self):
        full_batches, remainder = divmod(len(self.indices), self.batch_size)
        return full_batches if self.drop_last or remainder == 0 else full_batches + 1


class TorchDataset:
    """Compatibility wrapper with no Torch dependency or hidden backend fallback."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        enable_distributed_sampler=False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if num_workers != 0:
            raise_unsupported(
                "sam3_mlx.train.data.torch_dataset.TorchDataset(num_workers)",
                reason="training-loop",
                detail="Multi-worker DataLoader requires torch worker processes.",
            )
        if pin_memory:
            raise_unsupported(
                "sam3_mlx.train.data.torch_dataset.TorchDataset(pin_memory=True)",
                reason="training-loop",
                detail="pin_memory is a PyTorch host-memory API.",
            )
        if worker_init_fn is not None:
            raise_unsupported(
                "sam3_mlx.train.data.torch_dataset.TorchDataset(worker_init_fn)",
                reason="training-loop",
                detail="worker_init_fn is unused without worker processes.",
            )
        if enable_distributed_sampler:
            raise_unsupported(
                "sam3_mlx.train.data.torch_dataset.TorchDataset(enable_distributed_sampler=True)",
                reason="training-loop",
                detail="Distributed sampling requires torch.distributed.",
            )
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise_unsupported(
                "sam3_mlx.train.data.torch_dataset.TorchDataset(iterable dataset)",
                reason="training-loop",
                detail="The MLX wrapper only supports map-style Python datasets.",
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn

    def get_loader(self, epoch) -> Iterable:
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        if hasattr(self.dataset, "set_curr_epoch"):
            self.dataset.set_curr_epoch(epoch)

        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(epoch)
            rng.shuffle(indices)
        return _PythonBatchLoader(
            dataset=self.dataset,
            indices=indices,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )


__all__ = [
    "MLX_TORCH_DATASET_BASE_COMMIT",
    "TorchDataset",
]
