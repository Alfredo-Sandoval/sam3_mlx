# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Official-shaped Trainer API for sam3_mlx.

Full SAM3 training is still a PyTorch/DDP surface upstream.  This file keeps
Hydra config targets and dataclass names importable while failing before any
implicit backend fallback can happen.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Never, TypeVar, cast

from sam3_mlx._unsupported import UPSTREAM_COMMIT, raise_unsupported


CORE_LOSS_KEY = "core_loss"

_UNSUPPORTED_TRAINER_MESSAGE = (
    "Official SAM3 Torch trainer/distributed training behavior is not "
    "implemented in sam3_mlx. The official trainer at commit "
    f"{UPSTREAM_COMMIT} depends on PyTorch modules, AMP, "
    "DDP, torch dataloaders, and torch checkpoint state. Use the "
    "inference/runtime paths that are explicitly ported to MLX, or port "
    "training end-to-end before instantiating Trainer."
)


def _raise_trainer_unsupported(feature: str) -> Never:
    raise_unsupported(
        feature,
        reason="training-loop",
        detail=_UNSUPPORTED_TRAINER_MESSAGE,
    )


_Model = TypeVar("_Model")


def unwrap_ddp_if_wrapped(model: _Model) -> _Model:
    return model


@dataclass
class OptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "float16"


def _normalize_amp(amp: object) -> OptimAMPConf:
    if isinstance(amp, OptimAMPConf):
        return amp
    if amp is None:
        return OptimAMPConf()
    if not isinstance(amp, Mapping):
        raise TypeError("amp must be a mapping or OptimAMPConf")

    amp_mapping = cast(Mapping[object, object], amp)
    keys = list(amp_mapping)
    if not all(isinstance(key, str) for key in keys):
        raise TypeError("amp keys must be strings")
    extra_keys = set(cast(list[str], keys)) - {"enabled", "amp_dtype"}
    if extra_keys:
        raise TypeError(f"Unexpected amp keys: {sorted(extra_keys)}")
    enabled = amp_mapping.get("enabled", False)
    amp_dtype = amp_mapping.get("amp_dtype", "float16")
    if not isinstance(enabled, bool):
        raise TypeError("amp enabled must be bool")
    if not isinstance(amp_dtype, str):
        raise TypeError("amp amp_dtype must be str")
    return OptimAMPConf(enabled=enabled, amp_dtype=amp_dtype)


@dataclass
class OptimConf:
    optimizer: object = None
    options: Mapping[str, object] | None = None
    param_group_modifiers: list[object] | None = None
    amp: OptimAMPConf | Mapping[str, object] | None = None
    gradient_clip: object = None
    gradient_logger: object = None

    def __post_init__(self) -> None:
        self.amp = _normalize_amp(self.amp)


@dataclass
class DistributedConf:
    backend: str | None = None
    comms_dtype: str | None = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30
    gradient_as_bucket_view: bool = False
    static_graph: bool = False


@dataclass
class AcceleratorConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False
    matmul_allow_tf32: bool | None = None
    cudnn_allow_tf32: bool | None = None


@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int
    save_list: list[int] = field(default_factory=lambda: [])
    model_weight_initializer: object = None
    save_best_meters: list[str] | None = None
    skip_saving_parameters: list[str] = field(default_factory=lambda: [])
    initialize_after_preemption: bool | None = None
    resume_from: str | None = None

    def infer_missing(self) -> "CheckpointConf":
        if self.initialize_after_preemption is None:
            with_skip_saving = len(self.skip_saving_parameters) > 0
            self.initialize_after_preemption = with_skip_saving
        return self


@dataclass
class LoggingConf:
    log_dir: str
    log_freq: int
    tensorboard_writer: object
    log_level_primary: str = "INFO"
    log_level_secondary: str = "ERROR"
    log_scalar_frequency: int = 100
    log_visual_frequency: int = 100
    scalar_keys_to_log: Mapping[str, object] | None = None
    log_batch_stats: bool = False
    wandb_writer: object = None


class Trainer:
    """Official-shaped trainer placeholder for unsupported PyTorch training."""

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Mapping[str, object],
        model: Mapping[str, object],
        logging: Mapping[str, object],
        checkpoint: Mapping[str, object],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "mlx",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Mapping[str, bool] | None = None,
        accelerator_config: Mapping[str, bool] | None = None,
        env_variables: Mapping[str, object] | None = None,
        optim: Mapping[str, object] | None = None,
        optim_overrides: list[Mapping[str, object]] | None = None,
        meters: Mapping[str, object] | None = None,
        loss: Mapping[str, object] | None = None,
        skip_first_val: bool = False,
        skip_saving_ckpts: bool = False,
        empty_gpu_mem_cache_after_eval: bool = True,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self._setup_env_variables(env_variables)
        self.data_conf = data
        self.model_conf = model
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.max_epochs = max_epochs
        self.mode = mode
        self.accelerator = accelerator
        self.seed_value = seed_value
        self.val_epoch_freq = val_epoch_freq
        self.distributed_conf = distributed or {}
        self.accelerator_conf = accelerator_config or {}
        self.optim_conf = optim
        self.optim_overrides = optim_overrides
        self.meters_conf = meters
        self.loss_conf = loss
        self.skip_first_val = skip_first_val
        self.skip_saving_ckpts = skip_saving_ckpts
        self.empty_gpu_mem_cache_after_eval = empty_gpu_mem_cache_after_eval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        _raise_trainer_unsupported("Trainer.__init__")

    def _setup_env_variables(
        self, env_variables_conf: Mapping[str, object] | None
    ) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = str(value)

    def run(self) -> Never:
        _raise_trainer_unsupported("Trainer.run")

    def run_train(self) -> Never:
        _raise_trainer_unsupported("Trainer.run_train")

    def run_val(self) -> Never:
        _raise_trainer_unsupported("Trainer.run_val")

    def train_epoch(self, train_loader: object) -> Never:
        del train_loader
        _raise_trainer_unsupported("Trainer.train_epoch")

    def val_epoch(self, val_loader: object, phase: str) -> Never:
        del val_loader, phase
        _raise_trainer_unsupported("Trainer.val_epoch")


def print_model_summary(model: object, log_dir: str = "") -> Never:
    del model, log_dir
    _raise_trainer_unsupported("print_model_summary")


def get_human_readable_count(number: int) -> str:
    del number
    _raise_trainer_unsupported("get_human_readable_count")
