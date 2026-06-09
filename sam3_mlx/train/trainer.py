# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Official-shaped Trainer API for sam3_mlx.

Full SAM3 training is still a PyTorch/DDP surface upstream.  This file keeps
Hydra config targets and dataclass names importable while failing before any
implicit backend fallback can happen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

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


def _raise_trainer_unsupported(feature: str) -> None:
    raise_unsupported(
        feature,
        reason="training-loop",
        detail=_UNSUPPORTED_TRAINER_MESSAGE,
    )


def unwrap_ddp_if_wrapped(model):
    return model


@dataclass
class OptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "float16"


@dataclass
class OptimConf:
    optimizer: Any = None
    options: Optional[Dict[str, Any]] = None
    param_group_modifiers: Optional[List] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None
    gradient_logger: Any = None

    def __post_init__(self):
        if not isinstance(self.amp, OptimAMPConf):
            if self.amp is None:
                self.amp = {}
            if not isinstance(self.amp, Mapping):
                raise AssertionError("amp must be a mapping or OptimAMPConf")
            self.amp = OptimAMPConf(**self.amp)


@dataclass
class DistributedConf:
    backend: Optional[str] = None
    comms_dtype: Optional[str] = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30
    gradient_as_bucket_view: bool = False
    static_graph: bool = False


@dataclass
class AcceleratorConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False
    matmul_allow_tf32: Optional[bool] = None
    cudnn_allow_tf32: Optional[bool] = None


@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int
    save_list: List[int] = field(default_factory=list)
    model_weight_initializer: Any = None
    save_best_meters: List[str] = None
    skip_saving_parameters: List[str] = field(default_factory=list)
    initialize_after_preemption: Optional[bool] = None
    resume_from: Optional[str] = None

    def infer_missing(self):
        if self.initialize_after_preemption is None:
            with_skip_saving = len(self.skip_saving_parameters) > 0
            self.initialize_after_preemption = with_skip_saving
        return self


@dataclass
class LoggingConf:
    log_dir: str
    log_freq: int
    tensorboard_writer: Any
    log_level_primary: str = "INFO"
    log_level_secondary: str = "ERROR"
    log_scalar_frequency: int = 100
    log_visual_frequency: int = 100
    scalar_keys_to_log: Optional[Dict[str, Any]] = None
    log_batch_stats: bool = False
    wandb_writer: Optional[Any] = None


class Trainer:
    """Official-shaped trainer placeholder for unsupported PyTorch training."""

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "mlx",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        accelerator_config: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        optim_overrides: Optional[List[Dict[str, Any]]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        skip_first_val: bool = False,
        skip_saving_ckpts: bool = False,
        empty_gpu_mem_cache_after_eval: bool = True,
        gradient_accumulation_steps: int = 1,
    ):
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

    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = str(value)

    def run(self):
        _raise_trainer_unsupported("Trainer.run")

    def run_train(self):
        _raise_trainer_unsupported("Trainer.run_train")

    def run_val(self):
        _raise_trainer_unsupported("Trainer.run_val")

    def train_epoch(self, train_loader):
        _raise_trainer_unsupported("Trainer.train_epoch")

    def val_epoch(self, val_loader, phase):
        _raise_trainer_unsupported("Trainer.val_epoch")


def print_model_summary(model, log_dir: str = ""):
    _raise_trainer_unsupported("print_model_summary")


def get_human_readable_count(number: int) -> str:
    _raise_trainer_unsupported("get_human_readable_count")
