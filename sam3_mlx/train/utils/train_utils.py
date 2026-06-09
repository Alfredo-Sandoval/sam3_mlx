# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Pure Python training utilities kept importable for the MLX port."""

from __future__ import annotations

import logging
import math
import os
import random
import re
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from sam3_mlx._unsupported import UPSTREAM_COMMIT, raise_unsupported


_UNSUPPORTED_TRAIN_UTILS_MESSAGE = (
    "This official SAM3 train utility is not implemented for the MLX port yet. "
    "The upstream implementation at commit "
    f"{UPSTREAM_COMMIT} depends on PyTorch-only distributed "
    "training semantics."
)


def _raise_train_utils_unsupported(feature: str) -> None:
    raise_unsupported(
        feature,
        reason="training-loop",
        detail=_UNSUPPORTED_TRAIN_UTILS_MESSAGE,
    )


def multiply_all(*args):
    return np.prod(np.array(args)).item()


def collect_dict_keys(config):
    """Recursively collect collate ``dict_key`` values from a config object."""

    val_keys = []
    if isinstance(config, Mapping):
        if "_target_" in config and re.match(r".*collate_fn.*", config["_target_"]):
            val_keys.append(config["dict_key"])
            return val_keys
        values = config.values()
    elif isinstance(config, (list, tuple)):
        values = config
    elif hasattr(config, "values"):
        values = config.values()
    else:
        return val_keys

    for value in values:
        if isinstance(value, (Mapping, list, tuple)) or hasattr(value, "values"):
            val_keys.extend(collect_dict_keys(value))
    return val_keys


class Phase:
    TRAIN = "train"
    VAL = "val"


def register_omegaconf_resolvers():
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - optional config dependency
        raise NotImplementedError(
            "register_omegaconf_resolvers requires omegaconf, which is not a "
            "runtime dependency of sam3_mlx."
        ) from exc

    try:
        import hydra

        get_method = hydra.utils.get_method
        get_class = hydra.utils.get_class
    except ImportError:  # pragma: no cover - optional config dependency

        def get_method(*args, **kwargs):
            raise NotImplementedError("Hydra get_method resolver is unavailable")

        def get_class(*args, **kwargs):
            raise NotImplementedError("Hydra get_class resolver is unavailable")

    resolvers = {
        "get_method": get_method,
        "get_class": get_class,
        "add": lambda x, y: x + y,
        "times": multiply_all,
        "divide": lambda x, y: x / y,
        "pow": lambda x, y: x**y,
        "subtract": lambda x, y: x - y,
        "range": lambda x: list(range(x)),
        "int": lambda x: int(x),
        "ceil_int": lambda x: int(math.ceil(x)),
        "merge": lambda *x: OmegaConf.merge(*x),
        "string": lambda x: str(x),
    }
    for name, resolver in resolvers.items():
        OmegaConf.register_new_resolver(name, resolver, replace=True)


def setup_distributed_backend(backend, timeout_mins):
    _raise_train_utils_unsupported("setup_distributed_backend")


def get_machine_local_and_dist_rank():
    """Return local and distributed ranks, defaulting to single-process rank 0."""

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed_rank = int(os.environ.get("RANK", 0))
    return local_rank, distributed_rank


def print_cfg(cfg):
    logging.info("Training with config:")
    try:
        from omegaconf import OmegaConf

        logging.info(OmegaConf.to_yaml(cfg))
    except ImportError:
        logging.info("%s", cfg)


def set_seeds(seed_value, max_epochs, dist_rank):
    """Set Python, NumPy, and MLX random seeds when MLX is available."""

    seed_value = (seed_value + dist_rank) * max_epochs
    logging.info(f"MACHINE SEED: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    try:
        import mlx.core as mx

        mx.random.seed(seed_value)
    except ImportError:  # pragma: no cover - MLX is platform dependent
        logging.debug("MLX is unavailable; only Python and NumPy seeds were set.")


def makedir(dir_path):
    """Create a directory if it does not exist."""

    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        logging.info(f"Error creating directory: {dir_path}")
        return False


def is_dist_avail_and_initialized():
    return False


def get_amp_type(amp_type: Optional[str] = None):
    if amp_type is None:
        return None
    if amp_type not in ["bfloat16", "float16"]:
        raise AssertionError("Invalid Amp type.")
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - MLX is platform dependent
        raise NotImplementedError(
            "get_amp_type requires MLX for MLX dtype objects."
        ) from exc
    return mx.bfloat16 if amp_type == "bfloat16" else mx.float16


def log_env_variables():
    env_keys = sorted(list(os.environ.keys()))
    st = ""
    for key in env_keys:
        st += f"{key}={os.environ[key]}\n"
    logging.info("Logging ENV_VARIABLES")
    logging.info(st)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, device, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self._allow_updates = True

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class MemMeter:
    """Official-shaped accelerator memory meter placeholder."""

    def __init__(self, name, device, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.peak = 0
        self.sum = 0
        self.count = 0
        self._allow_updates = True

    def update(self, n=1, reset_peak_usage=True):
        _raise_train_utils_unsupported("MemMeter.update")

    def __str__(self):
        fmtstr = (
            "{name}: {val"
            + self.fmt
            + "} ({avg"
            + self.fmt
            + "}/{peak"
            + self.fmt
            + "})"
        )
        return fmtstr.format(**self.__dict__)


def human_readable_time(time_seconds):
    time = int(time_seconds)
    minutes, seconds = divmod(time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days:02}d {hours:02}h {minutes:02}m"


class DurationMeter:
    def __init__(self, name, device, fmt=":f"):
        self.name = name
        self.device = device
        self.fmt = fmt
        self.val = 0

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def add(self, val):
        self.val += val

    def __str__(self):
        return f"{self.name}: {human_readable_time(self.val)}"


class ProgressMeter:
    def __init__(self, num_batches, meters, real_meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.real_meters = real_meters
        self.prefix = prefix

    def display(self, batch, enable_print=False):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += [
            " | ".join(
                [
                    f"{os.path.join(name, subname)}: {val:.4f}"
                    for subname, val in meter.compute().items()
                ]
            )
            for name, meter in self.real_meters.items()
        ]
        logging.info(" | ".join(entries))
        if enable_print:
            print(" | ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_resume_checkpoint(checkpoint_save_dir):
    checkpoint_dir = Path(checkpoint_save_dir)
    if not checkpoint_dir.is_dir():
        return None
    ckpt_file = checkpoint_dir / "checkpoint.pt"
    if not ckpt_file.is_file():
        return None
    return str(ckpt_file)
