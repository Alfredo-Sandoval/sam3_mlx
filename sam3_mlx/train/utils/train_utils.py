# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Pure Python training utilities kept importable for the MLX port."""

from __future__ import annotations

import logging
import math
import os
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import import_module
from numbers import Integral
from pathlib import Path
from typing import Never, Protocol, SupportsInt, cast

import numpy as np

from sam3_mlx._unsupported import UPSTREAM_COMMIT, raise_unsupported


_UNSUPPORTED_TRAIN_UTILS_MESSAGE = (
    "This official SAM3 train utility is not implemented for the MLX port yet. "
    "The upstream implementation at commit "
    f"{UPSTREAM_COMMIT} depends on PyTorch-only distributed "
    "training semantics."
)


def _raise_train_utils_unsupported(feature: str) -> Never:
    raise_unsupported(
        feature,
        reason="training-loop",
        detail=_UNSUPPORTED_TRAIN_UTILS_MESSAGE,
    )


class _ValuesConfig(Protocol):
    def values(self) -> Iterable[object]: ...


class _OmegaConf(Protocol):
    def merge(self, *configs: object) -> object: ...

    def register_new_resolver(
        self,
        name: str,
        resolver: Callable[..., object],
        *,
        replace: bool,
    ) -> None: ...

    def to_yaml(self, config: object) -> str: ...


class _HydraUtils(Protocol):
    def get_method(self, path: str) -> object: ...

    def get_class(self, path: str) -> type: ...


class _Hydra(Protocol):
    utils: _HydraUtils


class ComputedMeter(Protocol):
    def compute(self) -> Mapping[str, int | float]: ...


def _require_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _config_values(config: object) -> Iterable[object] | None:
    if isinstance(config, Mapping):
        return cast(Mapping[object, object], config).values()
    if isinstance(config, (list, tuple)):
        return cast(Sequence[object], config)
    values = getattr(config, "values", None)
    if callable(values):
        return cast(_ValuesConfig, config).values()
    return None


def multiply_all(*args: int | float) -> int | float:
    return cast(int | float, np.prod(np.asarray(args)).item())


def collect_dict_keys(config: object) -> list[object]:
    """Recursively collect collate ``dict_key`` values from a config object."""

    val_keys: list[object] = []
    if isinstance(config, Mapping):
        mapping = cast(Mapping[object, object], config)
        target = mapping.get("_target_")
        if isinstance(target, str) and re.match(r".*collate_fn.*", target):
            val_keys.append(mapping["dict_key"])
            return val_keys
    values = _config_values(cast(object, config))
    if values is None:
        return val_keys

    for value in values:
        if _config_values(value) is not None:
            val_keys.extend(collect_dict_keys(value))
    return val_keys


class Phase:
    TRAIN = "train"
    VAL = "val"


def _add(x: int | float, y: int | float) -> int | float:
    return x + y


def _divide(x: int | float, y: int | float) -> float:
    return x / y


def _power(x: int | float, y: int | float) -> object:
    return x**y


def _subtract(x: int | float, y: int | float) -> int | float:
    return x - y


def _integer_range(value: object) -> list[int]:
    return list(range(_require_integer(value, "range argument")))


def _to_integer(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("integer resolver does not accept booleans")
    return int(cast(str | bytes | SupportsInt, value))


def _ceil_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("ceil_int resolver expects a real number")
    return int(math.ceil(value))


def _to_string(value: object) -> str:
    return str(value)


def register_omegaconf_resolvers() -> None:
    try:
        omegaconf_module = import_module("omegaconf")
        omega_conf = cast(_OmegaConf, getattr(omegaconf_module, "OmegaConf"))
    except ImportError as exc:  # pragma: no cover - optional config dependency
        raise NotImplementedError(
            "register_omegaconf_resolvers requires omegaconf, which is not a "
            "runtime dependency of sam3_mlx."
        ) from exc

    try:
        hydra = cast(_Hydra, import_module("hydra"))
        get_method = cast(Callable[..., object], hydra.utils.get_method)
        get_class = cast(Callable[..., object], hydra.utils.get_class)
    except ImportError:  # pragma: no cover - optional config dependency

        def missing_get_method(*args: object, **kwargs: object) -> Never:
            raise NotImplementedError("Hydra get_method resolver is unavailable")

        def missing_get_class(*args: object, **kwargs: object) -> Never:
            raise NotImplementedError("Hydra get_class resolver is unavailable")

        get_method = missing_get_method
        get_class = missing_get_class

    def merge(*configs: object) -> object:
        return omega_conf.merge(*configs)

    resolvers: dict[str, Callable[..., object]] = {
        "get_method": get_method,
        "get_class": get_class,
        "add": _add,
        "times": multiply_all,
        "divide": _divide,
        "pow": _power,
        "subtract": _subtract,
        "range": _integer_range,
        "int": _to_integer,
        "ceil_int": _ceil_integer,
        "merge": merge,
        "string": _to_string,
    }
    for name, resolver in resolvers.items():
        omega_conf.register_new_resolver(name, resolver, replace=True)


def setup_distributed_backend(backend: str, timeout_mins: int) -> Never:
    _raise_train_utils_unsupported("setup_distributed_backend")


def get_machine_local_and_dist_rank() -> tuple[int, int]:
    """Return local and distributed ranks, defaulting to single-process rank 0."""

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed_rank = int(os.environ.get("RANK", 0))
    return local_rank, distributed_rank


def print_cfg(cfg: object) -> None:
    logging.info("Training with config:")
    try:
        omegaconf_module = import_module("omegaconf")
        omega_conf = cast(_OmegaConf, getattr(omegaconf_module, "OmegaConf"))
        logging.info(omega_conf.to_yaml(cfg))
    except ImportError:
        logging.info("%s", cfg)


def set_seeds(seed_value: object, max_epochs: object, dist_rank: object) -> None:
    """Set Python, NumPy, and MLX random seeds when MLX is available."""

    seed = _require_integer(seed_value, "seed_value")
    epochs = _require_integer(max_epochs, "max_epochs")
    rank = _require_integer(dist_rank, "dist_rank")
    seed_value = (seed + rank) * epochs
    logging.info(f"MACHINE SEED: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    try:
        import mlx.core as mx

        mx.random.seed(seed_value)
    except ImportError:  # pragma: no cover - MLX is platform dependent
        logging.debug("MLX is unavailable; only Python and NumPy seeds were set.")


def makedir(dir_path: str | os.PathLike[str]) -> bool:
    """Create a directory if it does not exist."""

    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        logging.info(f"Error creating directory: {dir_path}")
        return False


def is_dist_avail_and_initialized() -> bool:
    return False


def get_amp_type(amp_type: str | None = None) -> object:
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


def log_env_variables() -> None:
    env_keys = sorted(list(os.environ.keys()))
    st = ""
    for key in env_keys:
        st += f"{key}={os.environ[key]}\n"
    logging.info("Logging ENV_VARIABLES")
    logging.info(st)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str, device: object, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self._allow_updates = True

    def update(self, val: int | float, n: object = 1) -> None:
        count = _require_integer(n, "n")
        self.val = float(val)
        self.sum += self.val * count
        self.count += count
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class MemMeter:
    """Official-shaped accelerator memory meter placeholder."""

    def __init__(self, name: str, device: object, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.peak = 0.0
        self.sum = 0.0
        self.count = 0
        self._allow_updates = True

    def update(self, n: int = 1, reset_peak_usage: bool = True) -> Never:
        _raise_train_utils_unsupported("MemMeter.update")

    def __str__(self) -> str:
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


def human_readable_time(time_seconds: int | float) -> str:
    time = int(time_seconds)
    minutes, _seconds = divmod(time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days:02}d {hours:02}h {minutes:02}m"


class DurationMeter:
    def __init__(self, name: str, device: object, fmt: str = ":f") -> None:
        self.name = name
        self.device = device
        self.fmt = fmt
        self.val = 0.0

    def reset(self) -> None:
        self.val = 0.0

    def update(self, val: int | float) -> None:
        self.val = float(val)

    def add(self, val: int | float) -> None:
        self.val += float(val)

    def __str__(self) -> str:
        return f"{self.name}: {human_readable_time(self.val)}"


class ProgressMeter:
    def __init__(
        self,
        num_batches: int,
        meters: Sequence[object],
        real_meters: Mapping[str, ComputedMeter],
        prefix: str = "",
    ) -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.real_meters = real_meters
        self.prefix = prefix

    def display(self, batch: int, enable_print: bool = False) -> None:
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

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_resume_checkpoint(
    checkpoint_save_dir: str | os.PathLike[str],
) -> str | None:
    checkpoint_dir = Path(checkpoint_save_dir)
    if not checkpoint_dir.is_dir():
        return None
    ckpt_file = checkpoint_dir / "checkpoint.pt"
    if not ckpt_file.is_file():
        return None
    return str(ckpt_file)
