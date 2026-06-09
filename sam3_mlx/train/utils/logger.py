# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Logging utilities for official-shaped training config imports."""

from __future__ import annotations

import atexit
import functools
import logging
import sys
import uuid
from typing import Any, Dict, Optional, Union

import numpy as np

from sam3_mlx._unsupported import UPSTREAM_COMMIT, raise_unsupported
from sam3_mlx.train.utils.train_utils import get_machine_local_and_dist_rank, makedir


Scalar = Union[np.ndarray, int, float]

_UNSUPPORTED_LOGGER_MESSAGE = (
    "TensorBoard logging from official SAM3 training is not implemented in the "
    "MLX port yet. The upstream implementation at commit "
    f"{UPSTREAM_COMMIT} depends on torch.utils.tensorboard."
)


def _raise_logger_unsupported(feature: str) -> None:
    raise_unsupported(
        feature,
        reason="training-loop",
        detail=_UNSUPPORTED_LOGGER_MESSAGE,
    )


def make_tensorboard_logger(log_dir: str, **writer_kwargs: Any):
    _raise_logger_unsupported("make_tensorboard_logger")


class TensorBoardWriterWrapper:
    """Wrapper around a SummaryWriter-like object when one is provided."""

    def __init__(
        self,
        path: str,
        *args: Any,
        filename_suffix: str = None,
        summary_writer_method: Any = None,
        **kwargs: Any,
    ) -> None:
        if summary_writer_method is None:
            _raise_logger_unsupported("TensorBoardWriterWrapper")
        self._writer = None
        _, self._rank = get_machine_local_and_dist_rank()
        self._path: str = path
        if self._rank == 0:
            self._writer = summary_writer_method(
                log_dir=path,
                *args,
                filename_suffix=filename_suffix or str(uuid.uuid4()),
                **kwargs,
            )
        atexit.register(self.close)

    @property
    def writer(self):
        return self._writer

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        if not self._writer:
            return
        self._writer.flush()

    def close(self) -> None:
        if not self._writer:
            return
        self._writer.close()
        self._writer = None


class TensorBoardLogger(TensorBoardWriterWrapper):
    """A simple logger for SummaryWriter-like objects."""

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if not self._writer:
            return
        for key, value in payload.items():
            self.log(key, value, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if not self._writer:
            return
        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        if not self._writer:
            return
        self._writer.add_hparams(hparams, meters)


class Logger:
    """Official-shaped logger aggregator with TensorBoard disabled by default."""

    def __init__(self, logging_conf):
        tb_config = getattr(logging_conf, "tensorboard_writer", None)
        if isinstance(logging_conf, dict):
            tb_config = logging_conf.get("tensorboard_writer")
        if tb_config:
            should_log = (
                tb_config.get("should_log", True) if hasattr(tb_config, "get") else True
            )
            if should_log:
                _raise_logger_unsupported("Logger.tensorboard_writer")
        self.tb_logger: Optional[TensorBoardLogger] = None

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log_dict(payload, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log(name, data, step)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        if self.tb_logger:
            self.tb_logger.log_hparams(hparams, meters)


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    log_buffer_kb = 10 * 1024
    io = open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io


def setup_logging(
    name,
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
):
    """Set up stdout and optional file logging."""

    log_filename = None
    if output_dir:
        makedir(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    log_format = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(log_format)

    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.root.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if rank == 0:
        console_handler.setLevel(log_level_primary)
    else:
        console_handler.setLevel(log_level_secondary)

    if log_filename and rank == 0:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger


def shutdown_logging():
    """Close logging streams."""

    logging.info("Shutting down loggers...")
    handlers = logging.root.handlers
    for handler in handlers:
        handler.close()
