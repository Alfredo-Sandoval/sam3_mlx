# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Official-shaped SAM3 train launcher placeholder for the MLX port."""

from __future__ import annotations

import logging
import os
import sys
import traceback
from argparse import ArgumentParser, Namespace
from importlib import import_module
from typing import Never, Protocol, cast

from sam3_mlx._unsupported import UPSTREAM_COMMIT, raise_unsupported
from sam3_mlx.train.utils.train_utils import register_omegaconf_resolvers


os.environ["HYDRA_FULL_ERROR"] = "1"

_UNSUPPORTED_TRAIN_MESSAGE = (
    "The official SAM3 training launcher is not implemented in sam3_mlx yet. "
    "The upstream launcher at commit "
    f"{UPSTREAM_COMMIT} depends on Hydra, Submitit, torch.multiprocessing, "
    "and torch.distributed. This MLX fork currently exposes these names only so "
    "imports fail clearly instead of pulling in PyTorch."
)


class _OmegaConf(Protocol):
    def to_container(self, cfg: object, *, resolve: bool) -> object: ...

    def create(self, value: object) -> object: ...


class _ConfigGet(Protocol):
    def get(self, key: str, default: object = None) -> object: ...


def _raise_train_unsupported(feature: str) -> Never:
    raise_unsupported(
        feature,
        reason="training-loop",
        detail=_UNSUPPORTED_TRAIN_MESSAGE,
    )


class SlurmEvent:
    QUEUED = "QUEUED"
    START = "START"
    FINISH = "FINISH"
    JOB_ERROR = "JOB_ERROR"
    SLURM_SIGNAL = "SLURM_SIGNAL"


def handle_custom_resolving(cfg: object) -> object:
    try:
        omega_conf = cast(_OmegaConf, getattr(import_module("omegaconf"), "OmegaConf"))
    except ImportError:
        return cfg
    cfg_resolved = omega_conf.to_container(cfg, resolve=False)
    return omega_conf.create(cfg_resolved)


def single_proc_run(
    local_rank: int, main_port: int, cfg: object, world_size: int
) -> Never:
    del local_rank, main_port, cfg, world_size
    _raise_train_unsupported("single_proc_run")


def single_node_runner(cfg: object, main_port: int) -> Never:
    del cfg, main_port
    _raise_train_unsupported("single_node_runner")


def format_exception(e: Exception, limit: int | None = 20) -> str:
    traceback_str = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


class SubmititRunner:
    """Official-shaped Submitit runner placeholder."""

    def __init__(self, port: int, cfg: object) -> None:
        self.cfg = cfg
        self.port = port
        self.has_setup = False

    def run_trainer(self) -> Never:
        _raise_train_unsupported("SubmititRunner.run_trainer")

    def __call__(self) -> Never:
        _raise_train_unsupported("SubmititRunner")

    def setup_job_info(self, job_id: str, rank: int) -> None:
        cluster = (
            cast(_ConfigGet, self.cfg).get("cluster", None)
            if hasattr(self.cfg, "get")
            else None
        )
        self.job_info = {
            "job_id": job_id,
            "rank": rank,
            "cluster": cluster,
            "experiment_log_dir": getattr(
                getattr(self.cfg, "launcher", None), "experiment_log_dir", None
            ),
        }
        self.has_setup = True


def add_pythonpath_to_sys_path() -> None:
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def main(args: Namespace) -> Never:
    logging.info("Received train args: %s", args)
    _raise_train_unsupported("main")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="path to config file (e.g. configs/roboflow_v100_full_ft_100_images.yaml)",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUS per node"
    )
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")
    args = parser.parse_args()
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None
    try:
        register_omegaconf_resolvers()
    except NotImplementedError:
        pass
    main(args)
