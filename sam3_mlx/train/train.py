# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Official-shaped SAM3 train launcher placeholder for the MLX port."""

from __future__ import annotations

import logging
import os
import sys
import traceback
from argparse import ArgumentParser

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


def _raise_train_unsupported(feature: str) -> None:
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


def handle_custom_resolving(cfg):
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return cfg
    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    return OmegaConf.create(cfg_resolved)


def single_proc_run(local_rank, main_port, cfg, world_size):
    _raise_train_unsupported("single_proc_run")


def single_node_runner(cfg, main_port: int):
    _raise_train_unsupported("single_node_runner")


def format_exception(e: Exception, limit=20):
    traceback_str = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


class SubmititRunner:
    """Official-shaped Submitit runner placeholder."""

    def __init__(self, port, cfg):
        self.cfg = cfg
        self.port = port
        self.has_setup = False

    def run_trainer(self):
        _raise_train_unsupported("SubmititRunner.run_trainer")

    def __call__(self):
        _raise_train_unsupported("SubmititRunner")

    def setup_job_info(self, job_id, rank):
        self.job_info = {
            "job_id": job_id,
            "rank": rank,
            "cluster": self.cfg.get("cluster", None)
            if hasattr(self.cfg, "get")
            else None,
            "experiment_log_dir": getattr(
                getattr(self.cfg, "launcher", None), "experiment_log_dir", None
            ),
        }
        self.has_setup = True


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def main(args) -> None:
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
