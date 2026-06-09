"""SACO video eval compatibility entrypoints."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import FailFastEvaluator, raise_unsupported


class VEvalEvaluator(FailFastEvaluator):
    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.saco_veval_eval.VEvalEvaluator")


def run_main_all(dataset_name, args):
    raise_unsupported("eval.saco_veval_eval.run_main_all")


def main_all(args):
    raise_unsupported("eval.saco_veval_eval.main_all")


def main_one(args):
    raise_unsupported("eval.saco_veval_eval.main_one")


def main():
    raise_unsupported("eval.saco_veval_eval.main")
