from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Hashable, cast

import pytest

from sam3_mlx.train.optim.optimizer import (
    GradientClipper,
    Optimizer,
    ValueScaler,
    layer_decay_param_modifier,
    map_scheduler_cfgs_to_param_groups,
    set_default_parameters,
    unix_param_pattern_to_parameter_names,
    validate_param_group_params,
)


class _WhereScheduler:
    def __call__(self, where: float) -> float:
        return where + 1.0


class _StepScheduler:
    def __call__(self, *, step: int, where: float) -> float:
        return step + where


class _FakeOptimizer:
    defaults = {"lr": 0.0, "weight_decay": 0.0}

    def __init__(self) -> None:
        self.param_groups = [{"lr": 0.0}, {"weight_decay": 0.0}]
        self.step_closure: object = None
        self.zero_grad_args: tuple[tuple[object, ...], dict[str, object]] | None = None

    def step(self, closure: object = None) -> str:
        self.step_closure = closure
        return "stepped"

    def zero_grad(self, *args: object, **kwargs: object) -> str:
        self.zero_grad_args = (args, kwargs)
        return "cleared"


class _FakeModel:
    def __init__(self, parameters: Iterable[tuple[str, Hashable]]) -> None:
        self._parameters = list(parameters)

    def named_parameters(self, recurse: bool = True) -> Iterable[tuple[str, Hashable]]:
        del recurse
        return self._parameters


@dataclass
class _LayerModel:
    def get_num_layers(self) -> int:
        return 1

    def get_layer_id(self, parameter_name: str) -> int:
        return 0 if parameter_name == "block.weight" else 1


def test_optimizer_wrapper_updates_options_and_delegates_calls():
    raw_optimizer = _FakeOptimizer()
    optimizer = Optimizer(
        raw_optimizer,
        [{"lr": _WhereScheduler()}, {"weight_decay": _StepScheduler()}],
    )

    assert raw_optimizer.param_groups == [{"lr": 1.0}, {"weight_decay": 0.0}]
    assert optimizer.step(0.5, 3) == "stepped"
    assert raw_optimizer.param_groups == [{"lr": 1.5}, {"weight_decay": 3.5}]
    assert optimizer.zero_grad("set_to_none", enabled=True) == "cleared"
    assert raw_optimizer.zero_grad_args == (("set_to_none",), {"enabled": True})


def test_scheduler_configs_map_to_disjoint_parameter_groups():
    lr_configs: list[object] = [
        {
            "option": "lr",
            "scheduler": _WhereScheduler(),
            "parameter_names": {"encoder.weight"},
        },
        {"option": "lr", "scheduler": _WhereScheduler(), "parameter_names": None},
    ]
    set_default_parameters(lr_configs, {"encoder.weight", "decoder.weight"})
    weight_decay_configs: list[object] = [
        {
            "option": "weight_decay",
            "scheduler": _WhereScheduler(),
            "parameter_names": {"encoder.weight", "decoder.weight"},
        }
    ]
    encoder_parameter = object()
    decoder_parameter = object()

    schedulers, groups = map_scheduler_cfgs_to_param_groups(
        [lr_configs, weight_decay_configs],
        {
            "encoder.weight": encoder_parameter,
            "decoder.weight": decoder_parameter,
        },
    )

    assert len(schedulers) == 2
    assert {group["params"][0] for group in groups} == {
        encoder_parameter,
        decoder_parameter,
    }
    validate_param_group_params(
        groups,
        _FakeModel(
            [
                ("encoder.weight", encoder_parameter),
                ("decoder.weight", decoder_parameter),
            ]
        ),
    )


def test_pattern_and_layer_decay_helpers_preserve_selection_contracts():
    assert unix_param_pattern_to_parameter_names(
        ["encoder.*"], {"encoder.weight", "decoder.weight"}
    ) == {"encoder.weight"}

    scheduler = _WhereScheduler()
    result = layer_decay_param_modifier(
        [
            [
                {
                    "option": "lr",
                    "scheduler": scheduler,
                    "parameter_names": {"block.weight", "head.weight"},
                }
            ]
        ],
        _LayerModel(),
        0.5,
    )

    assert len(result[0]) == 2
    scaled_values: list[float] = []
    for config in result[0]:
        assert isinstance(config, dict)
        scheduler_value = cast(Mapping[str, object], config)["scheduler"]
        assert isinstance(scheduler_value, ValueScaler)
        scaled_values.append(scheduler_value(0.0))
    scaled_values.sort()
    assert scaled_values == [0.25, 0.5]


def test_gradient_clipper_rejects_boolean_max_norm():
    with pytest.raises(AssertionError, match="number or None"):
        GradientClipper(True)


def test_optimizer_wrapper_rejects_unknown_option():
    with pytest.raises(AssertionError, match="not found"):
        Optimizer(_FakeOptimizer(), [{"momentum": _WhereScheduler()}])


def test_gradient_clipper_keeps_training_boundary_unsupported():
    with pytest.raises(NotImplementedError, match="GradientClipper"):
        GradientClipper()(object())


def test_value_scaler_forwards_scheduler_arguments():
    assert ValueScaler(_StepScheduler(), 0.5)(step=3, where=1.0) == 2.0
