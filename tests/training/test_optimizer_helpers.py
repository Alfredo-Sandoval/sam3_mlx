from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Hashable, cast

import pytest

from sam3_mlx.train.optim.optimizer import (
    GradientClipper,
    Optimizer,
    ValueScaler,
    layer_decay_param_modifier,
    map_scheduler_cfgs_to_param_groups,
    name_constraints_to_parameters,
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


class _NestedStepScheduler:
    def __init__(self) -> None:
        self.scheduler = _StepScheduler()

    def __call__(self, **kwargs: int | float) -> float:
        step = kwargs["step"]
        where = kwargs["where"]
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer")
        return self.scheduler(step=step, where=float(where))


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


@dataclass
class _AttributeSchedulerConfig:
    option: str
    scheduler: object
    parameter_names: set[str] | None


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


def test_optimizer_wrapper_uses_nested_scheduler_signature():
    raw_optimizer = _FakeOptimizer()
    Optimizer(
        raw_optimizer,
        [{"lr": _NestedStepScheduler()}, {"weight_decay": _WhereScheduler()}],
    )

    assert raw_optimizer.param_groups == [{"lr": 0.0}, {"weight_decay": 1.0}]


@pytest.mark.parametrize("scheduler_count", [1, 3])
def test_optimizer_wrapper_rejects_scheduler_param_group_count_mismatch(
    scheduler_count: int,
):
    schedulers = [{"lr": _WhereScheduler()} for _ in range(scheduler_count)]

    with pytest.raises(ValueError, match="scheduler count must match.*2 groups"):
        Optimizer(_FakeOptimizer(), schedulers)


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


def test_scheduler_configs_support_attribute_objects():
    attribute_config = _AttributeSchedulerConfig("lr", _WhereScheduler(), None)
    scheduler_configs: list[object] = [attribute_config]

    set_default_parameters(scheduler_configs, {"weight"})
    schedulers, groups = map_scheduler_cfgs_to_param_groups(
        [scheduler_configs], {"weight": "parameter"}
    )

    assert attribute_config.parameter_names == {"weight"}
    assert list(schedulers[0]) == ["lr"]
    assert groups == [{"params": ["parameter"]}]


def test_default_scheduler_rejects_read_only_mapping():
    scheduler_configs: list[object] = [
        MappingProxyType(
            {
                "option": "lr",
                "scheduler": _WhereScheduler(),
                "parameter_names": None,
            }
        )
    ]

    with pytest.raises(TypeError, match="scheduler config mapping must be mutable"):
        set_default_parameters(scheduler_configs, {"weight"})


def test_default_scheduler_rejects_getter_without_mutation():
    class _ReadOnlyConfig:
        __slots__ = ()

        def get(self, key: str, default: object = None) -> object:
            del key
            return default

    with pytest.raises(TypeError, match="must support item or attribute assignment"):
        set_default_parameters([_ReadOnlyConfig()], {"weight"})


def test_default_scheduler_rejects_malformed_parameter_names():
    with pytest.raises(TypeError, match="parameter_names must be a set of strings"):
        set_default_parameters([{"parameter_names": ["weight"]}], {"weight"})


def test_default_scheduler_rejects_multiple_defaults():
    scheduler_configs: list[object] = [
        {"parameter_names": None},
        SimpleNamespace(parameter_names=None),
    ]

    with pytest.raises(AssertionError, match="Only one scheduler"):
        set_default_parameters(scheduler_configs, {"weight"})


def test_default_scheduler_appends_unmatched_constraint_group():
    scheduler_configs: list[object] = [{"parameter_names": {"encoder.weight"}}]

    set_default_parameters(scheduler_configs, {"encoder.weight", "decoder.weight"})

    assert scheduler_configs[-1] == {"parameter_names": {"decoder.weight"}}


def test_scheduler_mapping_accepts_constraint_without_scheduler_option():
    schedulers, groups = map_scheduler_cfgs_to_param_groups(
        [[{"parameter_names": {"weight"}}]], {"weight": "parameter"}
    )

    assert schedulers == [{}]
    assert groups == [{"params": ["parameter"]}]


def test_scheduler_mapping_rejects_noncallable_scheduler():
    with pytest.raises(TypeError, match="scheduler must be callable"):
        map_scheduler_cfgs_to_param_groups(
            [[{"option": "lr", "scheduler": 1, "parameter_names": {"weight"}}]],
            {"weight": object()},
        )


def test_empty_parameter_constraint_collection_fails_deliberately():
    with pytest.raises(TypeError, match="at least one constraint set"):
        name_constraints_to_parameters([], {"weight": object()})


def test_param_group_validation_rejects_overlap_and_incomplete_coverage():
    first = object()
    second = object()
    model = _FakeModel([("first", first), ("second", second)])

    with pytest.raises(AssertionError, match="should be disjoint"):
        validate_param_group_params(
            [{"params": [first]}, {"params": [first, second]}], model
        )
    with pytest.raises(AssertionError, match="must include all parameters"):
        validate_param_group_params([{"params": [first]}], model)


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


@pytest.mark.parametrize(
    ("layer_count", "layer_id", "exception", "message"),
    [
        (True, 0, TypeError, "layer count must be an integer"),
        (-1, 0, ValueError, "layer count must be non-negative"),
        (1, True, TypeError, "layer id must be an integer"),
        (1, 3, ValueError, "layer id must be between"),
    ],
)
def test_layer_decay_rejects_invalid_integer_boundaries(
    layer_count: object,
    layer_id: object,
    exception: type[Exception],
    message: str,
):
    class _MalformedLayerModel:
        def get_num_layers(self) -> object:
            return layer_count

        def get_layer_id(self, parameter_name: str) -> object:
            del parameter_name
            return layer_id

    with pytest.raises(exception, match=message):
        layer_decay_param_modifier(
            [
                [
                    {
                        "option": "lr",
                        "scheduler": _WhereScheduler(),
                        "parameter_names": {"weight"},
                    }
                ]
            ],
            _MalformedLayerModel(),
            0.5,
        )


def test_gradient_clipper_rejects_boolean_max_norm():
    with pytest.raises(AssertionError, match="number or None"):
        GradientClipper(True)


def test_optimizer_wrapper_rejects_unknown_option():
    with pytest.raises(AssertionError, match="not found"):
        Optimizer(_FakeOptimizer(), [{"momentum": _WhereScheduler()}, {}])


def test_gradient_clipper_keeps_training_boundary_unsupported():
    with pytest.raises(NotImplementedError, match="GradientClipper"):
        GradientClipper()(object())


def test_value_scaler_forwards_scheduler_arguments():
    assert ValueScaler(_StepScheduler(), 0.5)(step=3, where=1.0) == 2.0
