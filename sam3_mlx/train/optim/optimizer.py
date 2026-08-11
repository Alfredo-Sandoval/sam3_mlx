# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Optimizer helpers from the official SAM3 training surface.

Optimizer construction in upstream SAM3 is Hydra plus ``torch.optim`` specific.
This MLX fork keeps the pure parameter/scheduler bookkeeping helpers importable,
but the PyTorch optimizer construction and gradient clipping entry points fail
explicitly instead of importing torch.
"""

from __future__ import annotations

import fnmatch
import importlib
import inspect
import itertools
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from typing import Hashable, Never, Protocol, TypeGuard, TypedDict, cast

from sam3_mlx._unsupported import UPSTREAM_COMMIT, raise_unsupported


_UNSUPPORTED_OPTIMIZER_MESSAGE = (
    "SAM3 optimizer construction is not implemented in the MLX port yet. The "
    "official implementation at commit "
    f"{UPSTREAM_COMMIT} constructs torch.optim optimizers and "
    "uses torch.nn parameter objects. Add an explicit MLX optimizer path before "
    "using this training surface."
)


def _raise_optimizer_unsupported(feature: str) -> Never:
    raise_unsupported(
        feature,
        reason="training-loop",
        detail=_UNSUPPORTED_OPTIMIZER_MESSAGE,
    )


class Scheduler(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> float: ...


class _OptimizerStep(Protocol):
    def __call__(self, closure: Callable[[], object] | None = None) -> object: ...


class _ZeroGrad(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _ConfigGetter(Protocol):
    def __call__(self, key: str, default: object = None) -> object: ...


class _ConfigSetter(Protocol):
    def __call__(self, key: str, value: object) -> None: ...


class _NamedParameters(Protocol):
    def named_parameters(
        self, recurse: bool = True
    ) -> Iterable[tuple[str, Hashable]]: ...


class _NamedModules(_NamedParameters, Protocol):
    def named_modules(self) -> Iterable[tuple[str, _NamedParameters]]: ...


class _LayerDecayModel(Protocol):
    def get_num_layers(self) -> int: ...

    def get_layer_id(self, parameter_name: str) -> int: ...


class SchedulerConfig(TypedDict, total=False):
    option: str
    scheduler: Scheduler
    parameter_names: set[str]
    param_names: list[str] | None
    module_cls_names: list[str] | None


class ParamGroup(TypedDict):
    params: list[Hashable]


class LayerDecayOverride(TypedDict):
    pattern: str
    value: float


type SchedulerSet = Mapping[str, object]
type MutableSchedulerConfig = MutableMapping[str, object]


def _is_scheduler(value: object) -> TypeGuard[Scheduler]:
    return callable(value)


def _require_scheduler(value: object) -> Scheduler:
    if not callable(value):
        raise TypeError("scheduler must be callable")
    return cast(Scheduler, value)


def _is_layer_decay_model(value: object) -> TypeGuard[_LayerDecayModel]:
    return callable(getattr(value, "get_num_layers", None)) and callable(
        getattr(value, "get_layer_id", None)
    )


def _scheduler_signature(scheduler: Scheduler) -> inspect.Signature:
    return inspect.signature(scheduler.__call__)


def _nested_scheduler_parameters(scheduler: Scheduler) -> Mapping[str, object]:
    nested = getattr(scheduler, "scheduler", None)
    if not _is_scheduler(nested):
        return {}
    return cast(Mapping[str, object], _scheduler_signature(nested).parameters)


def _optimizer_defaults(optimizer: object) -> Mapping[str, object]:
    defaults = getattr(optimizer, "defaults", None)
    if not isinstance(defaults, Mapping):
        _raise_optimizer_unsupported(
            "Optimizer wrapper expected a torch.optim-style defaults mapping"
        )
    return cast(Mapping[str, object], defaults)


def _optimizer_param_groups(
    optimizer: object,
) -> list[MutableMapping[str, object]]:
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list):
        _raise_optimizer_unsupported(
            "Optimizer wrapper expected torch.optim-style param_groups"
        )
    group_values = cast(list[object], groups)
    if not all(isinstance(group, MutableMapping) for group in group_values):
        _raise_optimizer_unsupported(
            "Optimizer wrapper expected torch.optim-style param_groups"
        )
    return cast(list[MutableMapping[str, object]], groups)


def _cfg_get(scheduler_cfg: object, key: str, default: object = None) -> object:
    getter = getattr(scheduler_cfg, "get", None)
    if callable(getter):
        return cast(_ConfigGetter, getter)(key, default)
    return getattr(scheduler_cfg, key, default)


def _cfg_set(scheduler_cfg: object, key: str, value: object) -> None:
    setter = getattr(scheduler_cfg, "__setitem__", None)
    if callable(setter):
        cast(_ConfigSetter, setter)(key, value)
        return
    setattr(scheduler_cfg, key, value)


def _parameter_names(scheduler_cfg: object) -> set[str] | None:
    value = _cfg_get(scheduler_cfg, "parameter_names")
    if value is None:
        return None
    if not isinstance(value, set):
        raise TypeError("scheduler parameter_names must be a set of strings")
    names = cast(set[object], value)
    if not all(isinstance(name, str) for name in names):
        raise TypeError("scheduler parameter_names must be a set of strings")
    return cast(set[str], value)


def _scheduler_option(scheduler_cfg: object) -> str:
    value = _cfg_get(scheduler_cfg, "option")
    if not isinstance(value, str):
        raise TypeError("scheduler option must be a string")
    return value


def _scheduler_value(scheduler_cfg: object) -> Scheduler:
    value = _cfg_get(scheduler_cfg, "scheduler")
    return _require_scheduler(value)


def _optional_string_list(value: object, name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{name} must be a list of strings or None")
    items = list(cast(Iterable[object], value))
    if not all(isinstance(item, str) for item in items):
        raise TypeError(f"{name} must be a list of strings or None")
    return cast(list[str], items)


def _required_parameter_names(scheduler_cfg: object) -> set[str]:
    parameter_names = _parameter_names(scheduler_cfg)
    if parameter_names is None:
        raise KeyError("parameter_names")
    return parameter_names


def _union_sets[T](sets: Sequence[set[T]]) -> set[T]:
    if not sets:
        return set()
    return sets[0].union(*sets[1:])


def _intersect_sets[T](sets: Sequence[set[T]]) -> set[T]:
    if not sets:
        raise TypeError("at least one constraint set is required")
    return sets[0].intersection(*sets[1:])


class Optimizer:
    def __init__(
        self,
        optimizer: object,
        schedulers: Sequence[SchedulerSet] | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.schedulers = schedulers
        self._validate_optimizer_schedulers()
        self.step_schedulers(0.0, 0)

    def _validate_optimizer_schedulers(self) -> None:
        if self.schedulers is None:
            return
        defaults = _optimizer_defaults(self.optimizer)
        for set_of_schedulers in self.schedulers:
            for option, _ in set_of_schedulers.items():
                if option not in defaults:
                    raise AssertionError(
                        "Optimizer option "
                        f"{option} not found in {self.optimizer}. Valid options are "
                        f"{defaults.keys()}"
                    )

    def step_schedulers(self, where: float, step: int) -> None:
        if self.schedulers is None:
            return
        param_groups = _optimizer_param_groups(self.optimizer)
        for i, param_group in enumerate(param_groups):
            for option, scheduler_value in self.schedulers[i].items():
                scheduler = _require_scheduler(scheduler_value)
                if "step" in _scheduler_signature(scheduler).parameters:
                    new_value = scheduler(step=step, where=where)
                elif "step" in _nested_scheduler_parameters(scheduler):
                    new_value = scheduler(step=step, where=where)
                else:
                    new_value = scheduler(where)
                param_group[option] = new_value

    def step(
        self,
        where: float,
        step: int,
        closure: Callable[[], object] | None = None,
    ) -> object:
        self.step_schedulers(where, step)
        step_fn = getattr(self.optimizer, "step", None)
        if not callable(step_fn):
            _raise_optimizer_unsupported("Optimizer object has no step method")
        return cast(_OptimizerStep, step_fn)(closure)

    def zero_grad(self, *args: object, **kwargs: object) -> object:
        zero_grad = getattr(self.optimizer, "zero_grad", None)
        if not callable(zero_grad):
            _raise_optimizer_unsupported("Optimizer object has no zero_grad method")
        return cast(_ZeroGrad, zero_grad)(*args, **kwargs)


def set_default_parameters(
    scheduler_cfgs: list[object], all_parameter_names: set[str]
) -> None:
    """Set up the official "default" scheduler with the right parameters."""

    constraints = [
        parameter_names
        for scheduler_cfg in scheduler_cfgs
        if (parameter_names := _parameter_names(scheduler_cfg)) is not None
    ]
    default_params = (
        set(all_parameter_names)
        if len(constraints) == 0
        else all_parameter_names - _union_sets(constraints)
    )
    default_count = 0
    for scheduler_cfg in scheduler_cfgs:
        parameter_names = _parameter_names(scheduler_cfg)
        if parameter_names is None:
            _cfg_set(scheduler_cfg, "parameter_names", default_params)
            default_count += 1
    if default_count > 1:
        raise AssertionError("Only one scheduler per option can be default")
    if default_count == 0:
        scheduler_cfgs.append({"parameter_names": default_params})


def name_constraints_to_parameters[T](
    param_constraints: Sequence[set[str]], named_parameters: Mapping[str, T]
) -> list[T]:
    """Return parameters whose names match every constraint set."""

    matching_names = _intersect_sets(param_constraints)
    return [value for name, value in named_parameters.items() if name in matching_names]


def map_scheduler_cfgs_to_param_groups(
    all_scheduler_cfgs: Iterable[Sequence[object]],
    named_parameters: Mapping[str, Hashable],
) -> tuple[list[dict[str, Scheduler]], list[ParamGroup]]:
    """Produce official-style parameter groups for scheduler configs."""

    scheduler_cfgs_per_param_group = itertools.product(*all_scheduler_cfgs)
    schedulers: list[dict[str, Scheduler]] = []
    param_groups: list[ParamGroup] = []
    for scheduler_cfgs in scheduler_cfgs_per_param_group:
        param_constraints = [
            _required_parameter_names(scheduler_cfg) for scheduler_cfg in scheduler_cfgs
        ]
        matching_parameters = name_constraints_to_parameters(
            param_constraints, named_parameters
        )
        if len(matching_parameters) == 0:
            continue
        schedulers_for_group = {
            _scheduler_option(scheduler_cfg): _scheduler_value(scheduler_cfg)
            for scheduler_cfg in scheduler_cfgs
            if _cfg_get(scheduler_cfg, "option") is not None
        }
        schedulers.append(schedulers_for_group)
        param_groups.append({"params": matching_parameters})
    return schedulers, param_groups


def validate_param_group_params(
    param_groups: Sequence[ParamGroup], model: _NamedParameters
) -> None:
    """Check that official-style param groups are non-overlapping and complete."""

    for pg in param_groups:
        if len(pg["params"]) != len(set(pg["params"])):
            raise AssertionError("param_groups must not repeat params within a group")
    parameters = [set(param_group["params"]) for param_group in param_groups]
    model_parameters = {parameter for _, parameter in model.named_parameters()}
    for p1, p2 in itertools.permutations(parameters, 2):
        if not p1.isdisjoint(p2):
            raise AssertionError("Scheduler generated param_groups should be disjoint")
    covered_parameters = _union_sets(parameters)
    if covered_parameters != model_parameters:
        raise AssertionError(
            "Scheduler generated param_groups must include all parameters of the model."
            f" Found {len(covered_parameters)} params whereas model has"
            f" {len(model_parameters)} params"
        )


def _resolve_class(class_path: str) -> type:
    if class_path.startswith("torch."):
        _raise_optimizer_unsupported(
            f"module class constraint {class_path!r} is PyTorch-specific"
        )
    module_name, _, attr_name = class_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected a fully qualified class path, got {class_path!r}")
    module = importlib.import_module(module_name)
    resolved = getattr(module, attr_name)
    if not isinstance(resolved, type):
        raise TypeError(f"Resolved object {class_path!r} is not a class")
    return resolved


def unix_module_cls_pattern_to_parameter_names(
    filter_module_cls_names: list[str] | None,
    module_cls_to_param_names: Mapping[type, set[str]],
) -> set[str]:
    """Return param names passing fully-qualified module-class filters."""

    if filter_module_cls_names is None:
        return set()
    allowed_parameter_names: list[set[str]] = []
    for module_cls_name in filter_module_cls_names:
        module_cls = _resolve_class(module_cls_name)
        if module_cls not in module_cls_to_param_names:
            raise AssertionError(
                f"module_cls_name {module_cls_name} does not "
                "match any classes in the model"
            )
        matching_parameters = module_cls_to_param_names[module_cls]
        if len(matching_parameters) == 0:
            raise AssertionError(
                f"module_cls_name {module_cls_name} does not contain any parameters in the model"
            )
        allowed_parameter_names.append(matching_parameters)
    return _union_sets(allowed_parameter_names)


def unix_param_pattern_to_parameter_names(
    filter_param_names: list[str] | None,
    parameter_names: set[str],
) -> set[str]:
    """Return param names passing unix-style parameter filters."""

    if filter_param_names is None:
        return set()
    allowed_parameter_names: list[set[str]] = []
    for param_name in filter_param_names:
        matching_parameters = set(fnmatch.filter(parameter_names, param_name))
        if len(matching_parameters) < 1:
            raise AssertionError(
                f"param_name {param_name} does not match any parameters in the model"
            )
        allowed_parameter_names.append(matching_parameters)
    return _union_sets(allowed_parameter_names)


def unix_pattern_to_parameter_names(
    scheduler_cfg: object,
    parameter_names: set[str],
    module_cls_to_param_names: Mapping[type, set[str]],
) -> set[str] | None:
    """Return param names selected by a scheduler config."""

    if (
        _cfg_get(scheduler_cfg, "param_names") is None
        and _cfg_get(scheduler_cfg, "module_cls_names") is None
    ):
        return None
    return unix_param_pattern_to_parameter_names(
        _optional_string_list(_cfg_get(scheduler_cfg, "param_names"), "param_names"),
        parameter_names,
    ).union(
        unix_module_cls_pattern_to_parameter_names(
            _optional_string_list(
                _cfg_get(scheduler_cfg, "module_cls_names"),
                "module_cls_names",
            ),
            module_cls_to_param_names,
        )
    )


def get_module_cls_to_param_names(
    model: _NamedModules, param_allowlist: set[str] | None = None
) -> dict[type, set[str]]:
    """Produce a mapping from immediate module classes to owned param names."""

    module_cls_to_params: dict[type, set[str]] = {}
    for module_name, module in model.named_modules():
        module_cls = type(module)
        module_cls_to_params.setdefault(module_cls, set())
        for param_name, _ in module.named_parameters(recurse=False):
            full_param_name = get_full_parameter_name(module_name, param_name)
            if param_allowlist is None or full_param_name in param_allowlist:
                module_cls_to_params[module_cls].add(full_param_name)
    return module_cls_to_params


def construct_optimizer(
    model: object,
    optimizer_conf: object,
    options_conf: Mapping[str, Sequence[object]] | None = None,
    param_group_modifiers_conf: Sequence[Callable[..., object]] | None = None,
    param_allowlist: set[str] | None = None,
    validate_param_groups: bool = True,
) -> Optimizer:
    _raise_optimizer_unsupported("construct_optimizer")


def get_full_parameter_name(module_name: str, param_name: str) -> str:
    if module_name == "":
        return param_name
    return f"{module_name}.{param_name}"


class GradientClipper:
    """Official-shaped gradient clipper placeholder."""

    def __init__(self, max_norm: object = 1.0, norm_type: int = 2):
        if isinstance(max_norm, bool) or (
            not isinstance(max_norm, (int, float)) and max_norm is not None
        ):
            raise AssertionError("max_norm must be a number or None")
        self.max_norm: float | None = max_norm if max_norm is None else float(max_norm)
        self.norm_type = norm_type

    def __call__(self, model: object) -> None:
        if self.max_norm is None:
            return
        _raise_optimizer_unsupported("GradientClipper")


class ValueScaler:
    def __init__(self, scheduler: object, mult_val: float):
        self.scheduler = _require_scheduler(scheduler)
        self.mult_val = mult_val

    def __call__(self, *args: object, **kwargs: object) -> float:
        val = self.scheduler(*args, **kwargs)
        return val * self.mult_val


def rgetattr(obj: object, rattrs: str | None = None) -> object:
    """Like getattr(), but supports dotted notation for nested objects."""

    if rattrs is None:
        return obj
    attrs = rattrs.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def layer_decay_param_modifier(
    scheduler_cfgs: Sequence[Sequence[object]],
    model: object,
    layer_decay_value: float,
    layer_decay_min: float | None = None,
    apply_to: str | None = None,
    overrides: Sequence[LayerDecayOverride] = (),
) -> list[list[object]]:
    """Apply official SAM3 layer-decay rewriting to scheduler configs."""

    scoped_model = rgetattr(model, apply_to)
    if not _is_layer_decay_model(scoped_model):
        raise TypeError("layer-decay target must expose layer indexing methods")
    num_layers = scoped_model.get_num_layers() + 1
    layer_decays = [
        layer_decay_value ** (num_layers - i) for i in range(num_layers + 1)
    ]
    if layer_decay_min is not None:
        layer_decays = [max(val, layer_decay_min) for val in layer_decays]
    final_scheduler_cfgs: list[list[object]] = []
    prefix = apply_to or ""
    for scheduler_cfg_group in scheduler_cfgs:
        curr_cfg_group: list[object] = []
        for scheduler_cfg in scheduler_cfg_group:
            option = _scheduler_option(scheduler_cfg)
            if option != "lr":
                curr_cfg_group.append(scheduler_cfg)
                continue
            parameter_names_value = _parameter_names(scheduler_cfg)
            if parameter_names_value is None:
                raise ValueError("layer-decay scheduler requires parameter_names")
            parameter_names = sorted(parameter_names_value)
            layer_cfg_groups: dict[int | str, SchedulerConfig] = {}
            for param_name in parameter_names:
                layer_id = num_layers
                this_scale = layer_decays[layer_id]
                if param_name.startswith(prefix):
                    layer_id = scoped_model.get_layer_id(param_name)
                    this_scale = layer_decays[layer_id]
                    for override in overrides:
                        if fnmatch.fnmatchcase(param_name, override["pattern"]):
                            this_scale = float(override["value"])
                            layer_id = override["pattern"]
                            break

                curr_param: SchedulerConfig
                if layer_id not in layer_cfg_groups:
                    curr_param = {
                        "option": option,
                        "scheduler": ValueScaler(
                            _scheduler_value(scheduler_cfg), this_scale
                        ),
                        "parameter_names": {param_name},
                    }
                else:
                    curr_param = layer_cfg_groups[layer_id]
                    curr_parameter_names = _parameter_names(curr_param)
                    if curr_parameter_names is None:
                        raise AssertionError("layer config lost parameter_names")
                    curr_parameter_names.add(param_name)
                layer_cfg_groups[layer_id] = curr_param

            for layer_cfg in layer_cfg_groups.values():
                curr_cfg_group.append(layer_cfg)

        final_scheduler_cfgs.append(curr_cfg_group)
    return final_scheduler_cfgs
