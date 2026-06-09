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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from sam3_mlx._unsupported import UPSTREAM_COMMIT, raise_unsupported


_UNSUPPORTED_OPTIMIZER_MESSAGE = (
    "SAM3 optimizer construction is not implemented in the MLX port yet. The "
    "official implementation at commit "
    f"{UPSTREAM_COMMIT} constructs torch.optim optimizers and "
    "uses torch.nn parameter objects. Add an explicit MLX optimizer path before "
    "using this training surface."
)


def _raise_optimizer_unsupported(feature: str) -> None:
    raise_unsupported(
        feature,
        reason="training-loop",
        detail=_UNSUPPORTED_OPTIMIZER_MESSAGE,
    )


class Optimizer:
    def __init__(self, optimizer, schedulers=None) -> None:
        self.optimizer = optimizer
        self.schedulers = schedulers
        self._validate_optimizer_schedulers()
        self.step_schedulers(0.0, 0)

    def _validate_optimizer_schedulers(self):
        if self.schedulers is None:
            return
        defaults = getattr(self.optimizer, "defaults", None)
        if defaults is None:
            _raise_optimizer_unsupported(
                "Optimizer wrapper expected a torch.optim-style defaults mapping"
            )
        for _, set_of_schedulers in enumerate(self.schedulers):
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
        param_groups = getattr(self.optimizer, "param_groups", None)
        if param_groups is None:
            _raise_optimizer_unsupported(
                "Optimizer wrapper expected torch.optim-style param_groups"
            )
        for i, param_group in enumerate(param_groups):
            for option, scheduler in self.schedulers[i].items():
                if "step" in inspect.signature(scheduler.__call__).parameters:
                    new_value = scheduler(step=step, where=where)
                elif (
                    hasattr(scheduler, "scheduler")
                    and "step"
                    in inspect.signature(scheduler.scheduler.__call__).parameters
                ):
                    new_value = scheduler(step=step, where=where)
                else:
                    new_value = scheduler(where)
                param_group[option] = new_value

    def step(self, where, step, closure=None):
        self.step_schedulers(where, step)
        step_fn = getattr(self.optimizer, "step", None)
        if step_fn is None:
            _raise_optimizer_unsupported("Optimizer object has no step method")
        return step_fn(closure)

    def zero_grad(self, *args, **kwargs):
        zero_grad = getattr(self.optimizer, "zero_grad", None)
        if zero_grad is None:
            _raise_optimizer_unsupported("Optimizer object has no zero_grad method")
        return zero_grad(*args, **kwargs)


def set_default_parameters(
    scheduler_cfgs: List[Dict[str, Any]], all_parameter_names: Set[str]
) -> None:
    """Set up the official "default" scheduler with the right parameters."""

    constraints = [
        scheduler_cfg.parameter_names
        if hasattr(scheduler_cfg, "parameter_names")
        else scheduler_cfg.get("parameter_names")
        for scheduler_cfg in scheduler_cfgs
        if (
            scheduler_cfg.parameter_names
            if hasattr(scheduler_cfg, "parameter_names")
            else scheduler_cfg.get("parameter_names")
        )
        is not None
    ]
    default_params = (
        set(all_parameter_names)
        if len(constraints) == 0
        else all_parameter_names - set.union(*constraints)
    )
    default_count = 0
    for scheduler_cfg in scheduler_cfgs:
        parameter_names = (
            scheduler_cfg.parameter_names
            if hasattr(scheduler_cfg, "parameter_names")
            else scheduler_cfg.get("parameter_names")
        )
        if parameter_names is None:
            if hasattr(scheduler_cfg, "parameter_names"):
                scheduler_cfg.parameter_names = default_params
            else:
                scheduler_cfg["parameter_names"] = default_params
            default_count += 1
    if default_count > 1:
        raise AssertionError("Only one scheduler per option can be default")
    if default_count == 0:
        scheduler_cfgs.append({"parameter_names": default_params})


def name_constraints_to_parameters(
    param_constraints: List[Set[str]], named_parameters: Dict[str, Any]
) -> List[Any]:
    """Return parameters whose names match every constraint set."""

    matching_names = set.intersection(*param_constraints)
    return [value for name, value in named_parameters.items() if name in matching_names]


def map_scheduler_cfgs_to_param_groups(
    all_scheduler_cfgs: Iterable[List[Dict[str, Any]]],
    named_parameters: Dict[str, Any],
) -> Tuple[List[Dict[Any, Any]], List[Dict[str, List[Any]]]]:
    """Produce official-style parameter groups for scheduler configs."""

    scheduler_cfgs_per_param_group = itertools.product(*all_scheduler_cfgs)
    schedulers = []
    param_groups = []
    for scheduler_cfgs in scheduler_cfgs_per_param_group:
        param_constraints = [
            scheduler_cfg["parameter_names"] for scheduler_cfg in scheduler_cfgs
        ]
        matching_parameters = name_constraints_to_parameters(
            param_constraints, named_parameters
        )
        if len(matching_parameters) == 0:
            continue
        schedulers_for_group = {
            scheduler_cfg["option"]: scheduler_cfg["scheduler"]
            for scheduler_cfg in scheduler_cfgs
            if "option" in scheduler_cfg
        }
        schedulers.append(schedulers_for_group)
        param_groups.append({"params": matching_parameters})
    return schedulers, param_groups


def validate_param_group_params(param_groups: List[Dict[str, Any]], model):
    """Check that official-style param groups are non-overlapping and complete."""

    for pg in param_groups:
        if len(pg["params"]) != len(set(pg["params"])):
            raise AssertionError("param_groups must not repeat params within a group")
    parameters = [set(param_group["params"]) for param_group in param_groups]
    model_parameters = {parameter for _, parameter in model.named_parameters()}
    for p1, p2 in itertools.permutations(parameters, 2):
        if not p1.isdisjoint(p2):
            raise AssertionError("Scheduler generated param_groups should be disjoint")
    covered_parameters = set.union(*parameters) if parameters else set()
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
    return getattr(module, attr_name)


def unix_module_cls_pattern_to_parameter_names(
    filter_module_cls_names: List[str],
    module_cls_to_param_names: Dict[type, Set[str]],
) -> Set[str]:
    """Return param names passing fully-qualified module-class filters."""

    if filter_module_cls_names is None:
        return set()
    allowed_parameter_names = []
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
    return set.union(*allowed_parameter_names) if allowed_parameter_names else set()


def unix_param_pattern_to_parameter_names(
    filter_param_names: Optional[List[str]],
    parameter_names: Set[str],
) -> Set[str]:
    """Return param names passing unix-style parameter filters."""

    if filter_param_names is None:
        return set()
    allowed_parameter_names = []
    for param_name in filter_param_names:
        matching_parameters = set(fnmatch.filter(parameter_names, param_name))
        if len(matching_parameters) < 1:
            raise AssertionError(
                f"param_name {param_name} does not match any parameters in the model"
            )
        allowed_parameter_names.append(matching_parameters)
    return set.union(*allowed_parameter_names) if allowed_parameter_names else set()


def _cfg_get(scheduler_cfg: Any, key: str, default: Any = None) -> Any:
    return (
        scheduler_cfg.get(key, default)
        if hasattr(scheduler_cfg, "get")
        else getattr(scheduler_cfg, key, default)
    )


def _cfg_set(scheduler_cfg: Any, key: str, value: Any) -> None:
    if hasattr(scheduler_cfg, "__setitem__"):
        scheduler_cfg[key] = value
    else:
        setattr(scheduler_cfg, key, value)


def _unix_pattern_to_parameter_names(
    scheduler_cfg: Any,
    parameter_names: Set[str],
    module_cls_to_param_names: Dict[type, Set[str]],
) -> Optional[Set[str]]:
    """Return param names selected by a scheduler config."""

    if (
        _cfg_get(scheduler_cfg, "param_names") is None
        and _cfg_get(scheduler_cfg, "module_cls_names") is None
    ):
        return None
    return unix_param_pattern_to_parameter_names(
        _cfg_get(scheduler_cfg, "param_names"), parameter_names
    ).union(
        unix_module_cls_pattern_to_parameter_names(
            _cfg_get(scheduler_cfg, "module_cls_names"), module_cls_to_param_names
        )
    )


def get_module_cls_to_param_names(
    model, param_allowlist: Optional[Set[str]] = None
) -> Dict[type, Set[str]]:
    """Produce a mapping from immediate module classes to owned param names."""

    module_cls_to_params: Dict[type, Set[str]] = {}
    for module_name, module in model.named_modules():
        module_cls = type(module)
        module_cls_to_params.setdefault(module_cls, set())
        for param_name, _ in module.named_parameters(recurse=False):
            full_param_name = get_full_parameter_name(module_name, param_name)
            if param_allowlist is None or full_param_name in param_allowlist:
                module_cls_to_params[module_cls].add(full_param_name)
    return module_cls_to_params


def construct_optimizer(
    model,
    optimizer_conf: Any,
    options_conf: Mapping[str, List] = None,
    param_group_modifiers_conf: List[Callable] = None,
    param_allowlist: Optional[Set[str]] = None,
    validate_param_groups=True,
) -> Optimizer:
    _raise_optimizer_unsupported("construct_optimizer")


def get_full_parameter_name(module_name, param_name):
    if module_name == "":
        return param_name
    return f"{module_name}.{param_name}"


class GradientClipper:
    """Official-shaped gradient clipper placeholder."""

    def __init__(self, max_norm: float = 1.0, norm_type: int = 2):
        if not isinstance(max_norm, (int, float)) and max_norm is not None:
            raise AssertionError("max_norm must be a number or None")
        self.max_norm = max_norm if max_norm is None else float(max_norm)
        self.norm_type = norm_type

    def __call__(self, model):
        if self.max_norm is None:
            return
        _raise_optimizer_unsupported("GradientClipper")


class ValueScaler:
    def __init__(self, scheduler, mult_val: float):
        self.scheduler = scheduler
        self.mult_val = mult_val

    def __call__(self, *args, **kwargs):
        val = self.scheduler(*args, **kwargs)
        return val * self.mult_val


def rgetattr(obj, rattrs: str = None):
    """Like getattr(), but supports dotted notation for nested objects."""

    if rattrs is None:
        return obj
    attrs = rattrs.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def layer_decay_param_modifier(
    scheduler_cfgs: List[List[Dict[str, Any]]],
    model,
    layer_decay_value: float,
    layer_decay_min: Optional[float] = None,
    apply_to: Optional[str] = None,
    overrides: List[Dict[str, Any]] = (),
) -> List[List[Dict[str, Any]]]:
    """Apply official SAM3 layer-decay rewriting to scheduler configs."""

    scoped_model = rgetattr(model, apply_to)
    num_layers = scoped_model.get_num_layers() + 1
    layer_decays = [
        layer_decay_value ** (num_layers - i) for i in range(num_layers + 1)
    ]
    if layer_decay_min is not None:
        layer_decays = [max(val, layer_decay_min) for val in layer_decays]
    final_scheduler_cfgs = []
    prefix = apply_to or ""
    for scheduler_cfg_group in scheduler_cfgs:
        curr_cfg_group = []
        for scheduler_cfg in scheduler_cfg_group:
            if scheduler_cfg["option"] != "lr":
                curr_cfg_group.append(scheduler_cfg)
                continue
            parameter_names = sorted(scheduler_cfg["parameter_names"])
            layer_cfg_groups = {}
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

                if layer_id not in layer_cfg_groups:
                    curr_param = {
                        "option": scheduler_cfg["option"],
                        "scheduler": ValueScaler(
                            scheduler_cfg["scheduler"], this_scale
                        ),
                        "parameter_names": {param_name},
                    }
                else:
                    curr_param = layer_cfg_groups[layer_id]
                    curr_param["parameter_names"].add(param_name)
                layer_cfg_groups[layer_id] = curr_param

            for layer_cfg in layer_cfg_groups.values():
                curr_cfg_group.append(layer_cfg)

        final_scheduler_cfgs.append(curr_cfg_group)
    return final_scheduler_cfgs
