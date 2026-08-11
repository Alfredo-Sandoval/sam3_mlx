from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

from sam3_mlx.train.utils import checkpoint_utils, distributed, train_utils


class _Parameter:
    def __init__(self, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


class _StateModel:
    def __init__(self) -> None:
        self.weight = np.array([1.0, 2.0], dtype=np.float32)
        self.parameter = _Parameter(requires_grad=False)

    def state_dict(self) -> dict[str, object]:
        return {"encoder.weight": self.weight}

    def named_parameters(self):
        return [("encoder.weight", self.parameter)]


class _LoadModel:
    def __init__(self, result: object) -> None:
        self.result = result
        self.loaded: tuple[dict[str, object], bool] | None = None

    def load_state_dict(self, state_dict: dict[str, object], *, strict: bool) -> object:
        self.loaded = (state_dict, strict)
        return self.result


class _DropTemporaryKernel:
    def __call__(self, *, state_dict: dict[str, object]) -> dict[str, object]:
        return {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("temporary.")
        }


class _FakeOmegaConf:
    def __init__(self) -> None:
        self.resolvers: dict[str, Callable[..., object]] = {}

    def merge(self, *configs: object) -> object:
        return configs

    def register_new_resolver(
        self,
        name: str,
        resolver: Callable[..., object],
        *,
        replace: bool,
    ) -> None:
        assert replace
        self.resolvers[name] = resolver

    def to_yaml(self, config: object) -> str:
        return str(config)


class _ComputedMeter:
    def compute(self) -> dict[str, float]:
        return {"score": 0.75}


def test_checkpoint_pattern_filters_preserve_values_and_empty_exclusion_identity():
    state = {"encoder.weight": object(), "decoder.weight": object()}

    assert checkpoint_utils.filter_params_matching_unix_pattern(
        ["encoder.*"], state
    ) == {"encoder.weight": state["encoder.weight"]}
    assert checkpoint_utils.exclude_params_matching_unix_pattern([], state) is state
    assert checkpoint_utils.exclude_params_matching_unix_pattern(
        ["decoder.*"], state
    ) == {"encoder.weight": state["encoder.weight"]}


def test_get_state_dict_traverses_mapping_and_sequence_keys():
    checkpoint = {"models": [{"state_dict": {"weight": 3}}]}

    assert checkpoint_utils.get_state_dict(checkpoint, ["models", 0, "state_dict"]) == {
        "weight": 3
    }

    with pytest.raises(KeyError, match="sequence length 1"):
        checkpoint_utils.get_state_dict(checkpoint, ["models", 2])


def test_frozen_parameter_checks_detect_trainable_or_changed_values():
    model = _StateModel()
    checkpoint_utils.assert_skipped_parameters_are_frozen(model, ["encoder.*"])

    model.parameter.requires_grad = True
    with pytest.raises(ValueError, match="should be frozen"):
        checkpoint_utils.assert_skipped_parameters_are_frozen(model, ["encoder.*"])

    model.parameter.requires_grad = False
    with pytest.raises(ValueError, match="has initialized"):
        with checkpoint_utils.with_check_parameter_frozen(
            model, ["encoder.*"], disabled=False
        ):
            model.weight = model.weight + 1.0


def test_load_state_dict_applies_kernels_and_checks_result_keys():
    model = _LoadModel((list[str](), list[str]()))
    state = {"weight": object(), "temporary.buffer": object()}

    result = checkpoint_utils.load_state_dict_into_model(
        state, model, checkpoint_kernels=[_DropTemporaryKernel()]
    )

    assert result is model
    assert model.loaded == ({"weight": state["weight"]}, False)

    missing_model = _LoadModel((["missing.weight"], list[str]()))
    with pytest.raises(KeyError, match="Missing keys"):
        checkpoint_utils.load_state_dict_into_model(
            {"weight": object()}, missing_model, strict=True
        )


def test_single_process_distributed_helpers_preserve_identity_and_shape():
    value = {"payload": 1}

    assert distributed.all_gather(value) == [value]
    assert distributed.gather_tensors_from_all(value) == [value]
    assert distributed.broadcast_object(value) is value
    assert distributed.unwrap_ddp_if_wrapped(value) is value
    assert distributed.all_reduce_op(value, op="sum") is value
    assert (
        distributed.all_reduce_op(
            value, op="sum", after_op_func=lambda item: item["payload"] + 1
        )
        == 2
    )
    assert distributed.get_world_size() == 1
    assert distributed.get_rank() == 0
    assert not distributed.is_dist_avail_and_initialized()


def test_distributed_setup_boundaries_remain_unsupported():
    with pytest.raises(NotImplementedError, match="create_new_process_group"):
        distributed.create_new_process_group(2)
    with pytest.raises(NotImplementedError, match="_get_global_gloo_group"):
        distributed.get_global_gloo_group()


def test_collect_dict_keys_recurses_through_collate_configs():
    config = {
        "loader": [
            {"_target_": "package.collate_fn", "dict_key": "images"},
            {
                "nested": {
                    "_target_": "package.other_collate_fn",
                    "dict_key": "masks",
                }
            },
        ]
    }

    assert train_utils.collect_dict_keys(config) == ["images", "masks"]


def test_resolver_registration_keeps_optional_hydra_failure_explicit(
    monkeypatch: pytest.MonkeyPatch,
):
    omega_conf = _FakeOmegaConf()

    def fake_import(name: str) -> object:
        if name == "omegaconf":
            return SimpleNamespace(OmegaConf=omega_conf)
        raise ImportError(name)

    monkeypatch.setattr(train_utils, "import_module", fake_import)
    train_utils.register_omegaconf_resolvers()

    assert omega_conf.resolvers["add"](2, 3) == 5
    assert omega_conf.resolvers["range"](3) == [0, 1, 2]
    with pytest.raises(TypeError, match="booleans"):
        omega_conf.resolvers["int"](True)
    with pytest.raises(NotImplementedError, match="Hydra get_method"):
        omega_conf.resolvers["get_method"]("package.function")


def test_seed_and_meter_integer_boundaries_reject_booleans():
    with pytest.raises(TypeError, match="seed_value"):
        train_utils.set_seeds(True, 2, 0)

    meter = train_utils.AverageMeter("loss", device="cpu")
    meter.update(2.0, n=2)
    meter.update(4.0)
    assert meter.avg == pytest.approx(8.0 / 3.0)
    with pytest.raises(TypeError, match="n must be an integer"):
        meter.update(1.0, n=True)


def test_duration_and_progress_meter_formatting(caplog: pytest.LogCaptureFixture):
    duration = train_utils.DurationMeter("elapsed", device="cpu")
    duration.add(3661)
    assert str(duration) == "elapsed: 00d 01h 01m"

    progress = train_utils.ProgressMeter(
        10,
        meters=[train_utils.AverageMeter("loss", device="cpu")],
        real_meters={"validation": _ComputedMeter()},
        prefix="epoch ",
    )
    with caplog.at_level("INFO"):
        progress.display(2)
    assert "validation/score: 0.7500" in caplog.text
