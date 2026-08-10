from collections.abc import Callable
from typing import cast

import pytest

from sam3_mlx.model.model_misc import SAM3Output, Sam3StageSteps, Sam3StepDict


def _step(name: str) -> dict[str, object]:
    return {"name": name}


def _step_name(item: Sam3StageSteps | Sam3StepDict) -> object:
    assert isinstance(item, dict)
    return item["name"]


def _stage_names(item: Sam3StageSteps | Sam3StepDict) -> list[object]:
    assert isinstance(item, list)
    return [step["name"] for step in item]


def test_sam3_output_construct_append_and_len():
    out = SAM3Output()
    assert len(out) == 0
    out.append([_step("a1"), _step("a2")])
    out.append([_step("b1")])
    assert len(out) == 2
    assert _stage_names(out[0]) == ["a1", "a2"]


def test_sam3_output_is_coherent_composition_container():
    out = SAM3Output(output=[[_step("x")]])
    assert not isinstance(out, list)
    assert isinstance(out.output, list)

    out.append([_step("y")])
    assert len(out.output) == 2
    assert _stage_names(out.output[1]) == ["y"]


def test_sam3_output_iteration_modes():
    out = SAM3Output(
        output=[[_step("s0a"), _step("s0b")], [_step("s1a")]],
        iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
    )
    assert [_stage_names(stage) for stage in out] == [
        ["s0a", "s0b"],
        ["s1a"],
    ]
    assert [name for stage in out for name in _stage_names(stage)] == [
        "s0a",
        "s0b",
        "s1a",
    ]

    out.iter_mode = SAM3Output.IterMode.LAST_STEP_PER_STAGE
    assert [_step_name(step) for step in out] == ["s0b", "s1a"]
    assert _step_name(out[0]) == "s0b"
    assert [_stage_names(stage) for stage in out.output] == [
        ["s0a", "s0b"],
        ["s1a"],
    ]

    out.iter_mode = SAM3Output.IterMode.FLATTENED
    assert [_step_name(step) for step in out] == ["s0a", "s0b", "s1a"]
    assert len(out) == 3
    assert _step_name(out[-1]) == "s1a"
    assert _step_name(out[1]) == "s0b"
    assert _stage_names(out.output[0]) == ["s0a", "s0b"]

    out.iter_mode = SAM3Output.IterMode.ALL_STEPS_PER_STAGE
    assert _stage_names(out[1]) == ["s1a"]


def test_sam3_output_append_rejects_step_without_stage_container():
    out = SAM3Output()
    append_runtime_value = cast(Callable[[object], None], out.append)

    with pytest.raises(AssertionError, match="Only list items are supported"):
        append_runtime_value(_step("not-a-stage"))


def test_sam3_output_iteration_mode_restores_on_exception():
    out = SAM3Output(
        output=[[_step("only")]],
        iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
    )
    with pytest.raises(RuntimeError, match="boom"):
        with SAM3Output.iteration_mode(out, SAM3Output.IterMode.FLATTENED) as nested:
            assert nested.iter_mode is SAM3Output.IterMode.FLATTENED
            raise RuntimeError("boom")
    assert out.iter_mode is SAM3Output.IterMode.ALL_STEPS_PER_STAGE
