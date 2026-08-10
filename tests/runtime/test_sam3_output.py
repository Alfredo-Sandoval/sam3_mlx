import pytest

from sam3_mlx.model.model_misc import SAM3Output


def _step(name: str) -> dict[str, object]:
    return {"name": name}


def test_sam3_output_construct_append_and_len():
    out = SAM3Output()
    assert len(out) == 0
    out.append([_step("a1"), _step("a2")])
    out.append([_step("b1")])
    assert len(out) == 2
    assert out[0][0]["name"] == "a1"


def test_sam3_output_is_not_list_subclass():
    out = SAM3Output(output=[[_step("x")]])
    assert not isinstance(out, list)
    assert isinstance(out.output, list)


def test_sam3_output_iteration_modes():
    out = SAM3Output(
        output=[[_step("s0a"), _step("s0b")], [_step("s1a")]],
        iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
    )
    assert [step["name"] for stage in out for step in stage] == [
        "s0a",
        "s0b",
        "s1a",
    ]

    out.iter_mode = SAM3Output.IterMode.LAST_STEP_PER_STAGE
    assert [step["name"] for step in out] == ["s0b", "s1a"]
    assert out[0]["name"] == "s0b"

    out.iter_mode = SAM3Output.IterMode.FLATTENED
    assert [step["name"] for step in out] == ["s0a", "s0b", "s1a"]
    assert len(out) == 3
    assert out[-1]["name"] == "s1a"
    assert out[1]["name"] == "s0b"


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
