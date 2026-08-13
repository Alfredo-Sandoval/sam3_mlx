import pytest

from sam3_mlx.model.temporal_memory import (
    memory_trim_plan,
    previous_memory_frame_index,
)


@pytest.mark.parametrize(
    ("reverse", "distance", "expected"),
    [
        (False, 1, 16),
        (False, 2, 15),
        (False, 3, 12),
        (True, 1, 18),
        (True, 2, 21),
        (True, 3, 24),
    ],
)
def test_previous_memory_frame_index_preserves_stride_alignment(
    reverse: bool,
    distance: int,
    expected: int,
) -> None:
    assert (
        previous_memory_frame_index(
            frame_idx=17,
            temporal_distance=distance,
            stride=3,
            track_in_reverse=reverse,
        )
        == expected
    )


def test_previous_memory_frame_index_uses_ranked_selection() -> None:
    selected = [2, 7, 11]
    assert (
        previous_memory_frame_index(
            frame_idx=20,
            temporal_distance=2,
            stride=4,
            track_in_reverse=False,
            selected_indices=selected,
        )
        == 7
    )
    assert (
        previous_memory_frame_index(
            frame_idx=20,
            temporal_distance=4,
            stride=4,
            track_in_reverse=False,
            selected_indices=selected,
        )
        is None
    )


def test_memory_trim_plan_separates_selection_from_state_mutation() -> None:
    plan = memory_trim_plan(
        frame_idx=100,
        stride=3,
        num_maskmem=7,
        use_memory_selection=True,
        max_object_pointers=16,
    )

    assert plan.expired_frame_idx == 79
    assert plan.far_history_frame_idx == -220
