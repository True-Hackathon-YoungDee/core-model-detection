import numpy as np

from fall_detection.drawing import annotate_fall_state
from fall_detection.fall_fsm import FallState
from fall_detection.fall_state import FallEvent


def _event(state: FallState) -> FallEvent:
    return FallEvent(
        person_id=0,
        state=state,
        state_changed=False,
        t_seconds=0.0,
        torso_angle_deg=0.0,
        com_height=1.0,
        instability_index=None,
        vote_fraction=0.0,
    )


def test_annotate_fall_state_returns_same_shape_without_mutating_input():
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    original = canvas.copy()
    result = annotate_fall_state(canvas, [_event(FallState.UPRIGHT)])
    assert result.shape == canvas.shape
    assert np.array_equal(canvas, original)


def test_annotate_fall_state_draws_more_when_fall_confirmed():
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    upright = annotate_fall_state(canvas, [_event(FallState.UPRIGHT)])
    confirmed = annotate_fall_state(canvas, [_event(FallState.FALL_CONFIRMED)])
    upright_pixels = np.count_nonzero(upright)
    confirmed_pixels = np.count_nonzero(confirmed)
    assert confirmed_pixels > upright_pixels
