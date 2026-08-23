from fall_detection.fall_fsm import FallState
from fall_detection.fall_state import FallStateManager

from conftest import make_person
from synthetic_falls import fall_sequence


def test_fast_fall_sequence_escalates_to_fall_confirmed():
    manager = FallStateManager()
    states = []
    for t_seconds, person in fall_sequence():
        events = manager.update([person], t_seconds)
        states.append(events[0].state)

    assert states[-1] == FallState.FALL_CONFIRMED
    assert states == sorted(states)  # never regresses once escalation starts
    assert FallState.UPRIGHT in states  # actually started upright


def test_forget_discards_tracker_state_for_that_person():
    manager = FallStateManager()
    for t_seconds, person in fall_sequence(standing_s=0.2, transition_frames=5, lying_s=0.2):
        events = manager.update([person], t_seconds)
    assert events[0].state != FallState.UPRIGHT  # mid-escalation, not reset

    manager.forget(person_id=1)

    fresh_person = make_person(person_id=1)  # standing pose, fresh
    event = manager.update([fresh_person], t_seconds=1000.0)[0]
    assert event.state == FallState.UPRIGHT
