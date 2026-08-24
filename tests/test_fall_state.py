import dataclasses

from fall_detection.fall_fsm import FallState, FallThresholds
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


def _squashed_lying_person() -> "PersonPose":
    """A lying-flat person whose bbox is normalized against a 16:9 frame: the
    real pixel aspect ratio is 1.5 (wide), but MediaPipe's per-axis
    normalization (divide x by frame width, y by frame height) compresses
    that down to ~0.84 in normalized coordinates -- below the 1.2 threshold
    even though the person is unmistakably lying down."""
    _, lying_person = fall_sequence(standing_s=0.05, transition_frames=2, lying_s=0.05)[-1]
    return dataclasses.replace(lying_person, bbox=(0.3, 0.4, 0.553125, 0.7))


def test_normalized_aspect_ratio_understates_prone_pose_without_frame_dims():
    """Regression guard: omitting frame_width/frame_height must reproduce
    today's normalized-bbox behavior exactly -- existing callers (and tests)
    that never pass dims must see no change."""
    person = _squashed_lying_person()
    manager = FallStateManager()
    thresholds = FallThresholds()
    state = FallState.UPRIGHT
    for i in range(thresholds.static_prone_frames + 5):
        events = manager.update([person], t_seconds=i / 30.0)
        state = events[0].state
    assert state == FallState.UPRIGHT


def test_frame_dims_correct_pixel_aspect_ratio_unlocks_static_prone_path():
    """With real 1920x1080 frame dims supplied, the same squashed bbox
    corrects to its true pixel aspect ratio (1.5 > 1.2) and the static-prone
    path fires."""
    person = _squashed_lying_person()
    manager = FallStateManager()
    thresholds = FallThresholds()
    state = FallState.UPRIGHT
    for i in range(thresholds.static_prone_frames + 5):
        events = manager.update([person], t_seconds=i / 30.0, frame_width=1920, frame_height=1080)
        state = events[0].state
    assert state == FallState.SLUMPING
