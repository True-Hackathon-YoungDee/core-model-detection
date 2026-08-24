import numpy as np

from fall_detection.drawing import annotate_fall_state
from fall_detection.fall_fsm import FallAlertKind, FallEvidenceLevel, FallState
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


def test_fall_overlay_is_compact_by_default(monkeypatch):
    labels = []
    monkeypatch.setattr(
        "fall_detection.drawing.cv2.putText",
        lambda image, text, *args, **kwargs: labels.append(text),
    )

    annotate_fall_state(
        np.zeros((100, 100, 3), dtype=np.uint8),
        [_event(FallState.UPRIGHT)],
    )

    assert labels == ["#0 UPRIGHT"]


def test_debug_overlay_shows_incident_evidence_timing_coverage_and_staleness(
    monkeypatch,
):
    labels = []
    monkeypatch.setattr(
        "fall_detection.drawing.cv2.putText",
        lambda image, text, *args, **kwargs: labels.append(text),
    )
    event = FallEvent(
        person_id=4,
        state=FallState.UPRIGHT,
        state_changed=False,
        t_seconds=12.5,
        evidence_fraction=0.75,
        coverage_fraction=0.8,
        evidence_elapsed_s=1.2,
        evidence_required_s=1.0,
        observation_age_s=0.15,
        alert_kind=FallAlertKind.OBSERVED_FALL,
        evidence_level=FallEvidenceLevel.HIGH,
    )

    annotate_fall_state(
        np.zeros((100, 100, 3), dtype=np.uint8),
        [event],
        debug=True,
        additional_observation_age_s=0.25,
    )

    assert labels == [
        "#4 UPRIGHT OBSERVED_FALL/HIGH ev=75% 1.2/1.0s cov=80% stale=0.4s"
    ]
