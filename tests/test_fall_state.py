from __future__ import annotations

import dataclasses
import math
from dataclasses import FrozenInstanceError

import pytest

from fall_detection.fall_config import FallConfig
from fall_detection.fall_fsm import FallAlertKind, FallEvidenceLevel, FallState
from fall_detection.fall_state import FallEvent, FallIncident, FallStateManager
from fall_detection.pose import PoseLandmark

from conftest import (
    NUM_LANDMARKS,
    make_landmark,
    make_person,
    make_world_landmark,
)


def _person_with_torso(
    left_shoulder: tuple[float, float] = (0.4, 0.3),
    right_shoulder: tuple[float, float] = (0.5, 0.3),
    left_hip: tuple[float, float] = (0.5, 0.7),
    right_hip: tuple[float, float] = (0.6, 0.7),
    *,
    person_id: int = 1,
):
    landmarks = [make_landmark(0.5, 0.5) for _ in range(NUM_LANDMARKS)]
    for index, point in (
        (PoseLandmark.LEFT_SHOULDER, left_shoulder),
        (PoseLandmark.RIGHT_SHOULDER, right_shoulder),
        (PoseLandmark.LEFT_HIP, left_hip),
        (PoseLandmark.RIGHT_HIP, right_hip),
    ):
        landmarks[index] = make_landmark(*point)
    return make_person(person_id=person_id, landmarks=landmarks)


def test_manager_requires_frame_dimensions():
    manager = FallStateManager()

    with pytest.raises(TypeError):
        manager.update([], 0.0)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("frame_width", "frame_height"),
    [
        (0, 720),
        (-1, 720),
        (1280, 0),
        (1280, -1),
        (True, 720),
        (1280.0, 720),
    ],
)
def test_manager_rejects_non_positive_or_non_integer_dimensions(
    frame_width: object,
    frame_height: object,
):
    manager = FallStateManager()

    with pytest.raises(ValueError, match="positive integers"):
        manager.update(
            [],
            0.0,
            frame_width=frame_width,  # type: ignore[arg-type]
            frame_height=frame_height,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "world_landmarks",
    [
        [],
        [make_world_landmark(0.0, 0.0)],
    ],
)
def test_empty_or_short_world_landmarks_are_irrelevant(world_landmarks):
    person = dataclasses.replace(
        _person_with_torso(),
        world_landmarks=world_landmarks,
    )

    event = FallStateManager().update([person], 0.0, 1600, 900)[0]

    assert isinstance(event, FallEvent)
    assert event.features is not None
    assert event.features.valid is True
    assert event.state is FallState.UPRIGHT


def test_short_image_landmark_list_returns_safe_invalid_event():
    person = make_person(person_id=4, landmarks=[])

    event = FallStateManager().update([person], 0.0, 1280, 720)[0]

    assert event.person_id == 4
    assert event.features is not None
    assert event.features.valid is False
    assert event.evidence is not None
    assert not any(dataclasses.astuple(event.evidence))
    assert event.state is FallState.UPRIGHT


def test_manager_decision_geometry_uses_exact_frame_dimensions():
    person = _person_with_torso()

    event = FallStateManager(FallConfig()).update([person], 0.0, 1600, 900)[0]

    assert event.features is not None
    expected_angle = math.degrees(math.atan2(0.1 * 1600.0, 0.4 * 900.0))
    expected_aspect = (0.2 * 1600.0) / (0.4 * 900.0)
    assert event.features.torso_angle_deg == pytest.approx(expected_angle)
    assert event.features.bbox_aspect_ratio == pytest.approx(expected_aspect)
    assert event.com_height is None


def test_missing_known_person_ticks_gap_without_fabricating_pose_evidence():
    manager = FallStateManager()
    manager.update([_person_with_torso(person_id=7)], 0.0, 1280, 720)

    events = manager.update([], 0.25, 1280, 720)

    assert len(events) == 1
    event = events[0]
    assert event.person_id == 7
    assert event.features is None
    assert event.evidence is None
    assert event.decision.evidence is None
    assert event.observation_age_s == pytest.approx(0.25)


def _pixel_person(
    person_id: int,
    *,
    bbox: tuple[float, float, float, float],
    shoulder_center: tuple[float, float],
    hip_center: tuple[float, float],
):
    frame_size = 1000.0
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    landmarks = [
        make_landmark(center_x / frame_size, center_y / frame_size)
        for _ in range(NUM_LANDMARKS)
    ]
    for index, (x, y) in {
        PoseLandmark.NOSE: (bbox[0], bbox[1]),
        PoseLandmark.LEFT_FOOT_INDEX: (bbox[2], bbox[3]),
        PoseLandmark.LEFT_SHOULDER: (shoulder_center[0] - 50.0, shoulder_center[1]),
        PoseLandmark.RIGHT_SHOULDER: (shoulder_center[0] + 50.0, shoulder_center[1]),
        PoseLandmark.LEFT_HIP: (hip_center[0] - 50.0, hip_center[1]),
        PoseLandmark.RIGHT_HIP: (hip_center[0] + 50.0, hip_center[1]),
    }.items():
        landmarks[index] = make_landmark(x / frame_size, y / frame_size)
    return make_person(person_id=person_id, landmarks=landmarks, world_landmarks=[])


def _upright(person_id: int):
    return _pixel_person(
        person_id,
        bbox=(200.0, 100.0, 800.0, 900.0),
        shoulder_center=(500.0, 300.0),
        hip_center=(500.0, 700.0),
    )


def _prone(person_id: int):
    return _pixel_person(
        person_id,
        bbox=(200.0, 600.0, 800.0, 1000.0),
        shoulder_center=(400.0, 800.0),
        hip_center=(600.0, 800.0),
    )


def _drive_observed_fall(
    manager: FallStateManager,
    person_id: int,
    *,
    base_t: float = 0.0,
) -> list[FallEvent]:
    events = [manager.update([_upright(person_id)], base_t, 1000, 1000)[0]]
    events.append(manager.update([_prone(person_id)], base_t + 0.5, 1000, 1000)[0])
    events.append(manager.update([_prone(person_id)], base_t + 0.6, 1000, 1000)[0])
    events.append(manager.update([_prone(person_id)], base_t + 0.7, 1000, 1000)[0])
    for offset in (0.9, 1.1, 1.3, 1.5, 1.7, 1.9):
        events.append(manager.update([_prone(person_id)], base_t + offset, 1000, 1000)[0])
    assert events[-2].state is FallState.POST_STABILITY_EVALUATION
    assert events[-1].state is FallState.FALL_CONFIRMED
    return events


def _drive_persistent_prone(
    manager: FallStateManager,
    person_id: int,
    *,
    base_t: float,
) -> list[FallEvent]:
    events = []
    for offset in (0.0, 0.5, 1.0, 1.5, 2.0, 2.1, 2.2):
        events.append(manager.update([_prone(person_id)], base_t + offset, 1000, 1000)[0])
    assert events[-2].state is FallState.POST_STABILITY_EVALUATION
    assert events[-1].state is FallState.FALL_CONFIRMED
    return events


def test_observed_and_persistent_confirmations_create_unique_immutable_incidents():
    manager = FallStateManager()
    observed_events = _drive_observed_fall(manager, 11)
    manager.forget(11)
    persistent_events = _drive_persistent_prone(manager, 22, base_t=10.0)

    incidents = manager.incidents

    assert tuple(incident.incident_id for incident in incidents) == (
        "fall-000001",
        "fall-000002",
    )
    assert incidents[0] == FallIncident(
        incident_id="fall-000001",
        original_person_id=11,
        kind=FallAlertKind.OBSERVED_FALL,
        evidence_level=FallEvidenceLevel.HIGH,
        terminal_state=FallState.FALL_CONFIRMED,
        detected_at=1.9,
        recovered_at=None,
    )
    assert incidents[1].original_person_id == 22
    assert incidents[1].kind is FallAlertKind.PERSISTENT_PRONE
    assert incidents[1].evidence_level is FallEvidenceLevel.HIGH
    assert observed_events[-1].incident_event == "detected"
    assert persistent_events[-1].incident_event == "detected"
    with pytest.raises(FrozenInstanceError):
        incidents[0].recovered_at = 3.0  # type: ignore[misc]


def test_terminal_frames_emit_only_one_detected_event_for_active_incident():
    manager = FallStateManager()
    events = _drive_observed_fall(manager, 3)

    terminal = manager.update([_prone(3)], 2.0, 1000, 1000)[0]

    assert [event.incident_event for event in events].count("detected") == 1
    assert terminal.incident_event is None
    assert terminal.incident == manager.incidents[0]
    assert len(manager.incidents) == 1


def test_forget_and_default_reset_preserve_incident_history_until_explicit_clear():
    manager = FallStateManager()
    _drive_observed_fall(manager, 5)
    expected = manager.incidents

    manager.forget(5)
    assert manager.incidents == expected
    manager.reset()
    assert manager.incidents == expected
    manager.clear_incidents()
    assert manager.incidents == ()


@pytest.mark.parametrize("boundary", ["forget", "reset"])
def test_reused_person_id_after_runtime_boundary_creates_a_new_incident(boundary):
    manager = FallStateManager()
    _drive_persistent_prone(manager, 5, base_t=0.0)

    if boundary == "forget":
        manager.forget(5)
    else:
        manager.reset()
    later_events = _drive_persistent_prone(manager, 5, base_t=10.0)

    assert later_events[-1].incident_event == "detected"
    assert tuple(incident.incident_id for incident in manager.incidents) == (
        "fall-000001",
        "fall-000002",
    )
    assert all(incident.recovered_at is None for incident in manager.incidents)


def test_upright_recovery_updates_incident_once_and_emits_recovered_event():
    manager = FallStateManager()
    _drive_observed_fall(manager, 8)

    recovery_events = []
    for t_seconds in (2.0, 2.5, 3.0, 3.5, 4.0):
        recovery_events.append(manager.update([_upright(8)], t_seconds, 1000, 1000)[0])
    after = manager.update([_upright(8)], 4.1, 1000, 1000)[0]

    recovered = next(
        event for event in recovery_events if event.incident_event == "recovered"
    )
    assert recovered.incident_event == "recovered"
    assert recovered.state is FallState.UPRIGHT
    assert recovered.incident is not None
    assert recovered.incident.recovered_at == pytest.approx(3.0)
    assert manager.incidents[0] == recovered.incident
    assert [event.incident_event for event in recovery_events].count("recovered") == 1
    assert after.incident_event is None


def test_later_fall_after_recovery_creates_a_new_incident():
    manager = FallStateManager()
    _drive_observed_fall(manager, 9)
    for t_seconds in (2.0, 2.5, 3.0, 3.5, 4.0):
        manager.update([_upright(9)], t_seconds, 1000, 1000)

    later_events = _drive_observed_fall(manager, 9, base_t=5.0)

    assert later_events[-1].incident_event == "detected"
    assert tuple(incident.incident_id for incident in manager.incidents) == (
        "fall-000001",
        "fall-000002",
    )
    assert manager.incidents[0].recovered_at == pytest.approx(3.0)
    assert manager.incidents[1].recovered_at is None
