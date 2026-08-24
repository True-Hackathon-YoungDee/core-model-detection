import importlib
import io
import math
from dataclasses import replace

import pytest

from fall_detection.fall_evidence import FallEvidence, FallFeatures
from fall_detection.fall_fsm import (
    FallAlertKind,
    FallDecision,
    FallEvidenceLevel,
    FallState,
)
from fall_detection.fall_state import FallEvent, FallIncident


def _telemetry_module():
    return importlib.import_module("fall_detection.fall_telemetry")


def _event(*, incident_event="detected") -> FallEvent:
    features = FallFeatures(
        t_seconds=12.5,
        valid=True,
        torso_angle_deg=78.0,
        bbox_aspect_ratio=1.4,
        hip_downward_speed_bh_s=0.8,
        bbox_downward_speed_bh_s=0.7,
        torso_rotation_deg_s=82.0,
        height_collapse_fraction=0.3,
        motion_bh_s=0.04,
        visibility_quality=0.92,
        torso_centroid=(0.45, 0.7),
        furniture_roi=None,
        scale_source="upright_height",
        motion_available=True,
    )
    evidence = FallEvidence(
        dynamic_torso_angle=True,
        downward_motion=True,
        rapid_torso_rotation=True,
        height_collapse=True,
        posture_torso=True,
        posture_aspect=True,
        stillness=True,
        quality_ok=True,
    )
    decision = FallDecision(
        state=FallState.FALL_CONFIRMED,
        previous_state=FallState.POST_STABILITY_EVALUATION,
        state_changed=True,
        evidence=evidence,
        evidence_fraction=0.75,
        coverage_fraction=0.8,
        evidence_elapsed_s=1.2,
        evidence_required_s=1.0,
        alert_kind=FallAlertKind.OBSERVED_FALL,
        evidence_level=FallEvidenceLevel.HIGH,
        recovered=False,
    )
    incident = FallIncident(
        incident_id="fall-000007",
        original_person_id=40,
        kind=FallAlertKind.OBSERVED_FALL,
        evidence_level=FallEvidenceLevel.HIGH,
        terminal_state=FallState.FALL_CONFIRMED,
        detected_at=12.5,
        recovered_at=None,
    )
    return FallEvent(
        person_id=4,
        state=decision.state,
        state_changed=True,
        t_seconds=12.5,
        decision=decision,
        features=features,
        evidence=evidence,
        evidence_fraction=0.75,
        coverage_fraction=0.8,
        evidence_elapsed_s=1.2,
        evidence_required_s=1.0,
        observation_age_s=0.15,
        alert_kind=FallAlertKind.OBSERVED_FALL,
        evidence_level=FallEvidenceLevel.HIGH,
        incident=incident,
        incident_event=incident_event,
    )


def test_detected_incident_record_has_literal_schema():
    record = _telemetry_module().event_record(_event(), "detected")

    assert record == {
        "schema_version": 1,
        "incident_id": "fall-000007",
        "original_person_id": 40,
        "event": "detected",
        "person_id": 4,
        "terminal_state": "FALL_CONFIRMED",
        "state": "FALL_CONFIRMED",
        "t_seconds": 12.5,
        "kind": "OBSERVED_FALL",
        "evidence_level": "HIGH",
        "detected_at": 12.5,
        "recovered_at": None,
    }


def test_recovered_incident_record_has_literal_schema():
    detected = _event(incident_event="recovered")
    recovered_incident = replace(detected.incident, recovered_at=20.0)
    recovered = replace(
        detected,
        state=FallState.UPRIGHT,
        t_seconds=20.0,
        incident=recovered_incident,
    )

    record = _telemetry_module().event_record(recovered, "recovered")

    assert record == {
        "schema_version": 1,
        "incident_id": "fall-000007",
        "original_person_id": 40,
        "event": "recovered",
        "person_id": 4,
        "terminal_state": "FALL_CONFIRMED",
        "state": "UPRIGHT",
        "t_seconds": 20.0,
        "kind": "OBSERVED_FALL",
        "evidence_level": "HIGH",
        "detected_at": 12.5,
        "recovered_at": 20.0,
    }


def test_telemetry_record_has_raw_features_gates_and_decision_context():
    record = _telemetry_module().telemetry_record(_event())

    assert record == {
        "schema_version": 1,
        "person_id": 4,
        "t_seconds": 12.5,
        "previous_state": "POST_STABILITY_EVALUATION",
        "state": "FALL_CONFIRMED",
        "state_changed": True,
        "features": {
            "t_seconds": 12.5,
            "valid": True,
            "torso_angle_deg": 78.0,
            "bbox_aspect_ratio": 1.4,
            "hip_downward_speed_bh_s": 0.8,
            "bbox_downward_speed_bh_s": 0.7,
            "torso_rotation_deg_s": 82.0,
            "height_collapse_fraction": 0.3,
            "motion_bh_s": 0.04,
            "visibility_quality": 0.92,
            "torso_centroid": [0.45, 0.7],
            "furniture_roi": None,
            "scale_source": "upright_height",
            "motion_available": True,
        },
        "evidence": {
            "dynamic_torso_angle": True,
            "downward_motion": True,
            "rapid_torso_rotation": True,
            "height_collapse": True,
            "posture_torso": True,
            "posture_aspect": True,
            "stillness": True,
            "quality_ok": True,
        },
        "evidence_fraction": 0.75,
        "coverage_fraction": 0.8,
        "evidence_elapsed_s": 1.2,
        "evidence_required_s": 1.0,
        "observation_age_s": 0.15,
        "alert_kind": "OBSERVED_FALL",
        "evidence_level": "HIGH",
        "incident_id": "fall-000007",
        "incident_event": "detected",
    }


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_jsonl_writer_rejects_non_finite_values(non_finite):
    event = _event()
    event = replace(
        event,
        features=replace(event.features, torso_angle_deg=non_finite),
    )

    with pytest.raises(ValueError, match="JSON"):
        _telemetry_module().write_jsonl(
            io.StringIO(), _telemetry_module().telemetry_record(event)
        )
