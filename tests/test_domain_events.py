from fall_detection.domain.events import FallEvent, FallIncident
from fall_detection.fall_fsm import FallAlertKind, FallEvidenceLevel, FallState


def test_domain_event_keeps_durable_incident_information() -> None:
    """Fails if moving event models drops incident data required by delivery adapters."""
    incident = FallIncident(
        incident_id="fall-000001",
        original_person_id=7,
        kind=FallAlertKind.PERSISTENT_PRONE,
        evidence_level=FallEvidenceLevel.HIGH,
        terminal_state=FallState.FALL_CONFIRMED,
        detected_at=3.5,
    )

    event = FallEvent(
        person_id=7,
        state=FallState.FALL_CONFIRMED,
        state_changed=True,
        t_seconds=3.5,
        incident=incident,
        incident_event="detected",
    )

    assert event.incident == incident
    assert event.incident_event == "detected"
