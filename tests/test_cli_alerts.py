import json

from fall_detection.cli import format_alert
from fall_detection.fall_fsm import FallState
from fall_detection.fall_state import FallEvent, FallStateManager

from synthetic_falls import fall_sequence


def test_format_alert_produces_one_json_line_with_expected_fields():
    event = FallEvent(
        person_id=3,
        state=FallState.FALL_CONFIRMED,
        state_changed=True,
        t_seconds=12.5,
        torso_angle_deg=80.0,
        com_height=0.05,
        instability_index=0.1,
        vote_fraction=0.9,
    )
    line = format_alert(event)
    assert line.endswith("\n")
    payload = json.loads(line)
    assert payload == {"person_id": 3, "state": "FALL_CONFIRMED", "t_seconds": 12.5}


def test_real_fall_sequence_produces_exactly_one_fall_confirmed_alert_line():
    manager = FallStateManager()
    alert_lines = []
    for t_seconds, person in fall_sequence():
        for event in manager.update([person], t_seconds):
            if event.state_changed and event.state in (FallState.FALL_CONFIRMED, FallState.BED_REST):
                alert_lines.append(format_alert(event))

    assert len(alert_lines) == 1
    payload = json.loads(alert_lines[0])
    assert payload["state"] == "FALL_CONFIRMED"
    assert payload["person_id"] == 1
