import json
from dataclasses import replace

import numpy as np

from fall_detection.cli import format_alert, main
from fall_detection.fall_fsm import FallAlertKind, FallEvidenceLevel, FallState
from fall_detection.fall_state import FallEvent, FallIncident


def _incident_event(event_type):
    recovered_at = 3.0 if event_type == "recovered" else None
    incident = FallIncident(
        incident_id="fall-000001",
        original_person_id=2,
        kind=FallAlertKind.PERSISTENT_PRONE,
        evidence_level=FallEvidenceLevel.MEDIUM,
        terminal_state=FallState.FALL_CONFIRMED,
        detected_at=1.0,
        recovered_at=recovered_at,
    )
    return FallEvent(
        person_id=2,
        state=FallState.UPRIGHT if recovered_at is not None else FallState.FALL_CONFIRMED,
        state_changed=True,
        t_seconds=recovered_at or 1.0,
        incident=incident,
        incident_event=event_type,
    )


def test_format_alert_wraps_versioned_incident_record():
    line = format_alert(_incident_event("detected"))

    assert line.endswith("\n")
    assert json.loads(line) == {
        "schema_version": 1,
        "incident_id": "fall-000001",
        "original_person_id": 2,
        "event": "detected",
        "person_id": 2,
        "terminal_state": "FALL_CONFIRMED",
        "state": "FALL_CONFIRMED",
        "t_seconds": 1.0,
        "kind": "PERSISTENT_PRONE",
        "evidence_level": "MEDIUM",
        "detected_at": 1.0,
        "recovered_at": None,
    }


def test_cli_writes_every_telemetry_event_and_only_incident_alerts(
    monkeypatch, tmp_path
):
    source = tmp_path / "clip.mp4"
    source.touch()
    alert_path = tmp_path / "alerts.jsonl"
    telemetry_path = tmp_path / "telemetry.jsonl"
    ordinary = FallEvent(
        person_id=2,
        state=FallState.UPRIGHT,
        state_changed=False,
        t_seconds=0.0,
    )
    batches = [
        [ordinary],
        [_incident_event("detected")],
        [replace(ordinary, t_seconds=2.0)],
        [_incident_event("recovered")],
    ]

    class FakeManager:
        def __init__(self, config):
            pass

        def update(self, persons, t_seconds, frame_width, frame_height):
            return batches.pop(0)

        def forget(self, person_id):
            pass

    class FakeRunner:
        def __init__(self, config, source, **kwargs):
            self.on_frame = kwargs["on_frame"]

        def run(self):
            frame = np.zeros((60, 80, 3), dtype=np.uint8)
            for index in range(4):
                self.on_frame([], float(index), frame)
                if index == 1:
                    assert alert_path.read_text(encoding="utf-8").count("\n") == 1
            return 4

    monkeypatch.setattr("fall_detection.fall_state.FallStateManager", FakeManager)
    monkeypatch.setattr("fall_detection.runner.VideoFileRunner", FakeRunner)

    code = main(
        [
            "--source",
            str(source),
            "--fall-alert-log",
            str(alert_path),
            "--fall-telemetry-log",
            str(telemetry_path),
            "--no-display",
        ]
    )

    assert code == 0
    telemetry_lines = [
        json.loads(line) for line in telemetry_path.read_text().splitlines()
    ]
    alert_lines = [json.loads(line) for line in alert_path.read_text().splitlines()]
    assert len(telemetry_lines) == 4
    assert [line["event"] for line in alert_lines] == ["detected", "recovered"]
