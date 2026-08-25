"""Versioned JSONL records for fall decisions and durable incidents."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Literal, TextIO

from .domain.events import FallEvent

SCHEMA_VERSION = 1


def event_record(
    event: FallEvent,
    event_type: Literal["detected", "recovered"],
) -> dict[str, object]:
    """Return one schema-v1 incident boundary record."""
    if event_type not in ("detected", "recovered"):
        raise ValueError(f"unknown incident event: {event_type!r}")
    incident = event.incident
    if incident is None:
        raise ValueError("incident event requires an incident")
    return {
        "schema_version": SCHEMA_VERSION,
        "incident_id": incident.incident_id,
        "original_person_id": incident.original_person_id,
        "event": event_type,
        "person_id": event.person_id,
        "terminal_state": incident.terminal_state.name,
        "state": event.state.name,
        "t_seconds": event.t_seconds,
        "kind": incident.kind.value,
        "evidence_level": incident.evidence_level.value,
        "detected_at": incident.detected_at,
        "recovered_at": incident.recovered_at,
    }


def telemetry_record(event: FallEvent) -> dict[str, object]:
    """Return all raw evidence and FSM context for one manager event."""
    features: dict[str, object] | None = None
    if event.features is not None:
        features = asdict(event.features)
        features["torso_centroid"] = list(event.features.torso_centroid)

    evidence: dict[str, object] | None = None
    if event.evidence is not None:
        evidence = asdict(event.evidence)

    incident = event.incident
    alert_kind = event.alert_kind or (incident.kind if incident is not None else None)
    evidence_level = event.evidence_level or (
        incident.evidence_level if incident is not None else None
    )
    previous_state = (
        event.decision.previous_state if event.decision is not None else event.state
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "person_id": event.person_id,
        "t_seconds": event.t_seconds,
        "previous_state": previous_state.name,
        "state": event.state.name,
        "state_changed": event.state_changed,
        "features": features,
        "evidence": evidence,
        "evidence_fraction": event.evidence_fraction,
        "coverage_fraction": event.coverage_fraction,
        "evidence_elapsed_s": event.evidence_elapsed_s,
        "evidence_required_s": event.evidence_required_s,
        "observation_age_s": event.observation_age_s,
        "alert_kind": alert_kind.value if alert_kind is not None else None,
        "evidence_level": evidence_level.value if evidence_level is not None else None,
        "incident_id": incident.incident_id if incident is not None else None,
        "incident_event": event.incident_event,
    }


def jsonl_line(record: dict[str, object]) -> str:
    """Encode one strict JSON object followed by a newline."""
    return json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"


def write_jsonl(
    handle: TextIO,
    record: dict[str, object],
    *,
    flush: bool = False,
) -> None:
    """Append one strict JSON record to an already-managed text handle."""
    handle.write(jsonl_line(record))
    if flush:
        handle.flush()
