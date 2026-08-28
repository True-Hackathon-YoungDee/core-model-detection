"""Durable fall-domain event records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..fall_fsm import FallAlertKind, FallEvidenceLevel, FallState

if TYPE_CHECKING:
    from ..fall_evidence import FallEvidence, FallFeatures
    from ..fall_fsm import FallDecision


@dataclass(frozen=True)
class FallIncident:
    """Durable semantic record created by one terminal fall decision."""

    incident_id: str
    original_person_id: int
    kind: FallAlertKind
    evidence_level: FallEvidenceLevel
    terminal_state: FallState
    detected_at: float
    recovered_at: float | None = None


@dataclass(frozen=True)
class FallEvent:
    """One observed or gap-driven FSM decision for a tracked person."""

    person_id: int
    state: FallState
    state_changed: bool
    t_seconds: float
    decision: FallDecision | None = None
    features: FallFeatures | None = None
    evidence: FallEvidence | None = None
    evidence_fraction: float = 0.0
    coverage_fraction: float = 0.0
    evidence_elapsed_s: float = 0.0
    evidence_required_s: float = 0.0
    observation_age_s: float = 0.0
    alert_kind: FallAlertKind | None = None
    evidence_level: FallEvidenceLevel | None = None
    incident: FallIncident | None = None
    incident_event: Literal["detected", "recovered"] | None = None
    torso_angle_deg: float = 0.0
    com_height: float | None = None
    instability_index: float | None = None
    vote_fraction: float = 0.0
