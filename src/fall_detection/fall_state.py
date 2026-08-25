"""RGB-only per-person fall tracking and durable event boundaries."""

from __future__ import annotations

import warnings
from typing import Literal

from .fall_config import FallConfig
from .domain.events import FallEvent, FallIncident
from .fall_evidence import FallEvidence, FallFeatures, ImageEvidenceExtractor
from .fall_fsm import (
    FallAlertKind,
    FallDecision,
    FallEvidenceLevel,
    FallState,
    PersonFallFSM,
)
from .pose import PersonPose


class _PersonFallTracker:
    def __init__(self, person_id: int, config: FallConfig) -> None:
        self.person_id = person_id
        self.extractor = ImageEvidenceExtractor(config)
        self.fsm = PersonFallFSM(config)
        self.last_observed_at: float | None = None

    def update(
        self,
        person: PersonPose,
        t_seconds: float,
        frame_width: int,
        frame_height: int,
    ) -> tuple[FallFeatures, FallDecision]:
        features = self.extractor.update(
            person,
            t_seconds,
            frame_width,
            frame_height,
        )
        self.last_observed_at = t_seconds
        return features, self.fsm.step(features)

    def observe_gap(self, t_seconds: float) -> FallDecision:
        return self.fsm.observe_gap(t_seconds)


class FallStateManager:
    """Own RGB extractors and temporal FSMs keyed by stable person id."""

    def __init__(
        self,
        config: FallConfig | None = None,
        body_mass_kg: float | None = None,
    ) -> None:
        self.config = config or FallConfig()
        if body_mass_kg is not None:
            warnings.warn(
                "body_mass_kg is deprecated and has no effect on RGB fall decisions",
                DeprecationWarning,
                stacklevel=2,
            )
        self._trackers: dict[int, _PersonFallTracker] = {}
        self._incidents: list[FallIncident] = []
        self._active_incidents: dict[int, FallIncident] = {}
        self._next_incident_sequence = 1

    @property
    def incidents(self) -> tuple[FallIncident, ...]:
        return tuple(self._incidents)

    def update(
        self,
        persons: list[PersonPose],
        t_seconds: float,
        frame_width: int,
        frame_height: int,
    ) -> list[FallEvent]:
        _validate_dimensions(frame_width, frame_height)
        events: list[FallEvent] = []
        observed_ids: set[int] = set()
        for person in persons:
            person_id = person.person_id
            observed_ids.add(person_id)
            tracker = self._trackers.get(person_id)
            if tracker is None:
                tracker = _PersonFallTracker(person_id, self.config)
                self._trackers[person_id] = tracker
            features, decision = tracker.update(
                person,
                t_seconds,
                frame_width,
                frame_height,
            )
            events.append(
                self._event(
                    person_id,
                    t_seconds,
                    decision,
                    features=features,
                    observation_age_s=0.0,
                )
            )

        for person_id in sorted(set(self._trackers) - observed_ids):
            tracker = self._trackers[person_id]
            decision = tracker.observe_gap(t_seconds)
            observation_age_s = (
                max(0.0, t_seconds - tracker.last_observed_at)
                if tracker.last_observed_at is not None
                else 0.0
            )
            events.append(
                self._event(
                    person_id,
                    t_seconds,
                    decision,
                    features=None,
                    observation_age_s=observation_age_s,
                )
            )
        return events

    def forget(self, person_id: int) -> None:
        self._trackers.pop(person_id, None)
        self._active_incidents.pop(person_id, None)

    def reset(self, preserve_incidents: bool = True) -> None:
        self._trackers.clear()
        self._active_incidents.clear()
        if not preserve_incidents:
            self.clear_incidents()

    def clear_incidents(self) -> None:
        self._incidents.clear()
        self._active_incidents.clear()

    def _event(
        self,
        person_id: int,
        t_seconds: float,
        decision: FallDecision,
        *,
        features: FallFeatures | None,
        observation_age_s: float,
    ) -> FallEvent:
        incident, incident_event = self._apply_incident_decision(
            person_id,
            t_seconds,
            decision,
        )
        return FallEvent(
            person_id=person_id,
            state=decision.state,
            state_changed=decision.state_changed,
            t_seconds=t_seconds,
            decision=decision,
            features=features,
            evidence=decision.evidence,
            evidence_fraction=decision.evidence_fraction,
            coverage_fraction=decision.coverage_fraction,
            evidence_elapsed_s=decision.evidence_elapsed_s,
            evidence_required_s=decision.evidence_required_s,
            observation_age_s=observation_age_s,
            alert_kind=decision.alert_kind,
            evidence_level=decision.evidence_level,
            incident=incident,
            incident_event=incident_event,
            torso_angle_deg=(features.torso_angle_deg if features is not None else 0.0),
            vote_fraction=decision.evidence_fraction,
        )

    def _apply_incident_decision(
        self,
        person_id: int,
        t_seconds: float,
        decision: FallDecision,
    ) -> tuple[FallIncident | None, Literal["detected", "recovered"] | None]:
        active = self._active_incidents.get(person_id)
        if decision.recovered:
            if active is None:
                return None, None
            recovered = FallIncident(
                incident_id=active.incident_id,
                original_person_id=active.original_person_id,
                kind=active.kind,
                evidence_level=active.evidence_level,
                terminal_state=active.terminal_state,
                detected_at=active.detected_at,
                recovered_at=t_seconds,
            )
            incident_index = next(
                index
                for index, incident in enumerate(self._incidents)
                if incident.incident_id == active.incident_id
            )
            self._incidents[incident_index] = recovered
            del self._active_incidents[person_id]
            return recovered, "recovered"

        if decision.alert_kind is not None and decision.evidence_level is not None:
            if active is not None:
                return active, None
            incident = FallIncident(
                incident_id=f"fall-{self._next_incident_sequence:06d}",
                original_person_id=person_id,
                kind=decision.alert_kind,
                evidence_level=decision.evidence_level,
                terminal_state=decision.state,
                detected_at=t_seconds,
            )
            self._next_incident_sequence += 1
            self._incidents.append(incident)
            self._active_incidents[person_id] = incident
            return incident, "detected"

        return active, None


def _validate_dimensions(frame_width: object, frame_height: object) -> None:
    if (
        type(frame_width) is not int
        or type(frame_height) is not int
        or frame_width <= 0
        or frame_height <= 0
    ):
        raise ValueError("frame_width and frame_height must be positive integers")
