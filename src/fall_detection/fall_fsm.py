"""Seven-state temporal fall detection over RGB image evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from .fall_config import FallConfig
from .fall_evidence import FallEvidence, FallFeatures, classify_evidence


class FallState(IntEnum):
    """Stable fall states; ``IMPACT`` labels evidence peak, not measured force."""

    UPRIGHT = 0
    DESCENDING = 1
    IMPACT = 2
    SLUMPING = 3
    POST_STABILITY_EVALUATION = 4
    FALL_CONFIRMED = 5
    BED_REST = 6


class FallAlertKind(StrEnum):
    OBSERVED_FALL = "OBSERVED_FALL"
    PERSISTENT_PRONE = "PERSISTENT_PRONE"
    BED_REST = "BED_REST"


class FallEvidenceLevel(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"


@dataclass(frozen=True)
class FallDecision:
    state: FallState
    previous_state: FallState
    state_changed: bool
    evidence: FallEvidence | None
    evidence_fraction: float
    coverage_fraction: float
    evidence_elapsed_s: float
    evidence_required_s: float
    alert_kind: FallAlertKind | None
    evidence_level: FallEvidenceLevel | None
    recovered: bool


_DYNAMIC_CUES = (
    "downward_motion",
    "rapid_torso_rotation",
    "height_collapse",
)

_TERMINAL_STATES = (FallState.FALL_CONFIRMED, FallState.BED_REST)
_EPSILON = 1e-12


class _DurationAccumulator:
    """Integrate sample-and-hold evidence over valid bounded intervals."""

    def __init__(self, max_gap_s: float) -> None:
        self.max_gap_s = max_gap_s
        self.started_at: float | None = None
        self.last_at: float | None = None
        self.last_evidence: FallEvidence | None = None
        self.last_furniture_roi: str | None = None
        self.covered_s = 0.0
        self.posture_s = 0.0
        self.furniture_s = 0.0

    def observe(
        self,
        t_seconds: float,
        evidence: FallEvidence,
        furniture_roi: str | None,
        configured_roi_names: frozenset[str],
    ) -> None:
        if self.started_at is None:
            self.started_at = t_seconds
        if self.last_at is not None:
            interval_s = t_seconds - self.last_at
            if (
                0.0 < interval_s <= self.max_gap_s + _EPSILON
                and self.last_evidence is not None
                and self.last_evidence.quality_ok
            ):
                self.covered_s += interval_s
                if self.last_evidence.posture:
                    self.posture_s += interval_s
                    if self.last_furniture_roi in configured_roi_names:
                        self.furniture_s += interval_s
        self.last_at = t_seconds
        self.last_evidence = evidence
        self.last_furniture_roi = furniture_roi

    def gap(self, t_seconds: float) -> None:
        if self.started_at is not None:
            self.last_at = t_seconds
            self.last_evidence = None
            self.last_furniture_roi = None

    def clear(self) -> None:
        self.started_at = None
        self.last_at = None
        self.last_evidence = None
        self.last_furniture_roi = None
        self.covered_s = 0.0
        self.posture_s = 0.0
        self.furniture_s = 0.0

    def metrics(self, t_seconds: float) -> tuple[float, float, float]:
        elapsed_s = (
            max(0.0, t_seconds - self.started_at)
            if self.started_at is not None
            else 0.0
        )
        coverage_fraction = self.covered_s / elapsed_s if elapsed_s > 0.0 else 0.0
        evidence_fraction = (
            self.posture_s / self.covered_s if self.covered_s > 0.0 else 0.0
        )
        return evidence_fraction, coverage_fraction, elapsed_s


class PersonFallFSM:
    """Latch distinct dynamic cues and advance one person's fall state."""

    def __init__(self, config: FallConfig) -> None:
        self.config = config
        self.state = FallState.UPRIGHT
        self._dynamic_cue_times: dict[str, float] = {}
        self._candidate_since: float | None = None
        self._all_dynamic_cues_seen = False
        self._cooldown_until: float | None = None
        self._configured_roi_names = frozenset(roi.name for roi in config.furniture_rois)
        self._posture = _DurationAccumulator(config.max_observation_gap_s)
        self._candidate_alert_kind: FallAlertKind | None = None
        self._pending_alert_kind: FallAlertKind | None = None
        self._pending_evidence_level: FallEvidenceLevel | None = None
        self._active_alert_kind: FallAlertKind | None = None
        self._active_evidence_level: FallEvidenceLevel | None = None
        self._evidence_required_s = config.persistent_prone_dwell_s
        self._qualified_metrics: tuple[float, float, float] | None = None
        self._qualified_persistent_alert_kind: FallAlertKind | None = None
        self._alert_metrics: tuple[float, float, float] | None = None
        self._alert_evidence_required_s: float | None = None
        self._recovery_last_at: float | None = None
        self._recovery_elapsed_s = 0.0

    def step(self, features: FallFeatures) -> FallDecision:
        previous_state = self.state
        evidence = classify_evidence(features, self.config)
        t_seconds = features.t_seconds
        alert_kind = None
        evidence_level = None
        recovered = False

        if self.state is FallState.UPRIGHT:
            if self._cooldown_until is None or t_seconds >= self._cooldown_until:
                self._cooldown_until = None
                self._latch_dynamic_cues(t_seconds, evidence)
                if len(self._dynamic_cue_times) >= 2:
                    self.state = FallState.DESCENDING
                    self._candidate_since = t_seconds
                    self._all_dynamic_cues_seen = len(self._dynamic_cue_times) == 3
                    self._candidate_alert_kind = FallAlertKind.OBSERVED_FALL
                    self._evidence_required_s = (
                        self.config.observed_fall_postural_window_s
                    )
                    self._posture.clear()
                else:
                    self._advance_persistent_posture(features, evidence)
        elif self.state is FallState.DESCENDING:
            if self._candidate_timed_out(t_seconds):
                self._reject(t_seconds)
            else:
                self._latch_dynamic_cues(t_seconds, evidence)
                if len(self._dynamic_cue_times) == 3:
                    self._all_dynamic_cues_seen = True
                if evidence.dynamic_torso_angle:
                    self.state = FallState.IMPACT
        elif self.state is FallState.IMPACT:
            if self._candidate_timed_out(t_seconds):
                self._reject(t_seconds)
            elif evidence.quality_ok:
                self.state = FallState.SLUMPING
                self._posture.clear()
                self._posture.observe(
                    t_seconds,
                    evidence,
                    features.furniture_roi,
                    self._configured_roi_names,
                )
        elif self.state is FallState.SLUMPING:
            self._posture.observe(
                t_seconds,
                evidence,
                features.furniture_roi,
                self._configured_roi_names,
            )
            if self._candidate_alert_kind is FallAlertKind.PERSISTENT_PRONE:
                self._queue_evaluation(
                    self._qualified_persistent_alert_kind
                    or FallAlertKind.PERSISTENT_PRONE,
                    FallEvidenceLevel.HIGH,
                    t_seconds,
                    metrics=self._qualified_metrics,
                )
            elif self._posture_qualifies(
                t_seconds, self.config.observed_fall_postural_window_s
            ):
                self._queue_evaluation(
                    self._qualified_alert_kind(FallAlertKind.OBSERVED_FALL),
                    FallEvidenceLevel.HIGH,
                    t_seconds,
                )
            elif self._all_dynamic_cues_seen and self._postural_window_is_covered(
                t_seconds, self.config.observed_fall_postural_window_s
            ):
                self._queue_evaluation(
                    self._qualified_alert_kind(FallAlertKind.OBSERVED_FALL),
                    FallEvidenceLevel.MEDIUM,
                    t_seconds,
                )
            elif self._candidate_timed_out(t_seconds):
                self._reject(t_seconds)
        elif self.state is FallState.POST_STABILITY_EVALUATION:
            alert_kind, evidence_level = self._emit_pending_alert()
        elif self.state in _TERMINAL_STATES:
            recovered = self._advance_recovery(features, evidence)
            if recovered:
                alert_kind = self._active_alert_kind
                evidence_level = self._active_evidence_level

        return self._decision(
            previous_state,
            evidence,
            t_seconds,
            alert_kind=alert_kind,
            evidence_level=evidence_level,
            recovered=recovered,
        )

    def observe_gap(self, t_seconds: float) -> FallDecision:
        previous_state = self.state
        alert_kind = None
        evidence_level = None
        if self.state is FallState.SLUMPING:
            self._posture.gap(t_seconds)
        elif self.state is FallState.UPRIGHT:
            self._posture.gap(t_seconds)
        elif self.state in _TERMINAL_STATES:
            self._clear_recovery()
        if self.state in (FallState.DESCENDING, FallState.IMPACT, FallState.SLUMPING) and (
            self._candidate_timed_out(t_seconds)
        ):
            if self._all_dynamic_cues_seen:
                self._queue_evaluation(
                    self._qualified_alert_kind(FallAlertKind.OBSERVED_FALL),
                    FallEvidenceLevel.MEDIUM,
                    t_seconds,
                )
            else:
                self._reject(t_seconds)
        elif self.state is FallState.POST_STABILITY_EVALUATION:
            alert_kind, evidence_level = self._emit_pending_alert()
        return self._decision(
            previous_state,
            None,
            t_seconds,
            alert_kind=alert_kind,
            evidence_level=evidence_level,
        )

    def reset(self) -> None:
        self.state = FallState.UPRIGHT
        self._dynamic_cue_times.clear()
        self._candidate_since = None
        self._all_dynamic_cues_seen = False
        self._cooldown_until = None
        self._posture.clear()
        self._candidate_alert_kind = None
        self._pending_alert_kind = None
        self._pending_evidence_level = None
        self._active_alert_kind = None
        self._active_evidence_level = None
        self._evidence_required_s = self.config.persistent_prone_dwell_s
        self._qualified_metrics = None
        self._qualified_persistent_alert_kind = None
        self._alert_metrics = None
        self._alert_evidence_required_s = None
        self._clear_recovery()

    def _latch_dynamic_cues(self, t_seconds: float, evidence: FallEvidence) -> None:
        for cue_name in tuple(self._dynamic_cue_times):
            cue_time = self._dynamic_cue_times[cue_name]
            if t_seconds - cue_time > self.config.dynamic_cue_window_s:
                del self._dynamic_cue_times[cue_name]
        for cue_name in _DYNAMIC_CUES:
            if getattr(evidence, cue_name):
                self._dynamic_cue_times[cue_name] = t_seconds

    def _candidate_timed_out(self, t_seconds: float) -> bool:
        return self._candidate_since is not None and (
            t_seconds - self._candidate_since >= self.config.candidate_timeout_s
        )

    def _reject(self, t_seconds: float) -> None:
        self.state = FallState.UPRIGHT
        self._dynamic_cue_times.clear()
        self._candidate_since = None
        self._all_dynamic_cues_seen = False
        self._cooldown_until = t_seconds + self.config.rejection_cooldown_s
        self._posture.clear()
        self._candidate_alert_kind = None
        self._pending_alert_kind = None
        self._pending_evidence_level = None
        self._evidence_required_s = self.config.persistent_prone_dwell_s
        self._qualified_metrics = None
        self._qualified_persistent_alert_kind = None
        self._alert_metrics = None
        self._alert_evidence_required_s = None

    def _advance_persistent_posture(
        self,
        features: FallFeatures,
        evidence: FallEvidence,
    ) -> None:
        t_seconds = features.t_seconds
        if self._posture.started_at is None and not evidence.posture:
            return
        self._posture.observe(
            t_seconds,
            evidence,
            features.furniture_roi,
            self._configured_roi_names,
        )
        if self._posture_qualifies(t_seconds, self.config.persistent_prone_dwell_s):
            self.state = FallState.SLUMPING
            self._candidate_alert_kind = FallAlertKind.PERSISTENT_PRONE
            self._candidate_since = None
            self._dynamic_cue_times.clear()
            self._all_dynamic_cues_seen = False
            self._evidence_required_s = self.config.persistent_prone_dwell_s
            self._qualified_metrics = self._posture.metrics(t_seconds)
            self._qualified_persistent_alert_kind = self._qualified_alert_kind(
                FallAlertKind.PERSISTENT_PRONE
            )
            return

        _, _, elapsed_s = self._posture.metrics(t_seconds)
        if elapsed_s + _EPSILON >= self.config.persistent_prone_dwell_s:
            self._posture.clear()
            if evidence.posture:
                self._posture.observe(
                    t_seconds,
                    evidence,
                    features.furniture_roi,
                    self._configured_roi_names,
                )

    def _posture_qualifies(self, t_seconds: float, required_s: float) -> bool:
        evidence_fraction, coverage_fraction, elapsed_s = self._posture.metrics(t_seconds)
        return (
            elapsed_s + _EPSILON >= required_s
            and coverage_fraction + _EPSILON >= self.config.min_temporal_coverage
            and evidence_fraction + _EPSILON >= self.config.posture_evidence_fraction
        )

    def _postural_window_is_covered(self, t_seconds: float, required_s: float) -> bool:
        _, coverage_fraction, elapsed_s = self._posture.metrics(t_seconds)
        return (
            elapsed_s + _EPSILON >= required_s
            and coverage_fraction + _EPSILON >= self.config.min_temporal_coverage
        )

    def _queue_evaluation(
        self,
        alert_kind: FallAlertKind,
        evidence_level: FallEvidenceLevel,
        t_seconds: float,
        *,
        metrics: tuple[float, float, float] | None = None,
    ) -> None:
        self.state = FallState.POST_STABILITY_EVALUATION
        self._pending_alert_kind = alert_kind
        self._pending_evidence_level = evidence_level
        self._alert_metrics = metrics or self._posture.metrics(t_seconds)
        self._alert_evidence_required_s = self._evidence_required_s

    def _qualified_alert_kind(self, base_kind: FallAlertKind) -> FallAlertKind:
        if not self._configured_roi_names or self._posture.posture_s <= 0.0:
            return base_kind
        furniture_fraction = self._posture.furniture_s / self._posture.posture_s
        if furniture_fraction + _EPSILON >= self.config.furniture_occupancy_fraction:
            return FallAlertKind.BED_REST
        return base_kind

    def _emit_pending_alert(
        self,
    ) -> tuple[FallAlertKind | None, FallEvidenceLevel | None]:
        alert_kind = self._pending_alert_kind
        evidence_level = self._pending_evidence_level
        if alert_kind is None or evidence_level is None:
            return None, None
        self.state = (
            FallState.BED_REST
            if alert_kind is FallAlertKind.BED_REST
            else FallState.FALL_CONFIRMED
        )
        self._active_alert_kind = alert_kind
        self._active_evidence_level = evidence_level
        self._candidate_alert_kind = None
        self._qualified_metrics = None
        self._qualified_persistent_alert_kind = None
        self._pending_alert_kind = None
        self._pending_evidence_level = None
        return alert_kind, evidence_level

    def _advance_recovery(
        self,
        features: FallFeatures,
        evidence: FallEvidence,
    ) -> bool:
        upright = (
            evidence.quality_ok
            and features.torso_angle_deg <= self.config.recovery_torso_angle_deg
        )
        if not upright:
            self._clear_recovery()
            return False

        t_seconds = features.t_seconds
        if self._recovery_last_at is None:
            self._recovery_last_at = t_seconds
            return False
        interval_s = t_seconds - self._recovery_last_at
        if 0.0 < interval_s <= self.config.max_observation_gap_s + _EPSILON:
            self._recovery_elapsed_s += interval_s
        else:
            self._recovery_elapsed_s = 0.0
        self._recovery_last_at = t_seconds
        if self._recovery_elapsed_s + _EPSILON < self.config.recovery_dwell_s:
            return False

        self.state = FallState.UPRIGHT
        self._dynamic_cue_times.clear()
        self._candidate_since = None
        self._all_dynamic_cues_seen = False
        self._cooldown_until = None
        self._candidate_alert_kind = None
        self._pending_alert_kind = None
        self._pending_evidence_level = None
        self._posture.clear()
        self._evidence_required_s = self.config.persistent_prone_dwell_s
        self._qualified_persistent_alert_kind = None
        self._clear_recovery()
        return True

    def _clear_recovery(self) -> None:
        self._recovery_last_at = None
        self._recovery_elapsed_s = 0.0

    def _decision(
        self,
        previous_state: FallState,
        evidence: FallEvidence | None,
        t_seconds: float,
        *,
        alert_kind: FallAlertKind | None = None,
        evidence_level: FallEvidenceLevel | None = None,
        recovered: bool = False,
    ) -> FallDecision:
        use_alert_snapshot = recovered or self.state in (
            FallState.POST_STABILITY_EVALUATION,
            *_TERMINAL_STATES,
        )
        if use_alert_snapshot and self._alert_metrics is not None:
            evidence_fraction, coverage_fraction, elapsed_s = self._alert_metrics
            evidence_required_s = (
                self._alert_evidence_required_s
                if self._alert_evidence_required_s is not None
                else self._evidence_required_s
            )
        else:
            evidence_fraction, coverage_fraction, elapsed_s = self._posture.metrics(
                t_seconds
            )
            evidence_required_s = self._evidence_required_s
        return FallDecision(
            state=self.state,
            previous_state=previous_state,
            state_changed=self.state is not previous_state,
            evidence=evidence,
            evidence_fraction=evidence_fraction,
            coverage_fraction=coverage_fraction,
            evidence_elapsed_s=elapsed_s,
            evidence_required_s=evidence_required_s,
            alert_kind=alert_kind,
            evidence_level=evidence_level,
            recovered=recovered,
        )
