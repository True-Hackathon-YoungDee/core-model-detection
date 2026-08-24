from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fall_detection.fall_config import FallConfig, FurnitureROI
from fall_detection.fall_evidence import FallFeatures
from fall_detection.fall_fsm import (
    FallAlertKind,
    FallDecision,
    FallEvidenceLevel,
    FallState,
    PersonFallFSM,
)


def _features(
    t_seconds: float,
    *,
    valid: bool = True,
    torso_angle_deg: float = 0.0,
    bbox_aspect_ratio: float = 0.5,
    downward_speed_bh_s: float = 0.0,
    torso_rotation_deg_s: float = 0.0,
    height_collapse_fraction: float = 0.0,
    motion_bh_s: float = 0.0,
    furniture_roi: str | None = None,
    scale_source: str = "upright_height",
) -> FallFeatures:
    return FallFeatures(
        t_seconds=t_seconds,
        valid=valid,
        torso_angle_deg=torso_angle_deg,
        bbox_aspect_ratio=bbox_aspect_ratio,
        hip_downward_speed_bh_s=downward_speed_bh_s,
        bbox_downward_speed_bh_s=0.0,
        torso_rotation_deg_s=torso_rotation_deg_s,
        height_collapse_fraction=height_collapse_fraction,
        motion_bh_s=motion_bh_s,
        visibility_quality=1.0 if valid else 0.0,
        torso_centroid=(0.5, 0.5),
        furniture_roi=furniture_roi,
        scale_source=scale_source,
    )


def test_fall_state_preserves_public_integer_mapping_and_peak_semantics():
    assert {state.name: int(state) for state in FallState} == {
        "UPRIGHT": 0,
        "DESCENDING": 1,
        "IMPACT": 2,
        "SLUMPING": 3,
        "POST_STABILITY_EVALUATION": 4,
        "FALL_CONFIRMED": 5,
        "BED_REST": 6,
    }
    documentation = (FallState.__doc__ or "").lower()
    assert "peak" in documentation
    assert "not" in documentation and "force" in documentation


def test_public_alert_enums_and_immutable_decision_contract():
    assert {kind.name: kind.value for kind in FallAlertKind} == {
        "OBSERVED_FALL": "OBSERVED_FALL",
        "PERSISTENT_PRONE": "PERSISTENT_PRONE",
        "BED_REST": "BED_REST",
    }
    assert {level.name: level.value for level in FallEvidenceLevel} == {
        "HIGH": "HIGH",
        "MEDIUM": "MEDIUM",
    }

    decision = PersonFallFSM(FallConfig()).step(_features(0.0))

    assert isinstance(decision, FallDecision)
    assert {
        "state",
        "previous_state",
        "state_changed",
        "evidence",
        "evidence_fraction",
        "coverage_fraction",
        "evidence_elapsed_s",
        "evidence_required_s",
        "alert_kind",
        "evidence_level",
        "recovered",
    } <= decision.__dataclass_fields__.keys()
    with pytest.raises(FrozenInstanceError):
        decision.state = FallState.IMPACT  # type: ignore[misc]


def test_one_dynamic_cue_does_not_leave_upright():
    fsm = PersonFallFSM(FallConfig())

    decision = fsm.step(_features(0.0, downward_speed_bh_s=0.6))

    assert decision.state is FallState.UPRIGHT
    assert decision.state_changed is False


def test_two_distinct_dynamic_cues_inside_window_enter_descending_from_zero_timestamp():
    fsm = PersonFallFSM(FallConfig())
    fsm.step(_features(0.0, downward_speed_bh_s=0.6))

    decision = fsm.step(_features(0.75, torso_rotation_deg_s=61.0))

    assert decision.previous_state is FallState.UPRIGHT
    assert decision.state is FallState.DESCENDING
    assert decision.state_changed is True


def test_stale_dynamic_cues_do_not_combine():
    fsm = PersonFallFSM(FallConfig())
    fsm.step(_features(0.0, downward_speed_bh_s=0.6))

    decision = fsm.step(_features(0.751, torso_rotation_deg_s=61.0))

    assert decision.state is FallState.UPRIGHT


def test_dynamic_torso_angle_advances_descending_to_evidence_peak():
    fsm = PersonFallFSM(FallConfig())
    fsm.step(
        _features(
            0.0,
            downward_speed_bh_s=0.6,
            torso_rotation_deg_s=61.0,
        )
    )

    decision = fsm.step(_features(0.1, torso_angle_deg=45.0))

    assert decision.previous_state is FallState.DESCENDING
    assert decision.state is FallState.IMPACT


def test_candidate_timeout_rejects_and_enforces_cooldown():
    fsm = PersonFallFSM(FallConfig())
    fsm.step(
        _features(
            0.0,
            downward_speed_bh_s=0.6,
            torso_rotation_deg_s=61.0,
        )
    )

    rejected = fsm.step(_features(2.0, downward_speed_bh_s=0.6))
    during_cooldown = fsm.step(
        _features(
            2.49,
            downward_speed_bh_s=0.6,
            torso_rotation_deg_s=61.0,
        )
    )

    assert rejected.previous_state is FallState.DESCENDING
    assert rejected.state is FallState.UPRIGHT
    assert during_cooldown.state is FallState.UPRIGHT


def test_observation_gap_advances_timeout_without_fabricating_evidence():
    fsm = PersonFallFSM(FallConfig())
    fsm.step(
        _features(
            0.0,
            downward_speed_bh_s=0.6,
            torso_rotation_deg_s=61.0,
        )
    )

    decision = fsm.observe_gap(2.0)

    assert decision.previous_state is FallState.DESCENDING
    assert decision.state is FallState.UPRIGHT
    assert decision.evidence is None


def test_reset_clears_latched_dynamic_cues():
    fsm = PersonFallFSM(FallConfig())
    fsm.step(_features(0.0, downward_speed_bh_s=0.6))

    fsm.reset()
    decision = fsm.step(_features(0.1, torso_rotation_deg_s=61.0))

    assert decision.state is FallState.UPRIGHT


def _start_observed_fall(
    fsm: PersonFallFSM,
    *,
    all_three_dynamic_cues: bool = False,
    slumping_posture: bool = True,
    furniture_roi: str | None = None,
    base_t: float = 0.0,
) -> float:
    dynamic = {
        "downward_speed_bh_s": 0.6,
        "torso_rotation_deg_s": 61.0,
    }
    if all_three_dynamic_cues:
        dynamic["height_collapse_fraction"] = 0.16
    descending = fsm.step(_features(base_t, **dynamic))
    assert descending.state is FallState.DESCENDING

    impact = fsm.step(_features(base_t + 0.05, torso_angle_deg=45.0))
    assert impact.state is FallState.IMPACT

    t_seconds = base_t + 0.1
    slumping = fsm.step(
        _features(
            t_seconds,
            torso_angle_deg=60.0 if slumping_posture else 0.0,
            bbox_aspect_ratio=1.2 if slumping_posture else 0.5,
            furniture_roi=furniture_roi,
        )
    )
    assert slumping.state is FallState.SLUMPING
    return t_seconds


def test_impact_waits_for_a_valid_observation_before_entering_slumping():
    fsm = PersonFallFSM(FallConfig())
    fsm.step(
        _features(
            0.0,
            downward_speed_bh_s=0.6,
            torso_rotation_deg_s=61.0,
        )
    )
    fsm.step(_features(0.05, torso_angle_deg=45.0))

    invalid = fsm.step(_features(0.1, valid=False))
    valid = fsm.step(_prone(0.2))

    assert invalid.state is FallState.IMPACT
    assert valid.previous_state is FallState.IMPACT
    assert valid.state is FallState.SLUMPING


@pytest.mark.parametrize("observations_per_second", [5, 15, 30])
def test_observed_fall_confirmation_is_identical_across_observation_rates(
    observations_per_second: int,
):
    fsm = PersonFallFSM(FallConfig())
    started_at = _start_observed_fall(fsm)
    final_window_decision = None
    for index in range(1, observations_per_second + 1):
        final_window_decision = fsm.step(
            _features(
                started_at + index / observations_per_second,
                torso_angle_deg=60.0,
                bbox_aspect_ratio=1.2,
            )
        )

    assert final_window_decision is not None
    assert final_window_decision.state is FallState.POST_STABILITY_EVALUATION
    assert final_window_decision.evidence_elapsed_s == pytest.approx(1.0)
    assert final_window_decision.coverage_fraction == pytest.approx(1.0)
    assert final_window_decision.evidence_fraction == pytest.approx(1.0)

    alert = fsm.step(
        _features(
            started_at + 1.0 + 1.0 / observations_per_second,
            torso_angle_deg=60.0,
            bbox_aspect_ratio=1.2,
        )
    )
    duplicate = fsm.step(
        _features(
            started_at + 1.0 + 2.0 / observations_per_second,
            torso_angle_deg=60.0,
            bbox_aspect_ratio=1.2,
        )
    )

    assert alert.state is FallState.FALL_CONFIRMED
    assert alert.alert_kind is FallAlertKind.OBSERVED_FALL
    assert alert.evidence_level is FallEvidenceLevel.HIGH
    assert alert.evidence_elapsed_s == pytest.approx(
        final_window_decision.evidence_elapsed_s
    )
    assert alert.coverage_fraction == pytest.approx(
        final_window_decision.coverage_fraction
    )
    assert duplicate.evidence_elapsed_s == pytest.approx(alert.evidence_elapsed_s)
    assert duplicate.coverage_fraction == pytest.approx(alert.coverage_fraction)
    assert duplicate.state is FallState.FALL_CONFIRMED
    assert duplicate.alert_kind is None
    assert duplicate.evidence_level is None


def test_exact_minimum_temporal_coverage_qualifies():
    fsm = PersonFallFSM(FallConfig())
    started_at = _start_observed_fall(fsm)

    fsm.step(
        _features(
            started_at + 0.4,
            torso_angle_deg=60.0,
            bbox_aspect_ratio=1.2,
        )
    )
    fsm.step(_features(started_at + 0.8, valid=False))
    decision = fsm.step(_features(started_at + 1.0, valid=False))

    assert decision.state is FallState.POST_STABILITY_EVALUATION
    assert decision.evidence_elapsed_s == pytest.approx(1.0)
    assert decision.coverage_fraction == pytest.approx(0.8)
    assert decision.evidence_fraction == pytest.approx(1.0)


def test_observation_interval_over_maximum_gap_contributes_no_coverage():
    fsm = PersonFallFSM(FallConfig())
    started_at = _start_observed_fall(fsm)

    fsm.step(
        _features(
            started_at + 0.6,
            torso_angle_deg=60.0,
            bbox_aspect_ratio=1.2,
        )
    )
    decision = fsm.step(
        _features(
            started_at + 1.0,
            torso_angle_deg=60.0,
            bbox_aspect_ratio=1.2,
        )
    )

    assert decision.state is FallState.SLUMPING
    assert decision.evidence_elapsed_s == pytest.approx(1.0)
    assert decision.coverage_fraction == pytest.approx(0.4)
    assert decision.evidence_fraction == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("posture_duration_s", "expected_state"),
    [
        (0.600, FallState.POST_STABILITY_EVALUATION),
        (0.599, FallState.SLUMPING),
    ],
)
def test_profile_posture_fraction_boundary_is_duration_weighted(
    posture_duration_s: float,
    expected_state: FallState,
):
    fsm = PersonFallFSM(FallConfig(posture_evidence_fraction=0.60))
    started_at = _start_observed_fall(fsm)

    fsm.step(
        _features(
            started_at + 0.3,
            torso_angle_deg=60.0,
            bbox_aspect_ratio=1.2,
        )
    )
    fsm.step(
        _features(
            started_at + posture_duration_s,
            torso_angle_deg=0.0,
            bbox_aspect_ratio=0.5,
        )
    )
    decision = fsm.step(
        _features(
            started_at + 1.0,
            torso_angle_deg=0.0,
            bbox_aspect_ratio=0.5,
        )
    )

    assert decision.state is expected_state
    assert decision.coverage_fraction == pytest.approx(1.0)
    assert decision.evidence_fraction == pytest.approx(posture_duration_s)


def test_all_three_dynamic_cues_allow_medium_confirmation_after_observations_are_lost():
    fsm = PersonFallFSM(FallConfig())
    _start_observed_fall(fsm, all_three_dynamic_cues=True)

    evaluation = fsm.observe_gap(2.0)
    alert = fsm.observe_gap(2.1)

    assert evaluation.previous_state is FallState.SLUMPING
    assert evaluation.state is FallState.POST_STABILITY_EVALUATION
    assert evaluation.evidence is None
    assert alert.state is FallState.FALL_CONFIRMED
    assert alert.alert_kind is FallAlertKind.OBSERVED_FALL
    assert alert.evidence_level is FallEvidenceLevel.MEDIUM


def test_insufficient_postural_evidence_rejects_at_candidate_timeout():
    fsm = PersonFallFSM(FallConfig())
    _start_observed_fall(fsm, slumping_posture=False)

    decision = fsm.step(_features(2.0))

    assert decision.previous_state is FallState.SLUMPING
    assert decision.state is FallState.UPRIGHT
    assert decision.alert_kind is None


def _prone(t_seconds: float, *, furniture_roi: str | None = None) -> FallFeatures:
    return _features(
        t_seconds,
        torso_angle_deg=60.0,
        bbox_aspect_ratio=1.2,
        furniture_roi=furniture_roi,
    )


def _confirm_observed_fall(
    fsm: PersonFallFSM,
    *,
    base_t: float = 0.0,
    furniture_roi: str | None = None,
) -> tuple[float, FallDecision]:
    started_at = _start_observed_fall(
        fsm,
        furniture_roi=furniture_roi,
        base_t=base_t,
    )
    for index in range(1, 6):
        evaluation = fsm.step(
            _prone(
                started_at + index * 0.2,
                furniture_roi=furniture_roi,
            )
        )
    assert evaluation.state is FallState.POST_STABILITY_EVALUATION
    alert_t = started_at + 1.2
    return alert_t, fsm.step(_prone(alert_t, furniture_roi=furniture_roi))


def test_prone_from_first_observation_emits_persistent_prone_after_dwell():
    fsm = PersonFallFSM(FallConfig())

    for t_seconds in (0.0, 0.5, 1.0, 1.5, 2.0):
        dwell_decision = fsm.step(_prone(t_seconds))
    evaluation = fsm.step(_prone(2.1))
    alert = fsm.step(_prone(2.2))

    assert dwell_decision.previous_state is FallState.UPRIGHT
    assert dwell_decision.state is FallState.SLUMPING
    assert dwell_decision.evidence_elapsed_s == pytest.approx(2.0)
    assert dwell_decision.evidence_required_s == pytest.approx(2.0)
    assert evaluation.state is FallState.POST_STABILITY_EVALUATION
    assert alert.state is FallState.FALL_CONFIRMED
    assert alert.alert_kind is FallAlertKind.PERSISTENT_PRONE
    assert alert.evidence_level is FallEvidenceLevel.HIGH


def test_brief_prone_pose_does_not_trigger_persistent_prone():
    fsm = PersonFallFSM(FallConfig())

    fsm.step(_prone(0.0))
    fsm.step(_prone(0.5))
    fsm.step(_prone(0.9))
    for t_seconds in (1.0, 1.5, 2.0, 2.5):
        decision = fsm.step(_features(t_seconds))

    assert decision.state is FallState.UPRIGHT
    assert decision.alert_kind is None


def test_qualifying_furniture_fraction_changes_alert_kind_and_terminal_state():
    bed = FurnitureROI(
        "bed",
        ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)),
    )
    fsm = PersonFallFSM(FallConfig(furniture_rois=(bed,)))

    _, alert = _confirm_observed_fall(fsm, furniture_roi="bed")

    assert alert.state is FallState.BED_REST
    assert alert.alert_kind is FallAlertKind.BED_REST
    assert alert.evidence_level is FallEvidenceLevel.HIGH


@pytest.mark.parametrize(
    ("config", "feature_roi"),
    [
        (FallConfig(), "bed"),
        (
            FallConfig(
                furniture_rois=(
                    FurnitureROI(
                        "bed",
                        ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)),
                    ),
                )
            ),
            None,
        ),
    ],
)
def test_absent_configured_roi_or_feature_cannot_claim_bed_rest(
    config: FallConfig,
    feature_roi: str | None,
):
    fsm = PersonFallFSM(config)

    _, alert = _confirm_observed_fall(fsm, furniture_roi=feature_roi)

    assert alert.state is FallState.FALL_CONFIRMED
    assert alert.alert_kind is FallAlertKind.OBSERVED_FALL


def test_failed_evaluation_clears_static_evidence_and_stays_upright_through_cooldown():
    fsm = PersonFallFSM(FallConfig())
    _start_observed_fall(fsm, slumping_posture=False)
    rejected = fsm.step(_features(2.0))

    for t_seconds in (2.1, 2.2, 2.3, 2.4, 2.49):
        cooldown_decision = fsm.step(_prone(t_seconds))
    cooldown_finished = fsm.step(_prone(2.5))

    assert rejected.state is FallState.UPRIGHT
    assert cooldown_decision.state is FallState.UPRIGHT
    assert cooldown_finished.state is FallState.UPRIGHT
    assert cooldown_finished.evidence_elapsed_s == pytest.approx(0.0)


def test_terminal_alert_recovers_once_after_two_upright_seconds():
    fsm = PersonFallFSM(FallConfig())
    alert_t, alert = _confirm_observed_fall(fsm)
    assert alert.state is FallState.FALL_CONFIRMED

    fsm.step(_features(alert_t + 0.1))
    for offset in (0.6, 1.1, 1.6):
        before_recovery = fsm.step(_features(alert_t + offset))
    recovered = fsm.step(_features(alert_t + 2.1))
    after_recovery = fsm.step(_features(alert_t + 2.2))

    assert before_recovery.state is FallState.FALL_CONFIRMED
    assert before_recovery.recovered is False
    assert recovered.previous_state is FallState.FALL_CONFIRMED
    assert recovered.state is FallState.UPRIGHT
    assert recovered.recovered is True
    assert recovered.alert_kind is FallAlertKind.OBSERVED_FALL
    assert recovered.evidence_level is FallEvidenceLevel.HIGH
    assert recovered.evidence_fraction == pytest.approx(alert.evidence_fraction)
    assert recovered.coverage_fraction == pytest.approx(alert.coverage_fraction)
    assert recovered.evidence_elapsed_s == pytest.approx(alert.evidence_elapsed_s)
    assert recovered.evidence_required_s == pytest.approx(alert.evidence_required_s)
    assert after_recovery.state is FallState.UPRIGHT
    assert after_recovery.recovered is False


def test_later_fall_can_emit_a_new_alert_after_recovery():
    fsm = PersonFallFSM(FallConfig())
    first_alert_t, first_alert = _confirm_observed_fall(fsm)
    fsm.step(_features(first_alert_t + 0.1))
    for offset in (0.6, 1.1, 1.6, 2.1):
        recovery = fsm.step(_features(first_alert_t + offset))
    assert recovery.recovered is True

    _, second_alert = _confirm_observed_fall(fsm, base_t=4.0)

    assert first_alert.alert_kind is FallAlertKind.OBSERVED_FALL
    assert second_alert.state is FallState.FALL_CONFIRMED
    assert second_alert.alert_kind is FallAlertKind.OBSERVED_FALL
    assert second_alert.evidence_level is FallEvidenceLevel.HIGH
