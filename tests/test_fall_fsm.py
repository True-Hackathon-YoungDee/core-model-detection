import numpy as np
import pytest

from fall_detection.fall_fsm import (
    DiscriminatorFlags,
    FallFeatures,
    FallState,
    FallThresholds,
    PersonFallFSM,
    frame_vote,
)


def _features(
    t_seconds: float = 0.0,
    downward_speed: float = 0.0,
    accel: float = 0.0,
    torso_angle: float = 0.0,
    aspect_ratio: float = 1.0,
    com_height: float = 1.0,
) -> FallFeatures:
    return FallFeatures(
        t_seconds=t_seconds,
        com=np.zeros(3),
        com_velocity=np.array([0.0, downward_speed, 0.0]),
        com_acceleration=np.array([0.0, accel, 0.0]),
        com_height=com_height,
        torso_angle_deg=torso_angle,
        bbox_aspect_ratio=aspect_ratio,
        instability_index=None,
    )


def test_frame_vote_true_only_when_all_three_conditions_met():
    thresholds = FallThresholds()
    assert frame_vote(_features(torso_angle=70.0, aspect_ratio=1.5, com_height=0.1), thresholds) is True
    assert frame_vote(_features(torso_angle=10.0, aspect_ratio=1.5, com_height=0.1), thresholds) is False
    assert frame_vote(_features(torso_angle=70.0, aspect_ratio=0.5, com_height=0.1), thresholds) is False
    assert frame_vote(_features(torso_angle=70.0, aspect_ratio=1.5, com_height=1.5), thresholds) is False


def test_steady_upright_stays_upright():
    fsm = PersonFallFSM()
    state = FallState.UPRIGHT
    for i in range(10):
        state = fsm.step(_features(t_seconds=i * 0.033, downward_speed=0.0, torso_angle=5.0, aspect_ratio=0.5, com_height=1.5))
    assert state == FallState.UPRIGHT


def _drive_to_impact(fsm: PersonFallFSM, t: float, dt: float, discriminators=None, accel: float = 25.0) -> float:
    fsm.step(_features(t_seconds=t, downward_speed=0.0, torso_angle=5.0, aspect_ratio=0.5, com_height=1.5))
    t += dt
    state = fsm.step(_features(t_seconds=t, downward_speed=2.0, torso_angle=30.0, aspect_ratio=0.8, com_height=1.0))
    assert state == FallState.DESCENDING
    t += dt
    state = fsm.step(
        _features(t_seconds=t, downward_speed=2.0, accel=accel, torso_angle=70.0, aspect_ratio=1.5, com_height=0.1),
        discriminators,
    )
    assert state == FallState.IMPACT
    return t


def _drive_through_slumping_to_evaluation(fsm: PersonFallFSM, t: float, dt: float) -> float:
    state = None
    for _ in range(35):
        t += dt
        state = fsm.step(_features(t_seconds=t, torso_angle=80.0, aspect_ratio=1.5, com_height=0.1))
        if state == FallState.POST_STABILITY_EVALUATION:
            break
    assert state == FallState.POST_STABILITY_EVALUATION
    return t


def test_clean_fast_fall_confirms():
    fsm = PersonFallFSM()
    dt = 1.0 / 30.0
    t = _drive_to_impact(fsm, 0.0, dt)
    t = _drive_through_slumping_to_evaluation(fsm, t, dt)
    t += dt
    state = fsm.step(_features(t_seconds=t, torso_angle=80.0, aspect_ratio=1.5, com_height=0.1))
    assert state == FallState.FALL_CONFIRMED


def test_fall_then_quick_recovery_reverts_to_upright():
    fsm = PersonFallFSM()
    t = 0.0
    fsm.step(_features(t_seconds=t, downward_speed=0.0, torso_angle=5.0))
    t += 0.1
    state = fsm.step(_features(t_seconds=t, downward_speed=1.0, torso_angle=20.0))
    assert state == FallState.DESCENDING

    state = FallState.DESCENDING
    for _ in range(6):
        t += 0.5
        state = fsm.step(_features(t_seconds=t, downward_speed=0.0, torso_angle=5.0))
    assert state == FallState.UPRIGHT


def test_slow_slump_triggers_via_displacement_or_path():
    fsm = PersonFallFSM()
    dt = 1.0 / 30.0
    fsm.step(_features(t_seconds=0.0, downward_speed=0.0, torso_angle=5.0))
    t = 0.1
    state = fsm.step(_features(t_seconds=t, downward_speed=0.9, torso_angle=30.0))
    assert state == FallState.DESCENDING

    t += dt
    discriminators = DiscriminatorFlags(vertical_displacement_2s=0.5)
    state = fsm.step(
        _features(t_seconds=t, downward_speed=0.9, accel=1.0, torso_angle=60.0), discriminators
    )
    assert state == FallState.IMPACT


def test_soft_landing_triggers_via_dissipation_or_path():
    fsm = PersonFallFSM()
    dt = 1.0 / 30.0
    fsm.step(_features(t_seconds=0.0, downward_speed=0.0, torso_angle=5.0))
    t = 0.1
    state = fsm.step(_features(t_seconds=t, downward_speed=1.5, torso_angle=30.0))
    assert state == FallState.DESCENDING

    t += dt
    discriminators = DiscriminatorFlags(energy_dissipation_w=200.0)
    state = fsm.step(
        _features(t_seconds=t, downward_speed=1.5, accel=1.0, torso_angle=60.0), discriminators
    )
    assert state == FallState.IMPACT


def test_bed_rest_descent_confirms_as_bed_rest_not_fall():
    fsm = PersonFallFSM()
    dt = 1.0 / 30.0
    t = _drive_to_impact(fsm, 0.0, dt)
    t = _drive_through_slumping_to_evaluation(fsm, t, dt)
    t += dt
    state = fsm.step(
        _features(t_seconds=t, torso_angle=80.0, aspect_ratio=1.5, com_height=0.1),
        DiscriminatorFlags(bed_rest=True),
    )
    assert state == FallState.BED_REST


def test_ground_bound_forces_confirmation_even_below_vote_majority():
    fsm = PersonFallFSM()
    dt = 1.0 / 30.0
    t = _drive_to_impact(fsm, 0.0, dt)
    # Every SLUMPING frame fails the torso-angle vote condition (stays low),
    # so the plain majority-vote path would revert to UPRIGHT...
    state = None
    for _ in range(35):
        t += dt
        state = fsm.step(_features(t_seconds=t, torso_angle=10.0, aspect_ratio=0.5, com_height=0.1))
        if state == FallState.POST_STABILITY_EVALUATION:
            break
    assert state == FallState.POST_STABILITY_EVALUATION
    t += dt
    # ...but a ground_bound discriminator overrides that at evaluation time.
    state = fsm.step(
        _features(t_seconds=t, torso_angle=10.0, aspect_ratio=0.5, com_height=0.1),
        DiscriminatorFlags(ground_bound=True),
    )
    assert state == FallState.FALL_CONFIRMED


def test_confirmed_state_is_sticky_regardless_of_later_features():
    fsm = PersonFallFSM()
    dt = 1.0 / 30.0
    t = _drive_to_impact(fsm, 0.0, dt)
    t = _drive_through_slumping_to_evaluation(fsm, t, dt)
    t += dt
    state = fsm.step(_features(t_seconds=t, torso_angle=80.0, aspect_ratio=1.5, com_height=0.1))
    assert state == FallState.FALL_CONFIRMED

    t += dt
    state = fsm.step(_features(t_seconds=t, downward_speed=0.0, torso_angle=5.0, aspect_ratio=0.5, com_height=1.5))
    assert state == FallState.FALL_CONFIRMED


def test_reset_clears_state_and_buffer():
    fsm = PersonFallFSM()
    dt = 1.0 / 30.0
    _drive_to_impact(fsm, 0.0, dt)
    fsm.reset()
    assert fsm.state == FallState.UPRIGHT
    state = fsm.step(_features(t_seconds=100.0, downward_speed=0.0, torso_angle=5.0, aspect_ratio=0.5, com_height=1.5))
    assert state == FallState.UPRIGHT


def test_already_prone_subject_leaves_upright_without_velocity_trigger():
    """A subject who is prone from frame 1 (e.g. video opens on someone
    already down) never has a velocity spike, so the FSM must still leave
    UPRIGHT based on sustained static geometry (torso angle + aspect ratio)."""
    fsm = PersonFallFSM()
    dt = 1.0 / 30.0
    thresholds = FallThresholds()
    state = FallState.UPRIGHT
    t = 0.0
    for i in range(thresholds.static_prone_frames):
        t = i * dt
        state = fsm.step(
            _features(t_seconds=t, downward_speed=0.0, torso_angle=80.0, aspect_ratio=2.5, com_height=0.1)
        )
    assert state == FallState.SLUMPING

    for _ in range(35):
        t += dt
        state = fsm.step(_features(t_seconds=t, torso_angle=80.0, aspect_ratio=2.5, com_height=0.1))
        if state == FallState.POST_STABILITY_EVALUATION:
            break
    assert state == FallState.POST_STABILITY_EVALUATION

    t += dt
    state = fsm.step(_features(t_seconds=t, torso_angle=80.0, aspect_ratio=2.5, com_height=0.1))
    assert state == FallState.FALL_CONFIRMED


def test_brief_prone_glimpse_does_not_leave_upright():
    """A momentary prone-shaped read (occlusion glitch, bending over) that
    doesn't sustain for static_prone_frames should not trip the static path."""
    fsm = PersonFallFSM()
    dt = 1.0 / 30.0
    thresholds = FallThresholds()
    state = FallState.UPRIGHT
    for i in range(thresholds.static_prone_frames - 1):
        state = fsm.step(
            _features(t_seconds=i * dt, downward_speed=0.0, torso_angle=80.0, aspect_ratio=2.5, com_height=0.1)
        )
    assert state == FallState.UPRIGHT


def test_vote_fraction_reflects_buffer_contents_during_slumping():
    fsm = PersonFallFSM()
    dt = 1.0 / 30.0
    assert fsm.vote_fraction == 0.0
    t = _drive_to_impact(fsm, 0.0, dt)
    t += dt
    state = fsm.step(_features(t_seconds=t, torso_angle=80.0, aspect_ratio=1.5, com_height=0.1))
    assert state == FallState.SLUMPING
    fsm.step(_features(t_seconds=1000.0, torso_angle=80.0, aspect_ratio=1.5, com_height=0.1))
    assert fsm.vote_fraction == pytest.approx(1.0)
    fsm.step(_features(t_seconds=1000.1, torso_angle=5.0, aspect_ratio=0.5, com_height=1.5))
    assert fsm.vote_fraction == pytest.approx(0.5)


def test_streak_decays_instead_of_hard_reset_on_single_missed_frame():
    """A single occlusion/jitter frame amid an otherwise-sustained prone
    posture should cost the streak one frame, not erase it entirely -- a hard
    reset would force the full static_prone_frames count to restart from
    zero, which real detections can't tolerate given normal landmark noise."""
    fsm = PersonFallFSM()
    dt = 1.0 / 30.0
    thresholds = FallThresholds()
    t = 0.0
    state = FallState.UPRIGHT
    for i in range(thresholds.static_prone_frames - 1):
        t = i * dt
        state = fsm.step(
            _features(t_seconds=t, downward_speed=0.0, torso_angle=80.0, aspect_ratio=2.5, com_height=0.1)
        )
    assert state == FallState.UPRIGHT
    assert fsm._static_prone_streak == thresholds.static_prone_frames - 1

    t += dt
    state = fsm.step(
        _features(t_seconds=t, downward_speed=0.0, torso_angle=10.0, aspect_ratio=2.5, com_height=0.1)
    )
    assert state == FallState.UPRIGHT
    assert fsm._static_prone_streak == thresholds.static_prone_frames - 2

    for _ in range(2):
        t += dt
        state = fsm.step(
            _features(t_seconds=t, downward_speed=0.0, torso_angle=80.0, aspect_ratio=2.5, com_height=0.1)
        )
    assert state == FallState.SLUMPING
