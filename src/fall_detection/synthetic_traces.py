"""Deterministic, hand-authored ``FallFeatures`` streams for regression
scenarios that have no matching real video: ADL "hard negative" motions
(fast sit, slow lie-down, bend, squat, kneel, jump, brisk walk, deliberate
floor-sit) and a few additional fall geometries, plus two degenerate inputs.

**These are authored numbers, not recordings of a real subject.** Numeric
envelopes (torso angle, aspect ratio, downward speed, rotation, height
collapse, motion) are grounded in the summary statistics of the real,
committed ``evaluation/traces/local-regression-v2.jsonl`` trace so they stay
inside the range MediaPipe actually produces, then shaped per scenario to
represent a specific, physically plausible motion. Any recall/precision/F1
computed by replaying this corpus describes how ``PersonFallFSM`` responds
to these authored streams -- it is not a measurement of real-world system
accuracy. Do not quote it as one; real-world accuracy claims stay attached
to ``evaluation/manifests/local-falls.toml`` and other real-video datasets.

Each scenario is built from a small list of *keyframes*: ``(duration_s,
overrides)`` pairs. Numeric feature fields are linearly interpolated from
the previous keyframe's end state to the new target over ``duration_s``,
sampled at :data:`FPS`. This keeps every scenario a short, readable table
instead of a per-frame hand-written array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .fall_evidence import FallFeatures

FPS = 20.0
DT = 1.0 / FPS

# Numeric fields that get linearly interpolated between keyframes. The
# remaining FallFeatures fields (valid, torso_centroid, furniture_roi,
# scale_source, motion_available) are held constant within one scenario
# unless a keyframe overrides them.
_NUMERIC_FIELDS = (
    "torso_angle_deg",
    "bbox_aspect_ratio",
    "hip_downward_speed_bh_s",
    "bbox_downward_speed_bh_s",
    "torso_rotation_deg_s",
    "height_collapse_fraction",
    "motion_bh_s",
    "visibility_quality",
)

_BASELINE_STANDING: Mapping[str, object] = {
    "torso_angle_deg": 8.0,
    "bbox_aspect_ratio": 0.65,
    "hip_downward_speed_bh_s": 0.0,
    "bbox_downward_speed_bh_s": 0.0,
    "torso_rotation_deg_s": 0.0,
    "height_collapse_fraction": 0.0,
    "motion_bh_s": 0.05,
    "visibility_quality": 0.98,
    "valid": True,
    "torso_centroid": (0.5, 0.35),
    "furniture_roi": None,
    "scale_source": "upright_height",
    "motion_available": True,
}


@dataclass(frozen=True)
class LabelledEventSpec:
    """Author-declared ground truth for one scenario -- independent of what
    the FSM actually decides when the clip is replayed."""

    kind: str
    onset_s: float
    match_end_s: float
    recovered: bool


@dataclass(frozen=True)
class Scenario:
    clip_id: str
    subject: str
    trial: str
    frames: tuple[FallFeatures, ...]
    events: tuple[LabelledEventSpec, ...]

    @property
    def duration_s(self) -> float:
        return self.frames[-1].t_seconds if self.frames else 0.0


Keyframe = tuple[float, Mapping[str, object]]


def _build_stream(
    keyframes: Sequence[Keyframe],
    *,
    base: Mapping[str, object] = _BASELINE_STANDING,
) -> list[FallFeatures]:
    """Interpolate a sequence of keyframes into a per-frame feature stream."""
    state: dict[str, object] = dict(base)
    frames: list[FallFeatures] = []
    t = 0.0
    for duration_s, overrides in keyframes:
        target = dict(state)
        target.update(overrides)
        step_count = max(1, round(duration_s / DT))
        start_numeric = {name: float(state[name]) for name in _NUMERIC_FIELDS}
        end_numeric = {name: float(target[name]) for name in _NUMERIC_FIELDS}
        for step in range(1, step_count + 1):
            fraction = step / step_count
            values = {
                name: start_numeric[name]
                + (end_numeric[name] - start_numeric[name]) * fraction
                for name in _NUMERIC_FIELDS
            }
            frames.append(
                FallFeatures(
                    t_seconds=round(t + step * DT, 6),
                    valid=bool(target["valid"]),
                    torso_angle_deg=values["torso_angle_deg"],
                    bbox_aspect_ratio=values["bbox_aspect_ratio"],
                    hip_downward_speed_bh_s=values["hip_downward_speed_bh_s"],
                    bbox_downward_speed_bh_s=values["bbox_downward_speed_bh_s"],
                    torso_rotation_deg_s=values["torso_rotation_deg_s"],
                    height_collapse_fraction=values["height_collapse_fraction"],
                    motion_bh_s=values["motion_bh_s"],
                    visibility_quality=values["visibility_quality"],
                    torso_centroid=target["torso_centroid"],  # type: ignore[arg-type]
                    furniture_roi=target["furniture_roi"],  # type: ignore[arg-type]
                    scale_source=target["scale_source"],  # type: ignore[arg-type]
                    motion_available=bool(target["motion_available"]),
                )
            )
        t += step_count * DT
        state = target
    return frames


# ---------------------------------------------------------------------------
# Fall scenarios (labelled OBSERVED_FALL ground truth)
# ---------------------------------------------------------------------------


def _fall_forward_fast() -> Scenario:
    onset_s = 0.5
    frames = _build_stream(
        [
            (onset_s, {}),
            (
                0.3,
                {
                    "torso_angle_deg": 82.0,
                    "bbox_aspect_ratio": 2.6,
                    "hip_downward_speed_bh_s": 1.6,
                    "bbox_downward_speed_bh_s": 1.8,
                    "torso_rotation_deg_s": 320.0,
                    "height_collapse_fraction": 0.55,
                    "motion_bh_s": 1.2,
                },
            ),
            (
                0.2,
                {
                    "hip_downward_speed_bh_s": 0.0,
                    "bbox_downward_speed_bh_s": 0.0,
                    "torso_rotation_deg_s": 0.0,
                    "motion_bh_s": 0.05,
                },
            ),
            (2.0, {}),
        ]
    )
    return Scenario(
        "synth-fall-forward-fast",
        "synthetic-subject-1",
        "fall-forward-fast",
        tuple(frames),
        (LabelledEventSpec("OBSERVED_FALL", onset_s, onset_s + 2.0, False),),
    )


def _fall_backward_fast() -> Scenario:
    onset_s = 0.8
    frames = _build_stream(
        [
            (onset_s, {}),
            (
                0.25,
                {
                    "torso_angle_deg": 78.0,
                    "bbox_aspect_ratio": 2.3,
                    "hip_downward_speed_bh_s": 1.9,
                    "bbox_downward_speed_bh_s": 2.1,
                    "torso_rotation_deg_s": 410.0,
                    "height_collapse_fraction": 0.48,
                    "motion_bh_s": 1.4,
                },
            ),
            (
                0.2,
                {
                    "hip_downward_speed_bh_s": 0.0,
                    "bbox_downward_speed_bh_s": 0.0,
                    "torso_rotation_deg_s": 0.0,
                    "motion_bh_s": 0.05,
                },
            ),
            (2.0, {}),
        ]
    )
    return Scenario(
        "synth-fall-backward-fast",
        "synthetic-subject-2",
        "fall-backward-fast",
        tuple(frames),
        (LabelledEventSpec("OBSERVED_FALL", onset_s, onset_s + 2.0, False),),
    )


def _fall_lateral_recovered() -> Scenario:
    onset_s = 0.4
    frames = _build_stream(
        [
            (onset_s, {}),
            (
                0.3,
                {
                    "torso_angle_deg": 75.0,
                    "bbox_aspect_ratio": 2.4,
                    "hip_downward_speed_bh_s": 1.5,
                    "bbox_downward_speed_bh_s": 1.7,
                    "torso_rotation_deg_s": 260.0,
                    "height_collapse_fraction": 0.5,
                    "motion_bh_s": 1.1,
                },
            ),
            (
                0.2,
                {
                    "hip_downward_speed_bh_s": 0.0,
                    "bbox_downward_speed_bh_s": 0.0,
                    "torso_rotation_deg_s": 0.0,
                    "motion_bh_s": 0.05,
                },
            ),
            (1.2, {}),
            (
                0.4,
                {
                    "torso_angle_deg": 10.0,
                    "bbox_aspect_ratio": 0.65,
                    "motion_bh_s": 0.8,
                },
            ),
            (
                1.0,
                {
                    "motion_bh_s": 0.05,
                },
            ),
        ]
    )
    return Scenario(
        "synth-fall-lateral-recovered",
        "synthetic-subject-3",
        "fall-lateral-recovered",
        tuple(frames),
        (LabelledEventSpec("OBSERVED_FALL", onset_s, onset_s + 2.0, True),),
    )


def _fall_slow_slump() -> Scenario:
    onset_s = 0.6
    frames = _build_stream(
        [
            (onset_s, {}),
            (
                1.5,
                {
                    "torso_angle_deg": 70.0,
                    "bbox_aspect_ratio": 1.8,
                    "hip_downward_speed_bh_s": 0.6,
                    "bbox_downward_speed_bh_s": 0.65,
                    "torso_rotation_deg_s": 70.0,
                    "height_collapse_fraction": 0.3,
                    "motion_bh_s": 0.4,
                },
            ),
            (
                0.3,
                {
                    "hip_downward_speed_bh_s": 0.0,
                    "bbox_downward_speed_bh_s": 0.0,
                    "torso_rotation_deg_s": 0.0,
                    "motion_bh_s": 0.05,
                },
            ),
            (2.0, {}),
        ]
    )
    return Scenario(
        "synth-fall-slow-slump",
        "synthetic-subject-4",
        "fall-slow-slump",
        tuple(frames),
        # Gradual-onset falls take longer to confirm than fast ones (a
        # 1.5s collapse ramp precedes impact here, vs. ~0.3s for the fast
        # falls above), so this scenario gets a wider match window than
        # the onset+2.0s convention used elsewhere in this module.
        (LabelledEventSpec("OBSERVED_FALL", onset_s, onset_s + 3.0, False),),
    )


def _fall_low_visibility() -> Scenario:
    onset_s = 0.5
    frames = _build_stream(
        [
            (onset_s, {}),
            (
                0.3,
                {
                    "torso_angle_deg": 80.0,
                    "bbox_aspect_ratio": 2.5,
                    "hip_downward_speed_bh_s": 1.7,
                    "bbox_downward_speed_bh_s": 1.9,
                    "torso_rotation_deg_s": 300.0,
                    "height_collapse_fraction": 0.5,
                    "motion_bh_s": 1.2,
                    "visibility_quality": 0.35,
                },
            ),
            (
                0.4,
                {
                    "hip_downward_speed_bh_s": 0.0,
                    "bbox_downward_speed_bh_s": 0.0,
                    "torso_rotation_deg_s": 0.0,
                    "motion_bh_s": 0.05,
                    "visibility_quality": 0.9,
                },
            ),
            (2.0, {}),
        ]
    )
    return Scenario(
        "synth-fall-low-visibility",
        "synthetic-subject-5",
        "fall-low-visibility",
        tuple(frames),
        (LabelledEventSpec("OBSERVED_FALL", onset_s, onset_s + 2.0, False),),
    )


def _fall_after_long_quiet() -> Scenario:
    onset_s = 8.0
    frames = _build_stream(
        [
            (onset_s, {}),
            (
                0.3,
                {
                    "torso_angle_deg": 83.0,
                    "bbox_aspect_ratio": 2.7,
                    "hip_downward_speed_bh_s": 1.6,
                    "bbox_downward_speed_bh_s": 1.8,
                    "torso_rotation_deg_s": 330.0,
                    "height_collapse_fraction": 0.55,
                    "motion_bh_s": 1.2,
                },
            ),
            (
                0.2,
                {
                    "hip_downward_speed_bh_s": 0.0,
                    "bbox_downward_speed_bh_s": 0.0,
                    "torso_rotation_deg_s": 0.0,
                    "motion_bh_s": 0.05,
                },
            ),
            (2.0, {}),
        ]
    )
    return Scenario(
        "synth-fall-after-long-quiet",
        "synthetic-subject-6",
        "fall-after-long-quiet",
        tuple(frames),
        (LabelledEventSpec("OBSERVED_FALL", onset_s, onset_s + 2.0, False),),
    )


# ---------------------------------------------------------------------------
# ADL hard-negative scenarios (no labelled events)
# ---------------------------------------------------------------------------


def _adl_fast_sit() -> Scenario:
    frames = _build_stream(
        [
            (0.4, {}),
            (
                0.3,
                {
                    "torso_angle_deg": 32.0,
                    "bbox_aspect_ratio": 0.75,
                    "hip_downward_speed_bh_s": 0.6,
                    "bbox_downward_speed_bh_s": 0.55,
                    "torso_rotation_deg_s": 20.0,
                    "height_collapse_fraction": 0.05,
                    "motion_bh_s": 0.9,
                },
            ),
            (0.2, {"hip_downward_speed_bh_s": 0.0, "bbox_downward_speed_bh_s": 0.0, "motion_bh_s": 0.1}),
            (2.0, {}),
        ]
    )
    return Scenario(
        "synth-adl-fast-sit", "synthetic-subject-7", "adl-fast-sit", tuple(frames), ()
    )


def _adl_lie_on_bed_brief() -> Scenario:
    # Lies down (posture crosses both torso/aspect thresholds) but gets back
    # up before persistent_prone_dwell_s (2.0s default) elapses -- below the
    # dwell floor the FSM requires before flagging sustained stillness.
    frames = _build_stream(
        [
            (0.5, {}),
            (
                1.2,
                {
                    "torso_angle_deg": 78.0,
                    "bbox_aspect_ratio": 1.3,
                    "hip_downward_speed_bh_s": 0.15,
                    "bbox_downward_speed_bh_s": 0.15,
                    "torso_rotation_deg_s": 8.0,
                    "height_collapse_fraction": 0.08,
                    "motion_bh_s": 0.08,
                },
            ),
            (
                0.4,
                {
                    "torso_angle_deg": 10.0,
                    "bbox_aspect_ratio": 0.65,
                    "motion_bh_s": 0.7,
                },
            ),
            (1.5, {"motion_bh_s": 0.05}),
        ]
    )
    return Scenario(
        "synth-adl-lie-on-bed-brief",
        "synthetic-subject-8",
        "adl-lie-on-bed-brief",
        tuple(frames),
        (),
    )


def _adl_bend_pick_up() -> Scenario:
    frames = _build_stream(
        [
            (0.5, {}),
            (
                0.3,
                {
                    "torso_angle_deg": 55.0,
                    "bbox_aspect_ratio": 0.7,
                    "hip_downward_speed_bh_s": 0.3,
                    "bbox_downward_speed_bh_s": 0.3,
                    "torso_rotation_deg_s": 15.0,
                    "motion_bh_s": 0.6,
                },
            ),
            (0.3, {"torso_angle_deg": 8.0, "motion_bh_s": 0.5}),
            (1.5, {"motion_bh_s": 0.05}),
        ]
    )
    return Scenario(
        "synth-adl-bend-pick-up",
        "synthetic-subject-9",
        "adl-bend-pick-up",
        tuple(frames),
        (),
    )


def _adl_squat() -> Scenario:
    frames = _build_stream(
        [
            (0.5, {}),
            (
                0.4,
                {
                    "torso_angle_deg": 40.0,
                    "bbox_aspect_ratio": 0.8,
                    "hip_downward_speed_bh_s": 0.45,
                    "bbox_downward_speed_bh_s": 0.4,
                    "torso_rotation_deg_s": 10.0,
                    "motion_bh_s": 0.5,
                },
            ),
            (1.5, {"hip_downward_speed_bh_s": 0.0, "bbox_downward_speed_bh_s": 0.0, "motion_bh_s": 0.1}),
            (0.4, {"torso_angle_deg": 8.0, "bbox_aspect_ratio": 0.65, "motion_bh_s": 0.5}),
            (1.0, {"motion_bh_s": 0.05}),
        ]
    )
    return Scenario(
        "synth-adl-squat", "synthetic-subject-10", "adl-squat", tuple(frames), ()
    )


def _adl_jump() -> Scenario:
    frames = _build_stream(
        [
            (0.5, {}),
            (0.15, {"motion_bh_s": 1.0, "hip_downward_speed_bh_s": -0.6}),
            (0.1, {"hip_downward_speed_bh_s": 0.7, "bbox_downward_speed_bh_s": 0.6, "motion_bh_s": 1.1}),
            (0.25, {"hip_downward_speed_bh_s": 0.0, "bbox_downward_speed_bh_s": 0.0, "motion_bh_s": 0.2}),
            (1.5, {"motion_bh_s": 0.05}),
        ]
    )
    return Scenario(
        "synth-adl-jump", "synthetic-subject-11", "adl-jump", tuple(frames), ()
    )


def _adl_brisk_walk() -> Scenario:
    frames = _build_stream(
        [
            (0.3, {}),
            (0.4, {"torso_angle_deg": 18.0, "hip_downward_speed_bh_s": 0.25, "motion_bh_s": 0.55}),
            (0.4, {"torso_angle_deg": 10.0, "hip_downward_speed_bh_s": -0.15, "motion_bh_s": 0.45}),
            (0.4, {"torso_angle_deg": 18.0, "hip_downward_speed_bh_s": 0.25, "motion_bh_s": 0.55}),
            (0.4, {"torso_angle_deg": 10.0, "hip_downward_speed_bh_s": -0.15, "motion_bh_s": 0.45}),
            (1.1, {"torso_angle_deg": 8.0, "hip_downward_speed_bh_s": 0.0, "motion_bh_s": 0.05}),
        ]
    )
    return Scenario(
        "synth-adl-brisk-walk", "synthetic-subject-12", "adl-brisk-walk", tuple(frames), ()
    )


def _adl_sit_on_floor_deliberate() -> Scenario:
    frames = _build_stream(
        [
            (0.5, {}),
            (
                0.9,
                {
                    "torso_angle_deg": 46.0,
                    "bbox_aspect_ratio": 0.78,
                    "hip_downward_speed_bh_s": 0.35,
                    "bbox_downward_speed_bh_s": 0.3,
                    "torso_rotation_deg_s": 12.0,
                    "motion_bh_s": 0.35,
                },
            ),
            (1.6, {"hip_downward_speed_bh_s": 0.0, "bbox_downward_speed_bh_s": 0.0, "motion_bh_s": 0.08}),
        ]
    )
    return Scenario(
        "synth-adl-sit-on-floor",
        "synthetic-subject-13",
        "adl-sit-on-floor-deliberate",
        tuple(frames),
        (),
    )


def _adl_kneel() -> Scenario:
    frames = _build_stream(
        [
            (0.5, {}),
            (
                0.3,
                {
                    "torso_angle_deg": 25.0,
                    "bbox_aspect_ratio": 0.8,
                    "hip_downward_speed_bh_s": 0.55,
                    "bbox_downward_speed_bh_s": 0.5,
                    "torso_rotation_deg_s": 30.0,
                    "height_collapse_fraction": 0.08,
                    "motion_bh_s": 0.6,
                },
            ),
            (0.2, {"hip_downward_speed_bh_s": 0.0, "bbox_downward_speed_bh_s": 0.0, "motion_bh_s": 0.1}),
            (1.5, {}),
        ]
    )
    return Scenario(
        "synth-adl-kneel", "synthetic-subject-14", "adl-kneel", tuple(frames), ()
    )


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def _degenerate_all_invalid() -> Scenario:
    frames = _build_stream(
        [(2.0, {"valid": False, "visibility_quality": 0.1})],
    )
    return Scenario(
        "synth-degenerate-all-invalid",
        "synthetic-subject-15",
        "degenerate-all-invalid",
        tuple(frames),
        (),
    )


def _degenerate_single_observation() -> Scenario:
    frames = _build_stream([(DT, {})])
    return Scenario(
        "synth-degenerate-single-observation",
        "synthetic-subject-16",
        "degenerate-single-observation",
        tuple(frames),
        (),
    )


SCENARIO_BUILDERS = (
    _fall_forward_fast,
    _fall_backward_fast,
    _fall_lateral_recovered,
    _fall_slow_slump,
    _fall_low_visibility,
    _fall_after_long_quiet,
    _adl_fast_sit,
    _adl_lie_on_bed_brief,
    _adl_bend_pick_up,
    _adl_squat,
    _adl_jump,
    _adl_brisk_walk,
    _adl_sit_on_floor_deliberate,
    _adl_kneel,
    _degenerate_all_invalid,
    _degenerate_single_observation,
)


def build_scenarios() -> tuple[Scenario, ...]:
    """Return every synthetic scenario in a fixed, deterministic order."""
    return tuple(builder() for builder in SCENARIO_BUILDERS)
