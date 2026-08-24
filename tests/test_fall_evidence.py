import dataclasses
import math

import pytest

from fall_detection.fall_config import FallConfig, FurnitureROI
from fall_detection.fall_evidence import (
    STILLNESS_THRESHOLD_BH_S,
    ImageEvidenceExtractor,
    classify_evidence,
)
from fall_detection.pose import PoseLandmark

from conftest import NUM_LANDMARKS, make_landmark, make_person


def _person_with_torso(
    left_shoulder: tuple[float, float],
    right_shoulder: tuple[float, float],
    left_hip: tuple[float, float],
    right_hip: tuple[float, float],
):
    landmarks = [make_landmark(0.5, 0.5) for _ in range(NUM_LANDMARKS)]
    for index, point in (
        (PoseLandmark.LEFT_SHOULDER, left_shoulder),
        (PoseLandmark.RIGHT_SHOULDER, right_shoulder),
        (PoseLandmark.LEFT_HIP, left_hip),
        (PoseLandmark.RIGHT_HIP, right_hip),
    ):
        landmarks[index] = make_landmark(*point)
    return make_person(landmarks=landmarks)


def _pixel_person(
    *,
    frame_width: int = 1000,
    frame_height: int = 1000,
    bbox: tuple[float, float, float, float],
    shoulder_center: tuple[float, float],
    hip_center: tuple[float, float],
    torso_half_width: float = 50.0,
    visibility: float = 1.0,
):
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    landmarks = [
        make_landmark(center_x / frame_width, center_y / frame_height, visibility=visibility)
        for _ in range(NUM_LANDMARKS)
    ]
    pixel_points = {
        PoseLandmark.NOSE: (bbox[0], bbox[1]),
        PoseLandmark.LEFT_FOOT_INDEX: (bbox[2], bbox[3]),
        PoseLandmark.LEFT_SHOULDER: (
            shoulder_center[0] - torso_half_width,
            shoulder_center[1],
        ),
        PoseLandmark.RIGHT_SHOULDER: (
            shoulder_center[0] + torso_half_width,
            shoulder_center[1],
        ),
        PoseLandmark.LEFT_HIP: (hip_center[0] - torso_half_width, hip_center[1]),
        PoseLandmark.RIGHT_HIP: (hip_center[0] + torso_half_width, hip_center[1]),
    }
    for index, (x, y) in pixel_points.items():
        landmarks[index] = make_landmark(
            x / frame_width,
            y / frame_height,
            visibility=visibility,
        )
    return make_person(landmarks=landmarks)


def test_torso_angle_is_measured_from_pixel_vertical():
    extractor = ImageEvidenceExtractor(FallConfig())
    upright = _person_with_torso((0.4, 0.2), (0.6, 0.2), (0.4, 0.8), (0.6, 0.8))
    horizontal = _person_with_torso((0.2, 0.4), (0.2, 0.6), (0.8, 0.4), (0.8, 0.6))

    upright_features = extractor.update(upright, 0.0, 1600, 900)
    horizontal_features = extractor.update(horizontal, 1.0, 1600, 900)

    assert upright_features.torso_angle_deg == pytest.approx(0.0, abs=1e-9)
    assert horizontal_features.torso_angle_deg == pytest.approx(90.0, abs=1e-9)


@pytest.mark.parametrize("frame_width,frame_height", [(1600, 900), (900, 1600), (1000, 1000)])
def test_normalized_geometry_is_pixel_corrected_for_each_frame_shape(frame_width, frame_height):
    person = _person_with_torso((0.4, 0.3), (0.5, 0.3), (0.5, 0.7), (0.6, 0.7))

    features = ImageEvidenceExtractor(FallConfig()).update(
        person, 0.0, frame_width, frame_height
    )

    expected_angle = math.degrees(math.atan2(0.1 * frame_width, 0.4 * frame_height))
    expected_aspect = (0.2 * frame_width) / (0.4 * frame_height)
    assert features.torso_angle_deg == pytest.approx(expected_angle)
    assert features.bbox_aspect_ratio == pytest.approx(expected_aspect)


@pytest.mark.parametrize("frame_width,frame_height", [(0, 720), (-1, 720), (1280, 0), (1280, -1)])
def test_non_positive_frame_dimensions_are_rejected(frame_width, frame_height):
    person = _person_with_torso((0.4, 0.2), (0.6, 0.2), (0.4, 0.8), (0.6, 0.8))

    with pytest.raises(ValueError, match="frame_width and frame_height must be positive"):
        ImageEvidenceExtractor(FallConfig()).update(person, 0.0, frame_width, frame_height)


@pytest.mark.parametrize(
    "person",
    [
        make_person(landmarks=[]),
        make_person(
            landmarks=[make_landmark(float("nan"), 0.5) for _ in range(NUM_LANDMARKS)]
        ),
        make_person(
            landmarks=[make_landmark(1e308, 1e308) for _ in range(NUM_LANDMARKS)]
        ),
        make_person(
            landmarks=[make_landmark(0.5, 0.5, visibility=0.49) for _ in range(NUM_LANDMARKS)]
        ),
    ],
)
def test_malformed_or_low_quality_landmarks_return_invalid_finite_features(person):
    features = ImageEvidenceExtractor(FallConfig()).update(person, 0.0, 1280, 720)

    assert not features.valid
    numeric_values = (
        features.t_seconds,
        features.torso_angle_deg,
        features.bbox_aspect_ratio,
        features.hip_downward_speed_bh_s,
        features.bbox_downward_speed_bh_s,
        features.torso_rotation_deg_s,
        features.height_collapse_fraction,
        features.motion_bh_s,
        features.visibility_quality,
        *features.torso_centroid,
    )
    assert all(math.isfinite(value) for value in numeric_values)


def test_pixel_temporal_derivatives_use_upright_body_height_scale():
    extractor = ImageEvidenceExtractor(FallConfig())
    upright = _pixel_person(
        bbox=(200.0, 100.0, 800.0, 900.0),
        shoulder_center=(500.0, 300.0),
        hip_center=(500.0, 700.0),
    )
    fallen = _pixel_person(
        bbox=(200.0, 600.0, 800.0, 1000.0),
        shoulder_center=(400.0, 800.0),
        hip_center=(600.0, 800.0),
    )

    extractor.update(upright, 0.0, 1000, 1000)
    features = extractor.update(fallen, 0.5, 1000, 1000)

    assert features.scale_source == "upright_height"
    assert features.hip_downward_speed_bh_s == pytest.approx(100.0 / 0.5 / 800.0)
    assert features.bbox_downward_speed_bh_s == pytest.approx(300.0 / 0.5 / 800.0)
    assert features.torso_rotation_deg_s == pytest.approx(90.0 / 0.5)
    assert features.height_collapse_fraction == pytest.approx(0.5)


def test_height_collapse_uses_rolling_median_of_recent_upright_heights():
    extractor = ImageEvidenceExtractor(FallConfig())
    for t_seconds, height in ((0.0, 800.0), (0.1, 600.0), (0.2, 1000.0)):
        extractor.update(
            _pixel_person(
                bbox=(200.0, 500.0 - height / 2.0, 800.0, 500.0 + height / 2.0),
                shoulder_center=(500.0, 350.0),
                hip_center=(500.0, 650.0),
            ),
            t_seconds,
            1000,
            1000,
        )

    collapsed = extractor.update(
        _pixel_person(
            bbox=(200.0, 600.0, 800.0, 1000.0),
            shoulder_center=(400.0, 800.0),
            hip_center=(600.0, 800.0),
        ),
        0.3,
        1000,
        1000,
    )

    assert collapsed.height_collapse_fraction == pytest.approx((800.0 - 400.0) / 800.0)


def test_diagonal_scale_is_used_without_fabricating_prebaseline_collapse():
    extractor = ImageEvidenceExtractor(FallConfig())
    prone = _pixel_person(
        bbox=(200.0, 300.0, 500.0, 700.0),
        shoulder_center=(250.0, 500.0),
        hip_center=(450.0, 500.0),
    )
    moved = _pixel_person(
        bbox=(200.0, 350.0, 500.0, 750.0),
        shoulder_center=(250.0, 550.0),
        hip_center=(450.0, 550.0),
    )

    first = extractor.update(prone, 0.0, 1000, 1000)
    second = extractor.update(moved, 0.5, 1000, 1000)

    assert first.scale_source == "current_diagonal"
    assert first.height_collapse_fraction == 0.0
    assert second.scale_source == "current_diagonal"
    assert second.hip_downward_speed_bh_s == pytest.approx(50.0 / 0.5 / 500.0)
    assert second.bbox_downward_speed_bh_s == pytest.approx(50.0 / 0.5 / 500.0)
    assert second.motion_bh_s == pytest.approx(50.0 / 0.5 / 500.0)
    assert second.height_collapse_fraction == 0.0


def test_observation_gap_beyond_limit_resets_all_derivatives():
    extractor = ImageEvidenceExtractor(FallConfig())
    first = _pixel_person(
        bbox=(200.0, 300.0, 500.0, 700.0),
        shoulder_center=(250.0, 500.0),
        hip_center=(450.0, 500.0),
    )
    changed = _pixel_person(
        bbox=(100.0, 500.0, 700.0, 800.0),
        shoulder_center=(400.0, 600.0),
        hip_center=(400.0, 750.0),
    )

    extractor.update(first, 0.0, 1000, 1000)
    features = extractor.update(changed, 0.6, 1000, 1000)

    assert features.hip_downward_speed_bh_s == 0.0
    assert features.bbox_downward_speed_bh_s == 0.0
    assert features.torso_rotation_deg_s == 0.0
    assert features.motion_bh_s == 0.0


def test_invalid_observation_breaks_temporal_adjacency():
    extractor = ImageEvidenceExtractor(FallConfig())
    first = _pixel_person(
        bbox=(200.0, 300.0, 500.0, 700.0),
        shoulder_center=(250.0, 500.0),
        hip_center=(450.0, 500.0),
    )
    moved = _pixel_person(
        bbox=(200.0, 350.0, 500.0, 750.0),
        shoulder_center=(250.0, 550.0),
        hip_center=(450.0, 550.0),
    )

    extractor.update(first, 0.0, 1000, 1000)
    extractor.update(make_person(landmarks=[]), 0.2, 1000, 1000)
    features = extractor.update(moved, 0.4, 1000, 1000)

    assert features.hip_downward_speed_bh_s == 0.0
    assert features.motion_bh_s == 0.0


def test_extreme_finite_interframe_displacement_cannot_emit_infinity():
    extractor = ImageEvidenceExtractor(FallConfig())
    left = _pixel_person(
        frame_width=1,
        frame_height=1,
        bbox=(-1e308, 0.0, -9e307, 1.0),
        shoulder_center=(-9.8e307, 0.2),
        hip_center=(-9.8e307, 0.8),
        torso_half_width=1e305,
    )
    right = _pixel_person(
        frame_width=1,
        frame_height=1,
        bbox=(9e307, 0.0, 1e308, 1.0),
        shoulder_center=(9.8e307, 0.2),
        hip_center=(9.8e307, 0.8),
        torso_half_width=1e305,
    )

    extractor.update(left, 0.0, 1, 1)
    features = extractor.update(right, 0.5, 1, 1)

    assert not features.valid
    assert all(
        math.isfinite(value)
        for value in (
            features.bbox_aspect_ratio,
            features.hip_downward_speed_bh_s,
            features.bbox_downward_speed_bh_s,
            features.torso_rotation_deg_s,
            features.motion_bh_s,
        )
    )


def test_torso_centroid_matches_first_configured_furniture_roi():
    config = dataclasses.replace(
        FallConfig(),
        furniture_rois=(
            FurnitureROI("bed", ((0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7))),
            FurnitureROI("room", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
    )
    person = _pixel_person(
        bbox=(200.0, 300.0, 800.0, 700.0),
        shoulder_center=(400.0, 500.0),
        hip_center=(600.0, 500.0),
    )

    features = ImageEvidenceExtractor(config).update(person, 0.0, 1000, 1000)

    assert features.torso_centroid == pytest.approx((0.5, 0.5))
    assert features.furniture_roi == "bed"


def test_evidence_gates_include_exact_threshold_boundaries():
    config = FallConfig()
    base = ImageEvidenceExtractor(config).update(
        _pixel_person(
            bbox=(200.0, 300.0, 800.0, 700.0),
            shoulder_center=(400.0, 500.0),
            hip_center=(600.0, 500.0),
            visibility=config.min_torso_visibility,
        ),
        0.0,
        1000,
        1000,
    )
    features = dataclasses.replace(
        base,
        torso_angle_deg=config.posture_torso_angle_deg,
        bbox_aspect_ratio=config.posture_aspect_ratio,
        hip_downward_speed_bh_s=config.dynamic_downward_speed_bh_s,
        torso_rotation_deg_s=config.dynamic_torso_rotation_deg_s,
        height_collapse_fraction=config.dynamic_height_collapse_fraction,
        motion_bh_s=STILLNESS_THRESHOLD_BH_S,
        scale_source="upright_height",
    )

    evidence = classify_evidence(features, config)

    assert evidence.dynamic_torso_angle
    assert evidence.downward_motion
    assert evidence.rapid_torso_rotation
    assert evidence.height_collapse
    assert evidence.dynamic_cue_count == 3
    assert evidence.posture_torso
    assert evidence.posture_aspect
    assert evidence.stillness
    assert evidence.quality_ok
    assert evidence.posture
    just_moving = dataclasses.replace(
        features,
        motion_bh_s=math.nextafter(STILLNESS_THRESHOLD_BH_S, math.inf),
    )
    assert not classify_evidence(just_moving, config).stillness


def test_collapse_gate_is_unavailable_without_upright_baseline():
    config = FallConfig()
    base = ImageEvidenceExtractor(config).update(
        _pixel_person(
            bbox=(200.0, 300.0, 500.0, 700.0),
            shoulder_center=(250.0, 500.0),
            hip_center=(450.0, 500.0),
        ),
        0.0,
        1000,
        1000,
    )
    features = dataclasses.replace(
        base,
        height_collapse_fraction=config.dynamic_height_collapse_fraction,
        scale_source="current_diagonal",
    )

    assert not classify_evidence(features, config).height_collapse


def test_invalid_features_never_produce_evidence():
    config = FallConfig()
    invalid = ImageEvidenceExtractor(config).update(make_person(landmarks=[]), 0.0, 1000, 1000)
    evidence = classify_evidence(invalid, config)

    assert not any(dataclasses.astuple(evidence))
