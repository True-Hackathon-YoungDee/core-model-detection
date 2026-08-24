"""Pixel-corrected image evidence for temporal fall decisions."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Any

from .fall_config import FallConfig
from .pose import PersonPose, PoseLandmark

STILLNESS_THRESHOLD_BH_S = 0.10


@dataclass(frozen=True)
class FallFeatures:
    t_seconds: float
    valid: bool
    torso_angle_deg: float
    bbox_aspect_ratio: float
    hip_downward_speed_bh_s: float
    bbox_downward_speed_bh_s: float
    torso_rotation_deg_s: float
    height_collapse_fraction: float
    motion_bh_s: float
    visibility_quality: float
    torso_centroid: tuple[float, float]
    furniture_roi: str | None
    scale_source: str


@dataclass(frozen=True)
class FallEvidence:
    dynamic_torso_angle: bool
    downward_motion: bool
    rapid_torso_rotation: bool
    height_collapse: bool
    posture_torso: bool
    posture_aspect: bool
    stillness: bool
    quality_ok: bool

    @property
    def posture(self) -> bool:
        return self.quality_ok and self.posture_torso and self.posture_aspect

    @property
    def dynamic_cue_count(self) -> int:
        return sum(
            (
                self.downward_motion,
                self.rapid_torso_rotation,
                self.height_collapse,
            )
        )


@dataclass(frozen=True)
class _FrameGeometry:
    t_seconds: float
    frame_width: float
    frame_height: float
    torso_angle_deg: float
    bbox_width: float
    bbox_height: float
    bbox_center_y: float
    hip_center_y: float
    torso_points_px: tuple[tuple[float, float], ...]


class ImageEvidenceExtractor:
    """Extract current-frame image geometry and bounded temporal derivatives."""

    def __init__(self, config: FallConfig) -> None:
        self.config = config
        self._previous: _FrameGeometry | None = None
        self._upright_heights: deque[tuple[float, float]] = deque()
        self._history_window_s = max(
            config.dynamic_cue_window_s,
            config.observed_fall_postural_window_s,
            config.candidate_timeout_s,
            config.recovery_dwell_s,
            config.persistent_prone_dwell_s,
        )

    def update(
        self,
        person: PersonPose,
        t_seconds: float,
        frame_width: int,
        frame_height: int,
    ) -> FallFeatures:
        width = float(frame_width)
        height = float(frame_height)
        if not math.isfinite(width) or not math.isfinite(height) or width <= 0.0 or height <= 0.0:
            raise ValueError("frame_width and frame_height must be positive")

        timestamp = _finite_or_zero(t_seconds)
        self._prune_upright_heights(timestamp)
        torso = _torso_points(person.landmarks)
        visibility_quality = _torso_visibility(torso)
        if (
            torso is None
            or visibility_quality < self.config.min_torso_visibility
            or not all(_finite_xy(landmark) for landmark in torso)
        ):
            self._previous = None
            return _invalid_features(timestamp, visibility_quality)

        visible_points = [
            landmark
            for landmark in person.landmarks
            if _finite_xy(landmark)
            and _finite_or_zero(getattr(landmark, "visibility", None))
            >= self.config.min_torso_visibility
        ]
        if not visible_points:
            self._previous = None
            return _invalid_features(timestamp, visibility_quality)

        xs = [float(landmark.x) * width for landmark in visible_points]
        ys = [float(landmark.y) * height for landmark in visible_points]
        torso_points_px = tuple(
            (float(landmark.x) * width, float(landmark.y) * height) for landmark in torso
        )
        if not all(
            math.isfinite(coordinate)
            for point in (*zip(xs, ys), *torso_points_px)
            for coordinate in point
        ):
            self._previous = None
            return _invalid_features(timestamp, visibility_quality)

        bbox_width = max(xs) - min(xs)
        bbox_height = max(ys) - min(ys)
        if (
            not math.isfinite(bbox_width)
            or not math.isfinite(bbox_height)
            or bbox_width <= 0.0
            or bbox_height <= 0.0
        ):
            self._previous = None
            return _invalid_features(timestamp, visibility_quality)

        shoulder_x = torso_points_px[0][0] / 2.0 + torso_points_px[1][0] / 2.0
        shoulder_y = torso_points_px[0][1] / 2.0 + torso_points_px[1][1] / 2.0
        hip_x = torso_points_px[2][0] / 2.0 + torso_points_px[3][0] / 2.0
        hip_y = torso_points_px[2][1] / 2.0 + torso_points_px[3][1] / 2.0
        dx = hip_x - shoulder_x
        dy = hip_y - shoulder_y
        if not math.isfinite(dx) or not math.isfinite(dy) or math.hypot(dx, dy) <= 0.0:
            self._previous = None
            return _invalid_features(timestamp, visibility_quality)

        torso_angle_deg = math.degrees(math.atan2(abs(dx), abs(dy)))
        torso_centroid = (
            sum(float(landmark.x) / len(torso) for landmark in torso),
            sum(float(landmark.y) / len(torso) for landmark in torso),
        )
        matching_roi = next(
            (roi.name for roi in self.config.furniture_rois if roi.contains(torso_centroid)),
            None,
        )

        current = _FrameGeometry(
            t_seconds=timestamp,
            frame_width=width,
            frame_height=height,
            torso_angle_deg=torso_angle_deg,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            bbox_center_y=min(ys) / 2.0 + max(ys) / 2.0,
            hip_center_y=hip_y,
            torso_points_px=torso_points_px,
        )
        if torso_angle_deg <= self.config.recovery_torso_angle_deg:
            self._upright_heights.append((timestamp, bbox_height))

        if self._upright_heights:
            scale = statistics.median(height for _, height in self._upright_heights)
            scale_source = "upright_height"
            height_collapse_fraction = max(0.0, (scale - bbox_height) / scale)
        else:
            scale = math.hypot(bbox_width, bbox_height)
            scale_source = "current_diagonal"
            height_collapse_fraction = 0.0

        hip_downward_speed = 0.0
        bbox_downward_speed = 0.0
        torso_rotation = 0.0
        motion = 0.0
        previous = self._previous
        if previous is not None:
            dt = timestamp - previous.t_seconds
            same_dimensions = (
                width == previous.frame_width and height == previous.frame_height
            )
            if 0.0 < dt <= self.config.max_observation_gap_s and same_dimensions:
                hip_downward_speed = (current.hip_center_y - previous.hip_center_y) / dt / scale
                bbox_downward_speed = (
                    current.bbox_center_y - previous.bbox_center_y
                ) / dt / scale
                torso_rotation = abs(torso_angle_deg - previous.torso_angle_deg) / dt
                mean_displacement = sum(
                    math.hypot(current_x - previous_x, current_y - previous_y)
                    for (current_x, current_y), (previous_x, previous_y) in zip(
                        torso_points_px, previous.torso_points_px
                    )
                ) / len(torso_points_px)
                motion = mean_displacement / dt / scale
        bbox_aspect_ratio = bbox_width / bbox_height
        if not all(
            math.isfinite(value)
            for value in (
                bbox_aspect_ratio,
                hip_downward_speed,
                bbox_downward_speed,
                torso_rotation,
                height_collapse_fraction,
                motion,
                *torso_centroid,
            )
        ):
            self._previous = None
            return _invalid_features(timestamp, visibility_quality)
        self._previous = current

        return FallFeatures(
            t_seconds=timestamp,
            valid=True,
            torso_angle_deg=torso_angle_deg,
            bbox_aspect_ratio=bbox_aspect_ratio,
            hip_downward_speed_bh_s=hip_downward_speed,
            bbox_downward_speed_bh_s=bbox_downward_speed,
            torso_rotation_deg_s=torso_rotation,
            height_collapse_fraction=height_collapse_fraction,
            motion_bh_s=motion,
            visibility_quality=visibility_quality,
            torso_centroid=torso_centroid,
            furniture_roi=matching_roi,
            scale_source=scale_source,
        )

    def _prune_upright_heights(self, t_seconds: float) -> None:
        while (
            self._upright_heights
            and t_seconds - self._upright_heights[0][0] > self._history_window_s
        ):
            self._upright_heights.popleft()


def classify_evidence(features: FallFeatures, config: FallConfig) -> FallEvidence:
    """Classify one immutable feature sample into fall-decision gates."""
    quality_ok = (
        features.valid
        and features.visibility_quality >= config.min_torso_visibility
    )
    return FallEvidence(
        dynamic_torso_angle=quality_ok
        and features.torso_angle_deg >= config.dynamic_torso_angle_deg,
        downward_motion=quality_ok
        and max(
            features.hip_downward_speed_bh_s,
            features.bbox_downward_speed_bh_s,
        )
        >= config.dynamic_downward_speed_bh_s,
        rapid_torso_rotation=quality_ok
        and features.torso_rotation_deg_s >= config.dynamic_torso_rotation_deg_s,
        height_collapse=quality_ok
        and features.scale_source == "upright_height"
        and features.height_collapse_fraction >= config.dynamic_height_collapse_fraction,
        posture_torso=quality_ok
        and features.torso_angle_deg >= config.posture_torso_angle_deg,
        posture_aspect=quality_ok
        and features.bbox_aspect_ratio >= config.posture_aspect_ratio,
        stillness=quality_ok and features.motion_bh_s <= STILLNESS_THRESHOLD_BH_S,
        quality_ok=quality_ok,
    )


def _torso_points(landmarks: list[Any]) -> tuple[Any, Any, Any, Any] | None:
    indices = (
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.LEFT_HIP,
        PoseLandmark.RIGHT_HIP,
    )
    if len(landmarks) <= max(indices):
        return None
    return tuple(landmarks[index] for index in indices)  # type: ignore[return-value]


def _torso_visibility(torso: tuple[Any, Any, Any, Any] | None) -> float:
    if torso is None:
        return 0.0
    values = [_finite_or_zero(getattr(landmark, "visibility", None)) for landmark in torso]
    return max(0.0, min(1.0, sum(values) / len(values)))


def _finite_xy(landmark: Any) -> bool:
    try:
        return math.isfinite(float(landmark.x)) and math.isfinite(float(landmark.y))
    except (AttributeError, OverflowError, TypeError, ValueError):
        return False


def _finite_or_zero(value: object) -> float:
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _invalid_features(t_seconds: float, visibility_quality: float) -> FallFeatures:
    return FallFeatures(
        t_seconds=t_seconds,
        valid=False,
        torso_angle_deg=0.0,
        bbox_aspect_ratio=0.0,
        hip_downward_speed_bh_s=0.0,
        bbox_downward_speed_bh_s=0.0,
        torso_rotation_deg_s=0.0,
        height_collapse_fraction=0.0,
        motion_bh_s=0.0,
        visibility_quality=visibility_quality,
        torso_centroid=(0.0, 0.0),
        furniture_roi=None,
        scale_source="unavailable",
    )
