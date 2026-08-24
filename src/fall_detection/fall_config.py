"""Validated profile and TOML configuration for temporal fall detection."""

from __future__ import annotations

import dataclasses
import math
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class FallProfile(str, Enum):
    SENSITIVE = "sensitive"
    BALANCED = "balanced"
    PRECISION = "precision"


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _fraction(value: object, name: str) -> float:
    result = _number(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _positive(value: object, name: str) -> float:
    result = _number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _angle(value: object, name: str) -> float:
    result = _number(value, name)
    if not 0.0 <= result <= 180.0:
        raise ValueError(f"{name} must be between 0 and 180 degrees")
    return result


@dataclass(frozen=True)
class FurnitureROI:
    """A named normalized-image polygon used for furniture occupancy evidence."""

    name: str
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("furniture ROI name must be non-empty")
        normalized_points: list[tuple[float, float]] = []
        for index, point in enumerate(self.points):
            if isinstance(point, (str, bytes)) or not isinstance(point, (tuple, list)) or len(point) != 2:
                raise ValueError(f"furniture ROI point {index} must be an [x, y] pair")
            x = _fraction(point[0], f"furniture ROI point {index} x")
            y = _fraction(point[1], f"furniture ROI point {index} y")
            normalized_points.append((x, y))
        if len(set(normalized_points)) < 3:
            raise ValueError("furniture ROI must have at least three distinct vertices")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "points", tuple(normalized_points))

    def contains(self, point: tuple[float, float]) -> bool:
        """Return whether normalized *point* is inside this polygon, edges included."""
        if isinstance(point, (str, bytes)) or not isinstance(point, (tuple, list)) or len(point) != 2:
            return False
        try:
            x = _number(point[0], "point x")
            y = _number(point[1], "point y")
        except ValueError:
            return False
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            return False

        inside = False
        previous_x, previous_y = self.points[-1]
        for current_x, current_y in self.points:
            if _point_on_segment(x, y, previous_x, previous_y, current_x, current_y):
                return True
            crosses_ray = (current_y > y) != (previous_y > y)
            if crosses_ray:
                intersection_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
                if x < intersection_x:
                    inside = not inside
            previous_x, previous_y = current_x, current_y
        return inside


def _point_on_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    epsilon = 1e-9
    cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross_product) > epsilon:
        return False
    return min(x1, x2) - epsilon <= x <= max(x1, x2) + epsilon and min(y1, y2) - epsilon <= y <= max(y1, y2) + epsilon


@dataclass(frozen=True)
class FallConfig:
    """Immutable validated thresholds used by the RGB temporal-evidence FSM."""

    profile: FallProfile = FallProfile.BALANCED
    dynamic_torso_angle_deg: float = 45.0
    dynamic_downward_speed_bh_s: float = 0.50
    dynamic_torso_rotation_deg_s: float = 60.0
    dynamic_height_collapse_fraction: float = 0.15
    posture_torso_angle_deg: float = 50.0
    posture_aspect_ratio: float = 1.00
    posture_evidence_fraction: float = 0.60
    persistent_prone_dwell_s: float = 2.0
    dynamic_cue_window_s: float = 0.75
    observed_fall_postural_window_s: float = 1.0
    candidate_timeout_s: float = 2.0
    recovery_dwell_s: float = 2.0
    max_observation_gap_s: float = 0.5
    min_temporal_coverage: float = 0.80
    rejection_cooldown_s: float = 0.5
    min_torso_visibility: float = 0.50
    recovery_torso_angle_deg: float = 35.0
    furniture_occupancy_fraction: float = 0.60
    furniture_rois: tuple[FurnitureROI, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        try:
            selected_profile = self.profile if isinstance(self.profile, FallProfile) else FallProfile(self.profile)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unknown fall profile: {self.profile!r}") from error
        object.__setattr__(self, "profile", selected_profile)

        for name in ("dynamic_torso_angle_deg", "posture_torso_angle_deg", "recovery_torso_angle_deg"):
            object.__setattr__(self, name, _angle(getattr(self, name), name))
        for name in ("dynamic_downward_speed_bh_s", "dynamic_torso_rotation_deg_s", "posture_aspect_ratio"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in (
            "dynamic_height_collapse_fraction",
            "posture_evidence_fraction",
            "min_temporal_coverage",
            "min_torso_visibility",
            "furniture_occupancy_fraction",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        for name in (
            "persistent_prone_dwell_s",
            "dynamic_cue_window_s",
            "observed_fall_postural_window_s",
            "candidate_timeout_s",
            "recovery_dwell_s",
            "max_observation_gap_s",
            "rejection_cooldown_s",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))

        rois = tuple(self.furniture_rois)
        if not all(isinstance(roi, FurnitureROI) for roi in rois):
            raise ValueError("furniture_rois must contain FurnitureROI values")
        if len({roi.name for roi in rois}) != len(rois):
            raise ValueError("furniture ROI names must be unique")
        object.__setattr__(self, "furniture_rois", rois)


_PROFILE_DEFAULTS: dict[FallProfile, dict[str, float]] = {
    FallProfile.SENSITIVE: {
        "dynamic_torso_angle_deg": 40.0,
        "dynamic_downward_speed_bh_s": 0.40,
        "dynamic_torso_rotation_deg_s": 45.0,
        "dynamic_height_collapse_fraction": 0.10,
        "posture_torso_angle_deg": 45.0,
        "posture_aspect_ratio": 0.90,
        "posture_evidence_fraction": 0.50,
        "persistent_prone_dwell_s": 1.5,
        "dynamic_cue_window_s": 0.75,
        "observed_fall_postural_window_s": 1.0,
        "candidate_timeout_s": 2.0,
        "recovery_dwell_s": 2.0,
        "max_observation_gap_s": 0.5,
        "min_temporal_coverage": 0.80,
        "rejection_cooldown_s": 0.5,
        "min_torso_visibility": 0.50,
        "recovery_torso_angle_deg": 35.0,
        "furniture_occupancy_fraction": 0.60,
    },
    FallProfile.BALANCED: {
        "dynamic_torso_angle_deg": 45.0,
        "dynamic_downward_speed_bh_s": 0.50,
        "dynamic_torso_rotation_deg_s": 60.0,
        "dynamic_height_collapse_fraction": 0.15,
        "posture_torso_angle_deg": 50.0,
        "posture_aspect_ratio": 1.00,
        "posture_evidence_fraction": 0.60,
        "persistent_prone_dwell_s": 2.0,
        "dynamic_cue_window_s": 0.75,
        "observed_fall_postural_window_s": 1.0,
        "candidate_timeout_s": 2.0,
        "recovery_dwell_s": 2.0,
        "max_observation_gap_s": 0.5,
        "min_temporal_coverage": 0.80,
        "rejection_cooldown_s": 0.5,
        "min_torso_visibility": 0.50,
        "recovery_torso_angle_deg": 35.0,
        "furniture_occupancy_fraction": 0.60,
    },
    FallProfile.PRECISION: {
        "dynamic_torso_angle_deg": 55.0,
        "dynamic_downward_speed_bh_s": 0.70,
        "dynamic_torso_rotation_deg_s": 75.0,
        "dynamic_height_collapse_fraction": 0.25,
        "posture_torso_angle_deg": 60.0,
        "posture_aspect_ratio": 1.20,
        "posture_evidence_fraction": 0.75,
        "persistent_prone_dwell_s": 3.0,
        "dynamic_cue_window_s": 0.75,
        "observed_fall_postural_window_s": 1.0,
        "candidate_timeout_s": 2.0,
        "recovery_dwell_s": 2.0,
        "max_observation_gap_s": 0.5,
        "min_temporal_coverage": 0.80,
        "rejection_cooldown_s": 0.5,
        "min_torso_visibility": 0.50,
        "recovery_torso_angle_deg": 35.0,
        "furniture_occupancy_fraction": 0.60,
    },
}

_SECTION_FIELDS: dict[str, dict[str, str]] = {
    "dynamic": {
        "torso_angle_deg": "dynamic_torso_angle_deg",
        "downward_speed_bh_s": "dynamic_downward_speed_bh_s",
        "torso_rotation_deg_s": "dynamic_torso_rotation_deg_s",
        "height_collapse_fraction": "dynamic_height_collapse_fraction",
    },
    "posture": {
        "torso_angle_deg": "posture_torso_angle_deg",
        "aspect_ratio": "posture_aspect_ratio",
        "evidence_fraction": "posture_evidence_fraction",
        "recovery_torso_angle_deg": "recovery_torso_angle_deg",
        "furniture_occupancy_fraction": "furniture_occupancy_fraction",
    },
    "timing": {
        "persistent_prone_dwell_s": "persistent_prone_dwell_s",
        "dynamic_cue_window_s": "dynamic_cue_window_s",
        "observed_fall_postural_window_s": "observed_fall_postural_window_s",
        "candidate_timeout_s": "candidate_timeout_s",
        "recovery_dwell_s": "recovery_dwell_s",
        "max_observation_gap_s": "max_observation_gap_s",
        "rejection_cooldown_s": "rejection_cooldown_s",
    },
    "quality": {
        "min_temporal_coverage": "min_temporal_coverage",
        "min_torso_visibility": "min_torso_visibility",
    },
}


def _select_profile(value: FallProfile | str | None) -> FallProfile | None:
    if value is None:
        return None
    try:
        return value if isinstance(value, FallProfile) else FallProfile(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown fall profile: {value!r}") from error


def _table(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _parse_rois(raw_rois: object) -> tuple[FurnitureROI, ...]:
    if not isinstance(raw_rois, list):
        raise ValueError("furniture_rois must be an array of TOML tables")
    rois: list[FurnitureROI] = []
    for index, raw_roi in enumerate(raw_rois):
        roi = _table(raw_roi, f"furniture_rois[{index}]")
        unknown = set(roi) - {"name", "points"}
        if unknown:
            raise ValueError(f"unknown furniture ROI key: {sorted(unknown)[0]}")
        if set(roi) != {"name", "points"}:
            raise ValueError("each furniture ROI requires name and points")
        points = roi["points"]
        if not isinstance(points, list):
            raise ValueError("furniture ROI points must be an array")
        rois.append(FurnitureROI(name=roi["name"], points=tuple(points)))
    if len({roi.name for roi in rois}) != len(rois):
        raise ValueError("furniture ROI names must be unique")
    return tuple(rois)


def load_fall_config(path: Path | None = None, profile: FallProfile | str | None = None) -> FallConfig:
    """Load a validated TOML config, applying explicit profile selection first."""
    document: Mapping[str, Any] = {}
    if path is not None:
        with Path(path).open("rb") as config_file:
            document = tomllib.load(config_file)

    allowed_root_keys = {"profile", *_SECTION_FIELDS, "furniture_rois"}
    unknown_root_keys = set(document) - allowed_root_keys
    if unknown_root_keys:
        raise ValueError(f"unknown fall configuration key: {sorted(unknown_root_keys)[0]}")

    explicit_profile = _select_profile(profile)
    document_profile = _select_profile(document.get("profile")) if "profile" in document else None
    selected_profile = explicit_profile or document_profile or FallProfile.BALANCED
    config = FallConfig(profile=selected_profile, **_PROFILE_DEFAULTS[selected_profile])

    overrides: dict[str, Any] = {}
    for section, field_names in _SECTION_FIELDS.items():
        if section not in document:
            continue
        values = _table(document[section], section)
        unknown_section_keys = set(values) - set(field_names)
        if unknown_section_keys:
            raise ValueError(f"unknown [{section}] key: {sorted(unknown_section_keys)[0]}")
        overrides.update({field_names[key]: value for key, value in values.items()})
    if "furniture_rois" in document:
        overrides["furniture_rois"] = _parse_rois(document["furniture_rois"])
    return dataclasses.replace(config, **overrides)
