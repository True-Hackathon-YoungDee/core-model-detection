import dataclasses

import pytest

from fall_detection.fall_config import FallProfile, FurnitureROI, load_fall_config


def test_balanced_profile_has_all_specified_seed_values():
    """A wrong balanced seed value changes the detector's public defaults."""
    config = load_fall_config()

    assert config.profile is FallProfile.BALANCED
    assert config.dynamic_torso_angle_deg == 45.0
    assert config.dynamic_downward_speed_bh_s == 0.50
    assert config.dynamic_torso_rotation_deg_s == 60.0
    assert config.dynamic_height_collapse_fraction == 0.15
    assert config.posture_torso_angle_deg == 50.0
    assert config.posture_aspect_ratio == 1.00
    assert config.posture_evidence_fraction == 0.60
    assert config.persistent_prone_dwell_s == 2.0
    assert config.dynamic_cue_window_s == 0.75
    assert config.observed_fall_postural_window_s == 1.0
    assert config.candidate_timeout_s == 2.0
    assert config.recovery_dwell_s == 0.70
    assert config.max_observation_gap_s == 0.5
    assert config.min_temporal_coverage == 0.80
    assert config.rejection_cooldown_s == 0.5
    assert config.min_torso_visibility == 0.50
    assert config.recovery_torso_angle_deg == 35.0
    assert config.furniture_occupancy_fraction == 0.60
    assert config.furniture_rois == ()


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (
            FallProfile.SENSITIVE,
            (40.0, 0.40, 45.0, 0.10, 45.0, 0.90, 0.50, 1.5),
        ),
        (
            FallProfile.PRECISION,
            (55.0, 0.70, 75.0, 0.25, 60.0, 1.20, 0.75, 3.0),
        ),
    ],
)
def test_sensitive_and_precision_profiles_have_their_specified_values(profile, expected):
    """A profile table regression must not silently make profiles equivalent."""
    config = load_fall_config(profile=profile)

    assert (
        config.dynamic_torso_angle_deg,
        config.dynamic_downward_speed_bh_s,
        config.dynamic_torso_rotation_deg_s,
        config.dynamic_height_collapse_fraction,
        config.posture_torso_angle_deg,
        config.posture_aspect_ratio,
        config.posture_evidence_fraction,
        config.persistent_prone_dwell_s,
    ) == expected


@pytest.mark.parametrize(
    ("profile", "expected_recovery_dwell_s"),
    [
        (FallProfile.SENSITIVE, 0.50),
        (FallProfile.BALANCED, 0.70),
        (FallProfile.PRECISION, 1.00),
    ],
)
def test_empirical_recovery_seeds_remain_monotonic_across_profiles(
    profile, expected_recovery_dwell_s
):
    config = load_fall_config(profile=profile)

    assert config.recovery_dwell_s == expected_recovery_dwell_s


def test_explicit_profile_takes_precedence_over_toml_profile(tmp_path):
    """Reversing profile precedence would use the document's profile instead."""
    config_path = tmp_path / "fall.toml"
    config_path.write_text('profile = "precision"\n')

    config = load_fall_config(config_path, profile="sensitive")

    assert config.profile is FallProfile.SENSITIVE
    assert config.dynamic_torso_angle_deg == 40.0


def test_toml_fields_override_the_selected_profile_defaults(tmp_path):
    """Applying TOML before profile selection would discard these overrides."""
    config_path = tmp_path / "fall.toml"
    config_path.write_text(
        'profile = "sensitive"\n\n'
        "[dynamic]\n"
        "torso_angle_deg = 47.0\n\n"
        "[timing]\n"
        "candidate_timeout_s = 2.5\n"
    )

    config = load_fall_config(config_path)

    assert config.profile is FallProfile.SENSITIVE
    assert config.dynamic_torso_angle_deg == 47.0
    assert config.dynamic_downward_speed_bh_s == 0.40
    assert config.candidate_timeout_s == 2.5


@pytest.mark.parametrize(
    "toml",
    [
        "unknown = 1\n",
        "[dynamic]\nunknown = 1\n",
        "[quality]\nmin_torso_visibility = nan\n",
        "[quality]\nmin_temporal_coverage = inf\n",
        "[posture]\nevidence_fraction = 1.1\n",
        "[dynamic]\nheight_collapse_fraction = -0.1\n",
        "[timing]\ncandidate_timeout_s = 0\n",
    ],
)
def test_invalid_toml_values_are_rejected(tmp_path, toml):
    """Skipping schema or numeric validation would accept unusable detector settings."""
    config_path = tmp_path / "fall.toml"
    config_path.write_text(toml)

    with pytest.raises(ValueError):
        load_fall_config(config_path)


@pytest.mark.parametrize(
    "toml",
    [
        "[[furniture_rois]]\nname = \"bed\"\npoints = [[0.1, 0.1], [0.8, 0.1]]\n",
        "[[furniture_rois]]\nname = \"bed\"\npoints = [[0.1, 0.1], [0.1, 0.1], [0.8, 0.1]]\n",
        "[[furniture_rois]]\nname = \"bed\"\npoints = [[0.1, 0.1], [1.1, 0.1], [0.8, 0.8]]\n",
        "[[furniture_rois]]\nname = \"\"\npoints = [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8]]\n",
        (
            "[[furniture_rois]]\nname = \"bed\"\npoints = [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8]]\n\n"
            "[[furniture_rois]]\nname = \"bed\"\npoints = [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8]]\n"
        ),
    ],
)
def test_malformed_or_ambiguous_furniture_rois_are_rejected(tmp_path, toml):
    """Malformed polygons or duplicate names would make furniture evidence ambiguous."""
    config_path = tmp_path / "fall.toml"
    config_path.write_text(toml)

    with pytest.raises(ValueError):
        load_fall_config(config_path)


def test_collinear_furniture_roi_is_rejected(tmp_path):
    """Accepting three collinear vertices would create a zero-area furniture region."""
    config_path = tmp_path / "fall.toml"
    config_path.write_text(
        "[[furniture_rois]]\n"
        'name = "line"\n'
        "points = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]\n"
    )

    with pytest.raises(ValueError):
        load_fall_config(config_path)


def test_self_intersecting_furniture_roi_is_rejected(tmp_path):
    """Accepting a bow-tie polygon would make furniture containment ambiguous."""
    config_path = tmp_path / "fall.toml"
    config_path.write_text(
        "[[furniture_rois]]\n"
        'name = "bow-tie"\n'
        "points = [[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.7, 0.1]]\n"
    )

    with pytest.raises(ValueError):
        load_fall_config(config_path)


def test_closed_furniture_roi_loads_and_contains_points(tmp_path):
    """A conventional repeated closing vertex must not make a valid ROI self-intersect."""
    config_path = tmp_path / "fall.toml"
    config_path.write_text(
        "[[furniture_rois]]\n"
        'name = "triangle"\n'
        "points = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.1]]\n"
    )

    roi = load_fall_config(config_path).furniture_rois[0]

    assert roi.points == ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9))
    assert roi.contains((0.7, 0.2))
    assert roi.contains((0.5, 0.1))
    assert not roi.contains((0.2, 0.8))


def test_furniture_roi_contains_interior_and_boundary_points():
    """Changing edge handling would incorrectly exclude torso centroids on furniture borders."""
    roi = FurnitureROI(name="bed", points=((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)))

    assert roi.contains((0.5, 0.5))
    assert roi.contains((0.2, 0.5))
    assert roi.contains((0.2, 0.2))
    assert not roi.contains((0.1, 0.5))


def test_loaded_configuration_is_immutable():
    """Making a loaded config mutable would let runtime state alter detector policy."""
    config = load_fall_config()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.candidate_timeout_s = 9.0
