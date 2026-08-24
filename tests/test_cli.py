import importlib
import logging

import numpy as np
import pytest

from fall_detection.cli import _parse_source, build_parser, main
from fall_detection.fall_config import FallConfig, FallProfile
from fall_detection.fall_fsm import FallState
from fall_detection.fall_state import FallEvent


def test_parse_source_webcam_index_is_live():
    assert _parse_source("0") == (0, True)


def test_parse_source_dev_video_device_is_live():
    assert _parse_source("/dev/video0") == ("/dev/video0", True)


def test_parse_source_http_url_is_live():
    url = "http://192.168.1.5:4747/mjpegfeed"
    assert _parse_source(url) == (url, True)


def test_parse_source_rtsp_url_is_live():
    url = "rtsp://192.168.1.5:8080/h264_pcm.sdp"
    assert _parse_source(url) == (url, True)


def test_parse_source_https_url_is_live():
    url = "https://192.168.1.5:8080/video"
    assert _parse_source(url) == (url, True)


def test_parse_source_video_file_path_is_not_live():
    assert _parse_source("clip.mp4") == ("clip.mp4", False)


def test_fall_detection_flags_default_on_with_sane_defaults():
    args = build_parser().parse_args([])
    assert args.no_fall_detection is False
    assert args.body_mass_kg is None
    assert args.fall_alert_log is None
    assert args.fall_config is None
    assert args.fall_profile is None
    assert args.fall_telemetry_log is None
    assert args.fall_debug_overlay is False


def test_fall_detection_flags_can_be_overridden():
    args = build_parser().parse_args(
        [
            "--no-fall-detection",
            "--body-mass-kg",
            "55.5",
            "--fall-alert-log",
            "alerts.jsonl",
            "--fall-config",
            "fall.toml",
            "--fall-profile",
            "precision",
            "--fall-telemetry-log",
            "telemetry.jsonl",
            "--fall-debug-overlay",
        ]
    )
    assert args.no_fall_detection is True
    assert args.body_mass_kg == 55.5
    assert args.fall_alert_log == "alerts.jsonl"
    assert args.fall_config == "fall.toml"
    assert args.fall_profile == "precision"
    assert args.fall_telemetry_log == "telemetry.jsonl"
    assert args.fall_debug_overlay is True


class _CallbackRunner:
    def __init__(self, config, source, **kwargs):
        self.on_frame = kwargs.get("on_frame")

    def run(self):
        if self.on_frame is not None:
            self.on_frame([], 1.25, np.zeros((72, 128, 3), dtype=np.uint8))
        return 1


def _patch_runtime(monkeypatch, manager_type):
    monkeypatch.setattr("fall_detection.fall_state.FallStateManager", manager_type)
    monkeypatch.setattr("fall_detection.runner.VideoFileRunner", _CallbackRunner)
    monkeypatch.setattr("fall_detection.runner.LiveStreamRunner", _CallbackRunner)


def test_explicit_fall_profile_overrides_toml_profile(monkeypatch, tmp_path):
    source = tmp_path / "clip.mp4"
    source.touch()
    config_path = tmp_path / "fall.toml"
    config_path.write_text('profile = "sensitive"\n', encoding="utf-8")
    constructed = []

    class RecordingManager:
        def __init__(self, config):
            constructed.append(config)

        def update(self, persons, t_seconds, frame_width, frame_height):
            return []

        def forget(self, person_id):
            pass

    _patch_runtime(monkeypatch, RecordingManager)

    code = main(
        [
            "--source",
            str(source),
            "--fall-config",
            str(config_path),
            "--fall-profile",
            "precision",
            "--no-display",
        ]
    )

    assert code == 0
    assert constructed == [
        FallConfig(
            profile=FallProfile.PRECISION,
            dynamic_torso_angle_deg=55.0,
            dynamic_downward_speed_bh_s=0.70,
            dynamic_torso_rotation_deg_s=75.0,
            dynamic_height_collapse_fraction=0.25,
            posture_torso_angle_deg=60.0,
            posture_aspect_ratio=1.20,
            posture_evidence_fraction=0.75,
            persistent_prone_dwell_s=3.0,
            recovery_dwell_s=1.0,
        )
    ]


def test_invalid_fall_toml_exits_two_with_useful_log(caplog, tmp_path):
    config_path = tmp_path / "bad.toml"
    config_path.write_text("[timing\n", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        code = main(["--source", "0", "--fall-config", str(config_path), "--no-display"])

    assert code == 2
    assert "fall configuration" in caplog.text
    assert str(config_path) in caplog.text


@pytest.mark.parametrize("source_kind", ["file", "live"])
def test_positive_frame_dimensions_reach_fall_manager(
    source_kind, monkeypatch, tmp_path
):
    calls = []

    class RecordingManager:
        def __init__(self, config):
            pass

        def update(self, persons, t_seconds, frame_width, frame_height):
            calls.append((frame_width, frame_height))
            return []

        def forget(self, person_id):
            pass

    _patch_runtime(monkeypatch, RecordingManager)
    if source_kind == "file":
        source = tmp_path / "clip.mp4"
        source.touch()
        raw_source = str(source)
    else:
        raw_source = "0"

    assert main(["--source", raw_source, "--no-display"]) == 0
    assert calls == [(128, 72)]


def test_explicit_body_mass_warns_once_without_changing_fall_config(
    monkeypatch, tmp_path
):
    source = tmp_path / "clip.mp4"
    source.touch()
    constructed = []

    class RecordingManager:
        def __init__(self, config):
            constructed.append(config)

        def update(self, persons, t_seconds, frame_width, frame_height):
            return []

        def forget(self, person_id):
            pass

    _patch_runtime(monkeypatch, RecordingManager)

    with pytest.warns(DeprecationWarning) as warnings_seen:
        code = main(
            ["--source", str(source), "--body-mass-kg", "55.5", "--no-display"]
        )

    assert code == 0
    assert len(warnings_seen) == 1
    assert constructed == [FallConfig()]


def test_live_debug_overlay_adds_wall_time_since_last_inference(monkeypatch):
    cli_module = importlib.import_module("fall_detection.cli")
    event = FallEvent(
        person_id=1,
        state=FallState.UPRIGHT,
        state_changed=False,
        t_seconds=2.0,
        observation_age_s=0.2,
    )
    draw_calls = []
    clock = iter([10.0, 10.75])
    monkeypatch.setattr(cli_module, "monotonic", lambda: next(clock), raising=False)
    monkeypatch.setattr(
        "fall_detection.drawing.annotate_fall_state",
        lambda canvas, events, **kwargs: draw_calls.append((events, kwargs)) or canvas,
    )

    class FakeManager:
        def __init__(self, config):
            pass

        def update(self, persons, t_seconds, frame_width, frame_height):
            return [event]

        def forget(self, person_id):
            pass

    class FakeLiveRunner:
        def __init__(self, config, source, **kwargs):
            self.on_frame = kwargs["on_frame"]
            self.overlay = kwargs["overlay"]

        def run(self):
            frame = np.zeros((60, 80, 3), dtype=np.uint8)
            self.on_frame([], 2.0, frame)
            self.overlay(frame, [])
            return 1

    monkeypatch.setattr("fall_detection.fall_state.FallStateManager", FakeManager)
    monkeypatch.setattr("fall_detection.runner.LiveStreamRunner", FakeLiveRunner)

    assert main(["--source", "0", "--fall-debug-overlay", "--no-display"]) == 0
    assert draw_calls == [
        (
            [event],
            {"debug": True, "additional_observation_age_s": pytest.approx(0.75)},
        )
    ]


def test_output_flag_defaults_to_none():
    args = build_parser().parse_args([])
    assert args.output is None


def test_output_flag_parses_path():
    args = build_parser().parse_args(["--output", "out.mp4"])
    assert args.output == "out.mp4"


def test_main_rejects_output_flag_with_a_live_source():
    """--output re-opens a cv2.VideoWriter that can't safely survive the
    LiveStreamRunner's auto-restart-on-stall loop, so it's file-source only."""
    code = main(["--source", "0", "--output", "out.mp4", "--no-display"])
    assert code == 2
