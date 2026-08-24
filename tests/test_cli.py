from fall_detection.cli import _parse_source, build_parser, main


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
    assert args.body_mass_kg == 70.0
    assert args.fall_alert_log is None


def test_fall_detection_flags_can_be_overridden():
    args = build_parser().parse_args(
        ["--no-fall-detection", "--body-mass-kg", "55.5", "--fall-alert-log", "alerts.jsonl"]
    )
    assert args.no_fall_detection is True
    assert args.body_mass_kg == 55.5
    assert args.fall_alert_log == "alerts.jsonl"


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
