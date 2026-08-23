from fall_detection.cli import build_parser


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
