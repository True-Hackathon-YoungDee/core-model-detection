"""Command line entry point for the pose core."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import logging
import sys
import warnings
from pathlib import Path
from time import monotonic

from .fall_config import FallProfile, load_fall_config
from .fall_telemetry import event_record, jsonl_line, telemetry_record, write_jsonl
from .logging_config import setup_logging
from .models import DEFAULT_CACHE_DIR, DetectorVariant, ModelVariant
from .strategy import Strategy

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fall-detection",
        description="MediaPipe pose landmarking for one or many people.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help=(
            "camera index (e.g. 0), /dev/video* device, path to a video file, "
            "or a stream URL (e.g. rtsp://..., or DroidCam/IP Webcam "
            "http://<phone-ip>:4747/mjpegfeed)"
        ),
    )
    parser.add_argument(
        "--model",
        default=ModelVariant.FULL.value,
        choices=[variant.value for variant in ModelVariant],
        help="pose landmarker bundle to download and use (default: full)",
    )
    parser.add_argument("--model-path", help="use this .task bundle instead of downloading")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"where downloaded bundles live (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--num-poses", type=int, default=1, help="maximum people to track (default: 1)"
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="keep only the most confident person even when --num-poses > 1",
    )
    parser.add_argument("--gpu", action="store_true", help="try the GPU delegate (Linux only)")

    cascade = parser.add_argument_group(
        "detection strategy",
        "One landmarker over the whole frame reports a single person in a crowd. "
        "The cascade detects person boxes first and landmarks each body separately.",
    )
    cascade.add_argument(
        "--detector",
        default=Strategy.AUTO.value,
        choices=[strategy.value for strategy in Strategy],
        help="auto picks native at --num-poses 1 and cascade above it (default: auto)",
    )
    cascade.add_argument(
        "--person-model",
        default=DetectorVariant.LITE0.value,
        choices=[variant.value for variant in DetectorVariant],
        help="cascade person-box model (default: efficientdet_lite0, best measured recall)",
    )
    cascade.add_argument(
        "--person-score", type=float, default=0.4, help="person box confidence floor (default: 0.4)"
    )
    cascade.add_argument(
        "--crop-padding",
        type=float,
        default=0.15,
        help="grow each person box by this fraction so limbs are not clipped (default: 0.15)",
    )
    cascade.add_argument(
        "--crop-workers",
        type=int,
        default=0,
        help="landmarkers running crops in parallel (default: min(4, --num-poses))",
    )
    cascade.add_argument(
        "--detect-interval",
        type=int,
        default=1,
        help="run the person detector every Nth frame, carrying regions of interest "
        "between runs (default: 1, no carry)",
    )
    cascade.add_argument(
        "--min-box-px",
        type=int,
        default=48,
        help="ignore person boxes whose short side is under this (default: 48)",
    )
    fall = parser.add_argument_group(
        "fall detection",
        "Pixel-corrected RGB temporal fall-state layer on top of the pose keypoints.",
    )
    fall.add_argument(
        "--no-fall-detection",
        action="store_true",
        help="pose-only, skip the fall-state layer",
    )
    fall.add_argument(
        "--body-mass-kg",
        type=float,
        help="deprecated compatibility option; RGB fall decisions do not use body mass",
    )
    fall.add_argument("--fall-config", help="load fall thresholds and ROIs from TOML")
    fall.add_argument(
        "--fall-profile",
        choices=[profile.value for profile in FallProfile],
        help="override the fall profile selected by TOML",
    )
    fall.add_argument(
        "--fall-alert-log",
        help="append detected/recovered fall incidents as JSON lines to this file",
    )
    fall.add_argument(
        "--fall-telemetry-log",
        help="append per-person fall evidence and state decisions as JSON lines",
    )
    fall.add_argument(
        "--fall-debug-overlay",
        action="store_true",
        help="show fall evidence, timing, coverage, and observation age",
    )

    parser.add_argument(
        "--output",
        help="write annotated frames to this video file (e.g. out.mp4); file sources only",
    )
    parser.add_argument("--no-display", action="store_true", help="run headless, no cv2 window")
    parser.add_argument(
        "--display-max-width",
        type=int,
        default=1280,
        help="cap initial display window width in px, preserves aspect ratio (default: 1280)",
    )
    parser.add_argument("--no-smoothing", action="store_true", help="disable One-Euro filtering")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--tracking-confidence", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, help="stop after N frames (testing aid)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console verbosity (default: INFO)",
    )
    parser.add_argument("--log-file", help="also write logs to this file")
    return parser


def format_alert(event) -> str:
    """Compatibility wrapper for one schema-v1 incident JSON line."""
    if event.incident_event not in ("detected", "recovered"):
        raise ValueError("alert formatting requires a detected or recovered incident")
    return jsonl_line(event_record(event, event.incident_event))


def _parse_source(raw: str) -> tuple[int | str, bool]:
    """Return (source, is_live). Digits, /dev/video*, and stream URLs mean a live camera."""
    if raw.isdigit():
        return int(raw), True
    if raw.startswith("/dev/video"):
        return raw, True
    if raw.startswith(("rtsp://", "http://", "https://")):
        return raw, True
    return raw, False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, args.log_file)

    if args.body_mass_kg is not None:
        warnings.warn(
            "--body-mass-kg is deprecated and has no effect on RGB fall decisions",
            DeprecationWarning,
            stacklevel=2,
        )

    try:
        fall_config = load_fall_config(
            Path(args.fall_config) if args.fall_config else None,
            args.fall_profile,
        )
    except (OSError, ValueError) as error:
        location = f" {args.fall_config}" if args.fall_config else ""
        logger.error("invalid fall configuration%s: %s", location, error)
        return 2

    # Imported after logging is configured so the native log level takes effect.
    from .drawing import annotate_fall_state
    from .fall_fsm import FallState
    from .fall_state import FallStateManager
    from .pose import PoseConfig
    from .runner import LiveStreamRunner, VideoFileRunner

    if args.num_poses < 1:
        logger.error("--num-poses must be >= 1, got %d", args.num_poses)
        return 2
    if args.crop_workers < 0:
        logger.error("--crop-workers must be >= 0, got %d", args.crop_workers)
        return 2
    if args.detect_interval < 1:
        logger.error("--detect-interval must be >= 1, got %d", args.detect_interval)
        return 2
    if not 0.0 < args.person_score < 1.0:
        logger.error("--person-score must be between 0 and 1, got %.2f", args.person_score)
        return 2
    if args.crop_padding < 0.0:
        logger.error("--crop-padding must be >= 0, got %.2f", args.crop_padding)
        return 2
    if args.min_box_px < 1:
        logger.error("--min-box-px must be >= 1, got %d", args.min_box_px)
        return 2

    source, is_live = _parse_source(args.source)
    if not is_live and not Path(source).is_file():
        logger.error("video file not found: %s", source)
        return 2
    if args.output and is_live:
        logger.error("--output is only supported for file sources, not live streams")
        return 2

    config = PoseConfig(
        model_variant=ModelVariant(args.model),
        model_path=Path(args.model_path) if args.model_path else None,
        num_poses=args.num_poses,
        min_pose_detection_confidence=args.min_detection_confidence,
        min_pose_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.tracking_confidence,
        use_gpu=args.gpu,
        cache_dir=Path(args.cache_dir),
        strategy=Strategy(args.detector),
        detector_variant=DetectorVariant(args.person_model),
        person_score_threshold=args.person_score,
        crop_padding=args.crop_padding,
        crop_workers=args.crop_workers,
        detect_interval=args.detect_interval,
        min_box_px=args.min_box_px,
    )
    if args.num_poses > 1:
        logger.info(
            "multi-person mode: %d poses; ids come from centroid tracking "
            "(MediaPipe itself does not re-identify people)",
            args.num_poses,
        )

    runner_kwargs = dict(
        display=not args.no_display,
        display_max_width=args.display_max_width,
        smoothing=not args.no_smoothing,
        best_only=args.best_only,
        max_frames=args.max_frames,
        max_unseen_s=fall_config.identity_timeout_s,
    )

    resources = ExitStack()
    try:
        if not args.no_fall_detection:
            alert_log = (
                resources.enter_context(
                    Path(args.fall_alert_log).open("a", encoding="utf-8")
                )
                if args.fall_alert_log
                else None
            )
            telemetry_log = (
                resources.enter_context(
                    Path(args.fall_telemetry_log).open("a", encoding="utf-8")
                )
                if args.fall_telemetry_log
                else None
            )
            fall_manager = FallStateManager(fall_config)
            latest_events: list = []
            last_live_inference_at: float | None = None

            def on_frame(persons, t_seconds, frame) -> None:
                nonlocal last_live_inference_at
                if is_live:
                    last_live_inference_at = monotonic()
                frame_height, frame_width = frame.shape[:2]
                events = fall_manager.update(
                    persons, t_seconds, frame_width=frame_width, frame_height=frame_height
                )
                latest_events[:] = events
                for event in events:
                    if telemetry_log is not None:
                        write_jsonl(telemetry_log, telemetry_record(event))
                    if event.incident_event == "detected":
                        log = (
                            logger.warning
                            if event.state == FallState.BED_REST
                            else logger.error
                        )
                        log(
                            "%s person=%d t=%.2fs",
                            event.state.name,
                            event.person_id,
                            event.t_seconds,
                        )
                    elif event.incident_event == "recovered":
                        logger.info(
                            "RECOVERED person=%d t=%.2fs",
                            event.person_id,
                            event.t_seconds,
                        )
                    else:
                        continue
                    if alert_log is not None:
                        write_jsonl(
                            alert_log,
                            event_record(event, event.incident_event),
                            flush=True,
                        )

            def overlay(canvas, persons):
                additional_age_s = (
                    max(0.0, monotonic() - last_live_inference_at)
                    if is_live and last_live_inference_at is not None
                    else 0.0
                )
                return annotate_fall_state(
                    canvas,
                    latest_events,
                    debug=args.fall_debug_overlay,
                    additional_observation_age_s=additional_age_s,
                )

            runner_kwargs.update(
                on_frame=on_frame, overlay=overlay, on_person_lost=fall_manager.forget
            )

        if is_live:
            runner = LiveStreamRunner(config, source, **runner_kwargs)
        else:
            runner = VideoFileRunner(config, source, output=args.output, **runner_kwargs)
        frames = runner.run()
    except KeyboardInterrupt:
        logger.info("interrupted by user")
        return 130
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 2
    except OSError as error:
        logger.error("fall runtime I/O failed: %s", error)
        return 2
    except Exception as error:
        logger.error("pose run failed: %s", error)
        logger.debug("pose run failure detail", exc_info=True)
        return 1
    finally:
        resources.close()

    logger.info("done: %d frames processed", frames)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
