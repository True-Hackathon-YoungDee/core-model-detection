"""Command line entry point for the pose core."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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
        help="camera index (e.g. 0), /dev/video* device, or path to a video file",
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
        "Physics-informed fall-state layer on top of the pose keypoints.",
    )
    fall.add_argument(
        "--no-fall-detection",
        action="store_true",
        help="pose-only, skip the fall-state layer",
    )
    fall.add_argument(
        "--body-mass-kg",
        type=float,
        default=70.0,
        help="approximate subject mass, used for kinetic-energy discriminators (default: 70.0)",
    )
    fall.add_argument(
        "--fall-alert-log",
        help="append FALL_CONFIRMED/BED_REST state transitions as JSON lines to this file",
    )

    parser.add_argument("--no-display", action="store_true", help="run headless, no cv2 window")
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
    """One JSON line for a FALL_CONFIRMED/BED_REST transition (--fall-alert-log)."""
    return (
        json.dumps(
            {
                "person_id": event.person_id,
                "state": event.state.name,
                "t_seconds": event.t_seconds,
            }
        )
        + "\n"
    )


def _parse_source(raw: str) -> tuple[int | str, bool]:
    """Return (source, is_live). Digits and /dev/video* mean a live camera."""
    if raw.isdigit():
        return int(raw), True
    if raw.startswith("/dev/video"):
        return raw, True
    return raw, False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, args.log_file)

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
        smoothing=not args.no_smoothing,
        best_only=args.best_only,
        max_frames=args.max_frames,
    )

    if not args.no_fall_detection:
        fall_manager = FallStateManager(body_mass_kg=args.body_mass_kg)
        alert_log_path = Path(args.fall_alert_log) if args.fall_alert_log else None
        latest_events: list = []

        def on_frame(persons, t_seconds, frame) -> None:
            events = fall_manager.update(persons, t_seconds)
            latest_events[:] = events
            for event in events:
                if not event.state_changed:
                    continue
                if event.state == FallState.FALL_CONFIRMED:
                    logger.error(
                        "FALL_CONFIRMED person=%d t=%.2fs", event.person_id, event.t_seconds
                    )
                elif event.state == FallState.BED_REST:
                    logger.warning(
                        "BED_REST person=%d t=%.2fs", event.person_id, event.t_seconds
                    )
                else:
                    continue
                if alert_log_path is not None:
                    with alert_log_path.open("a") as handle:
                        handle.write(format_alert(event))

        def overlay(canvas, persons):
            return annotate_fall_state(canvas, latest_events)

        runner_kwargs.update(
            on_frame=on_frame, overlay=overlay, on_person_lost=fall_manager.forget
        )

    try:
        if is_live:
            runner = LiveStreamRunner(config, source, **runner_kwargs)
        else:
            runner = VideoFileRunner(config, source, **runner_kwargs)
        frames = runner.run()
    except KeyboardInterrupt:
        logger.info("interrupted by user")
        return 130
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 2
    except Exception as error:
        logger.error("pose run failed: %s", error)
        logger.debug("pose run failure detail", exc_info=True)
        return 1

    logger.info("done: %d frames processed", frames)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
