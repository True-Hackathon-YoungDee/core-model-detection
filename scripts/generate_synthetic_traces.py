#!/usr/bin/env python3
"""Generate the deterministic synthetic ADL/fall regression corpus.

Writes both the numeric feature trace (JSONL, schema_version 2) and the
manifest (TOML) that binds it, from the scenario catalog in
``fall_detection.synthetic_traces``. Every scenario is an authored
``FallFeatures`` stream, not a recording of a real subject -- see that
module's docstring before treating its accuracy metrics as a system
performance claim.

Deterministic: the scenario catalog has a fixed build order and no
randomness, so re-running this script reproduces byte-identical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fall_detection.evaluation import TRACE_SCHEMA_VERSION, fall_config_fingerprint
from fall_detection.fall_config import FallConfig
from fall_detection.synthetic_traces import build_scenarios

MANIFEST_SCHEMA_VERSION = 1
# Sentinel provenance for a trace with no real pose model behind it. Every
# other trace in this repo's model_sha256 field means "this real model
# bundle produced these numbers" -- this sentinel documents that no model
# was involved, rather than filling in a plausible-looking fake hash.
SYNTHETIC_MODEL_SHA256 = hashlib.sha256(b"synthetic-no-model").hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/traces/synthetic-adl-v1.jsonl"),
        help="trace JSONL output path",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("evaluation/manifests/synthetic-adl.toml"),
        help="manifest TOML output path",
    )
    return parser


def _encode_record(record: dict[str, object]) -> str:
    return json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"


def _trace_lines(config: FallConfig) -> list[str]:
    fingerprint = fall_config_fingerprint(config)
    # A stable per-scenario placeholder source checksum: this manifest has
    # no real source file, so this documents "synthetic scenario <id>"
    # rather than a byte-for-byte checksum of nonexistent bytes.
    lines: list[str] = []
    for scenario in build_scenarios():
        source_sha256 = hashlib.sha256(f"synthetic:{scenario.clip_id}".encode()).hexdigest()
        lines.append(
            _encode_record(
                {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "record_type": "clip",
                    "clip_id": scenario.clip_id,
                    "source_sha256": source_sha256,
                    "model_sha256": SYNTHETIC_MODEL_SHA256,
                    "fall_config_fingerprint": fingerprint,
                    "duration_s": scenario.duration_s,
                    "frame_width": 640,
                    "frame_height": 480,
                    "fps": 20.0,
                    "frame_count": len(scenario.frames),
                }
            )
        )
        for frame_index, features in enumerate(scenario.frames):
            lines.append(
                _encode_record(
                    {
                        "schema_version": TRACE_SCHEMA_VERSION,
                        "record_type": "observation",
                        "clip_id": scenario.clip_id,
                        "source_sha256": source_sha256,
                        "frame_index": frame_index,
                        "t_seconds": features.t_seconds,
                        "person_id": 0,
                        "features": {
                            "t_seconds": features.t_seconds,
                            "valid": features.valid,
                            "torso_angle_deg": features.torso_angle_deg,
                            "bbox_aspect_ratio": features.bbox_aspect_ratio,
                            "hip_downward_speed_bh_s": features.hip_downward_speed_bh_s,
                            "bbox_downward_speed_bh_s": features.bbox_downward_speed_bh_s,
                            "torso_rotation_deg_s": features.torso_rotation_deg_s,
                            "height_collapse_fraction": features.height_collapse_fraction,
                            "motion_bh_s": features.motion_bh_s,
                            "visibility_quality": features.visibility_quality,
                            "torso_centroid": list(features.torso_centroid),
                            "furniture_roi": features.furniture_roi,
                            "scale_source": features.scale_source,
                            "motion_available": features.motion_available,
                        },
                    }
                )
            )
    return lines, {
        scenario.clip_id: hashlib.sha256(f"synthetic:{scenario.clip_id}".encode()).hexdigest()
        for scenario in build_scenarios()
    }


def _manifest_text(trace_relpath: str, trace_sha256: str, source_sha256_by_id: dict[str, str]) -> str:
    lines = [
        "# GENERATED FILE -- do not hand-edit.",
        "# Produced by scripts/generate_synthetic_traces.py from",
        "# src/fall_detection/synthetic_traces.py. Every clip below is an",
        "# authored FallFeatures stream, not a recorded subject: source_sha256",
        "# is a placeholder digest of the scenario id, and model_sha256 in the",
        "# paired trace is the documented sentinel sha256(\"synthetic-no-model\"),",
        "# not a real pose-model bundle. Regenerate rather than hand-edit:",
        "#   uv run python scripts/generate_synthetic_traces.py",
        "schema_version = 1",
        'dataset = "synthetic-adl"',
        f'trace = "{trace_relpath}"',
        f'trace_sha256 = "{trace_sha256}"',
        "",
    ]
    for scenario in build_scenarios():
        lines.append("[[clips]]")
        lines.append(f'id = "{scenario.clip_id}"')
        lines.append(f'source = "synthetic/{scenario.clip_id}.synthetic"')
        lines.append(f'source_sha256 = "{source_sha256_by_id[scenario.clip_id]}"')
        lines.append(f'subject = "{scenario.subject}"')
        lines.append(f'trial = "{scenario.trial}"')
        lines.append('camera = "synthetic"')
        lines.append('split = "regression"')
        lines.append(f"duration_s = {scenario.duration_s}")
        lines.append("")
        for event in scenario.events:
            lines.append("[[clips.events]]")
            lines.append(f'kind = "{event.kind}"')
            lines.append(f"onset_s = {event.onset_s}")
            lines.append(f"match_end_s = {event.match_end_s}")
            lines.append(f"recovered = {'true' if event.recovered else 'false'}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = FallConfig()

    trace_lines, source_sha256_by_id = _trace_lines(config)
    trace_text = "".join(trace_lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(trace_text, encoding="utf-8")

    trace_sha256 = hashlib.sha256(trace_text.encode("utf-8")).hexdigest()
    trace_relpath = f"../traces/{args.output.name}"
    manifest_text = _manifest_text(trace_relpath, trace_sha256, source_sha256_by_id)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(manifest_text, encoding="utf-8")

    print(f"wrote {args.output} ({len(build_scenarios())} clips)")
    print(f"wrote {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
