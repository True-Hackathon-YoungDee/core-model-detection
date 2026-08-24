from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import fall_detection.evaluation as evaluation
from fall_detection.evaluation import (
    evaluate_manifest,
    load_manifest,
    replay_trace,
)
from fall_detection.fall_config import FallConfig


SOURCE_SHA = "a" * 64


def _features(
    t_seconds: float,
    *,
    torso_angle_deg: float = 0.0,
    bbox_aspect_ratio: float = 0.4,
    downward_speed: float = 0.0,
) -> dict[str, object]:
    return {
        "t_seconds": t_seconds,
        "valid": True,
        "torso_angle_deg": torso_angle_deg,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "hip_downward_speed_bh_s": downward_speed,
        "bbox_downward_speed_bh_s": 0.0,
        "torso_rotation_deg_s": 0.0,
        "height_collapse_fraction": 0.0,
        "motion_bh_s": 0.0,
        "visibility_quality": 1.0,
        "torso_centroid": [0.5, 0.5],
        "furniture_roi": None,
        "scale_source": "upright_height",
    }


def _write_trace(path: Path, clips: dict[str, tuple[float, list[dict[str, object]]]]) -> str:
    lines: list[str] = []
    for clip_id, (duration_s, observations) in clips.items():
        lines.append(
            json.dumps(
                {
                    "schema_version": 1,
                    "record_type": "clip",
                    "clip_id": clip_id,
                    "source_sha256": SOURCE_SHA,
                    "duration_s": duration_s,
                    "frame_width": 640,
                    "frame_height": 360,
                    "fps": 1.0,
                    "frame_count": int(duration_s) + 1,
                },
                sort_keys=True,
            )
        )
        for frame_index, observation in enumerate(observations):
            lines.append(
                json.dumps(
                    {
                        "schema_version": 1,
                        "record_type": "observation",
                        "clip_id": clip_id,
                        "source_sha256": SOURCE_SHA,
                        "frame_index": frame_index,
                        **observation,
                    },
                    sort_keys=True,
                )
            )
    data = ("\n".join(lines) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _observation(t_seconds: float, *, downward_speed: float = 0.0) -> dict[str, object]:
    return {
        "t_seconds": t_seconds,
        "person_id": 1,
        "features": _features(t_seconds, downward_speed=downward_speed),
    }


def _manifest_text(
    trace_name: str,
    trace_sha256: str,
    clips: list[dict[str, object]],
) -> str:
    chunks = [
        'schema_version = 1',
        'dataset = "unit-test"',
        f'trace = "{trace_name}"',
        f'trace_sha256 = "{trace_sha256}"',
    ]
    for clip in clips:
        chunks.extend(
            [
                "",
                "[[clips]]",
                f'id = "{clip["id"]}"',
                f'source = "video/input/{clip["id"]}.mp4"',
                f'source_sha256 = "{SOURCE_SHA}"',
                f'subject = "{clip.get("subject", clip["id"])}"',
                f'trial = "{clip.get("trial", "trial-1")}"',
                f'camera = "{clip.get("camera", "camera-1")}"',
                f'split = "{clip.get("split", "test")}"',
                f'duration_s = {clip.get("duration_s", 10.0)}',
            ]
        )
        for event in clip.get("events", []):
            chunks.extend(
                [
                    "[[clips.events]]",
                    f'kind = "{event.get("kind", "OBSERVED_FALL")}"',
                    f'onset_s = {event["onset_s"]}',
                    f'recovered = {str(event.get("recovered", False)).lower()}',
                ]
            )
    return "\n".join(chunks) + "\n"


@pytest.mark.parametrize("missing_key", ["subject", "trial", "camera"])
def test_manifest_requires_every_leakage_group_key(tmp_path: Path, missing_key: str):
    trace = tmp_path / "trace.jsonl"
    trace_sha = _write_trace(trace, {"clip-a": (1.0, [])})
    manifest = tmp_path / "manifest.toml"
    text = _manifest_text(trace.name, trace_sha, [{"id": "clip-a"}])
    text = "\n".join(
        line for line in text.splitlines() if not line.startswith(f"{missing_key} =")
    )
    manifest.write_text(text)

    with pytest.raises(ValueError, match=missing_key):
        load_manifest(manifest)


def test_manifest_rejects_subject_group_leakage_across_splits(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace_sha = _write_trace(trace, {"clip-a": (1.0, []), "clip-b": (1.0, [])})
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            trace.name,
            trace_sha,
            [
                {"id": "clip-a", "subject": "person-1", "split": "train"},
                {"id": "clip-b", "subject": "person-1", "split": "test"},
            ],
        )
    )

    with pytest.raises(ValueError, match="subject.*person-1.*train.*test"):
        load_manifest(manifest)


def test_manifest_rejects_boolean_schema_version(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace_sha = _write_trace(trace, {"clip-a": (1.0, [])})
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(trace.name, trace_sha, [{"id": "clip-a"}]).replace(
            "schema_version = 1", "schema_version = true"
        )
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_manifest(manifest)


def test_trace_rejects_boolean_schema_version(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, {"clip-a": (1.0, [])})
    trace.write_text(trace.read_text().replace('"schema_version": 1', '"schema_version": true'))

    with pytest.raises(ValueError, match="schema_version"):
        replay_trace(trace, "temporal-fsm", FallConfig())


def test_evaluation_rejects_missing_trace_and_checksum_mismatch(tmp_path: Path):
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text("missing.jsonl", "0" * 64, [{"id": "clip-a"}])
    )

    with pytest.raises(ValueError, match="trace file does not exist"):
        evaluate_manifest(manifest, "temporal-fsm", FallConfig())

    trace = tmp_path / "missing.jsonl"
    _write_trace(trace, {"clip-a": (1.0, [])})
    with pytest.raises(ValueError, match="trace checksum mismatch"):
        evaluate_manifest(manifest, "temporal-fsm", FallConfig())


def test_replay_supports_exactly_the_four_declared_strategies(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, {"clip-a": (1.0, [_observation(0.0)])})

    for strategy in ("legacy-and", "relaxed-or", "k-of-n", "temporal-fsm"):
        result = replay_trace(trace, strategy, FallConfig())
        assert result["strategy"] == strategy

    with pytest.raises(ValueError, match="unknown replay strategy"):
        replay_trace(trace, "other", FallConfig())


def test_event_metrics_are_aggregated_from_executed_trace_replay(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace_sha = _write_trace(
        trace,
        {
            "clip-a": (
                10.0,
                [
                    _observation(0.0),
                    _observation(2.0, downward_speed=1.0),
                    _observation(3.0),
                    _observation(4.0),
                    _observation(7.0, downward_speed=1.0),
                    _observation(10.0),
                ],
            ),
            "clip-b": (10.0, [_observation(0.0), _observation(10.0)]),
        },
    )
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            trace.name,
            trace_sha,
            [
                {
                    "id": "clip-a",
                    "duration_s": 10.0,
                    "events": [{"onset_s": 1.0, "recovered": True}],
                },
                {
                    "id": "clip-b",
                    "duration_s": 10.0,
                    "events": [{"onset_s": 1.0}],
                },
            ],
        )
    )

    report = evaluate_manifest(
        manifest,
        "relaxed-or",
        FallConfig(recovery_dwell_s=1.0, max_observation_gap_s=1.0),
    )

    assert report["event_counts"] == {
        "labelled": 2,
        "detected": 2,
        "true_positive": 1,
        "false_positive": 1,
        "missed": 1,
    }
    assert report["metrics"] == {
        "event_sensitivity": 0.5,
        "precision": 0.5,
        "false_alerts_per_hour": 180.0,
        "miss_rate": 0.5,
        "alert_latencies_s": [1.0],
        "median_alert_latency_s": 1.0,
        "recovery_times_s": [2.0],
        "median_recovery_time_s": 2.0,
        "state_dwell_s": {"FALL_CONFIRMED": 5.0, "UPRIGHT": 15.0},
    }


def test_replay_rejects_non_finite_feature_values(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "clip",
                "clip_id": "clip-a",
                "source_sha256": SOURCE_SHA,
                "duration_s": 1.0,
                "frame_width": 640,
                "frame_height": 360,
                "fps": 1.0,
                "frame_count": 1,
            }
        )
        + "\n"
        + json.dumps(
            {
                "schema_version": 1,
                "record_type": "observation",
                "clip_id": "clip-a",
                "source_sha256": SOURCE_SHA,
                "frame_index": 0,
                "t_seconds": 0.0,
                "person_id": 1,
                "features": _features(0.0) | {"motion_bh_s": float("nan")},
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="finite"):
        replay_trace(trace, "temporal-fsm", FallConfig())


def test_console_entry_emits_one_json_document_to_stdout(tmp_path: Path, capsys):
    trace = tmp_path / "trace.jsonl"
    trace_sha = _write_trace(trace, {"clip-a": (1.0, [_observation(0.0)])})
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(trace.name, trace_sha, [{"id": "clip-a", "duration_s": 1.0}])
    )

    assert evaluation.main(
        ["--manifest", str(manifest), "--strategy", "temporal-fsm"]
    ) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == evaluate_manifest(
        manifest, "temporal-fsm", FallConfig()
    )


def test_committed_local_trace_executes_all_four_labelled_acceptance_events():
    manifest = Path(__file__).parents[1] / "evaluation" / "manifests" / "local-falls.toml"

    report = evaluate_manifest(manifest, "temporal-fsm", FallConfig())

    assert report["event_counts"] == {
        "labelled": 4,
        "detected": 4,
        "true_positive": 4,
        "false_positive": 0,
        "missed": 0,
    }
    expected = {
        "fall-example-1": ("HIGH", 5.754, None),
        "fall-example-2": ("HIGH", 3.366, None),
        "fall-example-3": ("MEDIUM", 2.166, None),
        "fall-example-4": ("MEDIUM", 2.2, 2.933),
    }
    for clip in report["clips"]:
        assert len(clip["incidents"]) == 1
        incident = clip["incidents"][0]
        evidence_level, detected_at, recovered_at = expected[clip["clip_id"]]
        assert incident["kind"] == "OBSERVED_FALL"
        assert incident["evidence_level"] == evidence_level
        assert incident["detected_at"] == pytest.approx(detected_at)
        assert incident["recovered_at"] == pytest.approx(recovered_at)
        assert clip["event_results"][0]["expected_recovery"] is (
            recovered_at is not None
        )


def test_public_dataset_templates_are_strict_manifest_examples():
    manifests = Path(__file__).parents[1] / "evaluation" / "manifests"

    for name in (
        "up-fall.example.toml",
        "urfd.example.toml",
        "le2i.example.toml",
    ):
        manifest = load_manifest(manifests / name)
        assert manifest.clips
        assert all(clip.subject and clip.trial and clip.camera for clip in manifest.clips)
