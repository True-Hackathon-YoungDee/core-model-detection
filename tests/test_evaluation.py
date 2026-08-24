from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import fall_detection.evaluation as evaluation
from fall_detection.evaluation import (
    evaluate_manifest,
    load_manifest,
    replay_trace,
)
from fall_detection.fall_config import FallConfig, FurnitureROI, load_fall_config


SOURCE_SHA = "a" * 64
MODEL_SHA = "b" * 64
BALANCED_CONFIG_FINGERPRINT = (
    "04b0aeafc288b37ceb47737075615882d10da6f6e6a06c3961a326ec6c177979"
)


def _features(
    t_seconds: float,
    *,
    torso_angle_deg: float = 0.0,
    bbox_aspect_ratio: float = 0.4,
    downward_speed: float = 0.0,
    furniture_roi: str | None = None,
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
        "furniture_roi": furniture_roi,
        "scale_source": "upright_height",
    }


def _write_trace(
    path: Path,
    clips: dict[str, tuple[float, list[dict[str, object]]]],
    *,
    fall_config: FallConfig | None = None,
) -> str:
    config_fingerprint = evaluation.fall_config_fingerprint(
        fall_config or FallConfig()
    )
    lines: list[str] = []
    for clip_id, (duration_s, observations) in clips.items():
        lines.append(
            json.dumps(
                {
                    "schema_version": 1,
                    "record_type": "clip",
                    "clip_id": clip_id,
                    "source_sha256": SOURCE_SHA,
                    "model_sha256": MODEL_SHA,
                    "fall_config_fingerprint": config_fingerprint,
                    "duration_s": duration_s,
                    "frame_width": 640,
                    "frame_height": 360,
                    "fps": 1.0,
                    "frame_count": max(1, len(observations)),
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
            onset_s = event["onset_s"]
            match_end_s = event.get(
                "match_end_s",
                min(float(onset_s) + 2.0, float(clip.get("duration_s", 10.0))),
            )
            chunks.extend(
                [
                    "[[clips.events]]",
                    f'kind = "{event.get("kind", "OBSERVED_FALL")}"',
                    f'onset_s = {onset_s}',
                    f'match_end_s = {match_end_s}',
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


@pytest.mark.parametrize(
    ("onset_s", "match_end_s"),
    [(2.0, 2.0), (2.0, 1.9), (2.0, 10.1), (2.0, 10.0000000000001)],
)
def test_manifest_requires_match_end_after_onset_within_clip(
    tmp_path: Path, onset_s: float, match_end_s: float
):
    trace = tmp_path / "trace.jsonl"
    trace_sha = _write_trace(trace, {"clip-a": (10.0, [_observation(0.0)])})
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            trace.name,
            trace_sha,
            [
                {
                    "id": "clip-a",
                    "events": [
                        {"onset_s": onset_s, "match_end_s": match_end_s}
                    ],
                }
            ],
        )
    )

    with pytest.raises(ValueError, match="match_end_s"):
        load_manifest(manifest)


def test_manifest_loads_required_event_match_end(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace_sha = _write_trace(trace, {"clip-a": (10.0, [_observation(0.0)])})
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        _manifest_text(
            trace.name,
            trace_sha,
            [
                {
                    "id": "clip-a",
                    "events": [
                        {"onset_s": 1.0, "match_end_s": 3.0}
                    ],
                }
            ],
        )
    )

    manifest = load_manifest(manifest_path)

    assert manifest.clips[0].events[0].match_end_s == 3.0


def test_manifest_rejects_event_without_match_end(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace_sha = _write_trace(trace, {"clip-a": (10.0, [_observation(0.0)])})
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        _manifest_text(
            trace.name,
            trace_sha,
            [{"id": "clip-a", "events": [{"onset_s": 1.0}]}],
        ).replace("match_end_s = 3.0\n", "")
    )

    with pytest.raises(ValueError, match="requires match_end_s"):
        load_manifest(manifest_path)


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


def test_fall_config_fingerprint_is_canonical_and_includes_rois():
    bed = FurnitureROI(
        "bed",
        ((0.1, 0.2), (0.9, 0.2), (0.9, 0.9), (0.1, 0.9)),
    )

    assert (
        evaluation.fall_config_fingerprint(FallConfig())
        == BALANCED_CONFIG_FINGERPRINT
    )
    assert evaluation.fall_config_fingerprint(
        replace(FallConfig(), furniture_rois=(bed,))
    ) != evaluation.fall_config_fingerprint(FallConfig())


def test_replay_rejects_fall_config_fingerprint_mismatch_with_reextract_hint(
    tmp_path: Path,
):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, {"clip-a": (1.0, [_observation(0.0)])})

    with pytest.raises(ValueError, match="fall config fingerprint.*re-extract"):
        replay_trace(trace, "temporal-fsm", load_fall_config(profile="precision"))


def test_replay_reports_model_hash_and_rejects_cross_clip_provenance_mismatch(
    tmp_path: Path,
):
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        {"clip-a": (1.0, [_observation(0.0)]), "clip-b": (1.0, [_observation(0.0)])},
    )
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    second_header = next(
        record
        for record in records
        if record["record_type"] == "clip" and record["clip_id"] == "clip-b"
    )
    second_header["model_sha256"] = "c" * 64
    trace.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(ValueError, match="model provenance.*consistent"):
        replay_trace(trace, "temporal-fsm", FallConfig())

    second_header["model_sha256"] = MODEL_SHA
    trace.write_text("".join(json.dumps(record) + "\n" for record in records))
    replay = replay_trace(trace, "temporal-fsm", FallConfig())
    assert replay["model_sha256"] == MODEL_SHA
    assert replay["fall_config_fingerprint"] == BALANCED_CONFIG_FINGERPRINT


def test_roi_bearing_config_replays_persistent_posture_as_bed_rest(tmp_path: Path):
    bed = FurnitureROI(
        "bed",
        ((0.1, 0.2), (0.9, 0.2), (0.9, 0.9), (0.1, 0.9)),
    )
    fall_config = replace(FallConfig(), furniture_rois=(bed,))
    observations = [
        _observation(t_seconds)
        for t_seconds in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.6)
    ]
    for observation in observations:
        observation["features"] = _features(
            float(observation["t_seconds"]),
            torso_angle_deg=60.0,
            bbox_aspect_ratio=1.2,
            furniture_roi="bed",
        )
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        {"clip-a": (3.0, observations)},
        fall_config=fall_config,
    )

    replay = replay_trace(trace, "temporal-fsm", fall_config)

    incidents = replay["clips"][0]["incidents"]
    assert len(incidents) == 1
    assert incidents[0]["kind"] == "BED_REST"


def test_event_metrics_are_aggregated_from_executed_trace_replay(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    fall_config = FallConfig(recovery_dwell_s=1.0, max_observation_gap_s=1.0)
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
        fall_config=fall_config,
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
        fall_config,
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


def test_same_kind_alert_after_label_match_end_is_a_miss_and_false_positive(
    tmp_path: Path,
):
    trace = tmp_path / "trace.jsonl"
    trace_sha = _write_trace(
        trace,
        {
            "clip-a": (
                10.0,
                [_observation(0.0), _observation(8.0, downward_speed=1.0)],
            )
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
                    "events": [
                        {"onset_s": 1.0, "match_end_s": 3.0}
                    ],
                }
            ],
        )
    )

    report = evaluate_manifest(manifest, "relaxed-or", FallConfig())

    assert report["event_counts"] == {
        "labelled": 1,
        "detected": 1,
        "true_positive": 0,
        "false_positive": 1,
        "missed": 1,
    }
    assert report["events"][0]["matched"] is False


def test_replay_rejects_non_finite_feature_values(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "clip",
                "clip_id": "clip-a",
                "source_sha256": SOURCE_SHA,
                "model_sha256": MODEL_SHA,
                "fall_config_fingerprint": BALANCED_CONFIG_FINGERPRINT,
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


def _three_frame_trace(path: Path) -> list[dict[str, object]]:
    _write_trace(
        path,
        {
            "clip-a": (
                3.0,
                [_observation(0.0), _observation(1.0), _observation(2.0)],
            )
        },
    )
    return [json.loads(line) for line in path.read_text().splitlines()]


def _replace_trace_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_trace_rejects_missing_decoded_frame_observation(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    records = _three_frame_trace(trace)
    records = [
        record
        for record in records
        if record.get("record_type") != "observation"
        or record.get("frame_index") != 1
    ]
    _replace_trace_records(trace, records)

    with pytest.raises(ValueError, match="missing frame indices.*1"):
        replay_trace(trace, "temporal-fsm", FallConfig())


def test_trace_rejects_duplicate_frame_person_observation_key(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    records = _three_frame_trace(trace)
    records.insert(2, dict(records[1]))
    _replace_trace_records(trace, records)

    with pytest.raises(ValueError, match="duplicate.*frame_index=0.*person_id=1"):
        replay_trace(trace, "temporal-fsm", FallConfig())


def test_trace_rejects_frame_index_outside_declared_count(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    records = _three_frame_trace(trace)
    records[-1]["frame_index"] = 3
    _replace_trace_records(trace, records)

    with pytest.raises(ValueError, match="frame_index.*outside.*frame_count"):
        replay_trace(trace, "temporal-fsm", FallConfig())


def test_trace_rejects_inconsistent_timestamps_for_same_frame(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    records = _three_frame_trace(trace)
    second_identity = dict(records[1])
    second_identity["person_id"] = 2
    second_identity["t_seconds"] = 0.1
    second_identity["features"] = _features(0.1)
    records.insert(2, second_identity)
    _replace_trace_records(trace, records)

    with pytest.raises(ValueError, match="frame.*timestamp.*consistent"):
        replay_trace(trace, "temporal-fsm", FallConfig())


def test_trace_rejects_nonmonotonic_frame_timestamps(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    records = _three_frame_trace(trace)
    records[2]["t_seconds"] = 2.0
    records[2]["features"] = _features(2.0)
    records[3]["t_seconds"] = 1.0
    records[3]["features"] = _features(1.0)
    _replace_trace_records(trace, records)

    with pytest.raises(ValueError, match="timestamps must be non-decreasing"):
        replay_trace(trace, "temporal-fsm", FallConfig())


@pytest.mark.parametrize("t_seconds", [-0.1, 3.1, 3.0000000000001])
def test_trace_rejects_timestamp_outside_clip_duration(
    tmp_path: Path, t_seconds: float
):
    trace = tmp_path / "trace.jsonl"
    records = _three_frame_trace(trace)
    records[1]["t_seconds"] = t_seconds
    records[1]["features"] = _features(t_seconds)
    _replace_trace_records(trace, records)

    with pytest.raises(ValueError, match="t_seconds.*clip duration"):
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
