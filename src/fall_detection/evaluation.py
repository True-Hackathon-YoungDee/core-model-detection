"""Deterministic replay and event-level evaluation for fall feature traces."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import logging
import math
import statistics
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .fall_config import FallConfig, load_fall_config
from .fall_evidence import FallEvidence, FallFeatures, classify_evidence
from .fall_fsm import FallAlertKind, FallEvidenceLevel, FallState, PersonFallFSM

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 1
REPLAY_STRATEGIES = ("legacy-and", "relaxed-or", "k-of-n", "temporal-fsm")
_FEATURE_FIELDS = {field.name for field in dataclasses.fields(FallFeatures)}
_SHA256_LENGTH = 64
_EPSILON = 1e-12


@dataclass(frozen=True)
class LabelledEvent:
    kind: FallAlertKind
    onset_s: float
    recovered: bool


@dataclass(frozen=True)
class ManifestClip:
    clip_id: str
    source: Path
    source_sha256: str
    subject: str
    trial: str
    camera: str
    split: str
    duration_s: float
    events: tuple[LabelledEvent, ...]


@dataclass(frozen=True)
class EvaluationManifest:
    path: Path
    dataset: str
    trace: Path
    trace_sha256: str
    clips: tuple[ManifestClip, ...]


@dataclass(frozen=True)
class _TraceObservation:
    frame_index: int
    t_seconds: float
    person_id: int | None
    features: FallFeatures | None


@dataclass(frozen=True)
class _TraceClip:
    clip_id: str
    source_sha256: str
    duration_s: float
    frame_width: int
    frame_height: int
    fps: float
    frame_count: int
    observations: tuple[_TraceObservation, ...]


@dataclass
class _Incident:
    incident_id: str
    person_id: int
    kind: FallAlertKind
    evidence_level: FallEvidenceLevel
    detected_at: float
    recovered_at: float | None = None

    def record(self) -> dict[str, object]:
        return {
            "incident_id": self.incident_id,
            "person_id": self.person_id,
            "kind": self.kind.value,
            "evidence_level": self.evidence_level.value,
            "detected_at": self.detected_at,
            "recovered_at": self.recovered_at,
        }


def _table(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_number(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, name: str) -> str:
    checksum = _nonempty_string(value, name).lower()
    if len(checksum) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in checksum
    ):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return checksum


def _relative_path(value: object, name: str) -> Path:
    raw = _nonempty_string(value, name)
    path = Path(raw)
    if path.is_absolute():
        raise ValueError(f"{name} must be a portable relative path")
    return path


def _require_exact_keys(
    table: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    name: str,
) -> None:
    missing = required - set(table)
    if missing:
        raise ValueError(f"{name} requires {sorted(missing)[0]}")
    unknown = set(table) - required - optional
    if unknown:
        raise ValueError(f"unknown {name} key: {sorted(unknown)[0]}")


def load_manifest(path: str | Path) -> EvaluationManifest:
    """Load and structurally validate a schema-v1 evaluation manifest."""
    manifest_path = Path(path)
    try:
        with manifest_path.open("rb") as manifest_file:
            document = tomllib.load(manifest_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot load manifest {manifest_path}: {error}") from error

    _require_exact_keys(
        document,
        required={"schema_version", "dataset", "trace", "trace_sha256", "clips"},
        optional=set(),
        name="manifest",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported manifest schema_version: {document['schema_version']!r}"
        )
    dataset = _nonempty_string(document["dataset"], "dataset")
    trace = _relative_path(document["trace"], "trace")
    trace_sha256 = _sha256(document["trace_sha256"], "trace_sha256")
    raw_clips = document["clips"]
    if not isinstance(raw_clips, list):
        raise ValueError("clips must be an array of TOML tables")

    clips: list[ManifestClip] = []
    seen_ids: set[str] = set()
    subject_splits: dict[str, str] = {}
    for index, raw_clip in enumerate(raw_clips):
        name = f"clips[{index}]"
        clip = _table(raw_clip, name)
        _require_exact_keys(
            clip,
            required={
                "id",
                "source",
                "source_sha256",
                "subject",
                "trial",
                "camera",
                "split",
                "duration_s",
            },
            optional={"events"},
            name=name,
        )
        clip_id = _nonempty_string(clip["id"], f"{name}.id")
        if clip_id in seen_ids:
            raise ValueError(f"duplicate clip id: {clip_id}")
        seen_ids.add(clip_id)
        source = _relative_path(clip["source"], f"{name}.source")
        source_sha256 = _sha256(
            clip["source_sha256"], f"{name}.source_sha256"
        )
        subject = _nonempty_string(clip["subject"], f"{name}.subject")
        trial = _nonempty_string(clip["trial"], f"{name}.trial")
        camera = _nonempty_string(clip["camera"], f"{name}.camera")
        split = _nonempty_string(clip["split"], f"{name}.split")
        previous_split = subject_splits.setdefault(subject, split)
        if previous_split != split:
            raise ValueError(
                f"subject group {subject!r} leaks across splits "
                f"{previous_split!r} and {split!r}"
            )
        duration_s = _finite_number(
            clip["duration_s"], f"{name}.duration_s", minimum=0.0
        )
        if duration_s <= 0.0:
            raise ValueError(f"{name}.duration_s must be positive")

        raw_events = clip.get("events", [])
        if not isinstance(raw_events, list):
            raise ValueError(f"{name}.events must be an array of TOML tables")
        events: list[LabelledEvent] = []
        for event_index, raw_event in enumerate(raw_events):
            event_name = f"{name}.events[{event_index}]"
            event = _table(raw_event, event_name)
            _require_exact_keys(
                event,
                required={"kind", "onset_s", "recovered"},
                optional=set(),
                name=event_name,
            )
            try:
                kind = FallAlertKind(
                    _nonempty_string(event["kind"], f"{event_name}.kind")
                )
            except ValueError as error:
                raise ValueError(f"{event_name}.kind is unknown") from error
            onset_s = _finite_number(
                event["onset_s"], f"{event_name}.onset_s", minimum=0.0
            )
            if onset_s > duration_s + _EPSILON:
                raise ValueError(f"{event_name}.onset_s exceeds clip duration")
            if type(event["recovered"]) is not bool:
                raise ValueError(f"{event_name}.recovered must be a boolean")
            events.append(LabelledEvent(kind, onset_s, event["recovered"]))
        if any(
            later.onset_s <= earlier.onset_s
            for earlier, later in itertools.pairwise(events)
        ):
            raise ValueError(f"{name}.events must have strictly increasing onset_s")
        clips.append(
            ManifestClip(
                clip_id=clip_id,
                source=source,
                source_sha256=source_sha256,
                subject=subject,
                trial=trial,
                camera=camera,
                split=split,
                duration_s=duration_s,
                events=tuple(events),
            )
        )

    return EvaluationManifest(
        path=manifest_path.resolve(),
        dataset=dataset,
        trace=(manifest_path.parent / trace).resolve(),
        trace_sha256=trace_sha256,
        clips=tuple(clips),
    )


def _load_json_line(line: str, path: Path, line_number: int) -> Mapping[str, Any]:
    try:
        record = json.loads(
            line,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"invalid trace JSON at {path}:{line_number}: {error}"
        ) from error
    if not isinstance(record, dict):
        raise ValueError(f"trace record at {path}:{line_number} must be an object")
    return record


def _features_from_record(value: object, context: str) -> FallFeatures:
    if not isinstance(value, dict):
        raise ValueError(f"{context}.features must be an object")
    if set(value) != _FEATURE_FIELDS:
        missing = _FEATURE_FIELDS - set(value)
        unknown = set(value) - _FEATURE_FIELDS
        detail = (
            f"missing {sorted(missing)[0]}"
            if missing
            else f"unknown key {sorted(unknown)[0]}"
        )
        raise ValueError(f"{context}.features {detail}")
    numeric_names = _FEATURE_FIELDS - {
        "valid",
        "torso_centroid",
        "furniture_roi",
        "scale_source",
    }
    parsed: dict[str, object] = {
        name: _finite_number(value[name], f"{context}.features.{name}")
        for name in numeric_names
    }
    if type(value["valid"]) is not bool:
        raise ValueError(f"{context}.features.valid must be a boolean")
    centroid = value["torso_centroid"]
    if not isinstance(centroid, list) or len(centroid) != 2:
        raise ValueError(f"{context}.features.torso_centroid must be [x, y]")
    parsed.update(
        valid=value["valid"],
        torso_centroid=(
            _finite_number(centroid[0], f"{context}.features.torso_centroid[0]"),
            _finite_number(centroid[1], f"{context}.features.torso_centroid[1]"),
        ),
        furniture_roi=(
            None
            if value["furniture_roi"] is None
            else _nonempty_string(
                value["furniture_roi"], f"{context}.features.furniture_roi"
            )
        ),
        scale_source=_nonempty_string(
            value["scale_source"], f"{context}.features.scale_source"
        ),
    )
    return FallFeatures(**parsed)  # type: ignore[arg-type]


def _read_trace(path: str | Path) -> tuple[_TraceClip, ...]:
    trace_path = Path(path)
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"trace file does not exist or cannot be read: {trace_path}") from error
    if not lines:
        raise ValueError(f"trace file is empty: {trace_path}")

    headers: dict[str, dict[str, object]] = {}
    observations: dict[str, list[_TraceObservation]] = defaultdict(list)
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank trace record at {trace_path}:{line_number}")
        record = _load_json_line(line, trace_path, line_number)
        context = f"trace record {line_number}"
        if (
            type(record.get("schema_version")) is not int
            or record.get("schema_version") != TRACE_SCHEMA_VERSION
        ):
            raise ValueError(f"{context} has unsupported schema_version")
        record_type = record.get("record_type")
        clip_id = _nonempty_string(record.get("clip_id"), f"{context}.clip_id")
        source_sha256 = _sha256(
            record.get("source_sha256"), f"{context}.source_sha256"
        )
        if record_type == "clip":
            _require_exact_keys(
                record,
                required={
                    "schema_version",
                    "record_type",
                    "clip_id",
                    "source_sha256",
                    "duration_s",
                    "frame_width",
                    "frame_height",
                    "fps",
                    "frame_count",
                },
                optional=set(),
                name=context,
            )
            if clip_id in headers:
                raise ValueError(f"duplicate trace clip header: {clip_id}")
            duration_s = _finite_number(
                record["duration_s"], f"{context}.duration_s", minimum=0.0
            )
            if duration_s <= 0.0:
                raise ValueError(f"{context}.duration_s must be positive")
            headers[clip_id] = {
                "source_sha256": source_sha256,
                "duration_s": duration_s,
                "frame_width": _positive_int(
                    record["frame_width"], f"{context}.frame_width"
                ),
                "frame_height": _positive_int(
                    record["frame_height"], f"{context}.frame_height"
                ),
                "fps": _finite_number(record["fps"], f"{context}.fps", minimum=0.0),
                "frame_count": _positive_int(
                    record["frame_count"], f"{context}.frame_count"
                ),
            }
            if headers[clip_id]["fps"] <= 0.0:
                raise ValueError(f"{context}.fps must be positive")
        elif record_type == "observation":
            _require_exact_keys(
                record,
                required={
                    "schema_version",
                    "record_type",
                    "clip_id",
                    "source_sha256",
                    "frame_index",
                    "t_seconds",
                    "person_id",
                    "features",
                },
                optional=set(),
                name=context,
            )
            if clip_id not in headers:
                raise ValueError(f"{context} appears before clip header {clip_id!r}")
            if source_sha256 != headers[clip_id]["source_sha256"]:
                raise ValueError(f"{context} source checksum differs from clip header")
            frame_index = record["frame_index"]
            if type(frame_index) is not int or frame_index < 0:
                raise ValueError(f"{context}.frame_index must be a non-negative integer")
            t_seconds = _finite_number(
                record["t_seconds"], f"{context}.t_seconds", minimum=0.0
            )
            person_id = record["person_id"]
            features_value = record["features"]
            if person_id is None:
                if features_value is not None:
                    raise ValueError(f"{context}.features must be null without a person")
                features = None
            else:
                if type(person_id) is not int or person_id < 0:
                    raise ValueError(f"{context}.person_id must be a non-negative integer or null")
                features = _features_from_record(features_value, context)
                if abs(features.t_seconds - t_seconds) > _EPSILON:
                    raise ValueError(f"{context} feature timestamp differs from observation")
            previous = observations[clip_id][-1] if observations[clip_id] else None
            if previous is not None and t_seconds + _EPSILON < previous.t_seconds:
                raise ValueError(f"{context} timestamps must be non-decreasing")
            observations[clip_id].append(
                _TraceObservation(frame_index, t_seconds, person_id, features)
            )
        else:
            raise ValueError(f"{context}.record_type must be 'clip' or 'observation'")

    clips: list[_TraceClip] = []
    for clip_id, header in headers.items():
        clips.append(
            _TraceClip(
                clip_id=clip_id,
                source_sha256=str(header["source_sha256"]),
                duration_s=float(header["duration_s"]),
                frame_width=int(header["frame_width"]),
                frame_height=int(header["frame_height"]),
                fps=float(header["fps"]),
                frame_count=int(header["frame_count"]),
                observations=tuple(observations[clip_id]),
            )
        )
    return tuple(clips)


class _TemporalMachine:
    def __init__(self, config: FallConfig) -> None:
        self.fsm = PersonFallFSM(config)

    @property
    def state(self) -> FallState:
        return self.fsm.state

    def step(
        self, features: FallFeatures | None, t_seconds: float
    ) -> tuple[FallAlertKind | None, FallEvidenceLevel | None, bool]:
        decision = (
            self.fsm.step(features)
            if features is not None
            else self.fsm.observe_gap(t_seconds)
        )
        return decision.alert_kind, decision.evidence_level, decision.recovered


class _VoteMachine:
    def __init__(self, strategy: str, config: FallConfig) -> None:
        self.strategy = strategy
        self.config = config
        self.state = FallState.UPRIGHT
        self._active = False
        self._recovery_last_at: float | None = None
        self._recovery_elapsed_s = 0.0

    def _triggered(self, evidence: FallEvidence) -> bool:
        votes = (
            evidence.dynamic_torso_angle,
            evidence.downward_motion,
            evidence.rapid_torso_rotation,
            evidence.height_collapse,
            evidence.posture,
        )
        if self.strategy == "legacy-and":
            return all(votes) and evidence.stillness
        if self.strategy == "relaxed-or":
            return any(votes)
        return sum(votes) >= 3

    def step(
        self, features: FallFeatures | None, t_seconds: float
    ) -> tuple[FallAlertKind | None, FallEvidenceLevel | None, bool]:
        if features is None:
            self._clear_recovery()
            return None, None, False
        evidence = classify_evidence(features, self.config)
        if not self._active:
            if self._triggered(evidence):
                self._active = True
                self.state = FallState.FALL_CONFIRMED
                return FallAlertKind.OBSERVED_FALL, FallEvidenceLevel.HIGH, False
            return None, None, False

        upright = (
            evidence.quality_ok
            and features.torso_angle_deg <= self.config.recovery_torso_angle_deg
        )
        if not upright:
            self._clear_recovery()
            return None, None, False
        if self._recovery_last_at is None:
            self._recovery_last_at = t_seconds
            return None, None, False
        interval_s = t_seconds - self._recovery_last_at
        if 0.0 < interval_s <= self.config.max_observation_gap_s + _EPSILON:
            self._recovery_elapsed_s += interval_s
        else:
            self._recovery_elapsed_s = 0.0
        self._recovery_last_at = t_seconds
        if self._recovery_elapsed_s + _EPSILON < self.config.recovery_dwell_s:
            return None, None, False
        self._active = False
        self.state = FallState.UPRIGHT
        self._clear_recovery()
        return FallAlertKind.OBSERVED_FALL, FallEvidenceLevel.HIGH, True

    def _clear_recovery(self) -> None:
        self._recovery_last_at = None
        self._recovery_elapsed_s = 0.0


def _new_machine(strategy: str, config: FallConfig) -> _TemporalMachine | _VoteMachine:
    if strategy == "temporal-fsm":
        return _TemporalMachine(config)
    return _VoteMachine(strategy, config)


def _replay_clip(clip: _TraceClip, strategy: str, config: FallConfig) -> dict[str, object]:
    machines: dict[int, _TemporalMachine | _VoteMachine] = {}
    last_times: dict[int, float] = {}
    dwell: defaultdict[str, float] = defaultdict(float)
    incidents: list[_Incident] = []
    active: dict[int, _Incident] = {}
    sequences: defaultdict[int, int] = defaultdict(int)

    grouped = itertools.groupby(clip.observations, key=lambda observation: observation.t_seconds)
    for t_seconds, group_iterator in grouped:
        group = tuple(group_iterator)
        observed_ids = {
            observation.person_id
            for observation in group
            if observation.person_id is not None
        }
        for person_id, machine in machines.items():
            previous_at = last_times[person_id]
            dwell[machine.state.name] += max(0.0, t_seconds - previous_at)
            last_times[person_id] = t_seconds
            if person_id not in observed_ids:
                machine.step(None, t_seconds)

        for observation in group:
            if observation.person_id is None:
                continue
            person_id = observation.person_id
            machine = machines.get(person_id)
            if machine is None:
                machine = _new_machine(strategy, config)
                machines[person_id] = machine
                last_times[person_id] = t_seconds
            alert_kind, evidence_level, recovered = machine.step(
                observation.features, t_seconds
            )
            if recovered:
                incident = active.pop(person_id, None)
                if incident is not None:
                    incident.recovered_at = t_seconds
            elif alert_kind is not None and evidence_level is not None:
                if person_id in active:
                    continue
                sequences[person_id] += 1
                incident = _Incident(
                    incident_id=(
                        f"{clip.clip_id}-person-{person_id}-"
                        f"{sequences[person_id]:06d}"
                    ),
                    person_id=person_id,
                    kind=alert_kind,
                    evidence_level=evidence_level,
                    detected_at=t_seconds,
                )
                incidents.append(incident)
                active[person_id] = incident

    for person_id, machine in machines.items():
        dwell[machine.state.name] += max(0.0, clip.duration_s - last_times[person_id])

    return {
        "clip_id": clip.clip_id,
        "source_sha256": clip.source_sha256,
        "duration_s": clip.duration_s,
        "incidents": [incident.record() for incident in incidents],
        "state_dwell_s": dict(sorted(dwell.items())),
    }


def replay_trace(
    path: str | Path,
    strategy: str,
    config: FallConfig | None = None,
) -> dict[str, object]:
    """Replay every clip in a finite schema-v1 feature trace."""
    if strategy not in REPLAY_STRATEGIES:
        raise ValueError(
            f"unknown replay strategy {strategy!r}; expected one of "
            + ", ".join(REPLAY_STRATEGIES)
        )
    selected_config = config or FallConfig()
    clips = _read_trace(path)
    logger.info("replaying %d clips with %s", len(clips), strategy)
    return {
        "schema_version": 1,
        "strategy": strategy,
        "clips": [
            _replay_clip(clip, strategy, selected_config) for clip in clips
        ],
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"trace file does not exist or cannot be read: {path}") from error
    return digest.hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_manifest(
    path: str | Path,
    strategy: str,
    config: FallConfig | None = None,
) -> dict[str, object]:
    """Validate a manifest and aggregate event metrics from real trace replay."""
    manifest = load_manifest(path)
    actual_trace_sha256 = _file_sha256(manifest.trace)
    if actual_trace_sha256 != manifest.trace_sha256:
        raise ValueError(
            "trace checksum mismatch: "
            f"expected {manifest.trace_sha256}, got {actual_trace_sha256}"
        )
    replay = replay_trace(manifest.trace, strategy, config)
    replay_clips = {
        str(clip["clip_id"]): clip for clip in replay["clips"]  # type: ignore[index]
    }
    manifest_ids = {clip.clip_id for clip in manifest.clips}
    trace_ids = set(replay_clips)
    if manifest_ids != trace_ids:
        missing = sorted(manifest_ids - trace_ids)
        extra = sorted(trace_ids - manifest_ids)
        raise ValueError(
            f"manifest/trace clip mismatch: missing={missing}, extra={extra}"
        )

    labelled_count = 0
    detected_count = 0
    true_positive = 0
    false_positive = 0
    missed = 0
    alert_latencies: list[float] = []
    recovery_times: list[float] = []
    event_results: list[dict[str, object]] = []
    dwell: defaultdict[str, float] = defaultdict(float)
    per_clip: list[dict[str, object]] = []

    for manifest_clip in manifest.clips:
        replay_clip = replay_clips[manifest_clip.clip_id]
        if replay_clip["source_sha256"] != manifest_clip.source_sha256:
            raise ValueError(
                f"source checksum mismatch for clip {manifest_clip.clip_id}: "
                f"manifest={manifest_clip.source_sha256}, "
                f"trace={replay_clip['source_sha256']}"
            )
        if abs(float(replay_clip["duration_s"]) - manifest_clip.duration_s) > 1e-6:
            raise ValueError(f"duration mismatch for clip {manifest_clip.clip_id}")
        incidents = list(replay_clip["incidents"])  # type: ignore[arg-type]
        detected_count += len(incidents)
        used_incidents: set[int] = set()
        clip_event_results: list[dict[str, object]] = []
        for event_index, event in enumerate(manifest_clip.events):
            labelled_count += 1
            next_onset = (
                manifest_clip.events[event_index + 1].onset_s
                if event_index + 1 < len(manifest_clip.events)
                else math.inf
            )
            match_index = next(
                (
                    index
                    for index, incident in enumerate(incidents)
                    if index not in used_incidents
                    and incident["kind"] == event.kind.value
                    and float(incident["detected_at"]) + _EPSILON >= event.onset_s
                    and float(incident["detected_at"]) < next_onset
                ),
                None,
            )
            if match_index is None:
                missed += 1
                event_result = {
                    "clip_id": manifest_clip.clip_id,
                    "event_index": event_index,
                    "kind": event.kind.value,
                    "onset_s": event.onset_s,
                    "expected_recovery": event.recovered,
                    "matched": False,
                    "detected_at": None,
                    "latency_s": None,
                    "recovered_at": None,
                    "recovery_time_s": None,
                }
            else:
                used_incidents.add(match_index)
                true_positive += 1
                incident = incidents[match_index]
                detected_at = float(incident["detected_at"])
                latency_s = detected_at - event.onset_s
                alert_latencies.append(latency_s)
                recovered_at = incident["recovered_at"]
                recovery_time_s = (
                    float(recovered_at) - detected_at
                    if recovered_at is not None
                    else None
                )
                if recovery_time_s is not None:
                    recovery_times.append(recovery_time_s)
                event_result = {
                    "clip_id": manifest_clip.clip_id,
                    "event_index": event_index,
                    "kind": event.kind.value,
                    "onset_s": event.onset_s,
                    "expected_recovery": event.recovered,
                    "matched": True,
                    "detected_at": detected_at,
                    "latency_s": latency_s,
                    "recovered_at": recovered_at,
                    "recovery_time_s": recovery_time_s,
                }
            event_results.append(event_result)
            clip_event_results.append(event_result)
        clip_false_positives = len(incidents) - len(used_incidents)
        false_positive += clip_false_positives
        for state, seconds in replay_clip["state_dwell_s"].items():  # type: ignore[union-attr]
            dwell[str(state)] += float(seconds)
        per_clip.append(
            {
                "clip_id": manifest_clip.clip_id,
                "split": manifest_clip.split,
                "duration_s": manifest_clip.duration_s,
                "incidents": incidents,
                "event_results": clip_event_results,
                "false_positives": clip_false_positives,
                "state_dwell_s": replay_clip["state_dwell_s"],
            }
        )

    total_hours = sum(clip.duration_s for clip in manifest.clips) / 3600.0
    metrics = {
        "event_sensitivity": _ratio(true_positive, labelled_count),
        "precision": _ratio(true_positive, detected_count),
        "false_alerts_per_hour": false_positive / total_hours if total_hours else 0.0,
        "miss_rate": _ratio(missed, labelled_count),
        "alert_latencies_s": alert_latencies,
        "median_alert_latency_s": (
            statistics.median(alert_latencies) if alert_latencies else None
        ),
        "recovery_times_s": recovery_times,
        "median_recovery_time_s": (
            statistics.median(recovery_times) if recovery_times else None
        ),
        "state_dwell_s": dict(sorted(dwell.items())),
    }
    return {
        "schema_version": 1,
        "dataset": manifest.dataset,
        "strategy": strategy,
        "trace_sha256": actual_trace_sha256,
        "event_counts": {
            "labelled": labelled_count,
            "detected": detected_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "missed": missed,
        },
        "metrics": metrics,
        "events": event_results,
        "clips": per_clip,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay finite fall-feature traces and report event metrics"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--strategy", choices=REPLAY_STRATEGIES, required=True)
    parser.add_argument("--fall-config", type=Path)
    parser.add_argument(
        "--fall-profile", choices=("sensitive", "balanced", "precision")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Console wrapper; library functions never write evaluation JSON to stdout."""
    args = _build_parser().parse_args(argv)
    try:
        config = load_fall_config(args.fall_config, args.fall_profile)
        report = evaluate_manifest(args.manifest, args.strategy, config)
    except ValueError as error:
        logger.error("evaluation failed: %s", error)
        return 2
    sys.stdout.write(json.dumps(report, allow_nan=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
