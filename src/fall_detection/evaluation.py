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
from typing import Any, Iterable, Mapping

from .fall_config import FallConfig, load_fall_config
from .fall_evidence import FallEvidence, FallFeatures, classify_evidence
from .fall_fsm import FallAlertKind, FallEvidenceLevel, FallState, PersonFallFSM

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 2
_SUPPORTED_TRACE_SCHEMA_VERSIONS = frozenset((1, TRACE_SCHEMA_VERSION))
REPLAY_STRATEGIES = ("legacy-and", "relaxed-or", "k-of-n", "temporal-fsm")
_FEATURE_FIELDS = {field.name for field in dataclasses.fields(FallFeatures)}
_V1_FEATURE_FIELDS = _FEATURE_FIELDS - {"motion_available"}
_SHA256_LENGTH = 64
_EPSILON = 1e-12


@dataclass(frozen=True)
class LabelledEvent:
    kind: FallAlertKind
    onset_s: float
    match_end_s: float
    recovered: bool


@dataclass(frozen=True)
class ManifestClip:
    clip_id: str
    source: Path
    source_sha256: str
    subject: str
    trial: str
    camera: str
    split: str | None
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
    model_sha256: str
    fall_config_fingerprint: str
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


def _fall_config_canonical_record(config: FallConfig) -> dict[str, object]:
    values: dict[str, object] = {}
    for config_field in dataclasses.fields(config):
        value = getattr(config, config_field.name)
        if config_field.name == "profile":
            value = config.profile.value
        elif config_field.name == "furniture_rois":
            value = [
                {
                    "name": roi.name,
                    "points": [[x, y] for x, y in roi.points],
                }
                for roi in config.furniture_rois
            ]
        values[config_field.name] = value
    return {"schema_version": 1, "fall_config": values}


def fall_config_fingerprint(config: FallConfig) -> str:
    """Return SHA-256 of the exact canonical schema-v1 FallConfig record."""
    canonical = json.dumps(
        _fall_config_canonical_record(config),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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
                "duration_s",
            },
            optional={"events", "split"},
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
        split = (
            _nonempty_string(clip["split"], f"{name}.split")
            if "split" in clip
            else None
        )
        if split is not None:
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
                required={"kind", "onset_s", "match_end_s", "recovered"},
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
            if onset_s > duration_s:
                raise ValueError(f"{event_name}.onset_s exceeds clip duration")
            match_end_s = _finite_number(
                event["match_end_s"],
                f"{event_name}.match_end_s",
                minimum=0.0,
            )
            if match_end_s <= onset_s:
                raise ValueError(
                    f"{event_name}.match_end_s must be greater than onset_s"
                )
            if match_end_s > duration_s:
                raise ValueError(f"{event_name}.match_end_s exceeds clip duration")
            if type(event["recovered"]) is not bool:
                raise ValueError(f"{event_name}.recovered must be a boolean")
            events.append(
                LabelledEvent(kind, onset_s, match_end_s, event["recovered"])
            )
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

    declared_split_count = sum(clip.split is not None for clip in clips)
    if declared_split_count not in (0, len(clips)):
        raise ValueError("manifest clips must either all declare split or all omit it")

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


def _features_from_record(
    value: object,
    context: str,
    schema_version: int,
) -> FallFeatures:
    if not isinstance(value, dict):
        raise ValueError(f"{context}.features must be an object")
    expected_fields = (
        _V1_FEATURE_FIELDS if schema_version == 1 else _FEATURE_FIELDS
    )
    if set(value) != expected_fields:
        missing = expected_fields - set(value)
        unknown = set(value) - expected_fields
        detail = (
            f"missing {sorted(missing)[0]}"
            if missing
            else f"unknown key {sorted(unknown)[0]}"
        )
        raise ValueError(f"{context}.features {detail}")
    numeric_names = expected_fields - {
        "valid",
        "motion_available",
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
    if schema_version == 1:
        motion_available = False
    else:
        if type(value["motion_available"]) is not bool:
            raise ValueError(
                f"{context}.features.motion_available must be a boolean"
            )
        motion_available = value["motion_available"]
    centroid = value["torso_centroid"]
    if not isinstance(centroid, list) or len(centroid) != 2:
        raise ValueError(f"{context}.features.torso_centroid must be [x, y]")
    parsed.update(
        valid=value["valid"],
        motion_available=motion_available,
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
    observation_keys: dict[str, set[tuple[int, int | None]]] = defaultdict(set)
    frame_timestamps: dict[str, dict[int, float]] = defaultdict(dict)
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank trace record at {trace_path}:{line_number}")
        record = _load_json_line(line, trace_path, line_number)
        context = f"trace record {line_number}"
        schema_version = record.get("schema_version")
        if (
            type(schema_version) is not int
            or schema_version not in _SUPPORTED_TRACE_SCHEMA_VERSIONS
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
                    "model_sha256",
                    "fall_config_fingerprint",
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
            model_sha256 = _sha256(
                record["model_sha256"], f"{context}.model_sha256"
            )
            config_fingerprint = _sha256(
                record["fall_config_fingerprint"],
                f"{context}.fall_config_fingerprint",
            )
            if headers:
                first_header = next(iter(headers.values()))
                if model_sha256 != first_header["model_sha256"]:
                    raise ValueError(
                        "trace model provenance must be consistent across clips"
                    )
                if config_fingerprint != first_header["fall_config_fingerprint"]:
                    raise ValueError(
                        "trace fall config provenance must be consistent across clips"
                    )
            duration_s = _finite_number(
                record["duration_s"], f"{context}.duration_s", minimum=0.0
            )
            if duration_s <= 0.0:
                raise ValueError(f"{context}.duration_s must be positive")
            headers[clip_id] = {
                "schema_version": schema_version,
                "source_sha256": source_sha256,
                "model_sha256": model_sha256,
                "fall_config_fingerprint": config_fingerprint,
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
            if schema_version != headers[clip_id]["schema_version"]:
                raise ValueError(
                    f"{context} schema_version differs from clip header"
                )
            if source_sha256 != headers[clip_id]["source_sha256"]:
                raise ValueError(f"{context} source checksum differs from clip header")
            frame_index = record["frame_index"]
            if type(frame_index) is not int or frame_index < 0:
                raise ValueError(f"{context}.frame_index must be a non-negative integer")
            frame_count = int(headers[clip_id]["frame_count"])
            if frame_index >= frame_count:
                raise ValueError(
                    f"{context}.frame_index={frame_index} is outside "
                    f"declared frame_count={frame_count}"
                )
            t_seconds = _finite_number(record["t_seconds"], f"{context}.t_seconds")
            duration_s = float(headers[clip_id]["duration_s"])
            if t_seconds < 0.0 or t_seconds > duration_s:
                raise ValueError(
                    f"{context}.t_seconds must be inside clip duration "
                    f"[0, {duration_s}]"
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
                features = _features_from_record(
                    features_value,
                    context,
                    int(schema_version),
                )
                if abs(features.t_seconds - t_seconds) > _EPSILON:
                    raise ValueError(f"{context} feature timestamp differs from observation")
            observation_key = (frame_index, person_id)
            if observation_key in observation_keys[clip_id]:
                raise ValueError(
                    f"duplicate observation key in {clip_id}: "
                    f"frame_index={frame_index}, person_id={person_id}"
                )
            observation_keys[clip_id].add(observation_key)
            existing_frame_timestamp = frame_timestamps[clip_id].get(frame_index)
            if (
                existing_frame_timestamp is not None
                and abs(existing_frame_timestamp - t_seconds) > _EPSILON
            ):
                raise ValueError(
                    f"frame {frame_index} timestamps must be consistent in clip {clip_id}"
                )
            frame_timestamps[clip_id][frame_index] = t_seconds
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
        frame_count = int(header["frame_count"])
        missing_frame_indices = sorted(
            set(range(frame_count)) - set(frame_timestamps[clip_id])
        )
        if missing_frame_indices:
            raise ValueError(
                f"trace clip {clip_id} missing frame indices: "
                + ", ".join(str(index) for index in missing_frame_indices)
            )
        ordered_timestamps = [
            frame_timestamps[clip_id][index] for index in range(frame_count)
        ]
        if any(
            later + _EPSILON < earlier
            for earlier, later in itertools.pairwise(ordered_timestamps)
        ):
            raise ValueError(
                f"trace clip {clip_id} frame timestamps must be non-decreasing"
            )
        clips.append(
            _TraceClip(
                clip_id=clip_id,
                source_sha256=str(header["source_sha256"]),
                model_sha256=str(header["model_sha256"]),
                fall_config_fingerprint=str(header["fall_config_fingerprint"]),
                duration_s=float(header["duration_s"]),
                frame_width=int(header["frame_width"]),
                frame_height=int(header["frame_height"]),
                fps=float(header["fps"]),
                frame_count=frame_count,
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
    last_observed_times: dict[int, float] = {}
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
        for person_id, machine in list(machines.items()):
            previous_at = last_times[person_id]
            dwell[machine.state.name] += max(0.0, t_seconds - previous_at)
            last_times[person_id] = t_seconds
            if t_seconds - last_observed_times[person_id] > config.identity_timeout_s:
                del machines[person_id]
                del last_times[person_id]
                del last_observed_times[person_id]
                active.pop(person_id, None)
                continue
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
            last_observed_times[person_id] = t_seconds
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
        "model_sha256": clip.model_sha256,
        "fall_config_fingerprint": clip.fall_config_fingerprint,
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
    expected_config_fingerprint = fall_config_fingerprint(selected_config)
    if clips and clips[0].fall_config_fingerprint != expected_config_fingerprint:
        raise ValueError(
            "trace fall config fingerprint "
            f"{clips[0].fall_config_fingerprint} does not match replay config "
            f"{expected_config_fingerprint}; re-extract the trace with this config"
        )
    logger.info("replaying %d clips with %s", len(clips), strategy)
    return {
        "schema_version": 1,
        "strategy": strategy,
        "model_sha256": clips[0].model_sha256 if clips else None,
        "fall_config_fingerprint": expected_config_fingerprint,
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


def binary_classification_summary(
    classifications: Iterable[tuple[bool, bool]],
) -> dict[str, dict[str, int] | dict[str, float]]:
    """Aggregate clip-level ground-truth/prediction pairs into standard metrics."""
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for actual_positive, predicted_positive in classifications:
        if actual_positive and predicted_positive:
            counts["TP"] += 1
        elif predicted_positive:
            counts["FP"] += 1
        elif actual_positive:
            counts["FN"] += 1
        else:
            counts["TN"] += 1
    total = sum(counts.values())
    return {
        "clip_confusion": counts,
        "classification_metrics": {
            "accuracy": _ratio(counts["TP"] + counts["TN"], total),
            "precision": _ratio(counts["TP"], counts["TP"] + counts["FP"]),
            "recall": _ratio(counts["TP"], counts["TP"] + counts["FN"]),
            "f1_score": _ratio(
                2 * counts["TP"],
                2 * counts["TP"] + counts["FP"] + counts["FN"],
            ),
        },
    }


def _maximum_cardinality_event_matches(
    events: tuple[LabelledEvent, ...],
    incidents: list[Mapping[str, object]],
) -> dict[int, int]:
    """Match compatible incidents one-to-one without greedy overlap losses."""
    candidates: dict[int, tuple[int, ...]] = {}
    for event_index, event in enumerate(events):
        candidates[event_index] = tuple(
            incident_index
            for incident_index, incident in sorted(
                enumerate(incidents),
                key=lambda item: (float(item[1]["detected_at"]), item[0]),
            )
            if incident["kind"] == event.kind.value
            and event.onset_s <= float(incident["detected_at"]) <= event.match_end_s
        )

    incident_owner: dict[int, int] = {}
    event_match: dict[int, int] = {}

    def augment(event_index: int, visited_incidents: set[int]) -> bool:
        for incident_index in candidates[event_index]:
            if incident_index in visited_incidents:
                continue
            visited_incidents.add(incident_index)
            owner = incident_owner.get(incident_index)
            if owner is not None and not augment(owner, visited_incidents):
                continue
            incident_owner[incident_index] = event_index
            event_match[event_index] = incident_index
            return True
        return False

    for event_index in range(len(events)):
        augment(event_index, set())
    return event_match


def evaluate_manifest(
    path: str | Path,
    strategy: str,
    config: FallConfig | None = None,
    *,
    split: str | None = None,
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

    available_splits = sorted(
        {clip.split for clip in manifest.clips if clip.split is not None}
    )
    if split is not None:
        selected_split = _nonempty_string(split, "split")
        if selected_split not in available_splits:
            if not available_splits:
                raise ValueError("manifest has no declared splits; omit split selection")
            raise ValueError(
                f"unknown split {selected_split!r}; expected one of "
                + ", ".join(available_splits)
            )
    elif len(available_splits) > 1:
        raise ValueError(
            "manifest contains multiple splits; select one explicitly with split/--split"
        )
    else:
        selected_split = available_splits[0] if available_splits else None
    selected_clips = tuple(
        clip
        for clip in manifest.clips
        if selected_split is None or clip.split == selected_split
    )

    labelled_count = 0
    detected_count = 0
    true_positive = 0
    false_positive = 0
    missed = 0
    clip_classifications: list[tuple[bool, bool]] = []
    alert_latencies: list[float] = []
    recovery_times: list[float] = []
    event_results: list[dict[str, object]] = []
    dwell: defaultdict[str, float] = defaultdict(float)
    per_clip: list[dict[str, object]] = []

    for manifest_clip in selected_clips:
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
        actual_positive = bool(manifest_clip.events)
        predicted_positive = bool(incidents)
        clip_classifications.append((actual_positive, predicted_positive))
        detected_count += len(incidents)
        matches = _maximum_cardinality_event_matches(manifest_clip.events, incidents)
        used_incidents = set(matches.values())
        clip_event_results: list[dict[str, object]] = []
        for event_index, event in enumerate(manifest_clip.events):
            labelled_count += 1
            match_index = matches.get(event_index)
            if match_index is None:
                missed += 1
                event_result = {
                    "clip_id": manifest_clip.clip_id,
                    "event_index": event_index,
                    "kind": event.kind.value,
                    "onset_s": event.onset_s,
                    "match_end_s": event.match_end_s,
                    "expected_recovery": event.recovered,
                    "matched": False,
                    "detected_at": None,
                    "latency_s": None,
                    "recovered_at": None,
                    "recovery_time_s": None,
                }
            else:
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
                    "match_end_s": event.match_end_s,
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

    total_hours = sum(clip.duration_s for clip in selected_clips) / 3600.0
    classification_summary = binary_classification_summary(clip_classifications)
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
        "split": selected_split,
        "available_splits": available_splits,
        "trace_sha256": actual_trace_sha256,
        "model_sha256": replay["model_sha256"],
        "fall_config_fingerprint": replay["fall_config_fingerprint"],
        "event_counts": {
            "labelled": labelled_count,
            "detected": detected_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "missed": missed,
        },
        **classification_summary,
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
    parser.add_argument(
        "--split",
        help="evaluate one frozen manifest split (required when multiple exist)",
    )
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
        report = evaluate_manifest(
            args.manifest,
            args.strategy,
            config,
            split=args.split,
        )
        sys.stdout.write(json.dumps(report, allow_nan=False, sort_keys=True) + "\n")
    except (OSError, ValueError) as error:
        logger.error("evaluation failed: %s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
