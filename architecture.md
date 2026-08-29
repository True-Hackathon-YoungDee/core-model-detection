# Architecture

## 1. Project Identification

| Field | Value |
| --- | --- |
| Project | `fall-detection` |
| Repository | `True-Hackathon-YoungDee/core-model-detection` |
| Maintainer | `parinya-ao` |
| Package | `src/fall_detection/` |
| Language/runtime | Python 3.13 or newer |
| Primary dependency | `mediapipe>=1.0.1` (with OpenCV and NumPy transitively) |
| Updated | 2026-08-29 |

This repository is a local Python application and library for physics-informed,
RGB-only fall detection on top of MediaPipe Pose. It accepts cameras, streams,
and video files, tracks one or more people, and emits deterministic temporal
fall decisions. It is not a clinical or medical-device system.

## 2. System Context and Scope

The system has three console interfaces defined in `pyproject.toml`:

| Command | Entry point | Responsibility |
| --- | --- | --- |
| `fall-detection` | `fall_detection:main` | Live or file-based pose inference, fall detection, overlays, logs, JSONL, and optional annotated video |
| `fall-evaluate` | `fall_detection.evaluation:main` | Offline replay of committed or extracted feature traces and event-level metrics |
| `fall-data` | `fall_detection.data_lifecycle:main` | Manifest-bound data download, verification, cleanup, and batch execution |

The package is a single-process application. There is no frontend, backend
server, HTTP API, database, authentication layer, cloud deployment,
containerization, CI/CD pipeline, message broker, or centralized monitoring.
Operators interact through the command line, local windows, files, and logs.

The architecture is partially hexagonal. Domain records live in `domain/`,
application services and structural ports live in `application/`, and delivery
adapters live in `adapters/`. The migration is incomplete: runners, engines,
fall policy, configuration, telemetry, evaluation, and lifecycle orchestration
remain top-level package modules. The CLI uses `FallEventService` and
`CallableEventSink`; `FrameProcessingService` is tested but does not own the
current CLI frame loop.

## 3. Repository Structure

| Path | Contents and ownership |
| --- | --- |
| `src/fall_detection/domain/` | Durable `FallEvent` and `FallIncident` domain records |
| `src/fall_detection/application/` | `FallEventService`, `FrameProcessingService`, and protocol-based ports |
| `src/fall_detection/adapters/` | `CallableEventSink`, adapting callbacks to event publication |
| `src/fall_detection/` | Active engines, runners, tracking, smoothing, fall policy, CLI, evaluation, and data lifecycle modules |
| `tests/` | Pytest unit, integration, CLI, replay, lifecycle, and regression tests plus synthetic fixtures |
| `config/` | Annotated fall-detection TOML configuration example |
| `evaluation/manifests/` | Versioned evaluation dataset/label declarations and trace bindings |
| `evaluation/traces/` | Committed JSONL feature traces for deterministic replay |
| `evaluation/batches/` | Checksum-pinned manifest batches and labelled direct-URL batches |
| `scripts/` | Trace extraction and deterministic synthetic-trace generation |
| `docs/`, `README.md`, `docs.md` | Operating guide, command reference, and contributor reading map |
| `plan/` | Historical specifications and implementation plans |
| `models/` | Locally cached MediaPipe `.task` model bundles; created/downloaded at runtime |
| `.fall-data/` | Local batch input workspace, controlled by `fall-data` |
| `results/` | Local append-only batch result logs |
| `video/input/`, `video/output/` | Local source recordings and generated annotated videos |
| `pyproject.toml`, `uv.lock`, `mise.toml` | Package metadata, locked dependencies, environment defaults, and developer tasks |

`models/`, `.fall-data/`, `results/`, and `video/` are local data directories,
not application services or durable shared storage. Their presence and contents
vary by checkout.

## 4. Runtime Architecture

The main runtime flow is synchronous for video files and supervised,
queue-bounded asynchronous capture for live sources:

```text
camera / stream / video file
            |
            v
  cli.py -> VideoFileRunner or LiveStreamRunner
            |
            v
  engine.build_engine(PoseConfig)
       |                         |
       +-> NativeEngine          +-> CascadeEngine
           PoseDetector              PersonDetector -> cropped PoseDetectors
                 \                 /
                  v               v
                  list[PersonPose]
                         |
                         v
       PosePipeline: IdentityTracker -> LandmarkSmoother
                         |
                         v
  FallEventService (application) -> FallStateManager
                                      |
                                      v
                         ImageEvidenceExtractor
                           -> PersonFallFSM
                           -> durable incident state
                                      |
                                      v
                         FallEvent / FallIncident
                                      |
                                      v
                CallableEventSink -> CLI publisher
                    |          |          |
                    v          v          v
                 logging   alert/telemetry JSONL
                                      |
                       drawing overlay / video output
```

`fall_detection:main` deliberately imports `cli.py` lazily after native log
quieting, because MediaPipe's native layer reads its logging environment at
import time. `PoseConfig` and `build_engine` are the stable configuration and
engine-construction seam. `Strategy.AUTO` selects native inference for one pose
and cascade inference for multiple poses unless explicitly overridden.

`IdentityTracker` assigns stable process-local person identifiers; it is
centroid tracking, not biometric re-identification. `LandmarkSmoother` applies
a One-Euro filter. `FallStateManager` owns one evidence extractor and FSM per
active identifier, retains durable incident history, and forgets expired
runtime bindings without deleting recorded incidents.

## 5. Core Components and Contracts

| Component | Contract or invariant |
| --- | --- |
| `pose.py`, `engine.py` | `PoseConfig`; `build_engine`; engines expose `infer(frame, timestamp_ms)` and `close()` |
| `runner.py` | `VideoFileRunner`, `LiveStreamRunner`, and `PosePipeline`; timestamps are monotonic within a run and live queues are bounded |
| `tracking.py`, `smoothing.py` | Stable local identities and per-identity landmark filtering, with loss callbacks |
| `application/frame_processing.py` | Ports `PoseInferencePort`, `PersonPipelinePort`, `FallStatePort`, `EventSinkPort`; returns `FrameProcessingResult` |
| `application/fall_events.py` | `FallStatePort` and `EventSinkPort`; publishes every event returned by state policy |
| `domain/events.py` | Immutable `FallEvent` decision records and `FallIncident` semantic records |
| `fall_evidence.py` | Converts normalized landmarks to pixel-aware temporal features and Boolean evidence |
| `fall_fsm.py` | Seven-state deterministic FSM, public numeric states, alert kind, and evidence level |
| `fall_state.py` | Per-person FSM ownership, identity expiry, incident creation, recovery, and history |
| `fall_telemetry.py` | Versioned JSONL serialization for per-decision telemetry and detected/recovered incidents |
| `fall_config.py` | Validated profiles, thresholds, timing, quality gates, and normalized furniture ROI polygons |
| `drawing.py` | Skeleton, state, and debug overlays; does not decide fall state |
| `evaluation.py` | Strict manifest/trace validation, four replay strategies, deterministic event matching, and metrics |
| `data_lifecycle.py`, `batch_processing.py` | Checksum-pinned download, receipt, resumable classification, and verified cleanup |
| `link_batch.py` | Sequential labelled HTTPS regression for unpinned public links, with a fresh result log |

The active CLI decision path is `cli.py -> runner.py -> fall_state.py ->
fall_fsm.py -> fall_evidence.py`. `biomechanics.py`, `discriminators.py`, and
`kalman.py` are tested offline physics/reference modules and are not called by
that path. `geometry.py` is a shared pure-NumPy utility used by reference
physics code. Changes to the offline modules therefore do not change CLI fall
decisions unless they are explicitly wired into the runtime.

## 6. Data Architecture and Lifecycle

The application uses local files rather than a database:

| Store/format | Producer | Consumer | Compatibility and durability |
| --- | --- | --- | --- |
| Fall configuration TOML | Operator | `fall_config.py`, CLI, extraction, batch inference | Strict keys and finite/range validation; CLI profile overrides file profile |
| Evaluation manifest TOML | Maintainer/operator | `evaluation.py`, extraction, lifecycle | Schema version, clip checksum, labels, subject/trial/camera, and split rules are validated |
| Feature trace JSONL | Extraction scripts or synthetic generator | `fall-evaluate` | Trace schema v2 is current; strict v1 remains readable with conservative unavailable motion |
| Alert JSONL | `fall-detection` | Operator tooling | Schema-v1 detected/recovered incident records, appended locally |
| Telemetry JSONL | `fall-detection` | Debugging/evaluation tooling | Schema-v1 decision records, appended locally |
| Batch result JSONL | `fall-data` batch runners | Resume logic and operator | Schema-v1 append-only records; pinned batches bind descriptor, manifest, clip, and source hashes |
| Model bundles | `models.py` downloader or operator | MediaPipe detectors | Cached `.task` files; downloads are staged and atomically replaced |
| Annotated video | `VideoFileRunner` | Operator | Optional file output; disallowed for live sources |

Evaluation flow:

```text
manifest TOML + source video + model + FallConfig
                |
                v
 extract_fall_traces.py -- checksum validation
                |
                v
 versioned feature-trace JSONL -- fingerprint/checksum validation
                |
                v
 fall-evaluate -- strategy replay -- event matching -- JSON metrics report
```

Pinned data lifecycle:

```text
batch TOML -> manifest TOML -> approved HTTP(S) mirrors
    |              |                 |
    +-- descriptor + manifest + source SHA-256 checks
                           |
                 atomic staged download
                           |
                 local verified source
                           |
           classify -> fsync'd resumable result JSONL
                           |
              delete only checksum-verified source
```

Whole-batch downloads write a matching receipt before an atomic directory
move. Deletion requires `--yes`, a contained path, a matching receipt, and
verified checksums. The resumable runner keeps its result log outside the
removable batch directory and records download, classification, deletion, and
completion stages. Retries skip already successful pinned work.

The separate `run-links` path consumes labelled direct HTTPS URLs without
source checksums. It rejects non-HTTPS inputs and redirects, downloads and
classifies clips sequentially, removes each temporary input, and requires an
empty result log because those inputs cannot be identity-pinned for safe
resume. It is a smoke/regression facility, not the checksum-pinned dataset
lifecycle.

## 7. External Integrations

| Integration | Use | Boundary behavior |
| --- | --- | --- |
| MediaPipe Tasks Pose Landmarker | 33 image and world landmarks per detected person | Local inference; model bundle may be downloaded on first use |
| MediaPipe object detector | Person boxes for cascade mode | Local inference, followed by one pose landmarker per crop |
| OpenCV/FFmpeg | Camera, file, HTTP/MJPEG, and RTSP capture; display and video encoding | Live sources use bounded buffering and low-latency defaults; operator environment can override FFmpeg options |
| Model hosting URLs | Pose/person model bundle retrieval | Cached locally through staged, atomic downloads |
| Approved HTTP(S) dataset mirrors | Checksum-pinned batch acquisition | Content must match manifest SHA-256 before use or deletion |
| Direct public HTTPS URLs | Labelled link-batch smoke regression | HTTPS-only, no safe resume, and not treated as immutable evidence |

There are no third-party alerting, identity, analytics, storage, or monitoring
services. Logs and JSONL files are the operational integration surfaces.

## 8. Deployment and Runtime Environment

The supported deployment is a developer/operator machine with Python 3.13+,
`uv`, MediaPipe-compatible native libraries, and an accessible camera or video
file. `uv sync` creates the locked environment. `mise.toml` selects current
Python and `uv`, exposes `test`, `regression_test`, `batch_regression`, and
`run` tasks, and defaults link-batch paths through `FALL_DATA_BATCH` and
`FALL_DATA_RESULT_LOG`.

Inference runs in one OS process. File input uses synchronous VIDEO-mode
inference. Live input uses a capture/producer path, a bounded result queue, and
a supervisor that can recover from stalls. Native mode can request a Linux GPU
delegate; cascade mode remains CPU-bound because crop inference is parallel and
delegate objects are thread-affine. There is no deployment manifest, service
manager, container image, autoscaling, remote configuration, or production
rollout mechanism in the repository.

## 9. Security and Operational Controls

Security is local and boundary-focused rather than account-based:

- CLI and TOML parsing reject invalid enum values, ranges, unknown keys, and
  non-finite numeric values.
- Manifest and trace readers enforce schema versions, exact checksums,
  configuration fingerprints, source identity, complete frame coverage, and
  leakage-safe subject splits.
- Batch paths must remain relative and contained under the selected data root;
  result logs must remain outside removable data directories.
- Pinned downloads use staging files/directories and atomic replacement.
  Cleanup verifies receipts and SHA-256 values and requires explicit `--yes`.
- Link batches accept HTTPS only and reject redirects to other schemes.
- JSONL progress/result records are flushed, and resumable batch records are
  `fsync`'d so partial failure is observable.
- Furniture contact is claimed only inside validated configured polygons;
  absent configuration produces no furniture claim.

There is no authorization or secret-management subsystem because the system
has no network service or user accounts. Operators are responsible for local
filesystem permissions, dataset licenses, trusted configuration, camera
privacy, URL provenance, and removal of sensitive recordings. URL credentials,
if supplied externally, can appear in descriptors or logs and should not be
committed.

## 10. Development and Testing

The package manager is `uv`; common verified commands are:

```bash
uv sync
uv run fall-detection --source 0
uv run pytest
uv run fall-evaluate --manifest evaluation/manifests/local-falls.toml --strategy temporal-fsm
mise run regression_test
mise run batch_regression
```

`pyproject.toml` scopes pytest to `tests/`. Tests cover pure geometry and
physics, engines and runners through fakes, tracking, smoothing, FSM/evidence,
domain records, application ports, CLI wiring, telemetry, evaluation, trace
extraction, batch lifecycle, and link-batch regression. Shared synthetic pose
fixtures live in `tests/conftest.py` and `tests/synthetic_falls.py`; authored
feature streams live in `src/fall_detection/synthetic_traces.py`.

The repository follows test-driven development. `tests/` intentionally has no
`__init__.py`, and shared fixtures use bare `from conftest import ...` imports.
Package modules use `logging`, not `print`. No linter, formatter, static type
checker, pre-commit hooks, Markdown linter, or CI workflow is configured.

## 11. Limitations, Ownership, and Glossary

### Architectural limitations

- The hexagonal separation is partial; top-level runtime modules still own
  significant orchestration and concrete types, and application protocols use
  broad `Any` shapes.
- `FrameProcessingService` is not yet the CLI frame loop, so there are two
  application orchestration shapes without full consolidation.
- Runtime state is process-local. Restarting loses active trackers; only files
  already written to logs, JSONL, model cache, or output video persist.
- Centroid identity tracking can swap identifiers and is not cross-camera or
  biometric identity.
- RGB-only pose is sensitive to occlusion, perspective, lighting, and camera
  placement. World landmarks are hip-centred, not room coordinates; the active
  detector deliberately uses image geometry for motion.
- Furniture ROIs indicate image-region occupancy, not physical contact.
- Committed and public sample regressions are small and staged; they do not
  demonstrate deployment fitness or clinical performance.
- There is no service hardening, authentication, centralized observability,
  high availability, or automated release pipeline.

### Ownership and change safety

This document describes the current branch and introduces no runtime reader,
writer, API, schema, dependency, or compatibility window. Its forward path is
adding this root file. Rollback is removal of `architecture.md`; runtime
behavior and stored data are unaffected. Source modules and their tests remain
the authority when this document and code disagree.

### Glossary

| Term | Meaning in this repository |
| --- | --- |
| ADL | Activity of daily living; a non-fall motion used as a hard negative |
| Cascade | Person detection followed by pose inference on each cropped body |
| Evidence | Boolean and duration-weighted cues derived from pixel-space pose features |
| FSM | Per-person finite-state machine that turns temporal evidence into incidents and recovery |
| Incident | Durable semantic record created once when a terminal fall decision is reached |
| Native | One MediaPipe pose landmarker applied to the full frame |
| Person ID | Process-local centroid-tracker identifier, not a real-world identity |
| Pinned batch | Dataset batch bound to descriptor, manifest, and source SHA-256 values |
| ROI | Validated normalized image polygon representing configured furniture |
| Trace | Versioned JSONL sequence of extracted or synthetic `FallFeatures` for replay |
