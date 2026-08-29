# Fall Detection

A Python 3.13+ fall-detection pipeline built on MediaPipe Pose and OpenCV. It
extracts 33 body landmarks per person from webcams, recorded video, and network
streams, then turns pixel-corrected geometry and duration-weighted evidence into
deterministic fall incidents. It supports multi-person inference, headless
operation, annotated video, JSONL alerts and telemetry, and replay evaluation
without a server or database.

This is an engineering and research implementation, not a medical device. Its
thresholds and small evaluation sets do not establish clinical or production
fitness.

## ✨ Features

- Webcam, network-stream, and recorded-video processing.
- Native single-person inference or a person-detection cascade for crowds.
- Stable identities, One-Euro smoothing, temporal evidence, and a per-person
  finite-state machine (FSM).
- Interactive overlays, optional evidence diagnostics, and annotated video.
- Schema-v1 JSON Lines for incidents, telemetry, and batch results.
- Deterministic feature-trace replay with four comparison strategies.
- Checksum-pinned data handling and sequential direct-link batch regression.

## 🧭 Table of contents

- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quickstart](#-quickstart)
- [Live and recorded inference](#-live-and-recorded-inference)
- [Configuration](#-configuration)
- [Runtime architecture](#-runtime-architecture)
- [Detector strategies and benchmark](#-detector-strategies-and-benchmark)
- [Alerts, telemetry, and output](#-alerts-telemetry-and-output)
- [Evaluation and replay](#-evaluation-and-replay)
- [Data lifecycle and batches](#-data-lifecycle-and-batches)
- [Programmatic use](#-programmatic-use)
- [Repository layout](#-repository-layout)
- [Operational gotchas](#-operational-gotchas)
- [Deployment](#-deployment)
- [Security and privacy](#-security-and-privacy)
- [Development and testing](#-development-and-testing)
- [Contributing](#-contributing)
- [Versioning, authorship, and license](#-versioning-authorship-and-license)
- [Limitations](#-limitations)
- [Further reading](#-further-reading)

## 🧰 Requirements

- Python 3.13 or newer, as declared in `pyproject.toml`.
- [`uv`](https://docs.astral.sh/uv/) for dependency and command management.
- An OpenCV-readable camera, video, or network stream for inference.
- A desktop display only for the interactive window; `--no-display` is
  headless.

MediaPipe 1.x supplies the Tasks API and pulls in
`opencv-contrib-python` and NumPy. Development uses pytest. There is no
frontend, server, database, container, CI/CD pipeline, formatter, linter, or
type checker configured in this repository.

GPU delegation is optional and Linux-only. It applies to native inference; the
parallel cascade uses CPU because delegates are bound to their creating thread.

## 📦 Installation

From the repository root, synchronize the locked environment:

```bash
uv sync
```

The first inference run downloads the selected MediaPipe Pose Landmarker
`.task` bundle into the gitignored `models/` directory. For offline or
controlled deployments, supply a local bundle with `--model-path`.

```bash
uv run fall-detection --help
uv run fall-evaluate --help
uv run fall-data --help
```

## 🚀 Quickstart

```bash
# Webcam with interactive overlay
uv run fall-detection --source 0

# Recorded video, headless, with annotated output
uv run fall-detection --source clip.mp4 --output out.mp4 --no-display

# Multi-person webcam processing
uv run fall-detection --source 0 --num-poses 4 --detect-interval 3

# Incident alerts and detailed decision telemetry
uv run fall-detection \
  --source clip.mp4 \
  --no-display \
  --fall-alert-log results/alerts.jsonl \
  --fall-telemetry-log results/telemetry.jsonl
```

Press `q` or `Esc` to close the window. Exit codes are `0` for success,
`1` for runtime failure, and `2` for invalid arguments.

## 🎥 Live and recorded inference

`fall-detection` accepts a camera index, `/dev/video*` device, file path,
or stream URL through `--source`.

### Webcam, video, and pose-only mode

```bash
uv run fall-detection --source 0
uv run fall-detection --source /dev/video0
uv run fall-detection --source clip.mp4 --max-frames 300
uv run fall-detection --source clip.mp4 --no-fall-detection
```

### Network streams

```bash
uv run fall-detection --source rtsp://camera.example/live
uv run fall-detection --source http://192.0.2.10:4747/mjpegfeed
```

Network sources receive low-latency FFmpeg defaults
(`rtsp_transport;tcp|fflags;nobuffer|max_delay;0`) and
`CAP_PROP_BUFFERSIZE=1`. They use `setdefault`, so operators can override
them:

```bash
OPENCV_FFMPEG_CAPTURE_OPTIONS='rtsp_transport;udp|stimeout;5000000' \
  uv run fall-detection --source rtsp://camera.example/live
```

OpenCV native logging stays quiet unless `--log-level DEBUG` is selected,
which helps diagnose stream handshakes and decoder failures.

### Multi-person cascade

```bash
# Auto selects cascade because num-poses is greater than one
uv run fall-detection \
  --source 0 \
  --num-poses 4 \
  --min-detection-confidence 0.6

# Carry detected regions between every-third-frame detector passes
uv run fall-detection --source 0 --num-poses 4 --detect-interval 3

# Track several poses but keep only the strongest downstream
uv run fall-detection --source 0 --num-poses 4 --best-only
```

### Headless output, logs, and debug overlay

```bash
uv run fall-detection \
  --source clip.mp4 \
  --output results/annotated.mp4 \
  --no-display \
  --log-level INFO \
  --log-file results/run.log

uv run fall-detection --source clip.mp4 --fall-debug-overlay --log-level DEBUG
```

`--output` supports file sources only and is rejected for cameras and network
streams. Annotation occurs only when display or output is enabled. The debug
overlay adds evidence duration/fraction/coverage and observation age.

### Option reference

| Option | Default | Purpose |
| --- | --- | --- |
| `--source` | `0` | Camera, device, video path, HTTP(S), or RTSP source. |
| `--model` | `full` | Use `lite`, `full`, or `heavy`. |
| `--model-path` | none | Use a local `.task` bundle. |
| `--cache-dir` | `models` | Downloaded model cache. |
| `--num-poses` | `1` | Maximum people to track. |
| `--best-only` | off | Keep only the most confident person. |
| `--gpu` | off | Try the Linux GPU delegate. |
| `--detector` | `auto` | Use `auto`, `native`, or `cascade`. |
| `--person-model` | `efficientdet_lite0` | Cascade box model; `lite0` or `lite2`. |
| `--person-score` | `0.4` | Person-box confidence floor. |
| `--crop-padding` | `0.15` | Padding around each person box. |
| `--crop-workers` | `min(4, --num-poses)` | Parallel crop landmarkers. |
| `--detect-interval` | `1` | Detect every Nth frame and carry regions. |
| `--min-box-px` | `48` | Minimum box short side. |
| `--no-fall-detection` | off | Run pose estimation without fall decisions. |
| `--body-mass-kg` | none | Deprecated; no effect on RGB decisions. |
| `--fall-config` | none | Load validated TOML thresholds and ROIs. |
| `--fall-profile` | `balanced` | `sensitive`, `balanced`, or `precision`. |
| `--fall-alert-log` | none | Append incident JSONL. |
| `--fall-telemetry-log` | none | Append decision telemetry JSONL. |
| `--fall-debug-overlay` | off | Display evidence timing and coverage. |
| `--output` | none | Write annotated video for a file source. |
| `--no-display` | off | Disable the OpenCV window. |
| `--display-max-width` | `1280` | Cap display width, preserving aspect. |
| `--no-smoothing` | off | Disable One-Euro filtering. |
| `--min-detection-confidence` | `0.5` | BlazePose detection threshold. |
| `--min-presence-confidence` | `0.5` | BlazePose presence threshold. |
| `--tracking-confidence` | `0.5` | BlazePose tracking threshold. |
| `--max-frames` | none | Stop after N frames. |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |
| `--log-file` | none | Mirror logs to a file. |

## ⚙️ Configuration

Copy the annotated example and edit it:

```bash
cp config/fall_detection.example.toml config/local-fall.toml
uv run fall-detection --source 0 --fall-config config/local-fall.toml
```

Configuration precedence is:

1. Explicit `--fall-profile`.
2. The TOML document's `profile`.
3. `balanced`.
4. Explicit TOML fields override the selected profile seed.

Profiles are `sensitive`, `balanced`, and `precision`. TOML supports
`[dynamic]`, `[posture]`, `[timing]`, `[quality]`, and optional
`[[furniture_rois]]` sections. Unknown keys, non-finite values, invalid
ranges, and non-positive durations fail validation.

```toml
[[furniture_rois]]
name = "bed"
points = [[0.10, 0.25], [0.90, 0.25], [0.90, 0.95], [0.10, 0.95]]
```

Without a furniture ROI, the system never claims `BED_REST`. See the
[example configuration](config/fall_detection.example.toml) and
[configuration guide](docs/fall-detection.md#profiles-and-configuration).

## 🏗️ Runtime architecture

```text
fall-detection CLI
    |
    +-- source + PoseConfig
    |
    +-- VideoFileRunner / LiveStreamRunner
          |
          +-- build_engine
          |     +-- NativeEngine: whole-frame Pose Landmarker
          |     `-- CascadeEngine: person boxes -> crops -> landmarkers
          |
          +-- IdentityTracker -> LandmarkSmoother -> list[PersonPose]
          |
          +-- frame-processing application layer
          |     `-- FallStateManager: one PersonFallFSM per identity
          |           `-- pixel features -> evidence -> temporal state
          |
          `-- overlays + annotated video + alert/telemetry JSONL

feature traces     -> fall-evaluate -> deterministic event metrics
batch descriptors -> fall-data     -> download/classify/cleanup JSONL
```

The engine produces `PersonPose` values with 33 normalized landmarks, metric
hip-centred world landmarks, a score, bounding box, and centroid. The identity
tracker uses nearest-centroid association, and One-Euro smoothing reduces
jitter unless disabled.

For each identity, `ImageEvidenceExtractor` derives finite pixel-space torso
angle, aspect ratio, downward speed, rotation, height collapse, motion,
visibility, and centroid features. `classify_evidence` creates explicit
evidence gates. `PersonFallFSM` latches them over time, weighting observation
duration and coverage. A terminal confirmation creates one durable incident;
recovery updates it without retracting the alert. See the
[FSM transition guide](docs/fall-detection.md#states-transitions-and-incidents).

## 📊 Detector strategies and benchmark

`--detector auto` uses native inference for `--num-poses 1` and cascade
above one. Force either with `--detector native|cascade`.

**Native** sends the whole frame to one landmarker. BlazePose tends to propose
one dominant region, so other people may never produce landmarks.

**Cascade** detects person boxes, pads and letterboxes each crop, and sends each
to an independent single-person landmarker. Crops run across
`--crop-workers`.

The repository benchmark used 120 frames at 1280×720, three separated bodies,
the `full` model, CPU, and 16 cores:

| Run | FPS | Frames with all 3 people |
| --- | ---: | ---: |
| `--detector native` | 17.2 | 63 / 120 |
| `--detector cascade` | 13.9 | 120 / 120 |
| `--detector cascade --crop-workers 1` | 8.2 | 120 / 120 |
| `--detector cascade --detect-interval 3` | 17.7 | 120 / 120 |
| `--detector cascade --model lite --detect-interval 3` | 17.0 | 120 / 120 |

These are measurements from one machine and clip, not performance guarantees.
FPS varied roughly one point; recall did not. Native also reported a fourth,
nonexistent person on three frames. Still-image probes returned one native
pose where cascade returned three or four.

Design choices behind the result:

- `efficientdet_lite0` int8 recalled people that `lite2` float16 missed.
- Crops are padded and letterboxed, not stretched; this protects limbs without
  pulling neighbours into the crop.
- Crop-local `z` is rescaled with normalized `x`; metric, hip-centred
  `world_landmarks` remain unchanged.
- Cascade stays on CPU due to parallel, thread-bound landmarkers. Native can
  use `--gpu`.

## 🚨 Alerts, telemetry, and output

```bash
uv run fall-detection \
  --source clip.mp4 \
  --output results/annotated.mp4 \
  --no-display \
  --fall-alert-log results/incidents.jsonl \
  --fall-telemetry-log results/decisions.jsonl
```

`--fall-alert-log` writes compact schema-v1 `detected` and `recovered`
records containing incident ID, original/current person IDs, state, timestamp,
kind, evidence level, `detected_at`, and `recovered_at`.

`--fall-telemetry-log` writes one schema-v1 record per person and processing
step: raw features, evidence gates, prior/current state, duration-weighted
evidence and coverage, observation age, and incident data. JSONL is append-only,
so long-running deployments need log rotation. Annotated video contains
skeletons, boxes, IDs, state HUD, and optional debug evidence.

## 🧪 Evaluation and replay

`fall-evaluate` replays finite feature traces without MediaPipe, source video,
or a model bundle:

```bash
uv run fall-evaluate \
  --manifest evaluation/manifests/local-falls.toml \
  --strategy temporal-fsm
```

| Strategy | Decision model |
| --- | --- |
| `legacy-and` | All five dynamic/posture votes plus stillness on one sample. |
| `relaxed-or` | Any dynamic/posture vote on one sample. |
| `k-of-n` | At least three of five votes on one sample. |
| `temporal-fsm` | Production latching, duration, gap, level, and recovery. |

Use `--split NAME` for a manifest with multiple frozen splits.
`--fall-config` and `--fall-profile` follow production precedence. Replay
rejects a configuration fingerprint different from extraction.

Reports contain event counts, sensitivity, precision, false alerts per hour,
miss rate, matched latency, recovery timing, and state dwell. Matching is
event-level and deterministic. `fall-evaluate` prints JSON and exits zero
regardless of its metrics; pytest supplies the pass/fail regression gates.

### Committed replay sets

**`local-falls`** contains real MediaPipe traces from 10 labelled falls and 2
person-present negatives. Balanced replay detects 8 falls; two short clips are
committed misses. This small staged set guards behavior, not real-world
accuracy.

**`synthetic-adl`** contains 16 authored `FallFeatures` streams: six fall
geometries, eight ADL hard negatives, and two degenerate inputs. Its metrics
describe authored scenarios and are not measured system accuracy.

```bash
uv run fall-evaluate \
  --manifest evaluation/manifests/synthetic-adl.toml \
  --strategy temporal-fsm

uv run python scripts/generate_synthetic_traces.py
```

Regenerate real-video traces when raw assets live outside the checkout:

```bash
uv run python scripts/extract_fall_traces.py \
  --manifest evaluation/manifests/local-falls.toml \
  --output evaluation/traces/local-regression-v2.jsonl \
  --source-root /path/to/repository-root \
  --model-path /path/to/pose_landmarker_full.task \
  --fall-profile balanced --force
```

Extraction verifies source SHA-256 and records model checksum plus a canonical
configuration fingerprint. The repository commits traces, checksums, and
labels—not videos or model bundles. Public-dataset templates under
`evaluation/manifests/` require lawful local assets and real checksums.

## 🗄️ Data lifecycle and batches

`fall-data` exposes five subcommands:

| Subcommand | Required options | Behavior |
| --- | --- | --- |
| `probe` | `--batch` | Probe mirrors and report status, latency, type, and selection. |
| `download` | `--batch` | Atomically download and checksum-verify a pinned batch. |
| `delete` | `--batch --yes` | Delete only a verified complete batch. |
| `run` | `--batch --result-log` | Run a resumable checksum-pinned batch. |
| `run-links` | `--batch --result-log` | Run binary-labelled direct HTTPS links. |

All accept `--data-root`, defaulting to `datasets`.

### Probe, download, and delete

```bash
uv run fall-data probe \
  --batch evaluation/batches/urfd.example.toml --data-root datasets

uv run fall-data download \
  --batch evaluation/batches/urfd.example.toml --data-root datasets

uv run fall-data delete \
  --batch evaluation/batches/urfd.example.toml --data-root datasets --yes
```

Downloads stage before an atomic move, verify each manifest SHA-256, and write
a receipt tied to descriptor and manifest. Deletion requires `--yes`, path
containment, a matching receipt, and matching checksums.

### Run a checksum-pinned batch

```bash
uv run fall-data run \
  --batch evaluation/batches/urfd.example.toml \
  --result-log results/urfd-run.jsonl \
  --data-root datasets
```

The log must be outside the removable batch directory. Stdout and the
append-only log use schema-v1 JSONL with fingerprints, checksums, frame counts,
elapsed time, incidents, and progress. Re-running resumes successful records,
never reclassifies them, and retries failed cleanup after checksum validation.

### Run a direct-link batch

```bash
uv run fall-data run-links \
  --batch evaluation/batches/urfd-github-samples.toml \
  --data-root .fall-data \
  --result-log results/urfd-github-samples.jsonl
```

Direct-link descriptors require unique IDs, HTTPS URLs, and `fall` or
`normal` labels. They have no source checksums, so the log must be new or
empty and cannot prove remote file identity. A successful input is deleted
only after its result is flushed and synced. Failures remain for diagnosis,
cause a nonzero exit, and are excluded from metric denominators. The summary
reports TP/TN/FP/FN, accuracy, precision, recall, and F1.

```bash
mise run batch_regression
```

This convenience task runs pytest and both committed replays before the
direct-link sample. Defaults are in [mise.toml](mise.toml). The included
two-clip URFD batch is a sample, not the full 70-trial corpus.

## 🐍 Programmatic use

```python
from fall_detection.engine import build_engine
from fall_detection.logging_config import setup_logging
from fall_detection.models import ModelVariant
from fall_detection.pose import PoseConfig, RunningMode, best_person

setup_logging("INFO")
config = PoseConfig(model_variant=ModelVariant.FULL, num_poses=4)
engine = build_engine(config, RunningMode.IMAGE)

try:
    persons = engine.infer(bgr_frame, timestamp_ms=0)
    subject = best_person(persons)
finally:
    engine.close()
```

`build_engine` honors `config.strategy`; `num_poses=4` with `AUTO`
selects cascade. Use `PoseDetector` for the raw single-landmarker path.

## 🗂️ Repository layout

| Path | Responsibility |
| --- | --- |
| `src/fall_detection/pose.py` | Pose configuration, values, and MediaPipe wrapper. |
| `src/fall_detection/engine.py` | Unified native/cascade engines. |
| `src/fall_detection/person_detector.py` | Cascade person boxes. |
| `src/fall_detection/cascade.py` | Crops, parallel inference, coordinate remap. |
| `src/fall_detection/runner.py` | Video and self-healing live runners. |
| `src/fall_detection/tracking.py` | Stable nearest-centroid identities. |
| `src/fall_detection/smoothing.py` | One-Euro landmark filtering. |
| `src/fall_detection/application/` | Frame and fall-event orchestration. |
| `src/fall_detection/domain/` | Adapter-independent domain events. |
| `src/fall_detection/adapters/` | Event-sink adapter boundary. |
| `src/fall_detection/fall_config.py` | Profiles, overrides, and ROIs. |
| `src/fall_detection/fall_evidence.py` | Pixel features and evidence. |
| `src/fall_detection/fall_fsm.py` | Temporal state transitions. |
| `src/fall_detection/fall_state.py` | Tracker and incident ownership. |
| `src/fall_detection/fall_telemetry.py` | Incident/telemetry JSONL. |
| `src/fall_detection/drawing.py` | Skeleton, box, HUD, and debug overlays. |
| `src/fall_detection/evaluation.py` | Replay and event metrics. |
| `src/fall_detection/data_lifecycle.py` | Pinned data CLI and cleanup. |
| `src/fall_detection/batch_processing.py` | Resumable pinned batches. |
| `src/fall_detection/link_batch.py` | Direct-link batches. |
| `src/fall_detection/models.py` | Atomic model download and cache. |
| `src/fall_detection/logging_config.py` | Logging and native suppression. |
| `src/fall_detection/geometry.py` | Convex hull and polygon geometry. |
| `src/fall_detection/biomechanics.py` | Offline biomechanics utilities. |
| `src/fall_detection/discriminators.py` | Offline ADL discriminators. |
| `src/fall_detection/kalman.py` | Offline Kalman stabilization. |
| `config/` | Annotated TOML configuration. |
| `evaluation/` | Batches, manifests, and traces. |
| `scripts/` | Trace generation. |
| `tests/` | pytest suite. |

Biomechanics, discriminators, and Kalman utilities are tested but not on the
active CLI decision path. See [docs.md](docs.md) for a reading order.

## ⚠️ Operational gotchas

- **World-landmark `y` points down.** World landmarks are metric and
  hip-centred but downward-positive. Up is `-y`; ground is `(x, z)`.
- **World coordinates do not locate a person in the room.** The hip origin is
  reset per pose, so runtime descent uses pixel-corrected image geometry.
- **Timestamps strictly increase.** Video uses FPS; live mode uses rebased
  `time.monotonic_ns()`. Invalid timestamps are bumped with a warning.
- **Results arrive off the capture thread.** A queue of size two drops stale
  results; drawing and I/O stay on the main thread.
- **Tracking is not re-identification.** Greedy centroid IDs can change after
  occlusion, disappearance, or crossings.
- **`--detect-interval` carries regions.** Prior boxes are reused between
  detector passes; incomplete carry is abandoned.
- **Live runs self-heal.** Stalls rebuild capture and inference with
  exponential backoff, ending after five consecutive failures.
- **Native logs bypass Python logging at startup.** File descriptor 2 is
  suppressed around model creation unless DEBUG is active; one benign
  square-ROI message may still appear.
- **Package code uses logging.** Command JSON is deliberately written to
  stdout; package modules contain no `print()`.

## 🌐 Deployment

This repository ships a local process, not a hosted service. A minimal
deployment needs Python 3.13+, a synchronized `uv` environment, a pinned
model via `--model-path`, a readable source, writable outputs, and optionally
a process supervisor.

For unattended use, select `--no-display`, explicit log paths, log rotation,
and OS-level exit-code/disk monitoring. Validate camera permissions and codecs,
then benchmark the chosen model and detector settings on target hardware.

No Dockerfile, Compose file, web API, database migration, deployment manifest,
or CI/CD configuration is provided.

## 🔒 Security and privacy

- Restrict access and retention for frames, annotated video, incident times,
  telemetry, and backups.
- Telemetry exposes movement, identity continuity, timestamps, and configured
  location names even without video.
- Stream credentials in URLs may appear in process listings or logs; prefer
  protected configuration and isolated camera networks.
- Review every batch descriptor. Pinned batches verify SHA-256; direct links
  enforce HTTPS but cannot prove content identity.
- Conservative pinned deletion requires `--yes`, containment, receipt, and
  checksum checks. Direct-link cleanup removes only a durably classified input.
- Verify external models and follow dataset licensing and consent requirements.

There is no authentication layer because no server is exposed. Apply access
control at the operating-system and deployment boundaries.

## 🛠️ Development and testing

```bash
uv sync
uv run pytest
```

pytest is scoped to `tests/`. Test counts are omitted because they evolve.
There is no configured formatter, linter, type checker, pre-commit hook, or CI.

- Use TDD for behavior changes: observe a focused failure, implement, then run
  the focused and relevant suite.
- `tests/` intentionally lacks `__init__.py`; use bare
  `from conftest import ...` imports.
- Builders live in `tests/conftest.py`; fall sequences live in
  `tests/synthetic_falls.py`.
- Keep MediaPipe imports lazy at the package entry point so native log quieting
  happens before initialization.

```bash
mise run regression_test
mise run batch_regression
```

## 🤝 Contributing

There is no formal `CONTRIBUTING.md`. Until one exists:

1. Describe behavior, data assumptions, and acceptance criteria before a large
   change.
2. Preserve the boundaries between pose inference, fall decisions, evaluation,
   and data lifecycle.
3. Add focused pytest coverage and preserve deterministic traces unless
   intentionally recalibrating them.
4. Document CLI, TOML, JSONL, or replay-semantic changes.
5. Never commit subject video, models, secrets, camera URLs, or runtime logs.

## 🏷️ Versioning, authorship, and license

The package version is `0.1.0`. There is no release automation or published
compatibility policy, so consumers should pin the revision they validate.
Treat CLI, configuration, trace, and JSONL changes as compatibility-sensitive.

`pyproject.toml` names **parinya-ao** as author. Contributions and research
foundations are acknowledged through Git history and the upstream projects
cited below.

There is currently no committed license file. Do not assume MIT or another
license; obtain permission from the copyright holder before copying,
redistributing, or deploying beyond what applicable law permits.

## 🚧 Limitations

- Not clinically validated; never use as the sole emergency or safety monitor.
- RGB inference is sensitive to occlusion, light, angle, clothing, blur,
  subject size, and stream quality.
- Native multi-pose inference can miss people; cascade costs more compute.
- Identity tracking is short-term association, not biometric identification.
- Furniture classification uses manual 2D ROIs; no depth, force, or physical
  contact is measured. `IMPACT` names image evidence, not impact force.
- Small real and synthetic replay sets are regression evidence, not
  population-level accuracy.
- Direct-link batches cannot prove which remote file version was evaluated.
- Offline physics utilities do not affect production CLI decisions.

## 📚 Further reading

| Topic | Resource |
| --- | --- |
| Guided codebase reading order | [docs.md](docs.md) |
| FSM, configuration, telemetry, replay, and safeguards | [docs/fall-detection.md](docs/fall-detection.md) |
| Annotated TOML template | [config/fall_detection.example.toml](config/fall_detection.example.toml) |
| Repository conventions | [CLAUDE.md](CLAUDE.md) |
| MediaPipe Pose Landmarker | [Google AI Edge documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) |
| OpenCV video I/O | [OpenCV documentation](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) |

This project builds on MediaPipe, OpenCV, NumPy, pytest, `uv`, and the public
fall-detection research datasets represented by the example manifests. Review
all upstream and dataset terms before use.
