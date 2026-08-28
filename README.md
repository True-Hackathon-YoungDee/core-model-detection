# fall-detection

MediaPipe pose core for a fall-detection pipeline: 33 body landmarks per person,
from a webcam or a video file, for one person or several. Fall decisions sit on
top of this keypoint layer — pixel-corrected RGB geometry and duration-weighted
evidence feed a deterministic temporal state machine. Everything reports
through `logging` — the package contains no `print`.

## Quickstart

```bash
uv sync                                                   # install
uv run fall-detection --source 0                          # webcam, overlay window
uv run fall-detection --source clip.mp4 --output out.mp4 --no-display
uv run pytest                                              # 258 tests
```

The first run downloads the pose model bundle into `models/` (gitignored).
Press `q` or `Esc` to close the overlay window.

## Repository map

| Path | What lives there |
| --- | --- |
| `src/fall_detection/` | the package — pose detection, fall FSM, CLI (see [Layout](#layout)) |
| `tests/` | pytest suite, TDD-developed; synthetic fixtures in `conftest.py` / `synthetic_falls.py` |
| `docs/fall-detection.md` | fall-detection operating guide: FSM, config, telemetry, replay, limits |
| `docs.md` | onboarding reading order / mental model for new contributors |
| `config/` | example `--fall-config` TOML (thresholds, furniture ROIs) |
| `evaluation/` | replay manifests (`manifests/`) and committed feature traces (`traces/`) |
| `scripts/` | `extract_fall_traces.py` — regenerate a trace from raw video |
| `models/` | downloaded `.task` bundles (gitignored, created on first run) |

## How it fits together

```
cli.py:main()
  -> PoseConfig
  -> VideoFileRunner / LiveStreamRunner        (runner.py)
       -> engine.py:build_engine picks:
            NativeEngine    (pose.py)
            CascadeEngine   (cascade.py + person_detector.py)
       -> PosePipeline:
            tracking.py:IdentityTracker        (stable ids across frames)
            smoothing.py:LandmarkSmoother      (One-Euro filter)
       -> per-frame list[PersonPose]
       -> fall_state.py:FallStateManager       (one tracker per stable id)
            -> fall_evidence.py:ImageEvidenceExtractor / classify_evidence
            -> fall_fsm.py:PersonFallFSM.step()
            -> FallEvent / FallIncident
       -> drawing.py overlay, VideoWriter, telemetry JSONL
```

Two concerns are layered here and separable: **pose estimation** (getting
landmarks reliably, one or many people) and **fall detection** (turning a
stream of landmarks into a state-machine decision). Run pose-only with
`--no-fall-detection`.

## Where to start reading

| If you want to change… | Start at |
| --- | --- |
| CLI flags / wiring | `cli.py` |
| how landmarks are produced | `pose.py`, then `engine.py` |
| multi-person recall | `cascade.py`, `person_detector.py` |
| fall decisions | `fall_evidence.py` → `fall_fsm.py` → `fall_state.py` |
| thresholds, without touching code | `config/fall_detection.example.toml` |

For a fully guided, file-by-file reading order with rationale — the version of
this section written for someone who has never opened the repo — see
[docs.md](docs.md).

## Install

```bash
uv sync
```

`mediapipe>=1.0.1` pulls `opencv-contrib-python` and `numpy`. MediaPipe 1.x ships
the Tasks API only (`mediapipe.solutions` no longer exists), so everything uses
`mediapipe.tasks.python.vision.PoseLandmarker`.

## Run

```bash
# webcam, single best person, overlay window
uv run fall-detection --source 0

# webcam, up to 4 people -- picks the cascade automatically
uv run fall-detection --source 0 --num-poses 4 --min-detection-confidence 0.6

# same, near native speed: person boxes every third frame, regions carried between
uv run fall-detection --source 0 --num-poses 4 --detect-interval 3

# recorded clip, headless, logs to a file
uv run fall-detection --source clip.mp4 --no-display --log-file run.log

# many poses tracked but only the strongest one kept downstream
uv run fall-detection --source 0 --num-poses 4 --best-only

# per-frame detail
uv run fall-detection --source clip.mp4 --log-level DEBUG
```

Press `q` or `Esc` to quit the window. The first run downloads the model bundle
into `models/` (gitignored).

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--source` | `0` | camera index, `/dev/video*`, a video file path, or a stream URL (`rtsp://...`, or DroidCam/IP Webcam `http://<phone-ip>:4747/mjpegfeed`) |
| `--model` | `full` | `lite` / `full` / `heavy` bundle |
| `--model-path` | – | use a local `.task` file instead of downloading |
| `--cache-dir` | `models/` | where downloaded bundles live |
| `--num-poses` | `1` | maximum people tracked |
| `--best-only` | off | keep only the highest-scoring person |
| `--gpu` | off | try the GPU delegate (Linux; falls back to CPU) |
| `--no-display` | off | headless, no cv2 window |
| `--display-max-width` | `1280` | cap initial display window width in px, preserves aspect ratio |
| `--no-smoothing` | off | disable One-Euro landmark filtering |
| `--min-detection-confidence` / `--min-presence-confidence` / `--tracking-confidence` | `0.5` | BlazePose thresholds |
| `--max-frames` | – | stop after N frames (testing aid) |
| `--detector` | `auto` | `auto` / `native` / `cascade` (see below) |
| `--person-model` | `efficientdet_lite0` | cascade person-box model |
| `--person-score` | `0.4` | person box confidence floor |
| `--crop-padding` | `0.15` | grow each box so limbs are not clipped |
| `--crop-workers` | `min(4, --num-poses)` | landmarkers running crops in parallel |
| `--detect-interval` | `1` | run the person detector every Nth frame |
| `--min-box-px` | `48` | ignore person boxes smaller than this |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `--log-file` | – | mirror logs to a file |
| `--output` | – | write annotated frames to this video file (e.g. `out.mp4`); file sources only, rejected for live sources |
| `--no-fall-detection` | off | pose-only, skip the fall-state layer |
| `--body-mass-kg` | – | deprecated compatibility flag; has no effect on RGB fall decisions |
| `--fall-config` | – | validated TOML threshold/ROI configuration |
| `--fall-profile` | `balanced` | `sensitive` / `balanced` / `precision`; overrides the TOML profile |
| `--fall-alert-log` | – | append schema-v1 detected/recovered incident JSONL |
| `--fall-telemetry-log` | – | append schema-v1 per-decision feature/evidence telemetry JSONL |
| `--fall-debug-overlay` | off | show evidence duration/fraction/coverage and observation age |

Exit codes: `0` success, `1` runtime failure, `2` bad arguments.

Network sources (`rtsp://`, `http://`) get low-latency FFmpeg flags by default
(`rtsp_transport;tcp|fflags;nobuffer|max_delay;0`) and `CAP_PROP_BUFFERSIZE=1`,
so a stalled phone stream drops old frames instead of piling up lag. OpenCV's
native logger stays silent unless `--log-level DEBUG`, which also turns it
verbose for diagnosing handshake stalls. Override the FFmpeg options (e.g. to
try `udp` transport, or add a `stimeout` for a flaky Wi-Fi link) by setting
`OPENCV_FFMPEG_CAPTURE_OPTIONS` yourself before running — the default only
applies via `setdefault`, so it never overrides an explicit value.

## Choosing a strategy

`--detector auto` (the default) runs **native** at `--num-poses 1` and the
**cascade** above it. Force either with `--detector native|cascade`.

**Native** hands the whole frame to one landmarker. BlazePose proposes a single
dominant region per frame, so extra people are silently dropped — the landmarks
never arrive and nothing warns you.

**Cascade** runs `efficientdet_lite0` over the frame, keeps the `person` class,
pads each box, letterboxes it to a square, and gives each one its own
single-person landmarker. Crops run in parallel across `--crop-workers`.

### Measured here

120 frames, 1280x720, three bodies standing apart, `full` model, CPU, 16 cores:

| run | fps | frames with all 3 people |
| --- | --- | --- |
| `--detector native` | 17.2 | 63 / 120 |
| `--detector cascade` | 13.9 | **120 / 120** |
| `--detector cascade --crop-workers 1` | 8.2 | 120 / 120 |
| `--detector cascade --detect-interval 3` | **17.7** | **120 / 120** |
| `--detector cascade --model lite --detect-interval 3` | 17.0 | 120 / 120 |

One batch, same machine and clip; fps wanders about a point run to run, the
recall column does not. Native also reported a fourth, non-existent person on 3
of those frames.
With `--detect-interval 3` the cascade costs nothing against native here and
still finds everybody, so that is the setting to reach for on a busy scene.

Single still frames, same bodies at several spacings: native returned 1 pose
where the cascade returned 3 or 4. The gap widens as people get closer.

### Choices behind the numbers

- **`efficientdet_lite0` int8 over `lite2` float16.** Probed side by side, the
  smaller bundle recalled people the larger one missed entirely.
- **Crops are letterboxed, not stretched or widened.** The landmarker squares its
  input, so a tall box gets squashed. Padding lifted torso confidence from 0.91 to
  0.99 on edge-clipped bodies; widening the box with real pixels dragged in the
  neighbour and scored worse.
- **`z` is rescaled with the crop.** It shares `x`'s scale, so a crop-local `z`
  copied straight out reads far too deep. `world_landmarks` are left alone —
  metric and hip-centred, they are already correct per crop.
- **The cascade stays on CPU.** GPU delegates are bound to their creating thread
  and crops are inferred in parallel; `--gpu --detector cascade` warns and uses
  CPU. `--detector native --gpu` does use the GPU.

## Layout

| Module | Role |
| --- | --- |
| `pose.py` | `PoseConfig`, `PersonPose`, `PoseDetector` — the core wrapper |
| `engine.py` | `NativeEngine` / `CascadeEngine` behind one `infer()` |
| `person_detector.py` | `PersonBox`, `PersonDetector` — the cascade's first stage |
| `cascade.py` | crops, letterboxing, parallel landmarking, coordinate remap |
| `strategy.py` | `Strategy` enum, importable without mediapipe |
| `models.py` | bundle URL resolution, atomic download, caching |
| `runner.py` | `VideoFileRunner` (VIDEO) and `LiveStreamRunner` (LIVE_STREAM) |
| `tracking.py` | `IdentityTracker` — stable ids across frames |
| `smoothing.py` | One-Euro filter for landmark jitter |
| `drawing.py` | skeleton / bbox / HUD overlay |
| `fall_config.py` | validated profiles, overrides, and furniture ROI polygons |
| `fall_evidence.py` | finite pixel geometry and temporal RGB derivatives |
| `fall_fsm.py` | temporal state transitions and alert evidence levels |
| `fall_state.py` | per-person extractors plus durable incident history |
| `fall_telemetry.py` | schema-v1 telemetry and incident JSONL records |
| `evaluation.py` | manifest validation, four replay strategies, event metrics |
| `logging_config.py` | `setup_logging`, rate limiting, native-log quieting |
| `cli.py` | argument parsing and wiring |
| `geometry.py` | pure-numpy 2D convex hull / point-to-polygon distance |
| `biomechanics.py` | De Leva center-of-mass, postural instability index *(offline)* |
| `discriminators.py` | physics-based ADL (activities-of-daily-living) discriminators *(offline)* |
| `kalman.py` | extended Kalman filter landmark stabilizer, occlusion-tolerant *(offline)* |

`geometry.py`, `biomechanics.py`, `discriminators.py`, and `kalman.py` are
unit-tested (`test_geometry.py`, `test_biomechanics.py`, `test_discriminators.py`,
`test_kalman.py`) but marked *(offline)* because nothing in
`cli.py → runner.py → fall_state.py → fall_fsm.py → fall_evidence.py` calls into
them — treat them as a physics reference / future-use layer, not active code.
`geometry.py` is the one exception reached at runtime: `biomechanics.py` imports
its convex-hull helpers.

## Configuration

`--fall-config` takes a TOML file of thresholds and furniture ROIs;
[`config/fall_detection.example.toml`](config/fall_detection.example.toml) is the
annotated starting point — copy it and edit. `profile` selects one of
`sensitive` / `balanced` / `precision` as a baseline (`--fall-profile` overrides
it from the CLI without editing the file). Full semantics of every threshold
group (`[dynamic]`, `[posture]`, `[timing]`, `[quality]`, `[[furniture_rois]]`)
are in [docs/fall-detection.md § Profiles and configuration](docs/fall-detection.md#profiles-and-configuration).

## Testing and development

- Package manager is **uv**: `uv sync` to install, `uv run <cmd>` to run
  anything.
- `uv run pytest` runs the suite (258 tests, scoped to `tests/` via
  `pyproject.toml`).
- This is TDD-developed code: write the failing test first, watch it fail,
  then implement.
- `tests/` intentionally has no `__init__.py`, so pytest's prepend import mode
  puts `tests/` on `sys.path`. Import shared fixtures with a **bare**
  `from conftest import make_person, standing_pose`, not
  `from tests.conftest import ...`. Don't add `tests/__init__.py`.
- Shared synthetic `PersonPose` fixtures: `tests/conftest.py` (basic builders)
  and `tests/synthetic_falls.py` (full keyframed standing→lying sequence for
  FSM/orchestrator tests).
- No lint, format, or type-check tooling is configured (no ruff/black/mypy/
  pre-commit), and no CI exists yet — don't invent commands for either.
- No `print()` anywhere in the package — every module uses
  `logger = logging.getLogger(__name__)`.

## Labelled URL batch regression

For a direct list of labelled video URLs, run the complete regression gate. The
default descriptor is [`urfd-github-samples.toml`](evaluation/batches/urfd-github-samples.toml):
two labelled RGB examples published by the URFD evaluation mirror. The task
keeps `mise run test` offline, runs the committed offline regressions first,
then downloads and classifies one URL at a time.

```bash
mise run batch_regression
```

The default `FALL_DATA_BATCH` and `FALL_DATA_RESULT_LOG` are configured in
[`mise.toml`](mise.toml). Change those values there to use another descriptor
or output path; `FALL_DATA_ROOT` optionally selects the download directory
(default `.fall-data`). The result log must be new or empty: without source
checksums, appending a later run could silently mix different versions of a
remote file. Successful inputs are deleted only after their per-clip result is
durably written; failed inputs remain under the data root for diagnosis.

The included batch is an executable two-clip sample, not the complete 70-trial
URFD corpus. A full-corpus descriptor needs accessible direct HTTPS URLs and a
binary label for every trial:

```toml
schema_version = 1
dataset = "my-dataset"
batch = "2026-08-28"

[[clips]]
id = "fall-001"
url = "https://example.org/fall-001.mp4"
label = "fall"

[[clips]]
id = "adl-001"
url = "https://example.org/adl-001.mp4"
label = "normal"
```

URLs must be HTTPS and each clip ID must be unique. Each `clip_result` JSONL
record contains the actual/predicted labels, `TP`/`TN`/`FP`/`FN` outcome, and
detected incidents. The final `summary` record is computed from those JSONL
records and contains the aggregate confusion matrix plus accuracy, precision,
recall, and F1. Download or inference failures are recorded but excluded from
the metric denominator, and make the command exit non-zero. This is a
clip-level metric: a clip predicts `fall` when it emits at least one detected
incident. No source checksum is recorded or verified, so the result cannot
prove which remote file version was evaluated.

## Approved checksum-pinned video batches

Use `fall-data run` for the complete manifest-bound lifecycle. It downloads
the batch, processes each clip headlessly with the default pose and fall
configuration, writes durable incident summaries, then removes only inputs
whose successful classification was recorded.

```bash
uv run fall-data run \
  --batch evaluation/batches/urfd.example.toml \
  --result-log results/urfd-run.jsonl \
  --data-root datasets
```

`--result-log` is required and must be outside the batch directory. Both
stdout and the append-only result log use schema-v1 JSON Lines. Stdout emits
`download`, `classify`, `delete`, and `complete` progress events with
`completed`, `total`, and `percent`; the result log additionally records the
descriptor and manifest fingerprints, checksums, frame counts, elapsed time,
and detected/recovered incidents. Re-running the same command resumes from
the durable records: it does not classify a successful clip again, and retries
only a failed cleanup after re-verifying its checksum.

## Replay regression

The repository commits numerical feature traces, checksums, and labels—not
videos or model bundles. There are two committed replay sets, and they mean
different things — don't conflate them.

**`local-falls`** — real MediaPipe traces extracted from real video (12
clips: 10 labelled falls, 2 person-present negatives). This is the one
accuracy claim in this repo that reflects real-world behavior.

```bash
uv run fall-evaluate \
  --manifest evaluation/manifests/local-falls.toml \
  --strategy temporal-fsm
```

To regenerate the trace from a checkout where raw assets live outside an
isolated worktree:

```bash
uv run python scripts/extract_fall_traces.py \
  --manifest evaluation/manifests/local-falls.toml \
  --output evaluation/traces/local-regression-v2.jsonl \
  --source-root /path/to/repository-root \
  --model-path /path/to/pose_landmarker_full.task \
  --fall-profile balanced --force
```

Extraction records the pose-model SHA-256 and a canonical fingerprint of the
exact fall configuration, including furniture ROIs. Replay must use that same
configuration; a profile, threshold, or ROI mismatch requires re-extraction.

**`synthetic-adl`** — hand-authored `FallFeatures` streams (6 fall
geometries, 8 ADL hard negatives — fast sit, brief lie-down, bend, squat,
kneel, jump, brisk walk, deliberate floor-sit — and 2 degenerate inputs).
These are **not recordings of a real subject**; they exist to pressure-test
the FSM against motions the small `local-falls` negative set doesn't cover,
and to pin exact per-clip behavior so a threshold change in `fall_fsm.py` /
`fall_evidence.py` can't silently flip clips in opposite directions and
still pass an aggregate metric. Its recall/precision/F1 describe how the
FSM responds to these authored inputs, not measured system accuracy — never
quote them as the latter.

```bash
uv run fall-evaluate \
  --manifest evaluation/manifests/synthetic-adl.toml \
  --strategy temporal-fsm
```

Regenerate deterministically (no video, no model bundle needed) from the
scenario catalog in `src/fall_detection/synthetic_traces.py`:

```bash
uv run python scripts/generate_synthetic_traces.py
```

Both regressions are gated by pytest (`tests/test_evaluation.py`,
`tests/test_synthetic_regression.py`), not by `fall-evaluate` itself —
`fall-evaluate` prints a JSON report and exits 0 regardless of the numbers
inside it; it is not a pass/fail gate on its own. `mise run regression_test`
runs the test suite and replays both manifests in sequence.

## Notes that bite

- **MediaPipe world landmarks are y-down, not y-up.** `world_landmarks`
  (metric, hip-centered) are **not** flipped to a standard "y-up" convention —
  `y` increases **downward**, same as normalized image landmarks. So
  "vertical"/"up" in any physics code is the `-y` direction, and the ground
  plane is `(x, z)`. Verified against a real MediaPipe inference and documented
  in both `src/fall_detection/biomechanics.py:8-11` and `tests/conftest.py:3-6`.
  Standard 3D-graphics/physics axis intuition is backwards here — don't assume it.
- **Timestamps must strictly increase.** VIDEO derives them from the file fps;
  LIVE_STREAM uses `time.monotonic_ns()` rebased to stream start, never the wall
  clock. `PoseDetector` bumps a non-increasing timestamp and warns instead of
  letting the C++ layer throw.
- **Results always arrive off the capture thread.** Native uses MediaPipe's
  dispatcher thread; the cascade gets its own worker, which also builds its
  landmarkers there so no delegate is ever touched from a foreign thread. Both
  stamp a heartbeat and push into a `queue.Queue(maxsize=2)`, dropping the stale
  result when full. Drawing and I/O happen on the main thread.
- **Native C++ logs are muted around model creation only.** MediaPipe writes
  those straight to file descriptor 2, before glog starts, so `GLOG_minloglevel`
  never sees them; `suppress_native_stderr()` redirects the fd instead, and steps
  aside at `--log-level DEBUG`. One benign line still escapes on first inference
  (`landmark_projection_calculator ... square ROI`) — the cascade's crops *are*
  square, so it is describing the supported case.
- **`--detect-interval > 1` carries regions forward.** Between detector runs the
  previous frame's pose boxes are reused. The carry is abandoned the moment it
  holds fewer regions than the detector last found, so someone walking into frame
  is not hidden until the next scheduled run.
- **No re-identification.** MediaPipe can swap the order of `pose_landmarks`
  between frames, so ids come from `IdentityTracker` (greedy nearest-centroid).
- **Live runs self-heal.** If no result arrives within the stall timeout, or the
  camera dies, the landmarker and capture are closed and rebuilt with exponential
  backoff, giving up after 5 consecutive failures.

## Programmatic use

```python
from fall_detection.logging_config import setup_logging
from fall_detection.models import ModelVariant
from fall_detection.pose import PoseConfig, RunningMode, best_person
from fall_detection.engine import build_engine

setup_logging("INFO")
config = PoseConfig(model_variant=ModelVariant.FULL, num_poses=4)   # -> cascade
engine = build_engine(config, RunningMode.IMAGE)
try:
    persons = engine.infer(bgr_frame, timestamp_ms=0)
    subject = best_person(persons)          # highest torso visibility
finally:
    engine.close()
```

`build_engine` honours `config.strategy`; `PoseDetector` is still there if you
want the raw single-landmarker path.

Each `PersonPose` carries `landmarks` (33 normalized), `world_landmarks` (metric),
`score`, `bbox` and `centroid` — the inputs a fall heuristic needs.

## Further reading

| Topic | Doc |
| --- | --- |
| Guided, file-by-file onboarding reading order | [docs.md](docs.md) |
| Fall-detection FSM, config profiles, telemetry, replay evaluation, limits | [docs/fall-detection.md](docs/fall-detection.md) |
| Dev/agent conventions and repo-specific gotchas | [CLAUDE.md](CLAUDE.md) |
