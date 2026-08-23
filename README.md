# fall-detection

MediaPipe pose core for a fall-detection pipeline: 33 body landmarks per person,
from a webcam or a video file, for one person or several. Everything reports
through `logging` — the package contains no `print`.

Two detection strategies ship side by side: the plain full-frame landmarker, and
a **cascade** that finds person boxes first and landmarks each body separately.
The cascade exists because the plain path loses people — see
[Choosing a strategy](#choosing-a-strategy).

Fall heuristics are **not** here yet; this is the keypoint layer they will sit on.

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
| `--source` | `0` | camera index, `/dev/video*`, or a video file path |
| `--model` | `full` | `lite` / `full` / `heavy` bundle |
| `--model-path` | – | use a local `.task` file instead of downloading |
| `--num-poses` | `1` | maximum people tracked |
| `--best-only` | off | keep only the highest-scoring person |
| `--gpu` | off | try the GPU delegate (Linux; falls back to CPU) |
| `--no-display` | off | headless, no cv2 window |
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

Exit codes: `0` success, `1` runtime failure, `2` bad arguments.

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
| `logging_config.py` | `setup_logging`, rate limiting, native-log quieting |
| `cli.py` | argument parsing and wiring |

## Notes that bite

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
