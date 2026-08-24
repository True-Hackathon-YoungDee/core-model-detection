"""Frame sources and processing loops.

Two loops, matched to the two data sources:

* :class:`VideoFileRunner` -- decoded files, synchronous inference.
* :class:`LiveStreamRunner` -- cameras, non-blocking inference plus a supervisor
  that rebuilds the engine and the capture whenever the stream stalls or the
  driver dies.

Both go through :mod:`fall_detection.engine`, so either can run the native
full-frame landmarker or the cascade without knowing which it got.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2

from .drawing import annotate
from .engine import CascadeEngine, build_engine, resolve_strategy
from .logging_config import RateLimiter
from .pose import (
    PersonPose,
    PoseConfig,
    PoseDetector,
    RunningMode,
    Strategy,
    best_person,
    build_persons,
)
from .smoothing import LandmarkSmoother
from .tracking import IdentityTracker

logger = logging.getLogger(__name__)

_QUIT_KEYS = (ord("q"), ord("Q"), 27)


class FpsMeter:
    """Periodic throughput logging; keeps a 30 fps loop from flooding INFO."""

    def __init__(self, label: str, interval: float = 5.0) -> None:
        self.label = label
        self.interval = interval
        self.frames = 0
        self._window_frames = 0
        self._started = time.monotonic()
        self._last_log = self._started

    def tick(self, persons: int = 0) -> None:
        self.frames += 1
        self._window_frames += 1
        now = time.monotonic()
        elapsed = now - self._last_log
        if elapsed < self.interval:
            return
        logger.info(
            "%s: %d frames total | %.1f fps | persons=%d",
            self.label,
            self.frames,
            self._window_frames / elapsed,
            persons,
        )
        self._window_frames = 0
        self._last_log = now

    def summary(self) -> None:
        elapsed = max(time.monotonic() - self._started, 1e-6)
        logger.info(
            "%s finished: %d frames in %.1fs (%.1f fps average)",
            self.label,
            self.frames,
            elapsed,
            self.frames / elapsed,
        )


class Display:
    """Optional cv2 window. Disables itself if the host has no GUI."""

    def __init__(
        self,
        enabled: bool,
        window: str = "fall-detection | pose",
        max_width: int = 1280,
    ) -> None:
        self.enabled = enabled
        self.window = window
        self.max_width = max_width
        self._opened = False

    def show(self, frame) -> bool:
        """Return False when the user asked to quit."""
        if not self.enabled:
            return True
        try:
            if not self._opened:
                cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
                height, width = frame.shape[:2]
                if width > self.max_width:
                    scale = self.max_width / width
                    cv2.resizeWindow(
                        self.window, self.max_width, int(height * scale)
                    )
                self._opened = True
            cv2.imshow(self.window, frame)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as error:
            logger.error("display unavailable (%s); continuing headless", error)
            self.enabled = False
            return True
        if key in _QUIT_KEYS:
            logger.info("quit key pressed")
            return False
        return True

    def close(self) -> None:
        if self._opened:
            cv2.destroyAllWindows()
            self._opened = False


class VideoWriter:
    """Optional annotated-frame video sink; mirrors Display's lazy-open pattern.

    Opens on the first frame (frame size isn't known until then) and stays
    open for the life of the object, so it's safe to hand the same instance
    across a caller's retry loop without truncating what's already written.
    """

    def __init__(self, path: str | Path | None, fps: float) -> None:
        self.path = str(path) if path else None
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def write(self, frame) -> None:
        if not self.enabled:
            return
        if self._writer is None:
            height, width = frame.shape[:2]
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.path, fourcc, self.fps, (width, height))
            if not writer.isOpened():
                logger.error("failed to open output video for writing: %s", self.path)
                self.path = None
                return
            self._writer = writer
            logger.info("writing annotated output to %s (%dx%d @ %.1f fps)", self.path, width, height, self.fps)
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class PosePipeline:
    """Detected people -> stable ids -> smoothed landmarks."""

    def __init__(
        self,
        smoothing: bool = True,
        best_only: bool = False,
        max_distance: float = 0.15,
        max_unseen_s: float = 2.5,
        on_person_lost: Callable[[int], None] | None = None,
    ) -> None:
        self.best_only = best_only
        self.smoother = LandmarkSmoother() if smoothing else None
        self.tracker = IdentityTracker(
            max_distance=max_distance,
            max_unseen_s=max_unseen_s,
            on_lost=self._make_on_lost(on_person_lost),
        )

    def _make_on_lost(
        self, on_person_lost: Callable[[int], None] | None
    ) -> Callable[[int], None] | None:
        callbacks = [cb for cb in (self.smoother.forget if self.smoother else None, on_person_lost) if cb]
        if not callbacks:
            return None

        def _fanout(person_id: int) -> None:
            for callback in callbacks:
                callback(person_id)

        return _fanout

    def process(self, persons: list[PersonPose], t_seconds: float) -> list[PersonPose]:
        if self.best_only and persons:
            chosen = best_person(persons)
            persons = [chosen] if chosen is not None else []
        if not persons:
            self.tracker.assign([], t_seconds)
            return []

        ids = self.tracker.assign([person.centroid for person in persons], t_seconds)
        for person, person_id in zip(persons, ids):
            person.person_id = person_id
            if self.smoother is not None:
                person.landmarks = self.smoother.smooth(
                    person_id, person.landmarks, t_seconds
                )
        return persons

    def reset(self) -> None:
        self.tracker.reset()
        if self.smoother is not None:
            self.smoother.reset()


def _configure_capture_logging() -> None:
    """Silence OpenCV's native logger unless the root logger is at DEBUG.

    Mirrors ``logging_config.suppress_native_stderr``'s DEBUG gate: normally
    OpenCV's C++ layer bypasses ``logging`` and spams stderr directly on every
    dropped/malformed frame from a flaky network source, so keep it silent
    unless the caller explicitly asked for debug output.
    """
    level = (
        cv2.utils.logging.LOG_LEVEL_DEBUG
        if logging.getLogger().isEnabledFor(logging.DEBUG)
        else cv2.utils.logging.LOG_LEVEL_SILENT
    )
    cv2.utils.logging.setLogLevel(level)


def _open_capture(source: int | str, label: str) -> cv2.VideoCapture:
    _configure_capture_logging()
    if isinstance(source, str) and source.startswith(("rtsp://", "http://", "https://")):
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|fflags;nobuffer|max_delay;0",
        )
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open {label} source: {source!r}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    logger.info("%s opened: %s (%dx%d @ %.1f fps)", label, source, width, height, fps)
    return capture


class VideoFileRunner:
    """Decoded-file processing, one frame at a time."""

    def __init__(
        self,
        config: PoseConfig,
        path: str | Path,
        display: bool = True,
        display_max_width: int = 1280,
        smoothing: bool = True,
        best_only: bool = False,
        max_frames: int | None = None,
        max_unseen_s: float = 2.5,
        on_frame: Callable[[list[PersonPose], float, Any], None] | None = None,
        overlay: Callable[[Any, list[PersonPose]], Any] | None = None,
        on_person_lost: Callable[[int], None] | None = None,
        output: str | Path | None = None,
    ) -> None:
        self.config = config
        self.path = str(path)
        self.display = Display(display, max_width=display_max_width)
        self.pipeline = PosePipeline(
            smoothing=smoothing,
            best_only=best_only,
            max_unseen_s=max_unseen_s,
            on_person_lost=on_person_lost,
        )
        self.max_frames = max_frames
        self.strategy = resolve_strategy(config)
        self.on_frame = on_frame
        self.overlay = overlay
        self.output = output

    def run(self) -> int:
        capture = _open_capture(self.path, "video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.warning("video reports fps=%.1f; assuming 30.0", fps)
            fps = 30.0

        meter = FpsMeter("video")
        writer = VideoWriter(self.output, fps)
        frame_index = 0
        engine = None
        try:
            engine = build_engine(self.config, RunningMode.VIDEO, self.strategy)
            while True:
                ok, frame = capture.read()
                if not ok:
                    logger.info("end of video after %d frames", frame_index)
                    break

                timestamp_ms = int(frame_index / fps * 1000)
                t_seconds = timestamp_ms / 1000.0
                persons = self.pipeline.process(engine.infer(frame, timestamp_ms), t_seconds)
                logger.debug(
                    "frame=%d ts=%dms persons=%d", frame_index, timestamp_ms, len(persons)
                )

                if self.on_frame:
                    self.on_frame(persons, t_seconds, frame)

                if self.display.enabled or writer.enabled:
                    hud = (
                        f"VIDEO {self.config.model_variant}/{engine.label} "
                        f"persons={len(persons)}"
                    )
                    canvas = annotate(frame, persons, hud)
                    if self.overlay:
                        canvas = self.overlay(canvas, persons)
                    writer.write(canvas)
                    if not self.display.show(canvas):
                        break

                frame_index += 1
                meter.tick(len(persons))
                if self.max_frames is not None and frame_index >= self.max_frames:
                    logger.info("reached --max-frames=%d", self.max_frames)
                    break
        finally:
            capture.release()
            if engine is not None:
                engine.close()
            self.display.close()
            writer.close()
            meter.summary()
        return frame_index


class _NativeProducer:
    """MediaPipe's own async path: submit and let the dispatcher thread answer."""

    label = "native"

    def __init__(self, config: PoseConfig, publish) -> None:
        self._publish = publish
        self.detector = PoseDetector(
            config, RunningMode.LIVE_STREAM, result_callback=self._on_result
        )

    def _on_result(self, result: Any, output_image: Any, timestamp_ms: int) -> None:
        """Runs on MediaPipe's dispatcher thread: no drawing, no I/O."""
        try:
            self._publish(build_persons(result), timestamp_ms)
        except Exception:  # never let an exception cross the ctypes boundary
            logger.exception("error inside result callback")

    def submit(self, frame, timestamp_ms: int) -> None:
        self.detector.detect_async(frame, timestamp_ms)

    def close(self) -> None:
        self.detector.close()


class _CascadeProducer:
    """The cascade is synchronous, so it gets its own thread.

    The capture loop drops frames into a one-slot inbox and moves on; the worker
    owns the estimator outright, which also keeps every delegate on the thread
    that created it.
    """

    label = "cascade"

    def __init__(self, config: PoseConfig, publish, start_timeout: float = 180.0) -> None:
        self._config = config
        self._publish = publish
        self._inbox: queue.Queue = queue.Queue(maxsize=1)
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self.engine: CascadeEngine | None = None
        self._thread = threading.Thread(target=self._run, name="cascade", daemon=True)
        self._thread.start()
        # First run downloads two model bundles, so the wait is generous.
        if not self._ready.wait(start_timeout):
            raise TimeoutError("cascade worker did not start in time")
        if self._error is not None:
            raise self._error
        logger.info("cascade worker started")

    def _run(self) -> None:
        try:
            self.engine = CascadeEngine(self._config)
        except BaseException as error:  # surfaced to the supervisor by start()
            self._error = error
            return
        finally:
            self._ready.set()

        while True:
            item = self._inbox.get()
            if item is None:
                break
            frame, timestamp_ms = item
            try:
                self._publish(self.engine.infer(frame, timestamp_ms), timestamp_ms)
            except Exception:
                logger.exception("cascade worker failed on frame ts=%d", timestamp_ms)

    def submit(self, frame, timestamp_ms: int) -> None:
        """Drop-oldest: a stale frame is worthless next to the one just captured."""
        item = (frame, timestamp_ms)
        try:
            self._inbox.put_nowait(item)
        except queue.Full:
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                pass
            try:
                self._inbox.put_nowait(item)
            except queue.Full:
                pass

    def close(self) -> None:
        if not self._thread.is_alive():
            return
        try:
            self._inbox.put_nowait(None)
        except queue.Full:
            try:
                self._inbox.get_nowait()
                self._inbox.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            logger.warning("cascade worker did not stop within 10s")
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        logger.info("cascade worker stopped")


class LiveStreamRunner:
    """Camera processing with auto-recovery.

    Whichever producer is in use, results arrive on another thread and land in a
    bounded queue; the capture loop never blocks on inference.
    """

    def __init__(
        self,
        config: PoseConfig,
        source: int | str = 0,
        display: bool = True,
        display_max_width: int = 1280,
        smoothing: bool = True,
        best_only: bool = False,
        max_frames: int | None = None,
        stall_timeout: float = 3.0,
        first_result_grace: float = 10.0,
        max_restarts: int = 5,
        max_unseen_s: float = 2.5,
        on_frame: Callable[[list[PersonPose], float, Any], None] | None = None,
        overlay: Callable[[Any, list[PersonPose]], Any] | None = None,
        on_person_lost: Callable[[int], None] | None = None,
    ) -> None:
        self.config = config
        self.source = source
        self.display = Display(display, max_width=display_max_width)
        self.pipeline = PosePipeline(
            smoothing=smoothing,
            best_only=best_only,
            max_unseen_s=max_unseen_s,
            on_person_lost=on_person_lost,
        )
        self.max_frames = max_frames
        self.stall_timeout = stall_timeout
        self.first_result_grace = first_result_grace
        self.max_restarts = max_restarts
        self.strategy = resolve_strategy(config)
        self.on_frame = on_frame
        self.overlay = overlay

        self._queue: queue.Queue[tuple[list[PersonPose], int]] = queue.Queue(maxsize=2)
        self._heartbeat = 0.0
        self._got_result = False
        self._dropped = 0
        self._limiter = RateLimiter(2.0)
        self._stop = False

    # --- called from the producer's thread --------------------------------
    def _publish(self, persons: list[PersonPose], timestamp_ms: int) -> None:
        """Stamp the heartbeat and enqueue, dropping the oldest when saturated."""
        self._heartbeat = time.monotonic()
        self._got_result = True
        item = (persons, timestamp_ms)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass
            self._dropped += 1
            if self._limiter.ready("drop"):
                logger.warning(
                    "result queue saturated; %d stale results dropped so far", self._dropped
                )

    def _build_producer(self):
        if self.strategy is Strategy.CASCADE:
            return _CascadeProducer(self.config, self._publish)
        return _NativeProducer(self.config, self._publish)

    # --- supervisor -------------------------------------------------------
    def run(self) -> int:
        backoff = 1.0
        total = 0
        failures = 0
        while not self._stop:
            producer = None
            capture = None
            try:
                producer = self._build_producer()
                capture = _open_capture(self.source, "camera")
                total += self._loop(producer, capture, total)
                self._stop = True
                backoff = 1.0
                failures = 0
            except KeyboardInterrupt:
                logger.info("interrupted by user")
                self._stop = True
            except Exception as error:
                failures += 1
                if failures >= self.max_restarts:
                    logger.error(
                        "stream failure: %s; giving up after %d attempts", error, failures
                    )
                    raise
                logger.warning(
                    "stream failure: %s; restart %d/%d in %.0fs",
                    error,
                    failures,
                    self.max_restarts,
                    backoff,
                )
                logger.debug("stream failure detail", exc_info=True)
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 8.0)
            finally:
                if capture is not None:
                    capture.release()
                if producer is not None:
                    producer.close()
                self._drain()
                self.pipeline.reset()
                self._got_result = False
        self.display.close()
        return total

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _loop(self, producer, capture: cv2.VideoCapture, already: int) -> int:
        meter = FpsMeter("camera")
        start_ns = time.monotonic_ns()
        self._heartbeat = time.monotonic()
        sent = 0
        read_failures = 0
        persons: list[PersonPose] = []

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    read_failures += 1
                    if read_failures >= 10:
                        raise RuntimeError("camera stopped delivering frames")
                    time.sleep(0.05)  # a dead device returns instantly; do not spin
                    continue
                read_failures = 0

                # Monotonic clock, not wall clock: NTP steps or camera retries
                # must never produce a non-increasing timestamp.
                timestamp_ms = (time.monotonic_ns() - start_ns) // 1_000_000
                producer.submit(frame, int(timestamp_ms))
                sent += 1

                grace = 0.0 if self._got_result else self.first_result_grace
                if time.monotonic() - self._heartbeat > self.stall_timeout + grace:
                    raise TimeoutError(
                        f"no results for {self.stall_timeout + grace:.0f}s after {sent} frames"
                    )

                # Nothing to consume means inference is still running: keep the
                # previous overlay rather than blocking the capture loop.
                try:
                    fresh, result_ts = self._queue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    result_t_seconds = result_ts / 1000.0
                    persons = self.pipeline.process(fresh, result_t_seconds)
                    logger.debug(
                        "result ts=%dms persons=%d ids=%s",
                        result_ts,
                        len(persons),
                        [person.person_id for person in persons],
                    )
                    if self.on_frame:
                        self.on_frame(persons, result_t_seconds, frame)

                if self.display.enabled:
                    hud = (
                        f"LIVE {self.config.model_variant}/{producer.label} "
                        f"persons={len(persons)}"
                    )
                    canvas = annotate(frame, persons, hud)
                    if self.overlay:
                        canvas = self.overlay(canvas, persons)
                    if not self.display.show(canvas):
                        return sent

                meter.tick(len(persons))
                if self.max_frames is not None and already + sent >= self.max_frames:
                    logger.info("reached --max-frames=%d", self.max_frames)
                    return sent
        finally:
            meter.summary()
            if self._dropped:
                logger.info("%d results dropped by backpressure this session", self._dropped)
