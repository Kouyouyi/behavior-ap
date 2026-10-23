"""Bounded, lossless background encoding; Isaac remains on its owning thread.

Inspired by the queue/drain design in rascal-yang/Isaac-sim-renderer,
commit eccdecde583b2e4fc74ef8bb2cc84ba9d6dc6d02, async_video.py.
This is an independent implementation for this runtime's PyAV writer contract.
It does not change rendering, image content, physics, or action frequency.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import threading
import time

import numpy as np


class AsyncVideoWriter:
    """Encode copied RGB frames in order, with backpressure and error propagation.

    The first frame specifies the image shape only: callers still explicitly
    submit it with write(). The worker owns the wrapped writer's entire lifetime.
    A failed encoder wakes blocked producers and close(), without relying on a
    queue sentinel that could be stranded behind a full queue.
    """

    def __init__(self, path, first_frame, fps=30, queue_size=8, *, writer_factory):
        if isinstance(queue_size, bool) or not isinstance(queue_size, int) or queue_size < 1:
            raise ValueError("queue_size must be a positive integer")
        if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
            raise ValueError("fps must be a positive integer")
        if not callable(writer_factory):
            raise TypeError("writer_factory must be callable")
        frame = self._copy_rgb(first_frame)
        self.path = Path(path)
        self.fps = fps
        self._shape = frame.shape
        self._capacity = queue_size
        self._condition = threading.Condition()
        self._pending = deque()
        self._ready = self._done = self._closing = False
        self._error = self._cleanup_error = None
        self._submitted = self._encoded = self._high_water = 0
        self._worker_s = self._enqueue_wait_s = self._close_s = 0.0
        self._snapshot = None
        self._worker = threading.Thread(
            target=self._run, args=(writer_factory, frame),
            name=f"video-encoder-{self.path.name}", daemon=True,
        )
        self._worker.start()
        with self._condition:
            self._condition.wait_for(lambda: self._ready or self._done or self._error is not None)
            error = self._error
        if error is not None:
            self._worker.join()
            raise RuntimeError(f"Video writer initialization failed: {self.path}") from error

    @staticmethod
    def _copy_rgb(frame):
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Video frames must be HxWx3 uint8 RGB arrays")
        if frame.shape[0] < 1 or frame.shape[1] < 1:
            raise ValueError("Video frames must have positive dimensions")
        return np.ascontiguousarray(frame).copy()

    def _fail(self, error, *, cleanup=False):
        with self._condition:
            if self._error is None:
                self._error = error
            elif cleanup:
                self._cleanup_error = repr(error)
            self._condition.notify_all()

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError(f"Video encoder failed: {self.path}") from self._error

    def _run(self, factory, first_frame):
        writer = None
        try:
            writer = factory(self.path, first_frame, fps=self.fps)
            with self._condition:
                self._ready = True
                self._condition.notify_all()
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._pending or self._closing)
                    if not self._pending:
                        break
                    frame = self._pending.popleft()
                    self._condition.notify_all()
                start = time.perf_counter()
                writer.write(frame)
                with self._condition:
                    self._encoded += 1
                    self._worker_s += time.perf_counter() - start
        except BaseException as error:
            self._fail(error)
        finally:
            if writer is not None:
                try:
                    writer.close()
                except BaseException as error:
                    self._fail(error, cleanup=True)
                try:
                    snapshot = dict(writer.metadata())
                    with self._condition:
                        self._snapshot = snapshot
                except BaseException as error:
                    self._fail(error, cleanup=True)
            with self._condition:
                if self._error is None and self._encoded != self._submitted:
                    self._error = RuntimeError("Video frame accounting mismatch after encoder drain")
                self._done = True
                self._condition.notify_all()

    def write(self, frame):
        with self._condition:
            self._raise_if_failed()
            if self._closing or self._done:
                raise RuntimeError("Cannot write video after close")
        owned_frame = self._copy_rgb(frame)
        if owned_frame.shape != self._shape:
            raise ValueError(f"Video resolution changed: {owned_frame.shape} != {self._shape}")
        with self._condition:
            start = time.perf_counter()
            while len(self._pending) >= self._capacity:
                self._raise_if_failed()
                if self._closing or self._done:
                    raise RuntimeError("Cannot write video after close")
                self._condition.wait()
            self._enqueue_wait_s += time.perf_counter() - start
            self._raise_if_failed()
            if self._closing or self._done:
                raise RuntimeError("Cannot write video after close")
            self._pending.append(owned_frame)
            self._submitted += 1
            self._high_water = max(self._high_water, len(self._pending))
            self._condition.notify_all()

    def close(self):
        start = time.perf_counter()
        with self._condition:
            already_done = self._done
            self._closing = True
            self._condition.notify_all()
            self._condition.wait_for(lambda: self._done)
        self._worker.join()
        with self._condition:
            if not already_done:
                self._close_s = max(self._close_s, time.perf_counter() - start)
            self._raise_if_failed()
            if self._submitted != self._encoded:
                raise RuntimeError("Video frame accounting mismatch after close")

    @property
    def frames(self):
        with self._condition:
            return self._encoded

    @property
    def closed(self):
        with self._condition:
            return self._done

    def metadata(self):
        """Return plain state only; never touch the live PyAV writer off-worker."""
        with self._condition:
            result = {
                "path": str(self.path), "frames": self._encoded, "fps": self.fps,
                "duration_seconds": self._encoded / self.fps,
                "width": self._shape[1], "height": self._shape[0],
                "codec": "h264", "profile": "High", "pixel_format": "yuv420p",
                "faststart_requested": True, "finalized": False,
            }
            if self._snapshot is not None:
                result.update(self._snapshot)
            result["finalized"] = bool(
                self._done and self._error is None and self._submitted == self._encoded
                and result.get("finalized", False)
            )
            result.update({
                "async_encoding": True,
                "frames_submitted": self._submitted,
                "frames_encoded": self._encoded,
                "queue_capacity_frames": self._capacity,
                "queue_high_water_frames": self._high_water,
                "queue_pending_frames": len(self._pending),
                "frames_unencoded": self._submitted - self._encoded,
                "dropped_frames": 0 if self._error is None else None,
                "encoder_worker_wall_s": self._worker_s,
                "enqueue_wait_wall_s": self._enqueue_wait_s,
                "close_and_drain_wall_s": self._close_s,
                "encoder_error": repr(self._error) if self._error is not None else None,
                "encoder_cleanup_error": self._cleanup_error,
                "timing_note": "Encoder worker time overlaps rollout; do not add it to elapsed wall time.",
            })
            return result
