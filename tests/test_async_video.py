import importlib.util
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np

from behavior_ap_sim.async_video import AsyncVideoWriter


class FakeWriter:
    def __init__(self, path, frame, fps=30):
        self.path, self.shape, self.fps = path, frame.shape, fps
        self.rows = []
        self.owner = threading.get_ident()
        self.closed = False
        self.close_calls = 0

    def write(self, frame):
        assert threading.get_ident() == self.owner
        self.rows.append(frame.copy())

    def close(self):
        assert threading.get_ident() == self.owner
        self.closed = True
        self.close_calls += 1

    def metadata(self):
        assert threading.get_ident() == self.owner
        return {"path": str(self.path), "frames": len(self.rows), "fps": self.fps,
                "width": self.shape[1], "height": self.shape[0], "finalized": self.closed}


class AsyncVideoTests(unittest.TestCase):
    def frame(self, value=0):
        return np.full((12, 16, 3), value, dtype=np.uint8)

    def make(self, factory=FakeWriter, queue_size=2):
        self.wrapped = None

        def create(*args, **kwargs):
            self.wrapped = factory(*args, **kwargs)
            return self.wrapped

        return AsyncVideoWriter("fake.mp4", self.frame(), writer_factory=create, queue_size=queue_size)

    def test_constructor_does_not_submit_and_close_is_idempotent(self):
        writer = self.make()
        self.assertEqual(writer.metadata()["frames_submitted"], 0)
        writer.close()
        writer.close()
        self.assertEqual(self.wrapped.rows, [])
        self.assertEqual(self.wrapped.close_calls, 1)
        self.assertTrue(writer.metadata()["finalized"])
        with self.assertRaisesRegex(RuntimeError, "after close"):
            writer.write(self.frame())

    def test_buffer_ownership_order_and_noncontiguous_input(self):
        writer = self.make()
        source = np.zeros((12, 32, 3), dtype=np.uint8)
        view = source[:, ::2, :]
        for value in range(25):
            source.fill(value)
            writer.write(view)
        source.fill(255)
        writer.close()
        self.assertEqual([int(row[0, 0, 0]) for row in self.wrapped.rows], list(range(25)))
        self.assertTrue(all(row.flags.c_contiguous for row in self.wrapped.rows))
        meta = writer.metadata()
        self.assertEqual((meta["frames_submitted"], meta["frames_encoded"], meta["dropped_frames"]), (25, 25, 0))
        self.assertEqual(meta["queue_pending_frames"], 0)
        self.assertTrue(meta["finalized"])
        self.assertNotEqual(self.wrapped.owner, threading.get_ident())

    def test_full_queue_applies_backpressure(self):
        entered, release, producer_done = threading.Event(), threading.Event(), threading.Event()

        class SlowWriter(FakeWriter):
            def write(inner, frame):
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("test gate timed out")
                super().write(frame)

        writer = self.make(SlowWriter, queue_size=2)
        try:
            writer.write(self.frame(1))
            self.assertTrue(entered.wait(1))
            writer.write(self.frame(2))
            writer.write(self.frame(3))
            producer = threading.Thread(target=lambda: (writer.write(self.frame(4)), producer_done.set()), daemon=True)
            producer.start()
            self.assertFalse(producer_done.wait(0.05))
            self.assertEqual(writer.metadata()["queue_high_water_frames"], 2)
            release.set()
            self.assertTrue(producer_done.wait(1))
            producer.join(1)
        finally:
            release.set()
            writer.close()
        self.assertEqual(len(self.wrapped.rows), 4)
        self.assertGreater(writer.metadata()["enqueue_wait_wall_s"], 0)

    def test_constructor_error_propagates(self):
        def broken(*args, **kwargs):
            raise OSError("constructor failure")

        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "initialization failed") as raised:
            AsyncVideoWriter("fake.mp4", self.frame(), writer_factory=broken)
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertLess(time.monotonic() - start, 2)

    def test_write_failure_wakes_blocked_producer_and_close(self):
        entered, release, producer_done = threading.Event(), threading.Event(), threading.Event()
        errors = []

        class BrokenWriter(FakeWriter):
            def write(inner, frame):
                entered.set()
                release.wait(2)
                raise OSError("write failure")

        writer = self.make(BrokenWriter, queue_size=1)
        writer.write(self.frame(1))
        self.assertTrue(entered.wait(1))
        writer.write(self.frame(2))

        def produce():
            try:
                writer.write(self.frame(3))
            except RuntimeError as error:
                errors.append(error)
            finally:
                producer_done.set()

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        self.assertFalse(producer_done.wait(0.05))
        release.set()
        self.assertTrue(producer_done.wait(1))
        producer.join(1)
        self.assertEqual(len(errors), 1)
        with self.assertRaisesRegex(RuntimeError, "encoder failed"):
            writer.close()
        with self.assertRaisesRegex(RuntimeError, "encoder failed"):
            writer.close()
        self.assertTrue(self.wrapped.closed)
        self.assertFalse(writer.metadata()["finalized"])
        self.assertEqual(writer.metadata()["frames_unencoded"], 2)

    def test_close_error_propagates(self):
        class BrokenClose(FakeWriter):
            def close(inner):
                super().close()
                raise OSError("close failure")

        writer = self.make(BrokenClose)
        writer.write(self.frame())
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "encoder failed") as raised:
            writer.close()
        self.assertEqual(str(raised.exception.__cause__), "close failure")
        self.assertLess(time.monotonic() - start, 2)
        self.assertFalse(writer.metadata()["finalized"])

    def test_metadata_failure_propagates(self):
        class BrokenMetadata(FakeWriter):
            def metadata(inner):
                raise OSError("metadata failure")

        writer = self.make(BrokenMetadata)
        with self.assertRaisesRegex(RuntimeError, "encoder failed"):
            writer.close()

    def test_invalid_frame_does_not_enter_queue(self):
        writer = self.make()
        try:
            with self.assertRaisesRegex(ValueError, "uint8 RGB"):
                writer.write(np.zeros((12, 16, 3), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "resolution changed"):
                writer.write(np.zeros((24, 16, 3), dtype=np.uint8))
        finally:
            writer.close()
        self.assertEqual(writer.metadata()["frames_submitted"], 0)

    @unittest.skipUnless(importlib.util.find_spec("av"), "PyAV not installed")
    def test_actual_h264_full_decode_preserves_frame_count(self):
        import av
        from behavior_ap_sim.backend import _VideoWriter

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "short.mp4"
            writer = AsyncVideoWriter(output, self.frame(), queue_size=2, writer_factory=_VideoWriter)
            for value in range(12):
                writer.write(self.frame(value * 15))
            writer.close()
            with av.open(str(output)) as container:
                stream = container.streams.video[0]
                self.assertEqual(stream.codec_context.name, "h264")
                self.assertEqual(stream.codec_context.profile, "High")
                self.assertEqual(stream.codec_context.pix_fmt, "yuv420p")
                frames = list(container.decode(video=0))
                self.assertEqual(len(frames), 12)
                self.assertEqual((frames[0].width, frames[0].height), (16, 12))
            self.assertEqual(writer.metadata()["frames_encoded"], 12)
            self.assertTrue(writer.metadata()["finalized"])


if __name__ == "__main__":
    unittest.main()
