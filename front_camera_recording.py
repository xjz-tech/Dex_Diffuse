"""Record an already-open front camera without pausing during policy inference."""

from datetime import datetime
from pathlib import Path
import threading
import time

from diffusion_policy.real_world.video_recorder import VideoRecorder


class RecordingFrontCamera:
    """Own camera reads in one thread and expose fresh frames to the policy."""

    def __init__(self, camera, output_dir, *, filename_prefix="front", fps=30, timeout_ms=1000):
        self.camera = camera
        self.path = Path(output_dir).expanduser().resolve() / (
            f"{filename_prefix}_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        )
        self.fps = fps
        self.timeout_ms = timeout_ms
        self.frames_written = 0
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest = None
        self._consumed = None
        self._error = None
        self._closed = False
        self._thread = None

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, name="front-camera-recording", daemon=True
        )
        self._thread.start()
        print(f"[record] front camera START | {self.fps} FPS | {self.path}", flush=True)

    def _run(self):
        writer = VideoRecorder.create_h264(
            fps=self.fps, codec="libx264", thread_count=1
        )
        try:
            while not self._stop.is_set():
                frame = self.camera.read(timeout_ms=self.timeout_ms)
                captured_at = time.monotonic()
                if self._stop.is_set():
                    break
                if not writer.is_ready():
                    writer.start(str(self.path), start_time=captured_at)
                # Resample by elapsed time so inference waits and camera jitter
                # do not speed up playback. The input stays full-resolution RGB.
                writer.write_frame(frame.color_rgb, frame_time=captured_at)
                self.frames_written = writer.next_global_idx
                with self._condition:
                    self._latest = frame
                    self._condition.notify_all()
        except Exception as exc:
            self._error = exc
        finally:
            try:
                # Flush delayed encoder packets and write the MP4 trailer.
                writer.stop()
            except Exception as exc:
                self._error = exc
            with self._condition:
                self._condition.notify_all()

    def raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError(f"Front camera recording failed: {self.path}") from self._error

    def read(self, timeout_ms):
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._latest is not self._consumed
                or self._error is not None
                or self._stop.is_set(),
                timeout=timeout_ms / 1000.0,
            )
            self.raise_if_failed()
            if self._stop.is_set():
                raise RuntimeError("Front camera recording has stopped")
            if not ready:
                raise TimeoutError(f"Front camera did not deliver a frame within {timeout_ms} ms")
            self._consumed = self._latest
            return self._latest

    def stop(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        try:
            if self._thread is not None:
                # The camera read has its own timeout; wait for encoding/flush
                # before stopping the pipeline used by the capture thread.
                self._thread.join()
        finally:
            self.camera.stop()
        if self._error is not None:
            print(f"[record] ERROR | {self._error} | {self.path}", flush=True)
        elif self.frames_written:
            print(
                f"[record] front camera SAVED | frames={self.frames_written} "
                f"duration={self.frames_written / self.fps:.3f}s | {self.path}",
                flush=True,
            )
        else:
            print("[record] no front camera frames captured; no video saved", flush=True)
