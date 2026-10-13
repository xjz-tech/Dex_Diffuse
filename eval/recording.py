"""Physics-isolated recording support for the bulb2 evaluator.

The object axes never become Isaac Gym actors. They are drawn with Isaac Gym's
native debug Lines API, so they cannot affect simulation physics or actor state.
The parallel evaluator registers those lines before rendering camera sensors,
matching the native rendering path used by the original serial recording.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from isaacgym import gymapi, gymtorch


AXIS_COLORS_FLOAT = np.asarray(
    (
        (1.0, 0.05, 0.05),
        (0.05, 1.0, 0.05),
        (0.05, 0.15, 1.0),
    ),
    dtype=np.float32,
)
CUBOID_EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)


def _draw_step_number(rgb: np.ndarray, control_step: int) -> None:
    """Draw the global simulation control-step number in-place."""
    height, width = rgb.shape[:2]
    resolution_scale = min(width / 1280.0, height / 720.0)
    font_scale = max(0.6, 1.25 * resolution_scale)
    thickness = max(1, int(round(2.0 * resolution_scale)))
    margin = max(8, int(round(0.02 * min(width, height))))
    text = str(int(control_step))
    (_text_width, text_height), _baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    cv2.putText(
        rgb,
        text,
        (margin, margin + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def parse_vec3(value: str) -> Tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("expected three comma-separated values, got %r" % value)
    try:
        result = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("invalid 3-vector: %r" % value) from exc
    if not all(np.isfinite(component) for component in result):
        raise ValueError("3-vector contains NaN or Inf: %r" % value)
    return result


@dataclass(frozen=True)
class RecordingConfig:
    output_dir: Path
    environment_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    camera_position: Tuple[float, float, float] = (-0.10, 0.55, 0.10)
    camera_target: Tuple[float, float, float] = (-0.10, 0.00, -0.14)
    camera_horizontal_fov: float = 60.0
    axis_length: float = 0.20
    axis_thickness: float = 0.008

    def validate(self) -> None:
        if self.environment_index < 0:
            raise ValueError("recording environment index must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("recording width and height must be positive")
        if self.width % 2 or self.height % 2:
            raise ValueError("recording width and height must be even for H.264")
        if self.fps <= 0:
            raise ValueError("recording FPS must be positive")
        if not 1.0 <= self.camera_horizontal_fov < 180.0:
            raise ValueError("camera horizontal FOV must be in [1, 180)")
        if self.axis_length <= 0.0 or self.axis_thickness <= 0.0:
            raise ValueError("axis dimensions must be positive")
        if self.axis_thickness >= self.axis_length:
            raise ValueError("axis thickness must be smaller than axis length")
        if np.allclose(self.camera_position, self.camera_target):
            raise ValueError("camera position and target cannot be identical")


class RecordingConstructionHooks:
    """Create one recording camera for the selected environment only."""

    def __init__(self, config: RecordingConfig):
        self.config = config
        self._task_cls = None
        self._original_create_camera = None
        self._original_create_multiview_cameras = None
        self._original_set_camera = None
        self._camera_call_index = 0

    def install(self, task_cls) -> None:
        if self._task_cls is not None:
            raise RuntimeError("recording construction hooks are already installed")
        self._task_cls = task_cls
        self._original_create_camera = task_cls.create_camera
        self._original_create_multiview_cameras = task_cls.create_multiview_cameras
        self._original_set_camera = task_cls.set_camera
        self._camera_call_index = 0
        config = self.config

        def create_record_camera(task, *, env, isaac_gym):
            del task
            env_index = self._camera_call_index
            self._camera_call_index += 1
            if env_index != config.environment_index:
                # The task appends this placeholder to camera_handlers. The
                # custom set_camera() below ignores every unselected entry.
                return None
            properties = gymapi.CameraProperties()
            properties.enable_tensors = True
            properties.width = config.width
            properties.height = config.height
            properties.horizontal_fov = config.camera_horizontal_fov
            camera = isaac_gym.create_camera_sensor(env, properties)
            isaac_gym.set_camera_location(
                camera,
                env,
                gymapi.Vec3(*config.camera_position),
                gymapi.Vec3(*config.camera_target),
            )
            return camera

        def create_no_multiview_cameras(task, *, env, isaac_gym):
            del task, env, isaac_gym
            return []

        def set_record_camera(task):
            env_index = config.environment_index
            if env_index >= len(task.envs):
                raise ValueError(
                    "recording environment %d is outside %d environments"
                    % (env_index, len(task.envs))
                )
            camera = task.camera_handlers[env_index]
            if camera is None:
                raise RuntimeError(
                    "recording camera was not created for environment %d"
                    % env_index
                )
            tensor = task.gym.get_camera_image_gpu_tensor(
                task.sim,
                task.envs[env_index],
                camera,
                gymapi.IMAGE_COLOR,
            )
            # Keep only the selected camera tensor. render_all_camera_sensors()
            # therefore renders one camera regardless of the total env count.
            task.camera_obs = [gymtorch.wrap_tensor(tensor)]
            task.multiview_camera_obs = []

        task_cls.create_camera = create_record_camera
        task_cls.create_multiview_cameras = create_no_multiview_cameras
        task_cls.set_camera = set_record_camera

    def restore(self) -> None:
        if self._task_cls is None:
            return
        self._task_cls.create_camera = self._original_create_camera
        self._task_cls.create_multiview_cameras = (
            self._original_create_multiview_cameras
        )
        self._task_cls.set_camera = self._original_set_camera
        self._task_cls = None
        self._camera_call_index = 0


def _ffmpeg_executable() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    bundled = Path(
        "/home/carus/miniforge3/envs/dp/lib/python3.10/site-packages/"
        "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
    )
    if bundled.is_file():
        return str(bundled)
    raise RuntimeError("ffmpeg was not found")


class Mp4Recorder:
    """Stream RGB frames to FFmpeg without encoding on the sim thread."""

    def __init__(self, config: RecordingConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.path: Optional[Path] = None
        self.frame_count = 0
        self._frame_queue = None
        self._writer_thread = None
        self._writer_error = None
        self._start_monotonic = None
        self._pending_payload = None

    @property
    def active(self) -> bool:
        return self.process is not None

    def _next_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_suffix = "_env%d" % self.config.environment_index
        candidate = self.config.output_dir / (timestamp + env_suffix + ".mp4")
        suffix = 1
        while candidate.exists():
            candidate = self.config.output_dir / (
                "%s_%02d%s.mp4" % (timestamp, suffix, env_suffix)
            )
            suffix += 1
        return candidate

    def start(self) -> None:
        if self.active:
            return
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self._next_path()
        command = [
            _ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            "%dx%d" % (self.config.width, self.config.height),
            "-framerate",
            str(self.config.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._frame_queue = queue.Queue(maxsize=8)
        self._writer_error = None
        self._writer_thread = threading.Thread(
            target=self._write_frames,
            name="eval-mp4-writer",
            daemon=True,
        )
        self._writer_thread.start()
        self.frame_count = 0
        self._start_monotonic = time.monotonic()
        self._pending_payload = None
        print("[record] START | %s" % self.path, flush=True)

    def _write_frames(self) -> None:
        assert self.process is not None and self.process.stdin is not None
        assert self._frame_queue is not None
        while True:
            payload = self._frame_queue.get()
            try:
                if payload is None:
                    return
                if self._writer_error is None:
                    try:
                        self.process.stdin.write(payload)
                    except Exception as exc:  # propagated by write()/stop()
                        self._writer_error = exc
            finally:
                self._frame_queue.task_done()

    def _enqueue_payload(self, payload: bytes, copies: int) -> None:
        if self._writer_error is not None:
            raise RuntimeError("ffmpeg writer failed: %s" % self._writer_error)
        assert self._frame_queue is not None
        for _ in range(copies):
            while True:
                try:
                    # Repeated frames share the same immutable bytes object;
                    # the queue does not duplicate the image allocation.
                    self._frame_queue.put(payload, timeout=0.5)
                    break
                except queue.Full:
                    if self._writer_error is not None:
                        raise RuntimeError(
                            "ffmpeg writer failed: %s" % self._writer_error
                        )
            self.frame_count += 1

    def write(self, rgb: np.ndarray, captured_at: Optional[float] = None) -> None:
        """Add a frame while preserving wall-clock time in a constant-FPS MP4.

        The previous image is repeated until ``captured_at``.  Consequently,
        serial policy inference appears as the same pause seen in the viewer,
        instead of making the saved motion play several times too fast.
        """
        if not self.active:
            return
        expected = (self.config.height, self.config.width, 3)
        if rgb.shape != expected or rgb.dtype != np.uint8:
            raise ValueError(
                "recording frame must be uint8 %r, got %s %s"
                % (expected, rgb.shape, rgb.dtype)
            )
        assert self._start_monotonic is not None
        if captured_at is None:
            captured_at = time.monotonic()
        target_frame_count = max(
            1,
            int(round((captured_at - self._start_monotonic) * self.config.fps)),
        )
        payload = np.ascontiguousarray(rgb).tobytes()
        if self._pending_payload is not None:
            self._enqueue_payload(
                self._pending_payload,
                max(0, target_frame_count - self.frame_count),
            )
        self._pending_payload = payload

    def stop(self) -> Optional[Path]:
        if not self.active:
            return None
        assert self.process is not None
        process = self.process
        path = self.path
        assert self._start_monotonic is not None
        if self._pending_payload is not None:
            final_target = max(
                self.frame_count + 1,
                int(
                    round(
                        (time.monotonic() - self._start_monotonic)
                        * self.config.fps
                    )
                ),
            )
            self._enqueue_payload(
                self._pending_payload,
                final_target - self.frame_count,
            )
            self._pending_payload = None
        frames = self.frame_count
        frame_queue = self._frame_queue
        writer_thread = self._writer_thread
        assert frame_queue is not None and writer_thread is not None
        frame_queue.put(None)
        writer_thread.join()
        writer_error = self._writer_error
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read()
        returncode = process.wait()

        self.process = None
        self.path = None
        self.frame_count = 0
        self._frame_queue = None
        self._writer_thread = None
        self._writer_error = None
        self._start_monotonic = None
        self._pending_payload = None
        if returncode != 0 or writer_error is not None:
            raise RuntimeError(
                "ffmpeg exited with status %d: %s"
                % (
                    returncode,
                    stderr.decode("utf-8", errors="replace").strip()
                    or str(writer_error),
                )
            )
        print(
            "[record] STOP | frames=%d duration=%.3fs saved=%s"
            % (frames, frames / float(self.config.fps), path),
            flush=True,
        )
        return path


def _quaternion_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = quaternion / norm
    return np.asarray(
        (
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ),
            (
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ),
            (
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def _local_axis_cuboids(length: float, thickness: float) -> np.ndarray:
    """Return three 8-corner cuboids extending along local +X/+Y/+Z."""
    half = 0.5 * thickness
    cuboids = []
    for axis in range(3):
        corners = []
        for along in (0.0, length):
            for side_a in (-half, half):
                for side_b in (-half, half):
                    point = np.zeros(3, dtype=np.float64)
                    point[axis] = along
                    point[(axis + 1) % 3] = side_a
                    point[(axis + 2) % 3] = side_b
                    corners.append(point)
        cuboids.append(corners)
    return np.asarray(cuboids, dtype=np.float64)


class RecordingRuntime:
    """Draw pose markers and continuously stream the fixed camera to MP4."""

    def __init__(self, env, config: RecordingConfig):
        self.env = env
        self.config = config
        self.recorder = Mp4Recorder(config)
        self.last_rgb = None

        self.environment_index = config.environment_index
        if self.environment_index >= env.num_envs:
            raise ValueError(
                "recording environment %d is outside %d environments"
                % (self.environment_index, env.num_envs)
            )
        if not env.camera_handlers or not env.camera_obs:
            raise RuntimeError("recording camera was not created")
        self.camera_env = env.envs[self.environment_index]
        self.camera_handle = env.camera_handlers[self.environment_index]
        self.camera_tensor = env.camera_obs[0]
        self.quit_requested = False
        self.quit_reason = None

        # Camera rendering is driven explicitly after the object pose has been
        # refreshed. This avoids a second automatic camera render in step().
        env.camera_obs = None
        env.multiview_camera_obs = None

        if env.viewer is not None:
            env.gym.subscribe_viewer_keyboard_event(
                env.viewer,
                gymapi.KEY_Q,
                "QUIT",
            )
            env.gym.viewer_camera_look_at(
                env.viewer,
                self.camera_env,
                gymapi.Vec3(*config.camera_position),
                gymapi.Vec3(*config.camera_target),
            )

        self.local_cuboids = _local_axis_cuboids(
            config.axis_length,
            config.axis_thickness,
        )

        if env.viewer is not None:
            self._install_viewer_render_override()
        self.update_axes()
        self.recorder.start()
        # Seed the wall-clock video timeline before the first serial inference
        # call, so startup inference time appears as an initial held frame.
        self.capture_if_active()
        print(
            "[record] recording automatically | timing=wall_clock | "
            "env=%d | press Q to save and quit | "
            "camera_pos=%s camera_target=%s"
            % (
                self.environment_index,
                config.camera_position,
                config.camera_target,
            ),
            flush=True,
        )

    def _install_viewer_render_override(self) -> None:
        runtime = self

        def render_with_record_quit(task, mode="rgb_array"):
            del mode
            if task.viewer is None:
                return
            runtime.poll_viewer_events()
            if runtime.quit_requested:
                return

            if task.device != "cpu":
                task.gym.fetch_results(task.sim, True)

            if task.enable_viewer_sync:
                task.gym.step_graphics(task.sim)
                runtime._draw_viewer_axes(task.control_steps)
                task.gym.draw_viewer(task.viewer, task.sim, True)
                task.gym.sync_frame_time(task.sim)

                now = time.time()
                delta = now - task.last_frame_time
                if task.render_fps < 0:
                    render_dt = task.dt * task.control_freq_inv
                else:
                    render_dt = 1.0 / task.render_fps
                if delta < render_dt:
                    time.sleep(render_dt - delta)
                task.last_frame_time = time.time()
            else:
                task.gym.poll_viewer_events(task.viewer)

        self.env.render = lambda mode="rgb_array": render_with_record_quit(
            self.env,
            mode,
        )

    def poll_viewer_events(self) -> bool:
        """Consume viewer events and request a clean exit after the current step."""
        if self.quit_requested or self.env.viewer is None:
            return self.quit_requested
        if self.env.gym.query_viewer_has_closed(self.env.viewer):
            self.quit_requested = True
            self.quit_reason = "viewer closed"
        else:
            for event in self.env.gym.query_viewer_action_events(self.env.viewer):
                if event.action == "QUIT" and event.value > 0:
                    self.quit_requested = True
                    self.quit_reason = "Q/Esc"
                    break
                if event.action == "toggle_viewer_sync" and event.value > 0:
                    self.env.enable_viewer_sync = not self.env.enable_viewer_sync
        if self.quit_requested:
            print(
                "[record] %s received; finishing current step and saving video"
                % self.quit_reason,
                flush=True,
            )
        return self.quit_requested

    def _world_cuboids(self) -> np.ndarray:
        obj_state = (
            self.env._manip_obj_root_state[self.environment_index, :7]
            .detach()
            .cpu()
            .numpy()
        )
        position = obj_state[:3].astype(np.float64, copy=False)
        rotation = _quaternion_matrix_xyzw(obj_state[3:7])
        return self.local_cuboids @ rotation.T + position[None, None, :]

    def update_axes(self) -> None:
        """Cache the current cuboids for the next viewer and camera render."""
        self._viewer_cuboids = self._world_cuboids()
        self._viewer_lines_step = None

    def _draw_viewer_axes(self, control_step: int) -> None:
        """Draw once per control step, after the task's latest clear_lines()."""
        if self._viewer_lines_step == control_step:
            return
        num_lines = 3 * len(CUBOID_EDGES)
        vertices = np.empty((num_lines, 2), dtype=gymapi.Vec3.dtype)
        colors = np.empty(num_lines, dtype=gymapi.Vec3.dtype)
        line_index = 0
        for axis_index, corners in enumerate(self._viewer_cuboids):
            for begin, end in CUBOID_EDGES:
                vertices[line_index][0] = tuple(corners[begin])
                vertices[line_index][1] = tuple(corners[end])
                colors[line_index] = tuple(AXIS_COLORS_FLOAT[axis_index])
                line_index += 1
        self.env.gym.add_lines(
            self.env.viewer,
            self.camera_env,
            num_lines,
            vertices,
            colors,
        )
        self._viewer_lines_step = control_step

    def capture_if_active(self) -> None:
        if not self.recorder.active:
            return
        self.env.gym.fetch_results(self.env.sim, True)
        self.env.gym.step_graphics(self.env.sim)
        self.env.gym.render_all_camera_sensors(self.env.sim)
        self.env.gym.start_access_image_tensors(self.env.sim)
        try:
            rgb = (
                self.camera_tensor[..., :3]
                .contiguous()
                .to(device="cpu")
                .numpy()
                .copy()
            )
        finally:
            self.env.gym.end_access_image_tensors(self.env.sim)
        _draw_step_number(rgb, self.env.control_steps)
        self.last_rgb = rgb
        self.recorder.write(rgb, captured_at=time.monotonic())

    def close(self) -> None:
        if self.recorder.active:
            self.recorder.stop()
