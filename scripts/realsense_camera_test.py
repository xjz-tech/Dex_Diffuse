#!/usr/bin/env python3
"""Preview every connected RealSense RGB camera in its own OpenCV window.

Devices are discovered at startup and identified by serial number, including
multiple cameras of the same model.  Disconnected cameras keep their windows
and reconnect automatically by serial number.  This script does not connect to robot,
tactile, or inference hardware.  Color capture helpers are reused from
``direct_robot_env.py`` in the project root.

Run from the project root with::

    python scripts/realsense_camera_test.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Direct script execution puts scripts/, rather than the project root, on sys.path.
sys.path.insert(0, str(PROJECT_ROOT))

from direct_robot_env import (  # noqa: E402
    CameraFrame,
    CameraSpec,
    ColorProfile,
    RealSenseColorCamera,
    _import_realsense,
    get_device_info,
    identify_camera_label,
    iter_color_profiles,
    parse_camera_resolution,
    rotate_image_180,
    select_color_profile,
    sensor_has_color_stream,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--front-serial", default=None,
        help="Apply front-camera settings to this serial; all cameras are still opened",
    )
    parser.add_argument(
        "--wrist-serial", default=None,
        help="Apply wrist-camera settings to this serial; all cameras are still opened",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", default="640x480", help="Preferred resolution for all cameras")
    parser.add_argument("--front-resolution", default=None, help="Override resolution for front cameras")
    parser.add_argument("--wrist-resolution", default=None, help="Override resolution for wrist cameras")
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--reconnect-interval", type=float, default=2.0,
        help="Seconds between reconnect attempts for disconnected cameras (default: 2)",
    )
    parser.add_argument(
        "--wrist-exposure",
        type=float,
        default=None,
        help="Manual wrist D405 exposure in microseconds; lower is darker (default: auto)",
    )
    parser.add_argument(
        "--rotate-wrist-camera-180",
        dest="rotate_wrist_camera_180",
        action="store_true",
        help="Rotate the wrist image by 180 degrees (default)",
    )
    parser.add_argument(
        "--no-rotate-wrist-camera-180",
        dest="rotate_wrist_camera_180",
        action="store_false",
        help="Show the raw wrist-camera orientation",
    )
    parser.set_defaults(rotate_wrist_camera_180=True)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=PROJECT_ROOT / "camera_snapshots",
        help="Directory used when 's' is pressed",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List detected RealSense cameras and exit",
    )
    return parser.parse_args()


def discover_all_cameras(args: argparse.Namespace) -> list[CameraSpec]:
    """Enumerate devices without requiring a model or a front/wrist pair."""
    rs = _import_realsense()
    context = rs.context()
    specs = []
    for device in context.devices:
        name = get_device_info(device, rs.camera_info.name, "RealSense")
        serial = get_device_info(device, rs.camera_info.serial_number)
        product_id = get_device_info(device, rs.camera_info.product_id)
        if not serial:
            print(f"[camera] skipping {name}: serial number unavailable")
            continue
        label = identify_camera_label(name, product_id) or "camera"
        if args.front_serial and serial == args.front_serial:
            label = "front"
        elif args.wrist_serial and serial == args.wrist_serial:
            label = "wrist"
        specs.append(CameraSpec(
            label=label,
            serial=serial,
            name=name,
            product_id=product_id,
            profiles=tuple(iter_color_profiles(device, fps=args.fps, rs=rs)),
        ))
    return sorted(specs, key=lambda spec: spec.serial)


def set_wrist_exposure(camera: RealSenseColorCamera, exposure: float) -> None:
    """Override startup auto exposure on the sensor supplying the wrist RGB stream."""
    if camera.pipeline is None:
        raise RuntimeError("The wrist camera is not started")
    rs = camera.rs
    device = camera.pipeline.get_active_profile().get_device()
    # D405 supplies RGB through its stereo sensor, so find it by stream type.
    sensors = [sensor for sensor in device.query_sensors() if sensor_has_color_stream(sensor, rs)]
    if not sensors:
        raise RuntimeError("The wrist camera has no color stream sensor")
    for sensor in sensors:
        if not sensor.supports(rs.option.exposure) or not sensor.supports(rs.option.enable_auto_exposure):
            raise RuntimeError("The wrist camera does not support manual exposure")
        limits = sensor.get_option_range(rs.option.exposure)
        if not limits.min <= exposure <= limits.max:
            raise ValueError(
                f"--wrist-exposure must be between {limits.min:g} and {limits.max:g} "
                f"microseconds (step={limits.step:g}), got {exposure:g}"
            )
        sensor.set_option(rs.option.enable_auto_exposure, 0.0)
        sensor.set_option(rs.option.exposure, exposure)
        print(
            f"[camera] wrist manual exposure={sensor.get_option(rs.option.exposure):g} us "
            f"auto_exposure={sensor.get_option(rs.option.enable_auto_exposure):.0f}"
        )


class CameraPreview:
    """Own a camera's SDK operations in a worker so reconnects cannot block imshow."""

    def __init__(
        self,
        spec: CameraSpec,
        profile: ColorProfile,
        requested_resolution: tuple[int, int],
        args: argparse.Namespace,
    ) -> None:
        self.spec = spec
        self.profile = profile
        self.requested_resolution = requested_resolution
        self.timeout_ms = args.timeout_ms
        self.reconnect_interval = args.reconnect_interval
        self.wrist_exposure = args.wrist_exposure
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame: CameraFrame | None = None
        self._status = "Connecting..."
        # Frame numbers from the device may restart after reconnecting.
        self._sequence = 0
        self._thread = threading.Thread(
            target=self._run, name=f"realsense-{spec.serial}", daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def latest(self) -> tuple[CameraFrame | None, str, int]:
        with self._lock:
            return self._frame, self._status, self._sequence

    def request_stop(self) -> None:
        self._stop_event.set()

    def join(self) -> None:
        if self._thread.ident is not None:
            # SDK calls may stall after a USB disconnect; keep quitting bounded.
            self._thread.join(timeout=max(3.0, self.timeout_ms / 1000.0 + 1.0))
            if self._thread.is_alive():
                print(f"[camera] {self.spec.serial}: SDK cleanup is still pending")

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._frame = None
            self._status = status

    def _run(self) -> None:
        rs = _import_realsense()
        while not self._stop_event.is_set():
            camera = None
            try:
                # Use fresh device discovery after unplug/replug; only reopen
                # the original serial, never substitute another camera.
                context = rs.context()
                present = any(
                    get_device_info(device, rs.camera_info.serial_number) == self.spec.serial
                    for device in context.devices
                )
                if not present:
                    self._set_status("Disconnected - waiting for camera...")
                    self._stop_event.wait(self.reconnect_interval)
                    continue
                self._set_status("Connecting...")
                camera = RealSenseColorCamera(
                    self.spec, self.profile, requested_resolution=self.requested_resolution,
                )
                camera.start()
                if self.spec.label == "wrist" and self.wrist_exposure is not None:
                    set_wrist_exposure(camera, self.wrist_exposure)
                first_frame = True
                while not self._stop_event.is_set():
                    frame = camera.read(timeout_ms=self.timeout_ms)
                    with self._lock:
                        self._frame = frame
                        self._status = "Streaming"
                        self._sequence += 1
                    if first_frame:
                        print(f"[camera] {self.spec.serial}: streaming")
                        first_frame = False
            except (TimeoutError, RuntimeError) as exc:
                self._set_status("Disconnected - reconnecting...")
                print(f"[camera] {self.spec.serial}: {exc}; will reconnect")
            except ValueError as exc:
                # An invalid manual exposure cannot be fixed by reconnecting.
                self._set_status("Configuration error - see console")
                print(f"[camera] {self.spec.serial}: {exc}")
                return
            finally:
                if camera is not None:
                    try:
                        camera.stop()
                    except RuntimeError as exc:
                        print(f"[camera] could not stop {self.spec.serial}: {exc}")
            self._stop_event.wait(self.reconnect_interval)


def add_label(image_bgr: np.ndarray, text: str) -> np.ndarray:
    output = image_bgr.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 255, 80),
        2,
        cv2.LINE_AA,
    )
    return output


def save_snapshot(
    snapshot_dir: Path,
    frames_bgr: dict[str, np.ndarray],
) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for serial, frame_bgr in frames_bgr.items():
        path = snapshot_dir / f"{stamp}_{serial}.png"
        if not cv2.imwrite(str(path), frame_bgr):
            raise RuntimeError(f"Failed to save {path}")
        print(f"[snapshot] {path}")


def any_window_closed(window_names: list[str], visible_windows: set[str]) -> bool:
    """Only trust window disappearance after visibility was supported once."""
    for name in window_names:
        try:
            visible = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = -1
        if visible >= 1:
            visible_windows.add(name)
        elif name in visible_windows:
            return True
        # Some backends always return -1 for unsupported window properties.
    return False


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be positive")
    if not math.isfinite(args.reconnect_interval) or args.reconnect_interval <= 0:
        raise ValueError("--reconnect-interval must be finite and positive")

    resolution = parse_camera_resolution(args.resolution)
    requested_resolutions = {
        "front": parse_camera_resolution(args.front_resolution) if args.front_resolution else resolution,
        "wrist": parse_camera_resolution(args.wrist_resolution) if args.wrist_resolution else resolution,
    }
    specs = discover_all_cameras(args)
    if not specs:
        print("[camera] No RealSense devices found.")
        return

    rs = _import_realsense()
    cameras: dict[str, CameraPreview] = {}
    window_names: dict[str, str] = {}
    try:
        for spec in specs:
            print(f"[camera] detected {spec.name} serial={spec.serial} role={spec.label}")
            if not spec.profiles:
                print(f"[camera] skipping {spec.serial}: no RGB/BGR color profile at {args.fps} Hz")
                continue
            requested = requested_resolutions.get(spec.label, resolution)
            available = {(profile.width, profile.height) for profile in spec.profiles}
            selected_resolution = min(
                available,
                key=lambda size: (abs(size[0] - requested[0]) + abs(size[1] - requested[1]), size),
            )
            if selected_resolution != requested:
                print(
                    f"[camera] {spec.serial}: {requested[0]}x{requested[1]} unavailable; "
                    f"using {selected_resolution[0]}x{selected_resolution[1]}"
                )
            profile = select_color_profile(spec.profiles, rs, requested_resolution=selected_resolution)
            if args.list_cameras:
                print(f"  color={profile.width}x{profile.height} {profile.fmt} {profile.fps}Hz")
                continue
            # Keep every initially detected RGB camera, even if its first
            # connection attempt fails. Windows belong to the GUI thread only.
            cameras[spec.serial] = CameraPreview(spec, profile, requested, args)
            window_name = f"RealSense - {spec.name} [{spec.serial}]"
            window_names[spec.serial] = window_name
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, min(profile.width, 960), min(profile.height, 720))
            cv2.imshow(window_name, add_label(
                np.zeros((profile.height, profile.width, 3), dtype=np.uint8), "Waiting for frames..."
            ))

        if args.list_cameras:
            return
        if not cameras:
            print("[camera] No compatible RealSense RGB cameras found.")
            return

        for camera in cameras.values():
            camera.start()
        print(
            "[controls] q/Esc: quit | s: save all camera PNG snapshots | "
            f"wrist_rotation={'180 degrees' if args.rotate_wrist_camera_180 else 'raw'}"
        )
        frame_counts = dict.fromkeys(cameras, 0)
        last_frames: dict[str, np.ndarray] = {}
        displayed: dict[str, tuple[int, str]] = {}
        visible_windows: set[str] = set()
        report_started = time.monotonic()

        while True:
            for serial, camera in cameras.items():
                frame, status, sequence = camera.latest()
                if displayed.get(serial) == (sequence, status):
                    continue
                displayed[serial] = (sequence, status)
                if frame is None:
                    last_frames.pop(serial, None)
                    cv2.imshow(window_names[serial], add_label(
                        np.zeros((camera.profile.height, camera.profile.width, 3), dtype=np.uint8),
                        status,
                    ))
                    continue
                image_rgb = frame.color_rgb
                rotated = camera.spec.label == "wrist" and args.rotate_wrist_camera_180
                if rotated:
                    image_rgb = rotate_image_180(image_rgb)
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                last_frames[serial] = image_bgr
                label = f"RGB frame={frame.frame_number}" + ("  rotated=180" if rotated else "")
                cv2.imshow(window_names[serial], add_label(image_bgr, label))
                frame_counts[serial] += 1

            # Keep processing GUI events even when all cameras are disconnected.
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break
            if any_window_closed(list(window_names.values()), visible_windows):
                break
            if key == ord("s"):
                save_snapshot(args.snapshot_dir, last_frames)

            elapsed = time.monotonic() - report_started
            if elapsed >= 5.0:
                rates = " | ".join(f"{serial}={count / elapsed:.1f} fps" for serial, count in frame_counts.items())
                print(f"[camera] preview_rate: {rates}")
                frame_counts = dict.fromkeys(cameras, 0)
                report_started = time.monotonic()
    except KeyboardInterrupt:
        print("\n[camera] interrupted")
    finally:
        for camera in cameras.values():
            camera.request_stop()
        for camera in cameras.values():
            camera.join()
        if window_names:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
