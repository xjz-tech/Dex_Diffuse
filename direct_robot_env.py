#!/usr/bin/env python3
"""Direct Franka/SharpA/RealSense environment for Diffusion Policy inference."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import cv2
import numpy as np

DEFAULT_FRANKA_HOST = "172.16.0.10"
DEFAULT_FRANKA_PORT = 9090
DEFAULT_HAND_HOST = "localhost"
DEFAULT_HAND_PORT = 5570
DEFAULT_CAMERA_RESOLUTION = "640x480"
DEFAULT_CAMERA_FPS = 30
DEFAULT_CAMERA_TIMEOUT_MS = 1000
DEFAULT_TACTILE_TIMEOUT_SECS = 5.0
DEFAULT_FRONT_CAMERA = "front"
DEFAULT_WRIST_CAMERA = "wrist"
DEFAULT_MAX_JOINT_STEP = 0.05
DEFAULT_MAX_HAND_STEP = 0.03

ACTION_DIM = 31
ARM_DIM = 9
HAND_DIM = 22
CAMERA_LABELS = ("front", "wrist")
TACTILE_CHANNELS = tuple(range(5))
TACTILE_DEFORM_SHAPE = (240, 240)
FRANKA_CONTROL_MODES = ("joints", "cartesian")

TACMP_ROOT = Path(__file__).resolve().parents[2]
DEX_SETUP_ROOT = TACMP_ROOT.parent
DEFAULT_SHARPAWAVE_SDK_ROOT = Path("/home/frankagvl/workspace/sharpawave/sharpa-wave-sdk")
PYTHON_TAG = f"python{sys.version_info.major}{sys.version_info.minor}"
LEGACY_SHARPAWAVE_PATH_MARKERS = (
    "/SharpaWaveSDK_4.6.6/",
    "/SharpaWaveSDK_4.6.6-new/",
    "/SharpaWave_4.6.6/",
)


def _non_legacy_env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name, "")
    if not value:
        return None
    if any(marker in value for marker in LEGACY_SHARPAWAVE_PATH_MARKERS):
        print(f"[tactile] ignoring legacy {name}={value}")
        return None
    return Path(value)


FR3_URDF_CANDIDATES = (
    DEX_SETUP_ROOT / "dex-controller_sapgv2" / "maniptrans_envs" / "assets" / "fr3_arm" / "fr3.urdf",
    DEX_SETUP_ROOT / "simtoolreal2teleop" / "assets" / "urdf" / "fr3_sharpa_description" / "fr3.urdf",
)
TACTILE_SDK_PYTHON_PATHS = (
    _non_legacy_env_path("SHARPAWAVE_SDK_PYTHON"),
    _non_legacy_env_path("SDK_PYTHON"),
    Path(os.environ.get("SHARPAWAVE_SDK_ROOT", "")) / "python" if os.environ.get("SHARPAWAVE_SDK_ROOT") else None,
    Path(os.environ.get("SHARPAWAVE_SDK_ROOT", "")) / "python" / PYTHON_TAG if os.environ.get("SHARPAWAVE_SDK_ROOT") else None,
    Path("/usr/lib/sharpa-wave-sdk/python"),
    Path("/usr/lib/sharpa-wave-sdk/python") / PYTHON_TAG,
    DEFAULT_SHARPAWAVE_SDK_ROOT / "python",
    DEFAULT_SHARPAWAVE_SDK_ROOT / "python" / PYTHON_TAG,
)
PRODUCT_ID_TO_LABEL = {
    "0b3a": "front",  # D435I
    "0b5b": "wrist",  # D405
    "0b5c": "front",  # D455
}
MODEL_TO_LABEL = {
    "d405": "wrist",
    "d435": "front",
    "d455": "front",
}
REAL_SHARPA_DOF_NAMES = (
    "right_thumb_CMC_FE", "right_thumb_CMC_AA",
    "right_thumb_MCP_FE", "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_index_DIP",
    "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP", "right_middle_DIP",
    "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP", "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE", "right_pinky_MCP_AA", "right_pinky_PIP", "right_pinky_DIP",
)
POLICY_SHARPA_DOF_NAMES = (
    "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_index_DIP",
    "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP", "right_middle_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE", "right_pinky_MCP_AA", "right_pinky_PIP", "right_pinky_DIP",
    "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP", "right_ring_DIP",
    "right_thumb_CMC_FE", "right_thumb_CMC_AA",
    "right_thumb_MCP_FE", "right_thumb_MCP_AA",
    "right_thumb_IP",
)
REAL2POLICY_DOF_INDICES = np.array(
    [REAL_SHARPA_DOF_NAMES.index(dof_name) for dof_name in POLICY_SHARPA_DOF_NAMES],
    dtype=np.int64,
)
POLICY2REAL_DOF_INDICES = np.array(
    [POLICY_SHARPA_DOF_NAMES.index(dof_name) for dof_name in REAL_SHARPA_DOF_NAMES],
    dtype=np.int64,
)
IK_W_POS_DIAG = np.array([20, 20, 20, 20, 1, 1, 1], dtype=np.float64)
IK_W_ROT_DIAG = np.array([1, 1, 1, 1, 20, 20, 20], dtype=np.float64)
IK_DAMPING = 1e-4
IK_DQ_MAX = 0.5


@dataclass(frozen=True)
class ColorProfile:
    width: int
    height: int
    fmt: Any
    fps: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class CameraSpec:
    label: str
    serial: str
    name: str
    product_id: str
    profiles: Tuple[ColorProfile, ...]


@dataclass
class CameraFrame:
    label: str
    color_rgb: np.ndarray
    host_time_ns: int
    frame_number: int


@dataclass
class IkStepResult:
    joint_positions: np.ndarray
    pos_err: np.ndarray
    rot_err: np.ndarray


class FrankaArmController:
    def __init__(self, host: str = DEFAULT_FRANKA_HOST, port: int = DEFAULT_FRANKA_PORT, timeout_ms: int = 2000) -> None:
        try:
            import msgpack
            import zmq
        except ImportError as exc:
            raise RuntimeError("msgpack and pyzmq are required for Franka state reading/execution.") from exc
        self._msgpack = msgpack
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self._sock.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self._sock.setsockopt(zmq.LINGER, 0)
        self._lock = threading.Lock()
        self._sock.connect(f"tcp://{host}:{port}")
        self._addr = f"tcp://{host}:{port}"
        print(f"[franka] connecting to {self._addr}")

    def _send(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._sock.send(self._msgpack.packb(msg))
            return self._msgpack.unpackb(self._sock.recv(), raw=False)

    def get_state(self) -> Dict[str, Any]:
        return self._send({"cmd": "get_state"})

    def get_current_tcp_pose(self) -> np.ndarray:
        state = self.get_state()
        if state.get("status") != "ok":
            raise RuntimeError(f"Failed to get Franka state: {state}")
        pose = np.asarray(state["tcp_pose"], dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"Expected Franka tcp_pose shape (4,4), got {pose.shape}")
        return pose

    def get_joint_positions(self) -> np.ndarray:
        state = self.get_state()
        if state.get("status") != "ok":
            raise RuntimeError(f"Failed to get Franka joint state: {state}")
        joints = np.asarray(state["joint_positions"], dtype=np.float64)
        if joints.shape != (7,):
            raise ValueError(f"Expected Franka joint_positions shape (7,), got {joints.shape}")
        return joints

    def send_move(self, mat4x4: list) -> Dict[str, Any]:
        return self._send({"cmd": "move", "tcp_pose": mat4x4})

    def send_move_joints(self, joint_positions: list) -> Dict[str, Any]:
        return self._send({"cmd": "move_joints", "joint_positions": joint_positions})

    def send_stop(self) -> Dict[str, Any]:
        return self._send({"cmd": "stop"})

    def close(self) -> None:
        self._sock.close()
        self._ctx.term()


class SharpaWaveController:
    def __init__(self, host: str = DEFAULT_HAND_HOST, port: int = DEFAULT_HAND_PORT, timeout_ms: int = 2000) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq is required for SharpaWave state reading/execution.") from exc
        self._zmq = zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self._sock.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self._sock.setsockopt(zmq.LINGER, 0)
        self._lock = threading.Lock()
        self._sock.connect(f"tcp://{host}:{port}")
        self._addr = f"tcp://{host}:{port}"
        print(f"[hand] connecting to {self._addr}")
        try:
            response = self._send({"cmd": "ping"})
        except (zmq.Again, TimeoutError) as exc:
            self.close()
            raise RuntimeError(
                f"SharpaWave hand controller did not respond at {self._addr} within {timeout_ms} ms. "
                "Start it in another terminal with: cd /home/frankagvl/workspace/sharpawave/sharpa-wave-sdk "
                "&& bash sample/python/run_server.sh. "
                "Then verify it is listening with: ss -ltnp | grep ':5570'. "
                "If it is listening elsewhere, pass HAND_HOST/HAND_PORT."
            ) from exc
        if not response.get("ok"):
            self.close()
            raise RuntimeError(f"SharpaWave ping failed at {self._addr}: {response}")

    def _send(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            try:
                self._sock.send_json(msg)
                return self._sock.recv_json()
            except self._zmq.Again as exc:
                raise TimeoutError(f"Timed out waiting for SharpaWave response from {self._addr} for command {msg.get('cmd')!r}") from exc

    def get_state(self) -> np.ndarray:
        response = self._send({"cmd": "get_state"})
        if not response.get("ok"):
            raise RuntimeError(f"SharpaWave get_state failed: {response}")
        angles = np.asarray(response["angles"], dtype=np.float32)
        if angles.shape != (HAND_DIM,):
            raise ValueError(f"Expected SharpaWave state shape ({HAND_DIM},), got {angles.shape}")
        return angles

    def set_action(self, angles: np.ndarray, interpolate: bool = False) -> None:
        angles = np.asarray(angles, dtype=np.float64)
        if angles.shape != (HAND_DIM,):
            raise ValueError(f"Expected hand action shape ({HAND_DIM},), got {angles.shape}")
        response = self._send({"cmd": "set_action", "angles": angles.tolist(), "interpolate": bool(interpolate)})
        if not response.get("ok"):
            raise RuntimeError(f"SharpaWave set_action failed: {response}")

    def close(self) -> None:
        self._sock.close()
        self._ctx.term()


class SharpaTactileReceiver:
    def __init__(self, channels: Tuple[int, ...] = TACTILE_CHANNELS) -> None:
        self.channels = tuple(channels)
        self.channel_to_index = {channel: idx for idx, channel in enumerate(self.channels)}
        self.lock = threading.Lock()
        self.latest_by_channel: Dict[int, Dict[str, Any]] = {}
        self.manager = None
        self.waves = []
        self.HandSide = None
        self.DeviceType = None
        self._printed_snapshot = False

    def _import_sdk(self):
        candidates = list(dict.fromkeys(path for path in TACTILE_SDK_PYTHON_PATHS if path is not None))
        existing_paths = [str(path) for path in candidates if path.exists()]
        missing_paths = [str(path) for path in candidates if not path.exists()]
        for sdk_path in reversed(existing_paths):
            if sdk_path in sys.path:
                sys.path.remove(sdk_path)
            sys.path.insert(0, sdk_path)
        try:
            import sharpa as sharpa_module
            from sharpa import SharpaWaveManager, HandSide, DeviceType
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import SharpA tactile SDK module 'sharpa'. "
                "Install the SDK or set SHARPAWAVE_SDK_PYTHON to the SDK python binding path. "
                f"SHARPAWAVE_SDK_ROOT={os.environ.get('SHARPAWAVE_SDK_ROOT', '')!r}; "
                f"SHARPAWAVE_SDK_PYTHON={os.environ.get('SHARPAWAVE_SDK_PYTHON', '')!r}; "
                f"existing python paths inserted={existing_paths}; missing candidates={missing_paths}"
            ) from exc
        self.HandSide = HandSide
        self.DeviceType = DeviceType
        module_path = getattr(sharpa_module, "__file__", "<unknown>")
        print(f"[tactile] imported SharpA SDK sharpa from {module_path}; python_paths={existing_paths}")
        return SharpaWaveManager, HandSide, DeviceType

    @staticmethod
    def _device_has_tactile(device_info: object) -> bool:
        checker = getattr(device_info, "has_fingertip_tactile", None)
        if checker is None:
            return True
        return bool(checker()) if callable(checker) else bool(checker)

    @staticmethod
    def _tactile_port(device_info: object) -> object:
        tactile_config = getattr(device_info, "tactile_data_config", None)
        return getattr(tactile_config, "target_port", "<unknown>")

    @staticmethod
    def _reshape_uint8_image(data: object, shape: Optional[object], fallback_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if data is None:
            return None
        array = np.asarray(data).squeeze()
        if array.size == 0:
            return None
        height, width = fallback_shape
        if shape is not None:
            try:
                height = int(shape[1])
                width = int(shape[2])
            except (TypeError, IndexError, ValueError):
                pass
        elif array.size != height * width:
            if array.size == 76800:
                height, width = 240, 320
            elif array.size == 57600:
                height, width = 240, 240
        try:
            return array.reshape(height, width).astype(np.uint8, copy=True)
        except ValueError:
            return None

    @staticmethod
    def _contact_points_to_array(data: object) -> np.ndarray:
        if data is None:
            return np.empty((0, 3), dtype=np.float32)
        array = np.asarray(data, dtype=np.float32).squeeze()
        if array.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if array.ndim == 1:
            point_dim = 3 if array.size % 3 == 0 else array.size
            return array.reshape(-1, point_dim).copy()
        return array.reshape(-1, array.shape[-1]).copy()

    def _normalize_frame(self, frames: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        channel = int(frames.get("channel", -1))
        if channel not in self.channel_to_index:
            return None
        content = frames.get("content", {}) or {}
        shape = frames.get("shape", {}) or {}
        f6 = np.asarray(content.get("F6", np.full(6, np.nan)), dtype=np.float32).reshape(-1)
        if f6.size < 6:
            padded_f6 = np.full(6, np.nan, dtype=np.float32)
            padded_f6[: f6.size] = f6
            f6 = padded_f6
        else:
            f6 = f6[:6].copy()
        deform = self._reshape_uint8_image(content.get("DEFORM"), shape.get("DEFORM"), TACTILE_DEFORM_SHAPE)
        if deform is None:
            deform = np.zeros(TACTILE_DEFORM_SHAPE, dtype=np.uint8)
        return {
            "channel": channel,
            "frame_id": int(frames.get("frame_id", -1)),
            "ts": float(frames.get("ts", np.nan)),
            "f6": f6,
            "deform": deform,
            "contact_points": self._contact_points_to_array(content.get("CONTACT_POINT")),
        }

    def _callback(self, frames: Dict[str, Any]) -> None:
        normalized = self._normalize_frame(frames)
        if normalized is None:
            return
        with self.lock:
            self.latest_by_channel[normalized["channel"]] = normalized

    def start(self) -> None:
        SharpaWaveManager, HandSide, DeviceType = self._import_sdk()
        self.manager = SharpaWaveManager.get_instance()
        device_infos = self.manager.get_all_devices()
        while not device_infos or not any(info.device_type == DeviceType.HAND for info in device_infos):
            print("Waiting for SharpA tactile device connection...")
            time.sleep(1.0)
            device_infos = self.manager.get_all_devices()

        hand_device_infos = [info for info in device_infos if info.device_type == DeviceType.HAND]
        found_right = False
        for device_info in hand_device_infos:
            wave = self.manager.connect(device_info.sn)
            connected_info = wave.get_device_info()
            if connected_info.hand_side != HandSide.RIGHT:
                wave.destroy()
                continue
            found_right = True
            if not self._device_has_tactile(connected_info):
                wave.destroy()
                raise RuntimeError(
                    f"RIGHT SharpA hand {connected_info.sn} does not report fingertip tactile support."
                )
            tactile_port = self._tactile_port(connected_info)
            print(
                f"[tactile] connecting RIGHT SharpA hand sn={connected_info.sn} "
                f"channels={self.channels} tactile_port={tactile_port}"
            )
            wave.set_tactile_callback(self._callback)
            if not wave.start():
                raise RuntimeError(
                    f"SharpA tactile startup failed for device {connected_info.sn}: "
                    f"check port {tactile_port}"
                )
            self.waves.append(wave)

        if not self.waves:
            if found_right:
                raise RuntimeError("RIGHT SharpA tactile hand was found but no tactile receiver started.")
            raise RuntimeError("No RIGHT SharpA tactile hand device found.")
        print(f"[tactile] SharpA tactile receiver started for channels {self.channels}.")

    def snapshot(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            if not all(channel in self.latest_by_channel for channel in self.channels):
                return None
            frames = [self.latest_by_channel[channel] for channel in self.channels]
            f6 = np.stack([frame["f6"] for frame in frames]).astype(np.float32)
            deform = np.stack([frame["deform"] for frame in frames]).astype(np.uint8)
            contact_points = [frame["contact_points"].copy() for frame in frames]
            timestamps = np.asarray([frame["ts"] for frame in frames], dtype=np.float64)
            frame_ids = np.asarray([frame["frame_id"] for frame in frames], dtype=np.int64)
            channels = np.asarray(self.channels, dtype=np.int64)
        if not self._printed_snapshot:
            self._printed_snapshot = True
            print(
                f"[tactile] first SDK snapshot f6={tuple(f6.shape)} "
                f"channels={channels.tolist()} frame_ids={frame_ids.tolist()}"
            )
        return {
            "f6": f6,
            "deform": deform,
            "contact_points": contact_points,
            "timestamps": timestamps,
            "frame_ids": frame_ids,
            "channels": channels,
        }

    def close(self) -> None:
        for wave in self.waves:
            try:
                wave.stop()
            except Exception:
                pass
            try:
                wave.destroy()
            except Exception:
                pass
        self.waves.clear()
        if self.manager is not None:
            try:
                self.manager.disconnect_all()
            except Exception:
                pass


def _import_realsense() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required for RealSense camera capture.") from exc
    return rs


def _import_pinocchio() -> Any:
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise RuntimeError(
            "pinocchio is required for --franka-control-mode joints. "
            "Use --franka-control-mode cartesian if you do not want local IK."
        ) from exc
    if not hasattr(pin, "buildModelFromUrdf"):
        module_path = getattr(pin, "__file__", "<unknown>")
        module_version = getattr(pin, "__version__", "<unknown>")
        raise RuntimeError(
            "The imported 'pinocchio' module is not the robotics Pinocchio library "
            f"(missing buildModelFromUrdf; path={module_path!r}; version={module_version!r}). "
            "Install the robotics package, e.g. 'conda install -c conda-forge pinocchio', "
            "or run with FRANKA_CONTROL_MODE=cartesian to avoid local IK."
        )
    return pin


def normalize_product_id(product_id: str) -> str:
    return product_id.strip().lower()


def identify_camera_label(name: str, product_id: str) -> Optional[str]:
    lowered_name = name.lower()
    if "405" in lowered_name:
        return MODEL_TO_LABEL["d405"]
    if "435" in lowered_name:
        return MODEL_TO_LABEL["d435"]
    if "455" in lowered_name:
        return MODEL_TO_LABEL["d455"]
    return PRODUCT_ID_TO_LABEL.get(normalize_product_id(product_id))


def parse_camera_resolution(value: str) -> Tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Expected resolution like 640x480, got {value!r}")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive, got {value!r}")
    return width, height


def get_device_info(device: Any, info: Any, default: str = "") -> str:
    try:
        if device.supports(info):
            return device.get_info(info)
    except Exception:
        pass
    return default


def iter_color_profiles(device: Any, fps: int, rs: Any) -> Iterable[ColorProfile]:
    seen = set()
    for sensor in device.sensors:
        for profile in sensor.get_stream_profiles():
            try:
                video = profile.as_video_stream_profile()
                stream = profile.stream_type()
                fmt = profile.format()
                profile_fps = int(profile.fps())
            except Exception:
                continue
            if stream != rs.stream.color or profile_fps != fps:
                continue
            if fmt not in (rs.format.rgb8, rs.format.bgr8):
                continue
            key = (int(video.width()), int(video.height()), fmt, profile_fps)
            if key in seen:
                continue
            seen.add(key)
            yield ColorProfile(width=int(video.width()), height=int(video.height()), fmt=fmt, fps=profile_fps)


def sensor_has_color_stream(sensor: Any, rs: Any) -> bool:
    for profile in sensor.get_stream_profiles():
        try:
            if profile.stream_type() == rs.stream.color:
                return True
        except Exception:
            continue
    return False


def enable_wrist_camera_auto_controls(device: Any, rs: Any) -> None:
    """Match camera_test.py's D405 exposure and white-balance state."""
    color_sensors = [sensor for sensor in device.query_sensors() if sensor_has_color_stream(sensor, rs)]
    if not color_sensors:
        raise RuntimeError("The wrist camera has no color sensor.")

    def enable_if_needed(sensor: Any, option: Any, name: str) -> str:
        if not sensor.supports(option):
            return "unsupported"

        current = None
        try:
            current = float(sensor.get_option(option))
        except RuntimeError as exc:
            print(f"[camera] wrist {name} state unavailable: {exc}")

        # Avoid an unnecessary V4L2 control write.  Some D405/USB controller
        # combinations time out when the already-enabled value is written
        # again even though color streaming itself is healthy.
        if current is not None and current >= 0.5:
            return f"{current:.0f}"

        try:
            sensor.set_option(option, 1.0)
        except RuntimeError as exc:
            current_text = "unknown" if current is None else f"{current:.0f}"
            print(
                f"[camera] warning: could not enable wrist {name}; "
                f"continuing with current={current_text}: {exc}"
            )
            return current_text

        try:
            return f"{sensor.get_option(option):.0f}"
        except RuntimeError as exc:
            print(f"[camera] wrist {name} verification unavailable: {exc}")
            return "enabled"

    for sensor in color_sensors:
        auto_exposure_text = enable_if_needed(
            sensor, rs.option.enable_auto_exposure, "auto exposure"
        )
        auto_white_balance_text = enable_if_needed(
            sensor, rs.option.enable_auto_white_balance, "auto white balance"
        )

        print(
            f"[camera] wrist controls auto_exposure={auto_exposure_text} "
            f"auto_white_balance={auto_white_balance_text}"
        )


def select_color_profile(
    profiles: Iterable[ColorProfile],
    rs: Any,
    requested_resolution: Optional[Tuple[int, int]] = None,
) -> ColorProfile:
    candidates = list(profiles)
    if requested_resolution is not None:
        req_width, req_height = requested_resolution
        candidates = [profile for profile in candidates if profile.width == req_width and profile.height == req_height]
    if not candidates:
        if requested_resolution is None:
            raise RuntimeError("No compatible RealSense color profile was found.")
        raise RuntimeError(
            "No compatible RealSense color profile was found at "
            f"{requested_resolution[0]}x{requested_resolution[1]}."
        )

    def sort_key(profile: ColorProfile) -> Tuple[int, int, int, int]:
        fmt_score = 1 if profile.fmt == rs.format.rgb8 else 0
        return (profile.area, fmt_score, profile.width, profile.height)

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


def discover_cameras(
    fps: int,
    front_serial: Optional[str],
    wrist_serial: Optional[str],
) -> Dict[str, CameraSpec]:
    rs = _import_realsense()
    ctx = rs.context()
    requested_serials = {"front": front_serial, "wrist": wrist_serial}
    discovered: Dict[str, CameraSpec] = {}

    for device in ctx.devices:
        name = get_device_info(device, rs.camera_info.name)
        serial = get_device_info(device, rs.camera_info.serial_number)
        product_id = get_device_info(device, rs.camera_info.product_id)
        label = None
        for requested_label, requested_serial in requested_serials.items():
            if requested_serial and serial == requested_serial:
                label = requested_label
                break
        if label is None:
            label = identify_camera_label(name, product_id)
        if label not in CAMERA_LABELS:
            continue
        if requested_serials[label] and serial != requested_serials[label]:
            continue
        if label in discovered:
            raise RuntimeError(f"Multiple {label} cameras found. Pass --{label}-serial.")
        profiles = tuple(iter_color_profiles(device, fps=fps, rs=rs))
        discovered[label] = CameraSpec(label=label, serial=serial, name=name, product_id=product_id, profiles=profiles)

    missing = [label for label in CAMERA_LABELS if label not in discovered]
    if missing:
        raise RuntimeError(f"Missing required camera(s): {', '.join(missing)}.")
    for label, spec in discovered.items():
        if not spec.profiles:
            raise RuntimeError(f"{label} ({spec.serial}) has no RGB/BGR profile at {fps} Hz.")
    return discovered


def list_cameras(fps: int, requested_resolutions: Optional[Mapping[str, Tuple[int, int]]] = None) -> None:
    rs = _import_realsense()
    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("No RealSense devices found.")
        return
    for idx, device in enumerate(ctx.devices):
        name = get_device_info(device, rs.camera_info.name)
        serial = get_device_info(device, rs.camera_info.serial_number)
        product_id = get_device_info(device, rs.camera_info.product_id)
        label = identify_camera_label(name, product_id) or "unknown"
        profiles = list(iter_color_profiles(device, fps=fps, rs=rs))
        print(f"device {idx}: label={label} name={name} serial={serial} product_id={product_id}")
        if not profiles:
            print(f"  no RGB/BGR color profile at {fps} Hz")
            continue
        requested_resolution = requested_resolutions.get(label) if requested_resolutions and label in CAMERA_LABELS else None
        selected = select_color_profile(profiles, rs, requested_resolution=requested_resolution)
        requested_text = f" requested={requested_resolution[0]}x{requested_resolution[1]}" if requested_resolution else ""
        print(f"  selected color profile:{requested_text} selected={selected.width}x{selected.height} {selected.fmt} {selected.fps}Hz")


class RealSenseColorCamera:
    def __init__(self, spec: CameraSpec, profile: ColorProfile, requested_resolution: Optional[Tuple[int, int]] = None) -> None:
        self.spec = spec
        self.profile = profile
        self.requested_resolution = requested_resolution
        self.rs = _import_realsense()
        self.pipeline = None

    def start(self) -> None:
        config = self.rs.config()
        config.enable_device(self.spec.serial)
        config.enable_stream(self.rs.stream.color, self.profile.width, self.profile.height, self.profile.fmt, self.profile.fps)
        self.pipeline = self.rs.pipeline()
        try:
            pipeline_profile = self.pipeline.start(config)
            if self.spec.label == "wrist":
                enable_wrist_camera_auto_controls(pipeline_profile.get_device(), self.rs)
        except Exception:
            try:
                self.pipeline.stop()
            finally:
                self.pipeline = None
            raise
        requested_text = f" requested={self.requested_resolution[0]}x{self.requested_resolution[1]}" if self.requested_resolution else ""
        print(
            f"[camera] {self.spec.label} serial={self.spec.serial} "
            f"selected={self.profile.width}x{self.profile.height}@{self.profile.fps}{requested_text}"
        )

    def read(self, timeout_ms: int) -> CameraFrame:
        if self.pipeline is None:
            raise RuntimeError(f"{self.spec.label} camera is not started")
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as exc:
            raise TimeoutError(
                f"{self.spec.label} camera serial={self.spec.serial} did not "
                f"deliver a color frame within {timeout_ms} ms. Check that "
                "the camera is connected through a working USB 3.x data "
                "port/cable (5000M or faster in `lsusb -t`)."
            ) from exc
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise TimeoutError(f"{self.spec.label} color frame is unavailable")
        color = np.asanyarray(color_frame.get_data()).copy()
        if self.profile.fmt == self.rs.format.bgr8:
            color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        else:
            color_rgb = color
        return CameraFrame(
            label=self.spec.label,
            color_rgb=color_rgb,
            host_time_ns=time.time_ns(),
            frame_number=int(color_frame.get_frame_number()),
        )

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            finally:
                self.pipeline = None


class RealSensePair:
    def __init__(self, cameras: Mapping[str, RealSenseColorCamera]) -> None:
        self.cameras = dict(cameras)

    @classmethod
    def start(
        cls,
        fps: int,
        front_serial: Optional[str],
        wrist_serial: Optional[str],
        requested_resolutions: Optional[Mapping[str, Tuple[int, int]]] = None,
    ) -> "RealSensePair":
        rs = _import_realsense()
        specs = discover_cameras(fps=fps, front_serial=front_serial, wrist_serial=wrist_serial)
        cameras: Dict[str, RealSenseColorCamera] = {}
        try:
            for label in CAMERA_LABELS:
                requested_resolution = requested_resolutions.get(label) if requested_resolutions else None
                profile = select_color_profile(specs[label].profiles, rs, requested_resolution=requested_resolution)
                camera = RealSenseColorCamera(specs[label], profile, requested_resolution=requested_resolution)
                camera.start()
                cameras[label] = camera
            return cls(cameras)
        except Exception:
            for camera in cameras.values():
                camera.stop()
            raise

    def read_pair(self, timeout_ms: int) -> Dict[str, CameraFrame]:
        return {label: self.cameras[label].read(timeout_ms=timeout_ms) for label in CAMERA_LABELS}

    def stop(self) -> None:
        for camera in self.cameras.values():
            camera.stop()


def rotate_image_180(image: np.ndarray) -> np.ndarray:
    return cv2.rotate(image, cv2.ROTATE_180)


def _normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return fallback.astype(np.float64, copy=True)
    return vector / norm


def rotation_6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64)
    if rot6d.shape != (6,):
        raise ValueError(f"Expected rotation6d shape (6,), got {rot6d.shape}")
    row1 = rot6d[:3]
    row2 = rot6d[3:6]
    r1 = _normalize_vector(row1, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    row2_ortho = row2 - np.dot(r1, row2) * r1
    if np.linalg.norm(row2_ortho) < 1e-8:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(r1, fallback))) > 0.95:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        row2_ortho = fallback - np.dot(r1, fallback) * r1
    r2 = _normalize_vector(row2_ortho, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    r3 = np.cross(r1, r2)
    return np.stack([r1, r2, r3], axis=0)


def arm9_to_pose_matrix(arm9: np.ndarray) -> np.ndarray:
    arm9 = np.asarray(arm9, dtype=np.float64)
    if arm9.shape != (ARM_DIM,):
        raise ValueError(f"Expected arm action shape ({ARM_DIM},), got {arm9.shape}")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = arm9[:3]
    pose[:3, :3] = rotation_6d_to_matrix(arm9[3:9])
    return pose


def pose_matrix_to_arm9(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected pose shape (4,4), got {pose.shape}")
    return np.concatenate([pose[:3, 3], pose[:2, :3].reshape(-1)]).astype(np.float32)


def invert_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected pose shape (4,4), got {pose.shape}")
    inv_pose = np.eye(4, dtype=np.float64)
    rot = pose[:3, :3]
    inv_pose[:3, :3] = rot.T
    inv_pose[:3, 3] = -rot.T @ pose[:3, 3]
    return inv_pose


def tcp_target_to_ee_target(target_tcp_pose: np.ndarray, ee_to_tcp_pose: np.ndarray) -> np.ndarray:
    target_tcp_pose = np.asarray(target_tcp_pose, dtype=np.float64)
    ee_to_tcp_pose = np.asarray(ee_to_tcp_pose, dtype=np.float64)
    if target_tcp_pose.shape != (4, 4):
        raise ValueError(f"Expected target_tcp_pose shape (4,4), got {target_tcp_pose.shape}")
    if ee_to_tcp_pose.shape != (4, 4):
        raise ValueError(f"Expected ee_to_tcp_pose shape (4,4), got {ee_to_tcp_pose.shape}")
    return target_tcp_pose @ invert_pose(ee_to_tcp_pose)


def _weighted_pinv(jacobian: np.ndarray, weights_diag: np.ndarray, damping: float) -> np.ndarray:
    weights_inv = np.diag(1.0 / np.asarray(weights_diag, dtype=np.float64))
    jwj = jacobian @ weights_inv @ jacobian.T + float(damping) * np.eye(jacobian.shape[0])
    return weights_inv @ jacobian.T @ np.linalg.solve(jwj, np.eye(jacobian.shape[0]))


class FrankaJointIkSolver:
    def __init__(
        self,
        initial_joint_positions: np.ndarray,
        urdf_path: Path,
        max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
        ik_dq_max: float = IK_DQ_MAX,
        damping: float = IK_DAMPING,
        w_pos: np.ndarray = IK_W_POS_DIAG,
        w_rot: np.ndarray = IK_W_ROT_DIAG,
        tcp_pose_at_initial_q: Optional[np.ndarray] = None,
        sync_qpos_fn: Optional[Any] = None,
        apply_joint_limits: bool = True,
    ) -> None:
        if max_joint_step <= 0:
            raise ValueError("max_joint_step must be positive")
        self.pin = _import_pinocchio()
        self.model = self.pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("end_effector")
        if self.ee_frame_id >= len(self.model.frames):
            raise RuntimeError("FR3 URDF frame 'end_effector' was not found.")
        self.q_current = np.asarray(initial_joint_positions, dtype=np.float64)
        if self.q_current.shape != (self.model.nq,):
            raise ValueError(f"Expected initial joint positions shape ({self.model.nq},), got {self.q_current.shape}")
        self.max_joint_step = float(max_joint_step)
        self.ik_dq_max = float(ik_dq_max)
        self.apply_joint_limits = bool(apply_joint_limits)
        self.damping = float(damping)
        self.w_pos = np.asarray(w_pos, dtype=np.float64)
        self.w_rot = np.asarray(w_rot, dtype=np.float64)
        self.ee_to_tcp_pose = np.eye(4, dtype=np.float64)
        if tcp_pose_at_initial_q is not None:
            tcp_pose_at_initial_q = np.asarray(tcp_pose_at_initial_q, dtype=np.float64)
            if tcp_pose_at_initial_q.shape != (4, 4):
                raise ValueError(f"Expected tcp_pose_at_initial_q shape (4,4), got {tcp_pose_at_initial_q.shape}")
            self.ee_to_tcp_pose = invert_pose(self._current_ee_pose()) @ tcp_pose_at_initial_q
        self.sync_qpos_fn = sync_qpos_fn

    def _current_ee_pose(self) -> np.ndarray:
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, self.q_current)
        pin.updateFramePlacements(self.model, self.data)
        oMee = self.data.oMf[self.ee_frame_id]
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = oMee.rotation
        pose[:3, 3] = oMee.translation
        return pose

    def _ik_step(self, target_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_pos = target_pose[:3, 3]
        target_rot = target_pose[:3, :3]
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, self.q_current)
        pin.updateFramePlacements(self.model, self.data)
        oMee = self.data.oMf[self.ee_frame_id]
        pos_err = target_pos - oMee.translation
        rot_err = pin.log3(target_rot @ oMee.rotation.T)
        jacobian = pin.computeFrameJacobian(self.model, self.data, self.q_current, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
        jacobian_pos = jacobian[:3, :]
        jacobian_rot = jacobian[3:, :]
        jacobian_pos_bar = _weighted_pinv(jacobian_pos, self.w_pos, self.damping)
        dq_pos = jacobian_pos_bar @ pos_err
        null_pos = np.eye(self.model.nq) - jacobian_pos_bar @ jacobian_pos
        jacobian_rot_null = jacobian_rot @ null_pos
        jacobian_rot_null_bar = _weighted_pinv(jacobian_rot_null, self.w_rot, self.damping)
        rot_residual = rot_err - jacobian_rot @ dq_pos
        dq_rot = null_pos @ (jacobian_rot_null_bar @ rot_residual)
        dq = dq_pos + dq_rot
        dq_norm = float(np.max(np.abs(dq)))
        if self.apply_joint_limits and dq_norm > self.ik_dq_max:
            dq *= self.ik_dq_max / dq_norm
        return dq, pos_err, rot_err

    def solve(self, target_pose: np.ndarray) -> IkStepResult:
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (4, 4):
            raise ValueError(f"Expected target pose shape (4,4), got {target_pose.shape}")
        if self.sync_qpos_fn is not None:
            synced = np.asarray(self.sync_qpos_fn(), dtype=np.float64)
            if synced.shape != self.q_current.shape:
                raise ValueError(f"Synced qpos shape {synced.shape} does not match {self.q_current.shape}")
            self.q_current = synced
        target_ee_pose = tcp_target_to_ee_target(target_pose, self.ee_to_tcp_pose)
        dq, pos_err, rot_err = self._ik_step(target_ee_pose)
        if self.apply_joint_limits:
            dq = np.clip(dq, -self.max_joint_step, self.max_joint_step)
        self.q_current = self.q_current + dq
        if self.apply_joint_limits:
            self.q_current = np.clip(self.q_current, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        return IkStepResult(self.q_current.copy(), np.asarray(pos_err, dtype=np.float64), np.asarray(rot_err, dtype=np.float64))


def resolve_fr3_urdf_path(path: Optional[Path]) -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"FR3 URDF not found: {resolved}")
        return resolved
    for candidate in FR3_URDF_CANDIDATES:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved
    searched = ", ".join(str(path) for path in FR3_URDF_CANDIDATES)
    raise FileNotFoundError(f"FR3 URDF not found. Searched: {searched}")


def split_action(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"Expected action shape ({ACTION_DIM},), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("Action contains NaN or Inf.")
    return action[:ARM_DIM].copy(), action[ARM_DIM:].copy()


def clamp_action_step(target_action: np.ndarray, previous_action: Optional[np.ndarray], max_arm_xyz_step: float, max_hand_step: float) -> np.ndarray:
    target = np.asarray(target_action, dtype=np.float64)
    if target.shape != (ACTION_DIM,):
        raise ValueError(f"Expected target action shape ({ACTION_DIM},), got {target.shape}")
    if previous_action is None:
        return target.copy()
    previous = np.asarray(previous_action, dtype=np.float64)
    if previous.shape != (ACTION_DIM,):
        raise ValueError(f"Expected previous action shape ({ACTION_DIM},), got {previous.shape}")
    clamped = target.copy()
    clamped[:3] = previous[:3] + np.clip(target[:3] - previous[:3], -float(max_arm_xyz_step), float(max_arm_xyz_step))
    clamped[ARM_DIM:] = previous[ARM_DIM:] + np.clip(target[ARM_DIM:] - previous[ARM_DIM:], -float(max_hand_step), float(max_hand_step))
    return clamped


def _check_controller_response(response: Any, command_name: str) -> None:
    if not isinstance(response, dict):
        return
    if response.get("ok") is False:
        raise RuntimeError(f"{command_name} failed: {response}")
    status = response.get("status")
    if status is not None and status != "ok":
        raise RuntimeError(f"{command_name} failed: {response}")


class DirectRobotEnv:
    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        *,
        live: bool = False,
        franka_host: str = DEFAULT_FRANKA_HOST,
        franka_port: int = DEFAULT_FRANKA_PORT,
        franka_timeout_ms: int = 2000,
        hand_host: str = DEFAULT_HAND_HOST,
        hand_port: int = DEFAULT_HAND_PORT,
        hand_timeout_ms: int = 2000,
        franka_control_mode: str = "joints",
        franka_urdf: Optional[str] = None,
        sync_qpos_each_step: bool = True,
        max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
        ik_dq_max: float = IK_DQ_MAX,
        ik_damping: float = IK_DAMPING,
        max_arm_xyz_step: float = 0.03,
        max_hand_step: float = DEFAULT_MAX_HAND_STEP,
        disable_clamp: bool = False,
        disable_ik_limits: bool = False,
        hand_interpolate: bool = False,
        action_chunk_steps: Optional[int] = 40,
        hz: float = 30.0,
        log_action_steps: bool = True,
        use_tactile: bool = True,
        tactile_source: str = "zeros",
        tactile_timeout_secs: float = DEFAULT_TACTILE_TIMEOUT_SECS,
        camera_fps: int = DEFAULT_CAMERA_FPS,
        camera_timeout_ms: int = DEFAULT_CAMERA_TIMEOUT_MS,
        front_serial: Optional[str] = None,
        wrist_serial: Optional[str] = None,
        front_resolution: str = DEFAULT_CAMERA_RESOLUTION,
        wrist_resolution: str = DEFAULT_CAMERA_RESOLUTION,
        rotate_wrist_camera_180: bool = True,
        show_camera_input: bool = False,
        camera_window: str = "Diffusion Policy direct robot camera input",
        show_obs_viz: bool = False,
        obs_viz_window: str = "Diffusion Policy live obs",
        obs_viz_hz: float = 15.0,
        obs_viz_deform_vmax: float = 80.0,
        obs_viz_f6_range: float = 5.0,
        obs_viz_show_state: bool = True,
        stop_on_close: bool = False,
    ) -> None:
        if action_dim != ACTION_DIM:
            raise ValueError(f"DirectRobotEnv expects action_dim={ACTION_DIM}, got {action_dim}")
        if franka_control_mode not in FRANKA_CONTROL_MODES:
            raise ValueError(f"Unsupported franka_control_mode: {franka_control_mode}")
        if hz <= 0:
            raise ValueError("hz must be positive")
        self.action_dim = action_dim
        self.live = bool(live)
        self.franka_control_mode = franka_control_mode
        self.sync_qpos_each_step = bool(sync_qpos_each_step)
        self.max_arm_xyz_step = float(max_arm_xyz_step)
        self.max_hand_step = float(max_hand_step)
        self.disable_clamp = bool(disable_clamp)
        self.hand_interpolate = bool(hand_interpolate)
        self.action_chunk_steps = action_chunk_steps
        self.period = 1.0 / float(hz)
        self.log_action_steps = bool(log_action_steps)
        self.use_tactile = bool(use_tactile)
        if tactile_source not in ("zeros", "sdk"):
            raise ValueError(f"Unsupported tactile_source={tactile_source!r}; expected 'zeros' or 'sdk'")
        self.tactile_source = tactile_source
        self.tactile_timeout_secs = float(tactile_timeout_secs)
        self.camera_timeout_ms = int(camera_timeout_ms)
        self.rotate_wrist_camera_180 = bool(rotate_wrist_camera_180)
        self.show_camera_input = bool(show_camera_input)
        self.camera_window = camera_window
        self.show_obs_viz = bool(show_obs_viz)
        self.obs_viz = None
        self.obs_viz_window = obs_viz_window
        self.obs_viz_hz = float(obs_viz_hz)
        self.obs_viz_deform_vmax = float(obs_viz_deform_vmax)
        self.obs_viz_f6_range = float(obs_viz_f6_range)
        self.obs_viz_show_state = bool(obs_viz_show_state)
        self.stop_on_close = bool(stop_on_close)
        self.obs: Dict[str, Any] = {}
        self.reward = 0.0
        self.chunk_idx = 0
        self.previous_action: Optional[np.ndarray] = None
        self.closed = False

        print("[mode] LIVE: Diffusion Policy actions will be sent to hardware." if self.live else "[mode] dry-run: Diffusion Policy actions will only be printed. Set LIVE=1 or pass --live to execute.")
        self.franka = FrankaArmController(host=franka_host, port=franka_port, timeout_ms=franka_timeout_ms)
        self.hand = SharpaWaveController(host=hand_host, port=hand_port, timeout_ms=hand_timeout_ms)
        self.ik_solver = None
        if self.franka_control_mode == "joints" and self.live:
            urdf_path = resolve_fr3_urdf_path(Path(franka_urdf) if franka_urdf else None)
            sync_fn = self.franka.get_joint_positions if self.sync_qpos_each_step else None
            self.ik_solver = FrankaJointIkSolver(
                initial_joint_positions=self.franka.get_joint_positions(),
                urdf_path=urdf_path,
                max_joint_step=max_joint_step,
                ik_dq_max=ik_dq_max,
                damping=ik_damping,
                tcp_pose_at_initial_q=self.franka.get_current_tcp_pose(),
                sync_qpos_fn=sync_fn,
                apply_joint_limits=not disable_ik_limits,
            )
            limit_text = "software_joint_limits=off" if disable_ik_limits else f"max_joint_step={max_joint_step}"
            print(f"[franka] control_mode=joints urdf={urdf_path} {limit_text} sync_qpos_each_step={self.sync_qpos_each_step}")
        else:
            print(f"[franka] control_mode={self.franka_control_mode}")

        requested_camera_resolutions = {
            "front": parse_camera_resolution(front_resolution),
            "wrist": parse_camera_resolution(wrist_resolution),
        }
        self.cameras = RealSensePair.start(
            fps=camera_fps,
            front_serial=front_serial,
            wrist_serial=wrist_serial,
            requested_resolutions=requested_camera_resolutions,
        )
        self.tactile = None
        if self.use_tactile and self.tactile_source == "sdk":
            self.tactile = SharpaTactileReceiver()
            self.tactile.start()
        elif self.use_tactile:
            print("[tactile] using zero tactile placeholder; SharpA SDK stays in the external hand server process.")
        if self.show_camera_input:
            try:
                cv2.namedWindow(self.camera_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.camera_window, 640 * 2, 480 + 24)
            except cv2.error as exc:
                print(f"[view] camera preview disabled because OpenCV highgui is unavailable: {exc}")
                self.show_camera_input = False
        if self.show_obs_viz:
            from live_obs_viz import LiveObsViz
            self.obs_viz = LiveObsViz(
                window_name=self.obs_viz_window,
                hz=self.obs_viz_hz,
                deform_vmax=self.obs_viz_deform_vmax,
                f6_range=self.obs_viz_f6_range,
                show_state=self.obs_viz_show_state,
            )

    def _wait_tactile_snapshot(self) -> Optional[Dict[str, Any]]:
        if self.tactile is None:
            return None
        deadline = time.monotonic() + self.tactile_timeout_secs
        while time.monotonic() < deadline:
            snapshot = self.tactile.snapshot()
            if snapshot is not None:
                return snapshot
            time.sleep(0.01)
        raise TimeoutError(f"Timed out waiting for SharpA tactile frames for channels {TACTILE_CHANNELS}")

    def _read_obs(self, *, clear: bool = False) -> Dict[str, Any]:
        frames = self.cameras.read_pair(timeout_ms=self.camera_timeout_ms)
        front_rgb = frames["front"].color_rgb
        wrist_rgb = frames["wrist"].color_rgb
        if self.rotate_wrist_camera_180:
            wrist_rgb = rotate_image_180(wrist_rgb)
        if self.show_camera_input:
            front_bgr = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
            wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.camera_window, np.concatenate([front_bgr, wrist_bgr], axis=1))
            cv2.waitKey(1)

        tcp_pose = self.franka.get_current_tcp_pose()
        hand_angles_real = self.hand.get_state()
        hand_angles_policy = hand_angles_real[REAL2POLICY_DOF_INDICES]
        obs: Dict[str, Any] = {
            "/observe/vision/front/rgb": front_rgb,
            "/observe/vision/wrist/rgb": wrist_rgb,
            "/state/arm/eef_pose": pose_matrix_to_arm9(tcp_pose),
            "/state/hand/joint_angle": hand_angles_policy.astype(np.float32),
            "reward": 0.0,
        }
        if clear:
            obs["clear"] = True

        tactile_snapshot = self._wait_tactile_snapshot()
        if tactile_snapshot is not None:
            f6 = np.asarray(tactile_snapshot["f6"], dtype=np.float32)
            if f6.shape != (5, 6):
                raise ValueError(f"Expected tactile f6 shape (5,6), got {f6.shape}")
            obs["/observe/tactile/deform"] = np.asarray(tactile_snapshot["deform"], dtype=np.uint8)
            obs["/observe/tactile/channels"] = np.asarray(tactile_snapshot["channels"], dtype=np.int64)
        elif self.use_tactile:
            f6 = np.zeros((5, 6), dtype=np.float32)
            obs["/observe/tactile/deform"] = np.zeros((5, *TACTILE_DEFORM_SHAPE), dtype=np.uint8)
            obs["/observe/tactile/channels"] = np.asarray(TACTILE_CHANNELS, dtype=np.int64)
        else:
            f6 = None
        if f6 is not None:
            for finger_idx in range(5):
                force = f6[finger_idx, :3].copy()
                torque = f6[finger_idx, 3:6].copy()
                obs[f"/observe/tactile/finger{finger_idx}/force"] = force
                obs[f"/observe/tactile/finger{finger_idx}/torque"] = torque
                # data_tactile.py's inference pipeline also concatenates the /next keys.
                # In live inference there is no future tactile frame yet, so use the same
                # current/placeholder value to satisfy the trained tactile pipeline shape.
                obs[f"/observe/tactile/finger{finger_idx}/force/next"] = force.copy()
                obs[f"/observe/tactile/finger{finger_idx}/torque/next"] = torque.copy()
            obs["/observe/tactile/f6"] = f6

        if not hasattr(self, "_printed_obs_shapes"):
            self._printed_obs_shapes = True
            tactile_shape = None if f6 is None else tuple(f6.shape)
            deform_shape = None if "/observe/tactile/deform" not in obs else tuple(obs["/observe/tactile/deform"].shape)
            print(
                "[obs] "
                f"front={front_rgb.shape} wrist={wrist_rgb.shape} "
                f"arm={obs['/state/arm/eef_pose'].shape} hand={obs['/state/hand/joint_angle'].shape} "
                f"tactile_f6={tactile_shape} tactile_deform={deform_shape}"
            )
        if self.obs_viz is not None:
            self.obs_viz.update(obs, chunk_idx=self.chunk_idx)
        return obs

    def reset(self, reset_meta: Optional[Dict[str, Any]] = None) -> None:
        del reset_meta
        self.obs = self._read_obs(clear=True)
        self.previous_action = np.concatenate([
            self.obs["/state/arm/eef_pose"],
            self.obs["/state/hand/joint_angle"],
        ]).astype(np.float64)

    def _validate_action_chunk(self, action: Any) -> np.ndarray:
        if not isinstance(action, dict):
            action = {"action": action}
        if "/action/arm/eef_pose" not in action or "/action/hand/joint_angle" not in action:
            raise KeyError("DirectRobotEnv expected action keys '/action/arm/eef_pose' and '/action/hand/joint_angle'.")
        arm = np.asarray(action["/action/arm/eef_pose"], dtype=np.float64)
        hand = np.asarray(action["/action/hand/joint_angle"], dtype=np.float64)
        if arm.ndim == 1:
            arm = arm.reshape(1, -1)
        if hand.ndim == 1:
            hand = hand.reshape(1, -1)
        if arm.shape[1] != ARM_DIM:
            raise ValueError(f"Expected arm action chunk shape (N,{ARM_DIM}), got {arm.shape}")
        if hand.shape[1] != HAND_DIM:
            raise ValueError(f"Expected hand action chunk shape (N,{HAND_DIM}), got {hand.shape}")
        steps = min(arm.shape[0], hand.shape[0])
        if self.action_chunk_steps is not None:
            steps = min(steps, int(self.action_chunk_steps))
        if steps <= 0:
            raise ValueError("Action chunk is empty.")
        actions = np.concatenate([arm[:steps], hand[:steps]], axis=-1)
        if not np.isfinite(actions).all():
            raise ValueError("Action chunk contains NaN or Inf.")
        return actions

    def _execute_action_step(self, raw_action: np.ndarray, step_idx: int) -> np.ndarray:
        target = raw_action
        if not self.disable_clamp:
            target = clamp_action_step(raw_action, self.previous_action, self.max_arm_xyz_step, self.max_hand_step)
        arm9, hand22 = split_action(target)
        pose = arm9_to_pose_matrix(arm9)
        hand22_real = hand22[POLICY2REAL_DOF_INDICES]
        franka_response = None
        ik_result = None

        if self.live:
            if self.franka_control_mode == "joints":
                if self.ik_solver is None:
                    raise RuntimeError("ik_solver is required when franka_control_mode='joints' and live=True")
                ik_result = self.ik_solver.solve(pose)
                franka_response = self.franka.send_move_joints(ik_result.joint_positions.tolist())
                _check_controller_response(franka_response, "franka.send_move_joints")
            else:
                franka_response = self.franka.send_move(pose.tolist())
                _check_controller_response(franka_response, "franka.send_move")
            self.hand.set_action(hand22_real, interpolate=self.hand_interpolate)

        if self.log_action_steps:
            ik_summary = ""
            if ik_result is not None:
                pos_err_mm = float(np.linalg.norm(ik_result.pos_err) * 1000.0)
                rot_err_deg = float(np.linalg.norm(ik_result.rot_err) * 180.0 / np.pi)
                ik_summary = f" ik_pos_err_mm={pos_err_mm:.2f} ik_rot_err_deg={rot_err_deg:.2f}"
            mode = "LIVE" if self.live else "dry-run"
            print(
                f"[action chunk {self.chunk_idx:04d} step {step_idx:02d} {mode}] "
                f"raw_shape={tuple(np.asarray(raw_action).shape)} target_shape={tuple(np.asarray(target).shape)} "
                f"xyz={np.round(arm9[:3], 4).tolist()} "
                f"hand_min={float(hand22.min()):.4f} hand_max={float(hand22.max()):.4f} "
                f"franka_response={franka_response}{ik_summary}"
            )
        return np.asarray(target, dtype=np.float64)

    def step(self, action: Optional[Any] = None, other_info: Optional[Dict[str, Any]] = None) -> None:
        del other_info
        if action is not None:
            actions = self._validate_action_chunk(action)
            print(
                f"[chunk {self.chunk_idx:04d} {'LIVE' if self.live else 'dry-run'}] "
                f"action_shape={tuple(actions.shape)} min={float(actions.min()):.4f} max={float(actions.max()):.4f} "
                f"first_xyz={np.round(actions[0, :3], 4).tolist()} last_xyz={np.round(actions[-1, :3], 4).tolist()}"
            )
            next_deadline = time.monotonic()
            for step_idx, raw_action in enumerate(actions):
                self.previous_action = self._execute_action_step(raw_action, step_idx)
                next_deadline += self.period
                remaining = next_deadline - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
            self.chunk_idx += 1
        self.obs = self._read_obs(clear=False)
        self.reward = 0.0

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.live and self.stop_on_close:
            try:
                _check_controller_response(self.franka.send_stop(), "franka.send_stop")
            except Exception as exc:
                print(f"[cleanup] franka.send_stop failed: {exc}")
        if self.obs_viz is not None:
            self.obs_viz.close()
        if self.show_camera_input:
            try:
                cv2.destroyWindow(self.camera_window)
            except cv2.error as exc:
                print(f"[cleanup] failed to close camera window: {exc}")
        if self.tactile is not None:
            self.tactile.close()
        self.cameras.stop()
        self.hand.close()
        self.franka.close()
