"""Local ZeroMQ clients and SharpA joint-order definitions.

This file intentionally contains the real-hardware communication code used by
the runner.  It does not import TacMP or dex-controller.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np


DEFAULT_FRANKA_HOST = "172.16.0.10"
DEFAULT_FRANKA_PORT = 9090
DEFAULT_HAND_HOST = "localhost"
DEFAULT_HAND_PORT = 5570
HAND_DIM = 22

# SharpA server/SDK order.
REAL_SHARPA_DOF_NAMES = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)

# Isaac Gym / checkpoint order.
POLICY_SHARPA_DOF_NAMES = (
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
)

REAL2POLICY_DOF_INDICES = np.asarray(
    [REAL_SHARPA_DOF_NAMES.index(name) for name in POLICY_SHARPA_DOF_NAMES],
    dtype=np.int64,
)
POLICY2REAL_DOF_INDICES = np.asarray(
    [POLICY_SHARPA_DOF_NAMES.index(name) for name in REAL_SHARPA_DOF_NAMES],
    dtype=np.int64,
)

# v3right_sharpa_wave-forhammer5.urdf limits in checkpoint order.
POLICY_LOWER_LIMITS = np.asarray(
    [
        -0.17453293,
        -0.3491,
        0.0,
        0.0,
        -0.17453293,
        -0.3491,
        0.0,
        0.0,
        0.0,
        -0.17453293,
        -0.3491,
        0.0,
        0.0,
        -0.17453293,
        -0.3491,
        0.0,
        0.0,
        -0.1745,
        -0.3491,
        -0.5236,
        -0.3491,
        0.0,
    ],
    dtype=np.float64,
)
POLICY_UPPER_LIMITS = np.asarray(
    [
        1.5708,
        0.3491,
        1.7453,
        1.3963,
        1.5708,
        0.3491,
        1.7453,
        1.3963,
        0.2618,
        1.5708,
        0.3491,
        1.7453,
        1.3963,
        1.5708,
        0.3491,
        1.7453,
        1.3963,
        1.9199,
        0.1309,
        1.3963,
        0.3491,
        1.7453,
    ],
    dtype=np.float64,
)


def real_to_policy(values: np.ndarray) -> np.ndarray:
    values = _joint_vector(values, "real-order joint vector")
    return values[REAL2POLICY_DOF_INDICES].copy()


def policy_to_real(values: np.ndarray) -> np.ndarray:
    values = _joint_vector(values, "policy-order joint vector")
    return values[POLICY2REAL_DOF_INDICES].copy()


def _joint_vector(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (HAND_DIM,):
        raise ValueError(f"{label} must have shape ({HAND_DIM},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or Inf")
    return array


class SharpaWaveController:
    """Thin JSON-over-ZeroMQ client for the SharpA position server."""

    def __init__(
        self,
        host: str = DEFAULT_HAND_HOST,
        port: int = DEFAULT_HAND_PORT,
        timeout_ms: int = 2000,
    ) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq is required for SharpA control") from exc
        if not host:
            raise ValueError("SharpA host cannot be empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError(f"invalid SharpA port: {port}")
        if int(timeout_ms) <= 0:
            raise ValueError("SharpA timeout must be positive")

        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.LINGER, 0)
        self._address = f"tcp://{host}:{int(port)}"
        self._socket.connect(self._address)
        self._lock = threading.Lock()
        self._closed = False
        print(f"[hand] connecting to {self._address}", flush=True)
        try:
            response = self._send({"cmd": "ping"})
            self._require_ok(response, "ping")
        except BaseException:
            self.close()
            raise
        print("[hand] SharpA server ready", flush=True)

    def _send(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("SharpA controller is closed")
        with self._lock:
            try:
                self._socket.send_json(message)
                response = self._socket.recv_json()
            except self._zmq.Again as exc:
                raise TimeoutError(
                    f"SharpA request {message.get('cmd')!r} timed out at "
                    f"{self._address}"
                ) from exc
        if not isinstance(response, dict):
            raise RuntimeError(f"SharpA returned an invalid response: {response!r}")
        return response

    @staticmethod
    def _require_ok(response: dict[str, Any], command: str) -> None:
        if response.get("ok") is not True:
            raise RuntimeError(f"SharpA {command} failed: {response}")

    def get_state(self) -> np.ndarray:
        response = self._send({"cmd": "get_state"})
        self._require_ok(response, "get_state")
        return _joint_vector(response.get("angles"), "SharpA state").astype(
            np.float32
        )

    def set_action(self, angles: np.ndarray, interpolate: bool = False) -> None:
        angles = _joint_vector(angles, "SharpA action")
        response = self._send(
            {
                "cmd": "set_action",
                "angles": angles.tolist(),
                "interpolate": bool(interpolate),
            }
        )
        self._require_ok(response, "set_action")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close(linger=0)
        self._context.term()


class FrankaArmController:
    """MsgPack-over-ZeroMQ client used only for optional ready-pose setup."""

    def __init__(
        self,
        host: str = DEFAULT_FRANKA_HOST,
        port: int = DEFAULT_FRANKA_PORT,
        timeout_ms: int = 2000,
    ) -> None:
        try:
            import msgpack
            import zmq
        except ImportError as exc:
            raise RuntimeError("msgpack and pyzmq are required for Franka setup") from exc
        if not host:
            raise ValueError("Franka host cannot be empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError(f"invalid Franka port: {port}")
        if int(timeout_ms) <= 0:
            raise ValueError("Franka timeout must be positive")

        self._msgpack = msgpack
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.LINGER, 0)
        self._address = f"tcp://{host}:{int(port)}"
        self._socket.connect(self._address)
        self._lock = threading.Lock()
        self._closed = False
        print(f"[franka] connecting to {self._address}", flush=True)

    def _send(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Franka controller is closed")
        with self._lock:
            try:
                self._socket.send(self._msgpack.packb(message))
                response = self._msgpack.unpackb(self._socket.recv(), raw=False)
            except self._zmq.Again as exc:
                raise TimeoutError(
                    f"Franka request {message.get('cmd')!r} timed out at "
                    f"{self._address}"
                ) from exc
        if not isinstance(response, dict):
            raise RuntimeError(f"Franka returned an invalid response: {response!r}")
        return response

    @staticmethod
    def _require_ok(response: dict[str, Any], command: str) -> None:
        if response.get("status") != "ok":
            raise RuntimeError(f"Franka {command} failed: {response}")

    def get_joint_positions(self) -> np.ndarray:
        response = self._send({"cmd": "get_state"})
        self._require_ok(response, "get_state")
        joints = np.asarray(response.get("joint_positions"), dtype=np.float64)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError(f"invalid Franka joint state: shape={joints.shape}")
        return joints

    def send_move_joints(self, joints: np.ndarray) -> None:
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError(f"Franka target must have shape (7,), got {joints.shape}")
        response = self._send(
            {"cmd": "move_joints", "joint_positions": joints.tolist()}
        )
        self._require_ok(response, "move_joints")

    def send_stop(self) -> None:
        response = self._send({"cmd": "stop"})
        self._require_ok(response, "stop")

    def hold_current(self) -> None:
        self.send_move_joints(self.get_joint_positions())
        self.send_stop()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close(linger=0)
        self._context.term()
