#!/usr/bin/env python3
"""Run the bulb image Diffusion Policy on the real Franka + SharpA setup."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "12")
os.environ.setdefault("OMP_NUM_THREADS", "12")

import cv2  # noqa: E402

cv2.setNumThreads(12)

import dill  # noqa: E402
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

DP_DIR = Path(__file__).resolve().parent
THIRD_PARTY_DIR = DP_DIR.parent
sys.path.insert(0, str(THIRD_PARTY_DIR / "ViTacFormer"))

from direct_robot_env import ACTION_DIM, DirectRobotEnv  # noqa: E402
from diffusion_policy.dataset.bulb_image_dataset import (  # noqa: E402
    ee_pose_relative_to,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)

EE_DIM = 9
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240


class DiffusionDirectRobotEnv(DirectRobotEnv):
    """Expose one-step execution so the observation history stays consecutive."""

    def reset(self, reset_meta=None):
        super().reset(reset_meta=reset_meta)
        return self.obs

    def step_single(self, action: np.ndarray, step_idx: int):
        self.previous_action = self._execute_action_step(
            np.asarray(action, dtype=np.float64), step_idx
        )
        self.obs = self._read_obs(clear=False)
        return self.obs


def load_policy(checkpoint: Path, device: torch.device, num_inference_steps: int | None):
    payload = torch.load(
        checkpoint.open("rb"), pickle_module=dill, map_location="cpu"
    )
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if bool(cfg.training.use_ema) and getattr(workspace, "ema_model", None) is not None:
        policy = workspace.ema_model
    if num_inference_steps is not None:
        policy.num_inference_steps = num_inference_steps
    policy.eval().to(device)
    return cfg, policy


def resize_chw_float(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        np.asarray(image),
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    return np.moveaxis(resized.astype(np.float32) / 255.0, -1, 0)


def capture_obs(obs: dict) -> dict:
    return {
        "front_image": resize_chw_float(obs["/observe/vision/front/rgb"]),
        "wrist_image": resize_chw_float(obs["/observe/vision/wrist/rgb"]),
        "ee_pose": np.asarray(obs["/state/arm/eef_pose"], dtype=np.float32),
        "hand_joint": np.asarray(
            obs["/state/hand/joint_angle"], dtype=np.float32
        ),
    }


def obs_keys_from_cfg(cfg) -> list[str]:
    return list(cfg.shape_meta.obs.keys())


def build_policy_obs(obs_history, device: torch.device, obs_keys: list[str]):
    observations = list(obs_history)
    base_ee_pose = observations[-1]["ee_pose"]
    policy_obs = {}
    if "ee_pose" in obs_keys:
        absolute_ee = np.stack([obs["ee_pose"] for obs in observations])
        relative_ee = ee_pose_relative_to(absolute_ee, base_ee_pose)
        policy_obs["ee_pose"] = torch.from_numpy(relative_ee[None]).float().to(device)

    def tensor(key):
        return torch.from_numpy(np.stack([obs[key] for obs in observations])[None]).float().to(device)

    for key in obs_keys:
        if key == "ee_pose":
            continue
        policy_obs[key] = tensor(key)
    return policy_obs, base_ee_pose


def mixed_actions_to_absolute(actions: np.ndarray, base_ee_pose: np.ndarray) -> np.ndarray:
    """Invert BulbImageDataset's relative-EEF transform; keep hand absolute."""
    actions = np.asarray(actions, dtype=np.float32)
    result = actions.copy()
    base_rotation = rotation_6d_to_matrix(base_ee_pose[None, 3:9])[0]
    relative_rotation = rotation_6d_to_matrix(actions[..., 3:9])

    result[..., :3] = base_ee_pose[:3] + np.einsum(
        "ij,...j->...i", base_rotation, actions[..., :3]
    )
    absolute_rotation = np.einsum(
        "ij,...jk->...ik", base_rotation, relative_rotation
    )
    result[..., 3:9] = matrix_to_rotation_6d(absolute_rotation)
    # result[..., 9:31] is already an absolute SharpA joint target.
    return result


def policy_action_to_robot(action: np.ndarray, base_ee_pose: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] == ACTION_DIM:
        return mixed_actions_to_absolute(action, base_ee_pose)
    if action.shape[-1] == ACTION_DIM - EE_DIM:
        out = np.repeat(base_ee_pose[None, :], action.shape[0], axis=0)
        result = np.concatenate([out, action], axis=-1)
        return result
    raise ValueError(f"policy action last dim must be 22 or 31, got {action.shape[-1]}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_inference_steps", type=int, default=16)
    parser.add_argument("--action_chunk_steps", type=int, default=None)
    parser.add_argument(
        "--serial_chunks",
        action="store_true",
        help=(
            "Require every action in the current chunk to finish before the next "
            "policy inference."
        ),
    )
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Load and validate the checkpoint without connecting to hardware.",
    )

    parser.add_argument("--franka_host", default="172.16.0.10")
    parser.add_argument("--franka_port", type=int, default=9090)
    parser.add_argument("--franka_timeout_ms", type=int, default=2000)
    parser.add_argument(
        "--franka_control_mode", choices=["joints", "cartesian"], default="joints"
    )
    parser.add_argument("--franka_urdf", default=None)
    parser.add_argument("--max_joint_step", type=float, default=0.05)
    parser.add_argument("--ik_dq_max", type=float, default=0.5)
    parser.add_argument("--ik_damping", type=float, default=1e-4)
    parser.add_argument("--max_arm_xyz_step", type=float, default=0.03)
    parser.add_argument("--max_hand_step", type=float, default=0.03)
    parser.add_argument("--disable_clamp", action="store_true")

    parser.add_argument("--hand_host", default="localhost")
    parser.add_argument("--hand_port", type=int, default=5570)
    parser.add_argument("--hand_timeout_ms", type=int, default=2000)
    parser.add_argument("--hand_interpolate", action="store_true")

    parser.add_argument("--camera_fps", type=int, default=30)
    parser.add_argument("--camera_timeout_ms", type=int, default=1000)
    parser.add_argument("--front_serial", default=None)
    parser.add_argument("--wrist_serial", default=None)
    parser.add_argument("--front_resolution", default="640x480")
    parser.add_argument("--wrist_resolution", default="640x480")
    parser.add_argument("--rotate_wrist_camera_180", action="store_true", default=True)
    parser.add_argument(
        "--no_rotate_wrist_camera_180",
        dest="rotate_wrist_camera_180",
        action="store_false",
    )
    parser.add_argument("--show_camera_input", action="store_true")
    parser.add_argument("--log_action_steps", action="store_true", default=True)
    parser.add_argument(
        "--no_log_action_steps", dest="log_action_steps", action="store_false"
    )
    parser.add_argument("--stop_on_close", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.action_chunk_steps is not None and args.action_chunk_steps <= 0:
        raise ValueError("--action_chunk_steps must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({device}), but CUDA is unavailable")

    print(f"[policy] loading {checkpoint}")
    cfg, policy = load_policy(checkpoint, device, args.num_inference_steps)
    obs_keys = obs_keys_from_cfg(cfg)
    print(f"[policy] obs_keys={obs_keys} action_dim={int(policy.action_dim)}")
    n_obs_steps = int(cfg.n_obs_steps)
    trained_action_steps = int(cfg.n_action_steps)
    action_chunk_steps = args.action_chunk_steps or trained_action_steps
    if action_chunk_steps > trained_action_steps:
        raise ValueError(
            f"--action_chunk_steps={action_chunk_steps} exceeds the policy output "
            f"length {trained_action_steps}"
        )
    print(
        f"[policy] loaded n_obs_steps={n_obs_steps} "
        f"n_action_steps={trained_action_steps} horizon={int(cfg.horizon)} "
        f"num_inference_steps={policy.num_inference_steps}"
    )
    if n_obs_steps <= 0:
        raise ValueError(f"Checkpoint has invalid n_obs_steps={n_obs_steps}")
    if int(policy.action_dim) not in (ACTION_DIM - EE_DIM, ACTION_DIM):
        raise ValueError(
            f"policy action_dim must be {ACTION_DIM - EE_DIM} or {ACTION_DIM}, "
            f"got {policy.action_dim}"
        )
    if args.check_only:
        dummy_obs = {}
        for key in obs_keys:
            if key in ("front_image", "wrist_image"):
                dummy_obs[key] = torch.zeros(
                    1, n_obs_steps, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=device
                )
            elif key == "ee_pose":
                dummy_obs[key] = torch.zeros(1, n_obs_steps, EE_DIM, device=device)
            elif key == "hand_joint":
                dummy_obs[key] = torch.zeros(
                    1, n_obs_steps, ACTION_DIM - EE_DIM, device=device
                )
            else:
                raise ValueError(f"Unsupported obs key for --check_only dummy: {key}")
        with torch.inference_mode():
            check_action = policy.predict_action(dummy_obs)["action"]
        expected_shape = (1, trained_action_steps, int(policy.action_dim))
        if tuple(check_action.shape) != expected_shape:
            raise ValueError(
                f"Expected policy action shape {expected_shape}, got "
                f"{tuple(check_action.shape)}"
            )
        if not bool(torch.isfinite(check_action).all()):
            raise RuntimeError("Checkpoint smoke inference produced NaN or Inf")
        print(
            f"[check] checkpoint smoke inference passed: "
            f"action_shape={tuple(check_action.shape)}"
        )
        return

    env = DiffusionDirectRobotEnv(
        action_dim=ACTION_DIM,
        live=args.live,
        franka_host=args.franka_host,
        franka_port=args.franka_port,
        franka_timeout_ms=args.franka_timeout_ms,
        hand_host=args.hand_host,
        hand_port=args.hand_port,
        hand_timeout_ms=args.hand_timeout_ms,
        franka_control_mode=args.franka_control_mode,
        franka_urdf=args.franka_urdf,
        max_joint_step=args.max_joint_step,
        ik_dq_max=args.ik_dq_max,
        ik_damping=args.ik_damping,
        max_arm_xyz_step=args.max_arm_xyz_step,
        max_hand_step=args.max_hand_step,
        disable_clamp=args.disable_clamp,
        hand_interpolate=args.hand_interpolate,
        action_chunk_steps=1,
        hz=args.hz,
        log_action_steps=args.log_action_steps,
        use_tactile=False,
        camera_fps=args.camera_fps,
        camera_timeout_ms=args.camera_timeout_ms,
        front_serial=args.front_serial,
        wrist_serial=args.wrist_serial,
        front_resolution=args.front_resolution,
        wrist_resolution=args.wrist_resolution,
        rotate_wrist_camera_180=args.rotate_wrist_camera_180,
        show_camera_input=args.show_camera_input,
        stop_on_close=args.stop_on_close,
    )

    try:
        first_obs = capture_obs(env.reset())
        obs_history = deque([first_obs] * n_obs_steps, maxlen=n_obs_steps)
        print(
            f"[mode] {'LIVE' if args.live else 'dry-run'}: "
            f"relative EEF + absolute hand, {args.hz:g} Hz, "
            f"{action_chunk_steps} actions/query"
        )
        if args.serial_chunks:
            print(
                f"[chunk mode] serial: infer -> execute {action_chunk_steps} "
                "actions -> observe -> infer"
            )

        while True:
            policy_obs, base_ee_pose = build_policy_obs(obs_history, device, obs_keys)
            with torch.inference_mode():
                prediction = policy.predict_action(policy_obs)
            mixed_or_hand = prediction["action"][0].detach().cpu().numpy()
            absolute_actions = policy_action_to_robot(
                mixed_or_hand, base_ee_pose
            )[:action_chunk_steps]
            if len(absolute_actions) != action_chunk_steps:
                raise RuntimeError(
                    f"Expected {action_chunk_steps} actions, got "
                    f"{len(absolute_actions)}"
                )
            if not np.isfinite(absolute_actions).all():
                raise RuntimeError("Policy produced NaN or Inf; refusing to execute")

            print(
                f"[chunk {env.chunk_idx:04d}] predicted={tuple(mixed_or_hand.shape)} "
                f"executing={tuple(absolute_actions.shape)} "
                f"first_xyz={np.round(absolute_actions[0, :3], 4).tolist()}"
            )
            next_deadline = time.monotonic()
            executed_steps = 0
            for step_idx, action in enumerate(absolute_actions):
                obs = env.step_single(action, step_idx)
                obs_history.append(capture_obs(obs))
                executed_steps += 1
                next_deadline += 1.0 / args.hz
                remaining = next_deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            if args.serial_chunks and executed_steps != action_chunk_steps:
                raise RuntimeError(
                    f"Serial chunk ended after {executed_steps}/"
                    f"{action_chunk_steps} actions"
                )
            if args.serial_chunks:
                print(
                    f"[serial] completed chunk {env.chunk_idx}: "
                    f"{executed_steps} actions executed; starting next inference"
                )
            env.chunk_idx += 1
    finally:
        env.close()


if __name__ == "__main__":
    main()
