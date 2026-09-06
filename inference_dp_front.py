#!/usr/bin/env python3
"""Run a front-image-only bulb checkpoint on the real Franka + SharpA setup."""

from collections import deque
from pathlib import Path
import time
import argparse

# Reuse checkpoint loading and coordinate transforms without changing the
# original two-camera entrypoint or its module globals.
from inference_dp import (
    ACTION_DIM, EE_DIM, OmegaConf, np, torch, ee_pose_relative_to,
    load_policy, mixed_actions_to_absolute, resize_chw_float,
)
from direct_robot_front_env import FrontDirectRobotEnv


def capture_obs(obs: dict, rgb_shapes: dict[str, tuple[int, int, int]]) -> dict:
    return {
        "front_image": resize_chw_float(
            obs["/observe/vision/front/rgb"], rgb_shapes["front_image"]
        ),
        "ee_pose": np.asarray(obs["/state/arm/eef_pose"], dtype=np.float32),
        "hand_joint": np.asarray(
            obs["/state/hand/joint_angle"], dtype=np.float32
        ),
    }


def build_policy_obs(obs_history, device: torch.device, relative_ee: bool):
    observations = list(obs_history)
    base_ee_pose = observations[-1]["ee_pose"]
    absolute_ee = np.stack([obs["ee_pose"] for obs in observations])
    policy_ee = (
        ee_pose_relative_to(absolute_ee, base_ee_pose)
        if relative_ee
        else absolute_ee
    )

    def tensor(values):
        return torch.from_numpy(np.stack(values)[None]).float().to(device)

    policy_obs = {
        "front_image": tensor([obs["front_image"] for obs in observations]),
        "ee_pose": torch.from_numpy(policy_ee[None]).float().to(device),
        "hand_joint": tensor([obs["hand_joint"] for obs in observations]),
    }
    return policy_obs, base_ee_pose


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
    parser.add_argument("--front_resolution", default="640x480")
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
    n_obs_steps = int(cfg.n_obs_steps)
    trained_action_steps = int(cfg.n_action_steps)
    inference_action_steps = int(cfg.horizon) - n_obs_steps + 1
    if inference_action_steps <= 0:
        raise ValueError(
            f"Invalid inference action length: horizon={int(cfg.horizon)}, "
            f"n_obs_steps={n_obs_steps}"
        )
    # The diffusion model predicts the full horizon. Training's A8 only selects
    # eight actions from it; deployment can consume all actions from To - 1 to T.
    policy.n_action_steps = inference_action_steps
    relative_ee = bool(OmegaConf.select(cfg, "task.dataset.relative", default=True))
    expected_obs = {"front_image", "ee_pose", "hand_joint"}
    actual_obs = set(cfg.shape_meta.obs.keys())
    if actual_obs != expected_obs:
        raise ValueError(
            f"Front-only inference requires observations {sorted(expected_obs)}, "
            f"but checkpoint has {sorted(actual_obs)}. "
            "For a front+wrist checkpoint, use inference_dp_dino.sh."
        )
    expected_shapes = {"ee_pose": (EE_DIM,), "hand_joint": (ACTION_DIM - EE_DIM,)}
    for key, shape in expected_shapes.items():
        if tuple(cfg.shape_meta.obs[key].shape) != shape:
            raise ValueError(f"Expected {key} shape {shape}")
    rgb_shapes = {
        key: tuple(int(x) for x in cfg.shape_meta.obs[key].shape)
        for key in ("front_image",)
    }
    action_chunk_steps = args.action_chunk_steps or inference_action_steps
    if action_chunk_steps > inference_action_steps:
        raise ValueError(
            f"--action_chunk_steps={action_chunk_steps} exceeds the policy output "
            f"length {inference_action_steps}"
        )
    print(
        f"[policy] loaded n_obs_steps={n_obs_steps} "
        f"trained_n_action_steps={trained_action_steps} "
        f"inference_n_action_steps={inference_action_steps} "
        f"horizon={int(cfg.horizon)} "
        f"num_inference_steps={policy.num_inference_steps} "
        f"ee_mode={'relative' if relative_ee else 'absolute'}"
    )
    if n_obs_steps <= 0:
        raise ValueError(f"Checkpoint has invalid n_obs_steps={n_obs_steps}")
    if int(policy.action_dim) != ACTION_DIM:
        raise ValueError(
            f"Direct robot expects action_dim={ACTION_DIM}, got {policy.action_dim}"
        )
    if args.check_only:
        dummy_obs = {
            "front_image": torch.zeros(
                1, n_obs_steps, *rgb_shapes["front_image"], device=device
            ),
            "ee_pose": torch.zeros(1, n_obs_steps, EE_DIM, device=device),
            "hand_joint": torch.zeros(
                1, n_obs_steps, ACTION_DIM - EE_DIM, device=device
            ),
        }
        with torch.inference_mode():
            check_action = policy.predict_action(dummy_obs)["action"]
        expected_shape = (1, inference_action_steps, ACTION_DIM)
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

    env = FrontDirectRobotEnv(args)

    try:
        first_obs = capture_obs(env.reset(), rgb_shapes)
        obs_history = deque([first_obs] * n_obs_steps, maxlen=n_obs_steps)
        print(
            f"[mode] {'LIVE' if args.live else 'dry-run'}: "
            f"{'relative' if relative_ee else 'absolute'} EEF + absolute hand, "
            f"{args.hz:g} Hz, "
            f"{action_chunk_steps} actions/query"
        )
        if args.serial_chunks:
            print(
                f"[chunk mode] serial: infer -> execute {action_chunk_steps} "
                "actions -> observe -> infer"
            )

        while True:
            policy_obs, base_ee_pose = build_policy_obs(
                obs_history, device, relative_ee
            )
            with torch.inference_mode():
                prediction = policy.predict_action(policy_obs)
            mixed_actions = prediction["action"][0].detach().cpu().numpy()
            if relative_ee:
                absolute_actions = mixed_actions_to_absolute(
                    mixed_actions, base_ee_pose
                )[:action_chunk_steps]
            else:
                absolute_actions = mixed_actions[:action_chunk_steps]
            if len(absolute_actions) != action_chunk_steps:
                raise RuntimeError(
                    f"Expected {action_chunk_steps} actions, got "
                    f"{len(absolute_actions)}"
                )
            if not np.isfinite(absolute_actions).all():
                raise RuntimeError("Policy produced NaN or Inf; refusing to execute")

            print(
                f"[chunk {env.chunk_idx:04d}] predicted={tuple(mixed_actions.shape)} "
                f"executing={tuple(absolute_actions.shape)} "
                f"first_xyz={np.round(absolute_actions[0, :3], 4).tolist()}"
            )
            next_deadline = time.monotonic()
            executed_steps = 0
            for step_idx, action in enumerate(absolute_actions):
                obs = env.step_single(action, step_idx)
                obs_history.append(capture_obs(obs, rgb_shapes))
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
