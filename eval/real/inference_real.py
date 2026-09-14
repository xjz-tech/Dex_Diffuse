#!/usr/bin/env python3
"""Run the 22/66-D Sim-Hand Diffusion Policy on a real SharpA hand."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

EVAL_DIR = Path(__file__).resolve().parent.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from hardware import (
    HAND_DIM,
    POLICY_LOWER_LIMITS,
    POLICY_SHARPA_DOF_NAMES,
    POLICY_UPPER_LIMITS,
    REAL_SHARPA_DOF_NAMES,
    SharpaWaveController,
    policy_to_real,
    real_to_policy,
)
from policy_loader import load_policy
from policy_observation import (
    OBSERVATION_DIMS,
    compose_policy_observation,
    observation_dim,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help=(
            "Compile the 1D UNet with TensorRT FP16 (batch-1 DDIM). "
            "First load is slow; changing the checkpoint requires a fresh compile."
        ),
    )
    parser.add_argument(
        "--observation-mode",
        choices=tuple(OBSERVATION_DIMS),
        default="qpos-target-residual",
    )
    parser.add_argument("--action-chunk-steps", type=int, default=None)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load the checkpoint and run synthetic inference without hardware.",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--reset-before-load",
        action="store_true",
        help="In live mode, finish hand initialization and Enter confirmation before loading the model.",
    )

    parser.add_argument("--hand-host", default="localhost")
    parser.add_argument("--hand-port", type=int, default=5570)
    parser.add_argument("--hand-timeout-ms", type=int, default=2000)
    parser.add_argument("--hand-interpolate", action="store_true")
    parser.add_argument(
        "--initial-pose-file",
        default="",
        help="JSON pose that SharpA reaches before policy execution.",
    )
    parser.set_defaults(move_initial_pose=True, wait_for_enter=True)
    parser.add_argument(
        "--no-move-initial-pose",
        action="store_false",
        dest="move_initial_pose",
    )
    parser.add_argument("--initial-pose-steps", type=int, default=50)
    parser.add_argument("--initial-pose-seconds", type=float, default=2.0)
    parser.add_argument("--initial-pose-tolerance", type=float, default=0.1)
    parser.add_argument("--initial-pose-timeout-seconds", type=float, default=3.0)
    parser.add_argument("--initial-pose-poll-seconds", type=float, default=0.1)
    parser.add_argument(
        "--no-wait-for-enter",
        action="store_false",
        dest="wait_for_enter",
    )

    parser.add_argument(
        "--max-hand-step",
        type=float,
        default=0.03,
        help="Per-command joint-target delta limit in radians.",
    )
    parser.add_argument(
        "--disable-step-clamp",
        action="store_true",
        help="Disable the per-command delta clamp; URDF joint limits remain active.",
    )
    parser.add_argument(
        "--joint-limit-margin",
        type=float,
        default=0.0,
        help="Extra distance kept from every URDF joint limit, in radians.",
    )
    parser.add_argument(
        "--max-tracking-error",
        type=float,
        default=0.35,
        help="Abort if measured qpos differs this much from the last target; 0 disables.",
    )
    parser.add_argument("--start-delay", type=float, default=3.0)
    parser.add_argument("--log-action-steps", action="store_true")
    parser.set_defaults(hold_current_on_exit=True)
    parser.add_argument(
        "--no-hold-current-on-exit",
        action="store_false",
        dest="hold_current_on_exit",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    args.checkpoint = str(checkpoint)
    if args.inference_steps <= 0:
        raise ValueError("--inference-steps must be positive")
    if args.action_chunk_steps is not None and args.action_chunk_steps <= 0:
        raise ValueError("--action-chunk-steps must be positive")
    if not np.isfinite(args.hz) or args.hz <= 0:
        raise ValueError("--hz must be finite and positive")
    if args.max_steps < 0:
        raise ValueError("--max-steps cannot be negative")
    if not np.isfinite(args.max_hand_step) or args.max_hand_step <= 0:
        raise ValueError("--max-hand-step must be finite and positive")
    if not np.isfinite(args.joint_limit_margin) or args.joint_limit_margin < 0:
        raise ValueError("--joint-limit-margin must be finite and non-negative")
    half_width = 0.5 * (POLICY_UPPER_LIMITS - POLICY_LOWER_LIMITS)
    if np.any(args.joint_limit_margin >= half_width):
        raise ValueError("--joint-limit-margin is too large for at least one joint")
    if not np.isfinite(args.max_tracking_error) or args.max_tracking_error < 0:
        raise ValueError("--max-tracking-error must be finite and non-negative")
    if not np.isfinite(args.start_delay) or args.start_delay < 0:
        raise ValueError("--start-delay must be finite and non-negative")
    if args.initial_pose_steps <= 0:
        raise ValueError("--initial-pose-steps must be positive")
    for name in (
        "initial_pose_seconds",
        "initial_pose_tolerance",
        "initial_pose_timeout_seconds",
        "initial_pose_poll_seconds",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and positive"
            )


def device_or_raise(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({device}), but CUDA is unavailable")
    return device


def predict(policy, history: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    history = np.asarray(history, dtype=np.float32, order="C")
    expected = (1, int(policy.n_obs_steps), int(policy.obs_dim))
    if history.shape != expected:
        raise ValueError(f"observation shape {history.shape}, expected {expected}")
    if not np.isfinite(history).all():
        raise ValueError("observation history contains NaN or Inf")

    observation = torch.from_numpy(history).to(device=device, dtype=torch.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        action = policy.predict_action({"obs": observation})["action"]
    elapsed = time.perf_counter() - started
    expected_action = (1, int(policy.n_action_steps), int(policy.action_dim))
    if tuple(action.shape) != expected_action:
        raise RuntimeError(
            f"policy action shape {tuple(action.shape)}, expected {expected_action}"
        )
    if not bool(torch.isfinite(action).all()):
        raise RuntimeError("policy returned NaN or Inf")
    return action[0].detach().cpu().float().numpy(), elapsed


def normalizer_observation_mean(policy) -> np.ndarray:
    stats = policy.normalizer.get_input_stats()
    mean = stats["obs"]["mean"].detach().cpu().float().numpy()
    expected = (int(policy.obs_dim),)
    if mean.shape != expected or not np.isfinite(mean).all():
        raise ValueError(
            f"observation normalizer mean shape {mean.shape}, expected {expected}"
        )
    return mean


def safe_target(
    raw_target: np.ndarray,
    previous_target: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, int]:
    raw_target = np.asarray(raw_target, dtype=np.float64)
    previous_target = np.asarray(previous_target, dtype=np.float64)
    if raw_target.shape != (HAND_DIM,) or previous_target.shape != (HAND_DIM,):
        raise ValueError("raw/previous hand targets must both have shape (22,)")
    if not np.isfinite(raw_target).all():
        raise RuntimeError("policy target contains NaN or Inf")

    lower = POLICY_LOWER_LIMITS + args.joint_limit_margin
    upper = POLICY_UPPER_LIMITS - args.joint_limit_margin
    limited = np.clip(raw_target, lower, upper)
    joint_limit_clips = int(np.count_nonzero(np.abs(limited - raw_target) > 1e-9))
    if args.disable_step_clamp:
        return limited, joint_limit_clips, 0
    stepped = previous_target + np.clip(
        limited - previous_target,
        -args.max_hand_step,
        args.max_hand_step,
    )
    stepped = np.clip(stepped, lower, upper)
    step_clips = int(np.count_nonzero(np.abs(stepped - limited) > 1e-9))
    return stepped, joint_limit_clips, step_clips


def measured_policy_qpos(hand: SharpaWaveController) -> np.ndarray:
    qpos = real_to_policy(hand.get_state()).astype(np.float32)
    if qpos.shape != (HAND_DIM,) or not np.isfinite(qpos).all():
        raise RuntimeError("invalid SharpA joint observation")
    return qpos


def load_initial_pose(path_text: str) -> np.ndarray:
    """Load and validate the saved initial pose in checkpoint joint order."""
    if not path_text:
        raise ValueError("--initial-pose-file is required when pose motion is enabled")
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"initial pose file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        policy_pose = payload["policy_checkpoint_order"]
        names = tuple(policy_pose["joint_names"])
        qpos = np.asarray(policy_pose["qpos_rad"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid SharpA initial pose file: {path}") from exc
    if names != tuple(POLICY_SHARPA_DOF_NAMES):
        raise ValueError(
            "initial pose policy joint names/order do not match the checkpoint"
        )
    if qpos.shape != (HAND_DIM,) or not np.isfinite(qpos).all():
        raise ValueError("initial pose must contain 22 finite policy qpos values")

    real_pose = payload.get("real_server_order")
    if real_pose is not None:
        try:
            real_names = tuple(real_pose["joint_names"])
            real_qpos = np.asarray(real_pose["qpos_rad"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid real-order pose in {path}") from exc
        if real_names != tuple(REAL_SHARPA_DOF_NAMES):
            raise ValueError("initial pose real joint names/order do not match SharpA")
        if real_qpos.shape != (HAND_DIM,) or not np.isfinite(real_qpos).all():
            raise ValueError("initial pose must contain 22 finite real qpos values")
        if not np.allclose(real_qpos, policy_to_real(qpos), atol=1e-6, rtol=0.0):
            raise ValueError("real-order and policy-order initial poses disagree")

    validate_measured_limits(qpos, tolerance=0.0)
    print(f"[initial-pose] loaded target from {path}", flush=True)
    return qpos


def move_hand_to_initial_pose(
    hand: SharpaWaveController,
    target_policy: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Move only SharpA smoothly to the saved initial pose and verify arrival."""
    target_real = policy_to_real(target_policy)
    start_real = np.asarray(hand.get_state(), dtype=np.float64)
    validate_measured_limits(real_to_policy(start_real))
    print(
        f"[initial-pose] moving SharpA: {args.initial_pose_steps} steps / "
        f"{args.initial_pose_seconds:g}s",
        flush=True,
    )
    period = args.initial_pose_seconds / args.initial_pose_steps
    next_deadline = time.monotonic()
    for step in range(1, args.initial_pose_steps + 1):
        ratio = step / args.initial_pose_steps
        command = start_real + ratio * (target_real - start_real)
        hand.set_action(command, interpolate=False)
        next_deadline += period
        remaining = next_deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    hand.set_action(target_real, interpolate=False)

    deadline = time.monotonic() + args.initial_pose_timeout_seconds
    while True:
        measured_policy = measured_policy_qpos(hand)
        validate_measured_limits(measured_policy)
        error = float(np.max(np.abs(measured_policy - target_policy)))
        print(
            f"\r[initial-pose] SharpA max error={error:.4f} rad",
            end="",
            flush=True,
        )
        if error <= args.initial_pose_tolerance:
            print("\n[initial-pose] SharpA initial pose reached", flush=True)
            return measured_policy
        if time.monotonic() >= deadline:
            print(flush=True)
            raise TimeoutError(
                "SharpA did not reach the initial pose within "
                f"{args.initial_pose_timeout_seconds:g}s; max error={error:.4f} rad"
            )
        time.sleep(args.initial_pose_poll_seconds)


def wait_for_policy_start() -> None:
    try:
        input("[ready] SharpA is at the initial pose. Press Enter to start policy execution...")
    except EOFError as exc:
        raise RuntimeError(
            "stdin closed while waiting for Enter; run in an interactive terminal or "
            "set WAIT_FOR_ENTER=0"
        ) from exc
    print("[ready] Enter received; starting policy inference and action execution", flush=True)


def validate_measured_limits(qpos: np.ndarray, tolerance: float = 0.05) -> None:
    below = POLICY_LOWER_LIMITS - np.asarray(qpos, dtype=np.float64)
    above = np.asarray(qpos, dtype=np.float64) - POLICY_UPPER_LIMITS
    violation = float(max(0.0, np.max(below), np.max(above)))
    if violation > tolerance:
        raise RuntimeError(
            f"measured SharpA qpos exceeds policy URDF limits by {violation:.4f} rad"
        )


def run_hardware(args: argparse.Namespace, policy, device: torch.device) -> int:
    if policy is not None:
        action_chunk_steps = args.action_chunk_steps or int(policy.n_action_steps)
    if policy is not None and action_chunk_steps > int(policy.n_action_steps):
        raise ValueError(
            f"--action-chunk-steps={action_chunk_steps} exceeds policy output "
            f"length {int(policy.n_action_steps)}"
        )

    hand = None
    steps_executed = 0
    chunk_index = 0
    total_inference_seconds = 0.0
    total_limit_clips = 0
    total_step_clips = 0
    started = time.monotonic()
    try:
        hand = SharpaWaveController(
            args.hand_host,
            args.hand_port,
            args.hand_timeout_ms,
        )
        if args.live and args.move_initial_pose:
            initial_pose_target = load_initial_pose(args.initial_pose_file)
            qpos = move_hand_to_initial_pose(hand, initial_pose_target, args)
            current_target = initial_pose_target.copy()
        else:
            qpos = measured_policy_qpos(hand)
            validate_measured_limits(qpos)
            current_target = np.clip(
                qpos.astype(np.float64),
                POLICY_LOWER_LIMITS + args.joint_limit_margin,
                POLICY_UPPER_LIMITS - args.joint_limit_margin,
            )
            if args.live:
                print("[initial-pose] motion disabled; using measured qpos", flush=True)
            else:
                print("[initial-pose] dry-run; no pose motion sent", flush=True)

        if args.live and args.wait_for_enter:
            wait_for_policy_start()
            qpos = measured_policy_qpos(hand)
            validate_measured_limits(qpos)
            if not args.move_initial_pose:
                current_target = np.clip(
                    qpos.astype(np.float64),
                    POLICY_LOWER_LIMITS + args.joint_limit_margin,
                    POLICY_UPPER_LIMITS - args.joint_limit_margin,
                )
        if policy is None:
            # Keep reset, loading, and execution under the same hardware cleanup.
            policy = load_inference_policy(args, device)
            action_chunk_steps = args.action_chunk_steps or int(policy.n_action_steps)
            if action_chunk_steps > int(policy.n_action_steps):
                raise ValueError(
                    f"--action-chunk-steps={action_chunk_steps} exceeds policy output "
                    f"length {int(policy.n_action_steps)}"
                )
            # Loading may take time: start inference with fresh measured state.
            qpos = measured_policy_qpos(hand)
            validate_measured_limits(qpos)
            if not args.move_initial_pose:
                current_target = np.clip(
                    qpos.astype(np.float64),
                    POLICY_LOWER_LIMITS + args.joint_limit_margin,
                    POLICY_UPPER_LIMITS - args.joint_limit_margin,
                )
        print(
            "[initial-state] measured policy-order qpos(rad)="
            + np.array2string(
                qpos,
                precision=6,
                separator=", ",
                max_line_width=1000,
            ),
            flush=True,
        )
        first_observation = compose_policy_observation(
            qpos,
            current_target,
            args.observation_mode,
        )
        history = deque(
            [first_observation.copy() for _ in range(int(policy.n_obs_steps))],
            maxlen=int(policy.n_obs_steps),
        )

        mode = "LIVE" if args.live else "dry-run (state read only)"
        print(
            f"[mode] {mode}; {int(policy.obs_dim)}-D obs -> "
            f"{int(policy.action_dim)}-D absolute SharpA targets; "
            f"serial {action_chunk_steps}-step chunks at {args.hz:g} Hz",
            flush=True,
        )
        print(
            "[schedule] infer -> execute chunk with per-step state reads -> infer",
            flush=True,
        )
        if args.live and args.start_delay > 0:
            print(
                f"[safety] live commands start in {args.start_delay:g}s; "
                "press Ctrl-C to cancel",
                flush=True,
            )
            time.sleep(args.start_delay)

        while args.max_steps == 0 or steps_executed < args.max_steps:
            history_batch = np.stack(tuple(history), axis=0)[None]
            actions, inference_seconds = predict(policy, history_batch, device)
            total_inference_seconds += inference_seconds
            remaining_steps = (
                action_chunk_steps
                if args.max_steps == 0
                else min(action_chunk_steps, args.max_steps - steps_executed)
            )
            actions = actions[:remaining_steps]
            print(
                f"[chunk {chunk_index:04d}] inference={inference_seconds:.3f}s "
                f"predicted={tuple(actions.shape)}",
                flush=True,
            )

            next_deadline = time.monotonic()
            for chunk_step, raw_target in enumerate(actions):
                target, limit_clips, step_clips = safe_target(
                    raw_target,
                    current_target,
                    args,
                )
                total_limit_clips += limit_clips
                total_step_clips += step_clips
                if args.live:
                    hand.set_action(
                        policy_to_real(target),
                        interpolate=args.hand_interpolate,
                    )

                # Observe after one control period, matching a completed real
                # action step rather than sampling immediately after sending.
                next_deadline += 1.0 / args.hz
                remaining = next_deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                qpos = measured_policy_qpos(hand)
                validate_measured_limits(qpos)
                current_target = target
                if args.live and args.max_tracking_error > 0:
                    tracking_error = float(
                        np.max(np.abs(qpos.astype(np.float64) - current_target))
                    )
                    if tracking_error > args.max_tracking_error:
                        raise RuntimeError(
                            f"SharpA tracking error {tracking_error:.4f} rad exceeds "
                            f"limit {args.max_tracking_error:.4f} rad"
                        )
                else:
                    tracking_error = float("nan")

                history.append(
                    compose_policy_observation(
                        qpos,
                        current_target,
                        args.observation_mode,
                    )
                )
                steps_executed += 1
                if args.log_action_steps:
                    print(
                        f"[action {steps_executed:06d} chunk_step={chunk_step}] "
                        f"raw=[{float(np.min(raw_target)):.4f},"
                        f"{float(np.max(raw_target)):.4f}] "
                        f"sent=[{float(np.min(target)):.4f},"
                        f"{float(np.max(target)):.4f}] "
                        f"limit_clips={limit_clips} step_clips={step_clips} "
                        f"tracking_max={tracking_error:.4f}",
                        flush=True,
                    )
            chunk_index += 1

        elapsed = time.monotonic() - started
        print(
            f"[done] steps={steps_executed} chunks={chunk_index} "
            f"elapsed={elapsed:.2f}s inference={total_inference_seconds:.2f}s "
            f"joint_limit_clips={total_limit_clips} step_clips={total_step_clips}",
            flush=True,
        )
        return 0
    finally:
        if hand is not None:
            if args.live and args.hold_current_on_exit:
                try:
                    measured_real = hand.get_state()
                    hand.set_action(measured_real, interpolate=False)
                    print("[cleanup] SharpA held at measured position", flush=True)
                except Exception as exc:
                    print(f"[cleanup] unable to hold SharpA: {exc}", flush=True)
            hand.close()


def load_inference_policy(args: argparse.Namespace, device: torch.device):
    print(f"[policy] loading {args.checkpoint}", flush=True)
    loaded, policy, spec = load_policy(
        args.checkpoint,
        device,
        args.sampler,
        args.inference_steps,
    )
    expected_obs_dim = observation_dim(args.observation_mode)
    if spec["obs_dim"] != expected_obs_dim:
        raise ValueError(
            f"checkpoint obs_dim={spec['obs_dim']} does not match "
            f"--observation-mode={args.observation_mode} ({expected_obs_dim})"
        )
    print(
        f"[policy] loaded {loaded.weight_source}; step={loaded.global_step} "
        f"epoch={loaded.epoch}; spec={spec}; sampler={args.sampler} "
        f"inference_steps={policy.num_inference_steps}",
        flush=True,
    )
    if args.tensorrt:
        from trt_unet import accelerate_policy_unet

        print(
            "[policy] compiling TensorRT FP16 UNet "
            "(weights are baked in; swap checkpoint => recompile)...",
            flush=True,
        )
        started = time.perf_counter()
        accelerate_policy_unet(policy, fp16=True)
        print(
            f"[policy] TensorRT UNet ready in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    # load_state_dict has copied the selected weights into the policy. Release
    # the checkpoint mapping before the long-running hardware loop.
    del loaded

    mean = normalizer_observation_mean(policy)
    smoke_history = np.repeat(mean[None, None, :], int(policy.n_obs_steps), axis=1)
    if args.check_only or not args.no_warmup:
        _, elapsed = predict(policy, smoke_history, device)
        print(f"[policy] synthetic inference passed in {elapsed:.3f}s", flush=True)
        # Keep the first real diffusion sample independent of the warm-up.
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
    return policy


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = device_or_raise(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.reset_before_load and args.live and not args.check_only:
        return run_hardware(args, None, device)
    policy = load_inference_policy(args, device)
    if args.check_only:
        print("[check] checkpoint check passed; hardware was not connected", flush=True)
        return 0
    return run_hardware(args, policy, device)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[real] stopped by Ctrl-C", flush=True)
        raise SystemExit(130)
