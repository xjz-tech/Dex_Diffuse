#!/usr/bin/env python3
"""Preflight and optionally initialize the Franka + SharpA real setup."""

from __future__ import annotations

import argparse
import time

import numpy as np

from direct_robot_env import FrankaArmController, SharpaWaveController


DEFAULT_READY_JOINTS = np.asarray(
    [0.5205, -0.7496, 0.6738, -2.2985, 0.4486, 1.6307, -1.2555],
    dtype=np.float64,
)
ARM_INTERPOLATION_STEPS = 30
ARM_INTERPOLATION_SECONDS = 3.0
# 全零初始姿态：启用这一行时，注释掉下方记录姿态的整个赋值块。
HAND_READY_JOINTS = np.zeros(22, dtype=np.float64)
# HAND_READY_JOINTS = np.asarray(
#     [
#         1.0299993753433228,
#         -0.09824000298976898,
#         0.36130866408348083,
#         -0.31451189517974854,
#         0.00842583179473877,
#         0.4360506236553192,
#         -0.055504534393548965,
#         0.837260901927948,
#         0.049354296177625656,
#         0.44111931324005127,
#         -0.06442071497440338,
#         0.6768842935562134,
#         0.10856100916862488,
#         0.33449938893318176,
#         0.09123406559228897,
#         0.9635206460952759,
#         0.3528777062892914,
#         0.015249253250658512,
#         0.7816686630249023,
#         -0.01308457925915718,
#         0.8393265604972839,
#         0.019550960510969162,
#     ],
#     dtype=np.float64,
# )


def _csv_vector(value: str, length: int, name: str) -> np.ndarray:
    try:
        result = np.asarray([float(item) for item in value.split(",")])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated floats") from exc
    if result.shape != (length,) or not np.isfinite(result).all():
        raise argparse.ArgumentTypeError(
            f"{name} must contain exactly {length} finite values"
        )
    return result.astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--franka-host", default="172.16.0.10")
    parser.add_argument("--franka-port", type=int, default=9090)
    parser.add_argument("--franka-timeout-ms", type=int, default=2000)
    parser.add_argument("--hand-host", default="localhost")
    parser.add_argument("--hand-port", type=int, default=5570)
    parser.add_argument("--hand-timeout-ms", type=int, default=2000)
    parser.add_argument("--skip-franka", action="store_true")
    parser.add_argument("--skip-hand", action="store_true")
    parser.add_argument("--no-move-franka", action="store_true")
    parser.add_argument("--no-move-hand", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify protocol/state only; never send a motion command.",
    )
    parser.add_argument(
        "--ready-joints",
        default=",".join(str(value) for value in DEFAULT_READY_JOINTS),
    )
    parser.add_argument("--hand-steps", type=int, default=50)
    parser.add_argument("--hand-seconds", type=float, default=2.0)
    parser.add_argument("--hand-tolerance", type=float, default=0.1)
    parser.add_argument("--hand-timeout-seconds", type=float, default=3.0)
    parser.add_argument("--arm-tolerance", type=float, default=0.02)
    parser.add_argument("--arm-settle-samples", type=int, default=3)
    parser.add_argument("--arm-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> np.ndarray:
    if HAND_READY_JOINTS.shape != (22,) or not np.isfinite(HAND_READY_JOINTS).all():
        raise ValueError("HAND_READY_JOINTS must contain 22 finite joint angles in radians")
    if args.skip_franka and args.skip_hand:
        raise ValueError("cannot skip both Franka and SharpA preflight")
    if args.hand_steps <= 0:
        raise ValueError("--hand-steps must be positive")
    positive = (
        "hand_seconds",
        "hand_tolerance",
        "hand_timeout_seconds",
        "arm_tolerance",
        "arm_timeout_seconds",
        "poll_seconds",
    )
    for name in positive:
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.arm_settle_samples <= 0:
        raise ValueError("--arm-settle-samples must be positive")
    return _csv_vector(args.ready_joints, 7, "--ready-joints")


def _check_franka_response(response: object, command: str) -> None:
    if not isinstance(response, dict):
        raise RuntimeError(f"{command} returned an invalid response: {response!r}")
    if response.get("status") != "ok":
        raise RuntimeError(f"{command} failed: {response}")


def hold_current_arm_position(arm: FrankaArmController) -> None:
    current = np.asarray(arm.get_joint_positions(), dtype=np.float64)
    _check_franka_response(
        arm.send_move_joints(current.tolist()), "Franka hold-current move_joints"
    )
    _check_franka_response(arm.send_stop(), "Franka stop")


def move_franka_to_ready(
    arm: FrankaArmController,
    target: np.ndarray,
    tolerance: float,
    settle_samples: int,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    start = np.asarray(arm.get_joint_positions(), dtype=np.float64)
    print(f"[init] Franka current: {np.round(start, 4).tolist()}", flush=True)
    print(f"[init] Franka target:  {np.round(target, 4).tolist()}", flush=True)
    print(
        f"[init] Franka moving through {ARM_INTERPOLATION_STEPS} interpolated targets / "
        f"{ARM_INTERPOLATION_SECONDS:.1f}s", flush=True,
    )
    period = ARM_INTERPOLATION_SECONDS / ARM_INTERPOLATION_STEPS
    next_deadline = time.monotonic()
    for step in range(1, ARM_INTERPOLATION_STEPS + 1):
        ratio = step / ARM_INTERPOLATION_STEPS
        interpolated_target = start + ratio * (target - start)
        response = arm.send_move_joints(interpolated_target.tolist())
        _check_franka_response(response, f"Franka move_joints step {step}/{ARM_INTERPOLATION_STEPS}")
        next_deadline += period
        remaining = next_deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    deadline = time.monotonic() + timeout_seconds
    settled = 0
    while True:
        current = arm.get_joint_positions()
        error = float(np.max(np.abs(current - target)))
        print(f"\r[init] Franka max error={error:.4f} rad", end="", flush=True)
        if error <= tolerance:
            settled += 1
            if settled >= settle_samples:
                print("\n[init] Franka ready pose reached", flush=True)
                return
        else:
            settled = 0
        if time.monotonic() >= deadline:
            print(flush=True)
            raise TimeoutError(
                f"Franka did not settle within {timeout_seconds:g}s; "
                f"max error={error:.4f} rad"
            )
        time.sleep(poll_seconds)


def move_hand_to_ready(
    hand: SharpaWaveController,
    steps: int,
    duration: float,
    tolerance: float,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    start = np.asarray(hand.get_state(), dtype=np.float64)
    if start.shape != HAND_READY_JOINTS.shape or not np.isfinite(start).all():
        raise ValueError("Expected 22 finite SharpA joint positions")
    print(
        f"[init] moving SharpA to HAND_READY_JOINTS: {steps} steps / {duration:g}s",
        flush=True,
    )
    period = duration / steps
    next_deadline = time.monotonic()
    for step in range(1, steps + 1):
        ratio = step / steps
        hand.set_action(start + ratio * (HAND_READY_JOINTS - start), interpolate=False)
        next_deadline += period
        remaining = next_deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    hand.set_action(HAND_READY_JOINTS, interpolate=False)

    deadline = time.monotonic() + timeout_seconds
    while True:
        current = np.asarray(hand.get_state(), dtype=np.float64)
        error = float(np.max(np.abs(current - HAND_READY_JOINTS)))
        print(f"\r[init] SharpA max error={error:.4f} rad", end="", flush=True)
        if error <= tolerance:
            print("\n[init] SharpA HAND_READY_JOINTS reached", flush=True)
            return
        if time.monotonic() >= deadline:
            print(flush=True)
            raise TimeoutError(
                f"SharpA did not settle within {timeout_seconds:g}s; "
                f"max error={error:.4f} rad"
            )
        time.sleep(poll_seconds)


def main() -> int:
    args = parse_args()
    ready_joints = _validate_args(args)
    arm = None
    hand = None
    arm_motion_started = False
    hand_motion_started = False
    try:
        if not args.skip_franka:
            arm = FrankaArmController(
                args.franka_host,
                args.franka_port,
                args.franka_timeout_ms,
            )
        if not args.skip_hand:
            hand = SharpaWaveController(
                args.hand_host,
                args.hand_port,
                args.hand_timeout_ms,
            )

        # Read every requested device before moving either one.
        if arm is not None:
            arm_state = arm.get_joint_positions()
            print(
                f"[preflight] Franka state OK: {np.round(arm_state, 4).tolist()}",
                flush=True,
            )
        if hand is not None:
            hand_state = hand.get_state()
            print(
                f"[preflight] SharpA state OK: {np.round(hand_state, 4).tolist()}",
                flush=True,
            )
        if args.check_only:
            print("[preflight] protocol checks passed; no motion sent", flush=True)
            return 0

        # Match inference_dp_dino.sh: settle the hand before moving the arm.
        if hand is not None and not args.no_move_hand:
            hand_motion_started = True
            move_hand_to_ready(
                hand,
                args.hand_steps,
                args.hand_seconds,
                args.hand_tolerance,
                args.hand_timeout_seconds,
                args.poll_seconds,
            )
        if arm is not None and not args.no_move_franka:
            arm_motion_started = True
            move_franka_to_ready(
                arm,
                ready_joints,
                args.arm_tolerance,
                args.arm_settle_samples,
                args.arm_timeout_seconds,
                args.poll_seconds,
            )
        print("[init] requested robot initialization completed", flush=True)
        return 0
    except BaseException:
        if hand is not None and hand_motion_started:
            try:
                measured_hand = hand.get_state()
                hand.set_action(measured_hand, interpolate=False)
                print("[cleanup] SharpA held at measured position", flush=True)
            except Exception as exc:
                print(f"[cleanup] SharpA hold failed: {exc}", flush=True)
        if arm is not None and arm_motion_started:
            try:
                hold_current_arm_position(arm)
                print("[cleanup] Franka held at measured position", flush=True)
            except Exception as exc:
                print(f"[cleanup] Franka hold/stop failed: {exc}", flush=True)
        raise
    finally:
        if hand is not None:
            hand.close()
        if arm is not None:
            arm.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[init] stopped by Ctrl-C", flush=True)
        raise SystemExit(130)
