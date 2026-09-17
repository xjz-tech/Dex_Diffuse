#!/usr/bin/env python3
"""Run Real-DP proposals through a guided Sim-Hand DDIM controller.

The image Diffusion Policy supplies the task-level 31-D action chunk.  Its
22-D hand part is used as an OpenAI-style score guide during every reverse
DDIM step of the simulation-trained hand policy.  The 9-D arm part is kept
unchanged.  One DP proposal guides multiple controller calls.  After each
short execution chunk, the controller uses fresh state and the next window of
the same DP proposal.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import dill
import hydra
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from omegaconf import OmegaConf

from inference_timing import InferenceTimer
from checkpoint_loader import build_policy as build_controller_policy
from checkpoint_loader import load_checkpoint as load_controller_checkpoint
from diffusion_policy.guidance.guided_ddim import sample_guided_trajectory
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from policy_observation import (
    QPOS_OBSERVATION,
    QPOS_TARGET_RESIDUAL_OBSERVATION,
    compose_policy_observation,
)


ARM_DIM = 9
HAND_DIM = 22
ACTION_DIM = ARM_DIM + HAND_DIM


@dataclass(frozen=True)
class GuidanceStats:
    mse_before: float
    mse_after: float
    mse_before_batch: np.ndarray
    mse_after_batch: np.ndarray


class GuidedDDIMController:
    """Simulation hand prior using switch-branch OpenAI-style DDIM guidance."""

    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        *,
        inference_steps: int,
        execution_steps: int,
        guidance_scale: float,
        eta: float,
        fixed_noise: bool,
        seed: int,
        allow_salvage: bool,
    ) -> None:
        loaded = load_controller_checkpoint(checkpoint, allow_salvage=allow_salvage)
        self.checkpoint_info = loaded
        policy, spec = build_controller_policy(loaded)
        policy = policy.to(device).eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)

        if inference_steps <= 0:
            raise ValueError("controller inference_steps must be positive")
        # DDIM generates the full horizon. n_action_steps is the training-time
        # execution default, not the number of available future predictions.
        action_start = int(spec["n_obs_steps"]) - 1
        max_execution_steps = int(spec["horizon"]) - action_start
        if execution_steps <= 0 or execution_steps > max_execution_steps:
            raise ValueError(
                "execution_steps must be in [1, %d], got %d "
                "(controller horizon=%d, action_start=%d)"
                % (max_execution_steps, execution_steps, int(spec["horizon"]), action_start)
            )
        if guidance_scale < 0.0:
            raise ValueError("guidance_scale must be non-negative")
        if eta != 0.0:
            raise ValueError("eta must be 0.0")
        if policy.noise_scheduler.config.prediction_type != "epsilon":
            raise ValueError("guided DDIM requires an epsilon-prediction controller")
        if not bool(policy.obs_as_global_cond):
            raise ValueError("guided DDIM requires global observation conditioning")

        self.policy = policy
        self.spec = {key: int(value) for key, value in spec.items()}
        self.device = device
        self.inference_steps = int(inference_steps)
        self.execution_steps = int(execution_steps)
        self.max_execution_steps = max_execution_steps
        self.guidance_scale = float(guidance_scale)
        self.eta = float(eta)
        self.fixed_noise = bool(fixed_noise)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))
        self._fixed_noise: torch.Tensor | None = None
        if bool(getattr(policy.noise_scheduler.config, "thresholding", False)):
            raise ValueError("Dynamic thresholding is not supported")
        self.scheduler = DDIMScheduler.from_config(
            policy.noise_scheduler.config,
            set_alpha_to_one=True,
            steps_offset=0,
        )

        self.action_start = self.spec["n_obs_steps"] - 1
        self.reference_steps = self.spec["n_pred_action_steps"]
        self.reference_slice = slice(
            self.action_start,
            self.action_start + self.reference_steps,
        )
        if self.reference_slice.stop > self.spec["horizon"]:
            raise ValueError("controller reference slice exceeds diffusion horizon")

        self.observation_mode = (
            QPOS_OBSERVATION
            if self.spec["obs_dim"] == HAND_DIM
            else QPOS_TARGET_RESIDUAL_OBSERVATION
        )
        print(
            "[controller] loaded %s | weights=%s obs_dim=%d horizon=%d "
            "guide_steps=%d execute_steps=%d max_execute_steps=%d sampler=DDIM ddim_steps=%d"
            % (
                checkpoint,
                loaded.weight_source,
                self.spec["obs_dim"],
                self.spec["horizon"],
                self.reference_steps,
                self.execution_steps,
                self.max_execution_steps,
                self.inference_steps,
            ),
            flush=True,
        )
        if loaded.salvaged:
            print(
                "[controller] WARNING: incomplete checkpoint; using strictly "
                "recovered base-model weights",
                flush=True,
            )

    def compose_observation(
        self,
        qpos: np.ndarray,
        target_before: np.ndarray,
    ) -> np.ndarray:
        return compose_policy_observation(qpos, target_before, self.observation_mode)

    def _noise(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        shape = (batch_size, self.spec["horizon"], HAND_DIM)
        if self.fixed_noise:
            if self._fixed_noise is None or tuple(self._fixed_noise.shape) != shape:
                self._fixed_noise = torch.randn(
                    shape,
                    device=self.device,
                    dtype=dtype,
                    generator=self.generator,
                )
            return self._fixed_noise.clone()
        return torch.randn(
            shape,
            device=self.device,
            dtype=dtype,
            generator=self.generator,
        )

    def reset_fixed_noise(self, seed: int | None = None) -> None:
        """Regenerate the shared DDIM noise, optionally from a new seed."""
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self._fixed_noise = None

    def set_fixed_noise_from_seeds(self, seeds: Sequence[int]) -> None:
        """Build one fixed-noise batch so every sample keeps its own seed."""
        if not seeds:
            raise ValueError("seeds must be non-empty")
        noises = []
        for seed in seeds:
            self.generator.manual_seed(int(seed))
            noises.append(
                torch.randn(
                    (1, self.spec["horizon"], HAND_DIM),
                    device=self.device,
                    dtype=self.policy.dtype,
                    generator=self.generator,
                )
            )
        self._fixed_noise = torch.cat(noises, dim=0)

    def set_guidance_horizon(self, steps: int) -> None:
        """Guide a prefix of the diffusion horizon; later steps stay unguided."""
        steps = int(steps)
        stop = self.action_start + steps
        if steps <= 0 or stop > self.spec["horizon"]:
            raise ValueError(
                "guidance horizon must be in [1, %d], got %d"
                % (self.spec["horizon"] - self.action_start, steps)
            )
        self.reference_steps = steps
        self.reference_slice = slice(self.action_start, stop)

    def _predict_epsilon(
        self,
        sample: torch.Tensor,
        timestep: int | torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        return self.policy.model(
            sample,
            timestep,
            local_cond=None,
            global_cond=global_cond,
        )

    def predict(
        self,
        observation_history: np.ndarray,
        hand_reference: np.ndarray,
    ) -> tuple[np.ndarray, GuidanceStats]:
        history = torch.as_tensor(
            observation_history,
            device=self.device,
            dtype=self.policy.dtype,
        )
        if history.ndim == 2:
            history = history.unsqueeze(0)
        expected_history = (
            history.shape[0],
            self.spec["n_obs_steps"],
            self.spec["obs_dim"],
        )
        if tuple(history.shape) != expected_history:
            raise ValueError(
                "controller observation history must have shape %s, got %s"
                % (expected_history, tuple(history.shape))
            )

        reference = torch.as_tensor(
            hand_reference,
            device=self.device,
            dtype=self.policy.dtype,
        )
        if reference.ndim == 2:
            reference = reference.unsqueeze(0)
        if reference.ndim != 3 or reference.shape[0] != history.shape[0]:
            raise ValueError("hand reference must have shape (batch, steps, 22)")
        if reference.shape[-1] != HAND_DIM:
            raise ValueError("hand reference last dimension must be 22")
        if reference.shape[1] < self.reference_steps:
            raise ValueError(
                "DP hand chunk has %d steps, but controller guidance needs %d"
                % (reference.shape[1], self.reference_steps)
            )
        if not torch.isfinite(history).all() or not torch.isfinite(reference).all():
            raise ValueError("controller history/reference contains NaN or Inf")

        nobs = self.policy.normalizer["obs"].normalize(history)
        global_cond = nobs[:, : self.spec["n_obs_steps"]].reshape(history.shape[0], -1)
        reference = self.policy.normalizer["action"].normalize(
            reference[:, : self.reference_steps]
        )
        trajectory = self._noise(history.shape[0], history.dtype)
        extra_kwargs = {}
        if getattr(self, "weak_guide", None) is not None:
            extra_kwargs["extra_guidance"] = self.weak_guide(history)
        sample = sample_guided_trajectory(
            model=self._predict_epsilon,
            scheduler=self.scheduler,
            initial_noise=trajectory,
            global_cond=global_cond,
            reference=reference,
            num_inference_steps=self.inference_steps,
            guidance_scale=self.guidance_scale,
            guidance_slice=self.reference_slice,
            eta=self.eta,
            **extra_kwargs,
        )
        if not sample.steps:
            raise RuntimeError("guided DDIM produced no reverse steps")

        action_norm = sample.trajectory[
            :,
            self.action_start : self.action_start + self.execution_steps,
            :,
        ]
        action = self.policy.normalizer["action"].unnormalize(action_norm)
        if not torch.isfinite(action).all():
            raise RuntimeError("guided DDIM controller produced NaN or Inf")
        last_step = sample.steps[-1]
        mse_before_batch = (
            last_step.guidance_loss_before.detach().reshape(-1).cpu().numpy()
        )
        mse_after_batch = (
            last_step.guidance_loss_after.detach().reshape(-1).cpu().numpy()
        )
        stats = GuidanceStats(
            mse_before=float(np.mean(mse_before_batch)),
            mse_after=float(np.mean(mse_after_batch)),
            mse_before_batch=np.asarray(mse_before_batch, dtype=np.float64),
            mse_after_batch=np.asarray(mse_after_batch, dtype=np.float64),
        )
        return action.detach().cpu().numpy(), stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp-checkpoint", required=True, type=Path)
    parser.add_argument("--controller-checkpoint", required=True, type=Path)
    parser.add_argument("--weak-guide-checkpoint", type=Path, default=None)
    parser.add_argument("--weak-guide-scale", type=float, default=25.0)
    parser.add_argument("--weak-guide-steps", type=int, default=2)
    parser.add_argument("--weak-guide-inference-steps", type=int, default=4)
    parser.add_argument("--weak-guide-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dp-inference-steps", type=int, default=16)
    parser.add_argument("--ddim-inference-steps", type=int, default=4)
    parser.add_argument(
        "--controller-action-chunk-size", "--execution-steps",
        dest="execution_steps", type=int, default=2,
        help="Actions executed per call, up to horizon - n_obs_steps + 1 (default: 2)",
    )
    parser.add_argument(
        "--guidance-steps",
        type=int,
        default=9,
        help="DP hand steps used as DDIM guidance (default: 9)",
    )
    parser.add_argument(
        "--controller-calls-per-dp", type=int, default=4,
        help="Controller calls reusing one DP proposal (default: 4)",
    )
    parser.add_argument("--guidance-scale", type=float, default=100.0)
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta (must be 0.0)")
    parser.add_argument("--fixed-noise", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-salvage", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Load models and validate execution/guidance lengths without hardware or sampling",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=0,
        help="Maximum executed controller chunks, not DP proposals (0: unlimited)",
    )
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--live", action="store_true")

    parser.add_argument("--franka-host", default="172.16.0.10")
    parser.add_argument("--franka-port", type=int, default=9090)
    parser.add_argument("--franka-timeout-ms", type=int, default=2000)
    parser.add_argument(
        "--franka-control-mode", choices=("joints", "cartesian"), default="joints"
    )
    parser.add_argument("--franka-urdf", default=None)
    parser.add_argument("--max-joint-step", type=float, default=0.05)
    parser.add_argument("--ik-dq-max", type=float, default=0.5)
    parser.add_argument("--ik-damping", type=float, default=1e-4)
    parser.add_argument("--max-arm-xyz-step", type=float, default=0.03)
    parser.add_argument("--max-hand-step", type=float, default=0.03)
    parser.add_argument("--disable-clamp", action="store_true")
    parser.add_argument("--hand-host", default="localhost")
    parser.add_argument("--hand-port", type=int, default=5570)
    parser.add_argument("--hand-timeout-ms", type=int, default=2000)
    parser.add_argument("--hand-interpolate", action="store_true")
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-timeout-ms", type=int, default=1000)
    parser.add_argument("--front-serial", default=None)
    parser.add_argument("--wrist-serial", default=None)
    parser.add_argument("--front-resolution", default="640x480")
    parser.add_argument("--wrist-resolution", default="640x480")
    parser.add_argument("--rotate-wrist-camera-180", action="store_true", default=True)
    parser.add_argument(
        "--no-rotate-wrist-camera-180",
        dest="rotate_wrist_camera_180",
        action="store_false",
    )
    parser.add_argument("--show-camera-input", action="store_true")
    parser.add_argument("--stop-on-close", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> torch.device:
    for name in ("dp_checkpoint", "controller_checkpoint"):
        path = Path(getattr(args, name)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("%s not found: %s" % (name, path))
        setattr(args, name, path)
    if args.weak_guide_checkpoint is not None:
        args.weak_guide_checkpoint = args.weak_guide_checkpoint.expanduser().resolve()
        if not args.weak_guide_checkpoint.is_file():
            raise FileNotFoundError(f"weak guide checkpoint not found: {args.weak_guide_checkpoint}")
        if not np.isfinite(args.weak_guide_scale) or args.weak_guide_scale < 0:
            raise ValueError("weak guide scale must be finite and non-negative")
        if not 1 <= args.weak_guide_steps <= 9 or args.weak_guide_inference_steps <= 0:
            raise ValueError("weak guide steps must be in 1..9 and inference steps positive")
    if args.dp_inference_steps <= 0 or args.ddim_inference_steps <= 0:
        raise ValueError("inference step counts must be positive")
    if args.guidance_scale < 0.0:
        raise ValueError("guidance_scale must be non-negative")
    if args.eta != 0.0:
        raise ValueError("eta must be 0.0")
    if args.execution_steps <= 0:
        raise ValueError("controller action chunk size must be positive")
    if args.guidance_steps <= 0:
        raise ValueError("guidance_steps must be positive")
    if args.controller_calls_per_dp <= 0:
        raise ValueError("controller_calls_per_dp must be positive")
    if args.max_chunks < 0:
        raise ValueError("max_chunks cannot be negative")
    if args.hz <= 0.0:
        raise ValueError("hz must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    return device


def _load_real_policy(
    checkpoint: Path,
    device: torch.device,
    inference_steps: int,
):
    payload = torch.load(
        checkpoint.open("rb"),
        pickle_module=dill,
        map_location="cpu",
    )
    if "cfg" not in payload:
        raise ValueError("real DP checkpoint has no embedded configuration")
    cfg = payload["cfg"]

    # Training checkpoints may retain absolute DINOv2 paths from another host.
    rgb_model_cfg = OmegaConf.select(cfg, "policy.obs_encoder.rgb_model")
    if rgb_model_cfg is not None:
        if os.environ.get("DINOV2_REPO_OR_DIR"):
            rgb_model_cfg.repo_or_dir = str(
                Path(os.environ["DINOV2_REPO_OR_DIR"]).expanduser().resolve()
            )
        if os.environ.get("DINOV2_WEIGHTS"):
            rgb_model_cfg.weights = str(
                Path(os.environ["DINOV2_WEIGHTS"]).expanduser().resolve()
            )
        if os.environ.get("DINOV2_SOURCE"):
            rgb_model_cfg.source = os.environ["DINOV2_SOURCE"]

    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if bool(cfg.training.use_ema) and getattr(workspace, "ema_model", None) is not None:
        policy = workspace.ema_model
    policy.num_inference_steps = int(inference_steps)
    policy = policy.to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return cfg, policy


def _real_policy_metadata(cfg, policy) -> tuple[int, int, bool, dict[str, tuple[int, ...]]]:
    n_obs_steps = int(policy.n_obs_steps)
    action_steps = int(policy.horizon) - n_obs_steps + 1
    if action_steps <= 0:
        raise ValueError("real DP has no usable action steps")
    if int(policy.action_dim) != ACTION_DIM:
        raise ValueError(
            "real DP action_dim must be %d, got %d" % (ACTION_DIM, policy.action_dim)
        )
    policy.n_action_steps = action_steps
    relative_ee = bool(OmegaConf.select(cfg, "task.dataset.relative", default=True))
    rgb_shapes = {
        key: tuple(int(value) for value in cfg.shape_meta.obs[key].shape)
        for key in ("front_image", "wrist_image")
    }
    return n_obs_steps, action_steps, relative_ee, rgb_shapes


def _controller_windows(
    action_steps: int,
    execution_steps: int,
    controller_calls_per_dp: int,
    reference_steps: int,
) -> tuple[tuple[slice, slice], ...]:
    """Return DP execution/guidance slices for each controller call.

    Every call needs the full guidance horizon, including the unexecuted tail.
    Never pad a short reference or silently reduce the requested call count.
    """
    if min(action_steps, execution_steps, controller_calls_per_dp, reference_steps) <= 0:
        raise ValueError("DP/controller lengths and controller call count must be positive")
    execution_total = controller_calls_per_dp * execution_steps
    if execution_total > action_steps:
        raise ValueError(
            "controller_calls_per_dp * controller_action_chunk_size = %d * %d = %d "
            "exceeds real DP action chunk size %d"
            % (controller_calls_per_dp, execution_steps, execution_total, action_steps)
        )
    required_steps = (controller_calls_per_dp - 1) * execution_steps + reference_steps
    if required_steps > action_steps:
        raise ValueError(
            "last controller call needs DP guidance [%d:%d], but DP only provides %d "
            "steps; require (controller_calls_per_dp - 1) * "
            "controller_action_chunk_size + guide_steps <= DP action chunk size "
            "(calls=%d, chunk_size=%d, guide_steps=%d, required=%d, available=%d). "
            "Parameters were not adjusted."
            % (required_steps - reference_steps, required_steps, action_steps,
               controller_calls_per_dp, execution_steps, reference_steps,
               required_steps, action_steps)
        )
    return tuple(
        (slice(start, start + execution_steps), slice(start, start + reference_steps))
        for start in range(0, execution_total, execution_steps)
    )


def _synthetic_real_observation(
    n_obs_steps: int,
    rgb_shapes: dict[str, tuple[int, ...]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "front_image": torch.zeros(1, n_obs_steps, *rgb_shapes["front_image"], device=device),
        "wrist_image": torch.zeros(1, n_obs_steps, *rgb_shapes["wrist_image"], device=device),
        "ee_pose": torch.zeros(1, n_obs_steps, ARM_DIM, device=device),
        "hand_joint": torch.zeros(1, n_obs_steps, HAND_DIM, device=device),
    }


def _run_check(
    real_policy,
    controller: GuidedDDIMController,
    n_obs_steps: int,
    action_steps: int,
    rgb_shapes: dict[str, tuple[int, ...]],
    device: torch.device,
    controller_windows: tuple[tuple[slice, slice], ...],
) -> None:
    synthetic_obs = _synthetic_real_observation(n_obs_steps, rgb_shapes, device)
    with InferenceTimer(device, "dp_check"), torch.inference_mode():
        proposal = real_policy.predict_action(synthetic_obs)["action"]
    expected = (1, action_steps, ACTION_DIM)
    if tuple(proposal.shape) != expected:
        raise RuntimeError(
            "real DP returned shape %s, expected %s" % (tuple(proposal.shape), expected)
        )
    observation_mean = (
        controller.policy.normalizer.get_input_stats()["obs"]["mean"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    history = np.repeat(
        observation_mean[None, :], controller.spec["n_obs_steps"], axis=0
    )
    hand_reference = proposal[0, :, ARM_DIM:].detach().cpu().numpy()
    expected_guided = (1, controller.execution_steps, HAND_DIM)
    # Synthetic shape/inference check only: no robot feedback is available here.
    for controller_idx, (execute_slice, guide_slice) in enumerate(controller_windows):
        hand_guide = hand_reference[guide_slice]
        with InferenceTimer(device, "controller_check"):
            guided, stats = controller.predict(history, hand_guide)
        if guided.shape != expected_guided:
            raise RuntimeError(
                "controller returned shape %s, expected %s" % (guided.shape, expected_guided)
            )
        print(
            "[check] passed: controller=%d/%d DP %s guide=[%d:%d] %s "
            "execute=[%d:%d] %s | final_mse %.6f -> %.6f"
            % (
                controller_idx + 1, len(controller_windows), expected,
                guide_slice.start, guide_slice.stop, hand_guide.shape,
                execute_slice.start, execute_slice.stop, guided.shape,
                stats.mse_before, stats.mse_after,
            ),
            flush=True,
        )


def main() -> int:
    args = parse_args()
    device = _validate_args(args)

    print("[dp] loading %s" % args.dp_checkpoint, flush=True)
    real_cfg, real_policy = _load_real_policy(
        args.dp_checkpoint,
        device,
        args.dp_inference_steps,
    )
    n_obs_steps, action_steps, relative_ee, rgb_shapes = _real_policy_metadata(
        real_cfg, real_policy
    )
    controller = GuidedDDIMController(
        args.controller_checkpoint,
        device,
        inference_steps=args.ddim_inference_steps,
        execution_steps=args.execution_steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        fixed_noise=bool(args.fixed_noise),
        seed=args.seed,
        allow_salvage=not args.no_salvage,
    )
    controller.set_guidance_horizon(args.guidance_steps)
    if args.weak_guide_checkpoint is not None:
        from dp_weak_guide import WeakTrajectoryGuide
        controller.weak_guide = WeakTrajectoryGuide(controller, args)
    controller_windows = _controller_windows(
        action_steps, controller.execution_steps,
        args.controller_calls_per_dp, controller.reference_steps,
    )
    print(
        "[pipeline] Real DP (%d-step task proposal) -> %d x "
        "(guided Sim-Hand DDIM, %d-step guide -> execute %d -> observe) -> replan DP"
        % (
            action_steps, args.controller_calls_per_dp,
            controller.reference_steps, controller.execution_steps,
        ),
        flush=True,
    )
    print(
        "[guidance] scale=%.4f eta=%.4f ddim_steps=%d fixed_noise=%s"
        % (
            controller.guidance_scale,
            controller.eta,
            controller.inference_steps,
            controller.fixed_noise,
        ),
        flush=True,
    )

    if args.validate_only:
        print("[check] model and schedule validation passed; hardware was not connected", flush=True)
        return 0

    if args.check_only:
        _run_check(
            real_policy,
            controller,
            n_obs_steps,
            action_steps,
            rgb_shapes,
            device,
            controller_windows,
        )
        return 0

    # Hardware/camera dependencies are needed only after both models pass the
    # model-only checks above.
    import inference_dp as real_dp

    env = real_dp.DiffusionDirectRobotEnv(
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
        log_action_steps=True,
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
        first_raw_obs = env.reset()
        first_obs = real_dp.capture_obs(first_raw_obs, rgb_shapes)
        real_history = deque([first_obs] * n_obs_steps, maxlen=n_obs_steps)
        qpos = first_obs["hand_joint"]
        controller_obs = controller.compose_observation(qpos, qpos)
        controller_history = deque(
            [controller_obs] * controller.spec["n_obs_steps"],
            maxlen=controller.spec["n_obs_steps"],
        )

        dp_idx = 0
        while args.max_chunks == 0 or env.chunk_idx < args.max_chunks:
            policy_obs, base_ee_pose = real_dp.build_policy_obs(
                real_history,
                device,
                relative_ee,
            )
            with InferenceTimer(device, "dp"), torch.inference_mode():
                proposal = real_policy.predict_action(policy_obs)["action"][0]
            mixed_actions = proposal.detach().cpu().numpy()
            absolute_dp_actions = (
                real_dp.mixed_actions_to_absolute(mixed_actions, base_ee_pose)
                if relative_ee
                else mixed_actions.copy()
            )
            if absolute_dp_actions.shape != (action_steps, ACTION_DIM):
                raise RuntimeError(
                    "real DP returned shape %s, expected %s"
                    % (absolute_dp_actions.shape, (action_steps, ACTION_DIM))
                )
            if not np.isfinite(absolute_dp_actions).all():
                raise RuntimeError("real DP proposal contains NaN or Inf")

            for controller_idx, (execute_slice, guide_slice) in enumerate(controller_windows):
                if args.max_chunks and env.chunk_idx >= args.max_chunks:
                    break
                # Replan from feedback after the previous execution chunk, while
                # advancing BOTH the arm actions and hand guide in the cached DP.
                hand_reference = absolute_dp_actions[guide_slice, ARM_DIM:]
                with InferenceTimer(device, "controller") as timer:
                    guided_hand, stats = controller.predict(
                        np.stack(controller_history),
                        hand_reference,
                    )
                inference_seconds = timer.elapsed
                guided_actions = absolute_dp_actions[execute_slice].copy()
                expected_hand = (1, controller.execution_steps, HAND_DIM)
                if guided_hand.shape != expected_hand:
                    raise RuntimeError(
                        "controller returned shape %s, expected %s"
                        % (guided_hand.shape, expected_hand)
                    )
                guided_actions[:, ARM_DIM:] = guided_hand[0]
                if not np.isfinite(guided_actions).all():
                    raise RuntimeError("combined guided action contains NaN or Inf")

                print(
                    "[guided chunk %04d] dp=%04d controller=%d/%d proposal=%s "
                    "guide=[%d:%d] execute=[%d:%d] %s "
                    "mse=%.6f->%.6f controller_s=%.3f"
                    % (
                        env.chunk_idx, dp_idx, controller_idx + 1, len(controller_windows),
                        tuple(absolute_dp_actions.shape),
                        guide_slice.start, guide_slice.stop,
                        execute_slice.start, execute_slice.stop,
                        tuple(guided_actions.shape),
                        stats.mse_before, stats.mse_after, inference_seconds,
                    ),
                    flush=True,
                )

                next_deadline = time.monotonic()
                for step_idx, action in enumerate(guided_actions):
                    raw_obs = env.step_single(action, step_idx)
                    captured = real_dp.capture_obs(raw_obs, rgb_shapes)
                    real_history.append(captured)
                    actual_qpos = captured["hand_joint"]
                    target_before = np.asarray(env.previous_action[ARM_DIM:], dtype=np.float32)
                    controller_history.append(
                        controller.compose_observation(actual_qpos, target_before)
                    )
                    next_deadline += 1.0 / args.hz
                    remaining = next_deadline - time.monotonic()
                    if remaining > 0.0:
                        time.sleep(remaining)
                env.chunk_idx += 1
            dp_idx += 1
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
