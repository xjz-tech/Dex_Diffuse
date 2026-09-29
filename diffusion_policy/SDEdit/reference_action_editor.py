"""Checkpoint-backed controller for reference-initialized SDEdit.

The action history comes from observation ``target_before`` fields, while the
future comes from the recorded reference. This module owns normalization and
checkpoint loading; the DDIM edit itself lives in
``diffusion_policy.SDEdit.reference_edit``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
EVAL_DIR = PROJECT_ROOT / "eval"
for import_path in (PROJECT_ROOT, EVAL_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from diffusion_policy.SDEdit.reference_edit import (  # noqa: E402
    ddim_transition,
    sample_reference_edit,
    select_edit_timesteps,
)
from inference_dp_controller import GuidedDDIMController  # noqa: E402


class ReferenceActionEditor:
    """Initialize DDIM from a reference, then edit its future action window."""

    def __init__(
        self,
        checkpoint: str | Path,
        noise_ratio: float,
        steps: int = 4,
        execution_steps: int = 2,
        *,
        device: str | torch.device = "cuda:0",
        fixed_noise: bool = True,
        seed: int = 42,
        allow_salvage: bool = True,
    ) -> None:
        self.controller = GuidedDDIMController(
            Path(checkpoint), torch.device(device),
            inference_steps=steps,
            execution_steps=execution_steps,
            guidance_scale=0.0,
            eta=0.0,
            fixed_noise=fixed_noise,
            seed=seed,
            allow_salvage=allow_salvage,
        )
        self.policy = self.controller.policy
        self.spec = self.controller.spec
        self.steps = int(steps)
        self.execution_steps = int(execution_steps)
        self.noise_ratio = float(noise_ratio)
        self.history_steps = int(self.spec["n_obs_steps"]) - 1
        self.future_steps = int(self.spec["n_pred_action_steps"])
        if (
            self.spec["obs_dim"] != 66
            or self.spec["action_dim"] != 22
            or self.spec["horizon"] not in (8, 12)
            or self.spec["horizon"] != self.history_steps + self.future_steps
        ):
            raise ValueError("reference edit requires a supported 66-D Sim-Hand checkpoint")
        self.timesteps, self.actual_noise_ratio = select_edit_timesteps(
            self.controller.scheduler.alphas_cumprod, self.noise_ratio, self.steps
        )
        self.metadata = dict(
            algorithm="reference_initialized_ddim",
            requested_noise_ratio=self.noise_ratio,
            actual_noise_ratio=self.actual_noise_ratio,
            timesteps=list(self.timesteps),
            num_train_timesteps=len(self.controller.scheduler.alphas_cumprod),
            inference_steps=self.steps,
            execution_steps=self.execution_steps,
            future_reference_steps=self.future_steps,
            known_history_steps=self.history_steps,
            eta=0.0,
            guidance_scale=0.0,
            clip_sample=bool(self.controller.scheduler.config.clip_sample),
            history_source="target_before from observation frames 1:4; equals actually issued previous three commands",
            weight_source=self.controller.checkpoint_info.weight_source,
        )

    @torch.no_grad()
    def predict(
        self,
        history: np.ndarray,
        future: np.ndarray,
        seeds: list[int] | tuple[int, ...] | None,
        return_full_plan: bool = False,
    ) -> tuple[np.ndarray, dict]:
        history = np.asarray(history, dtype=np.float32)
        future = np.asarray(future, dtype=np.float32)
        expected_history = (len(history), self.spec["n_obs_steps"], 66)
        expected_future = (len(history), self.future_steps, 22)
        if history.shape != expected_history or future.shape != expected_future:
            raise ValueError(
                f"expected history {expected_history} and future {expected_future}; "
                f"got {history.shape} and {future.shape}"
            )
        if not np.isfinite(history).all() or not np.isfinite(future).all():
            raise ValueError("history and reference must be finite")
        if seeds is not None and len(seeds) != len(history):
            raise ValueError("one noise seed is required per batch item")
        output_steps = self.future_steps if return_full_plan else self.execution_steps
        # Preserve exact reference actions for the zero-edit baseline.
        if self.noise_ratio == 0:
            return future[:, :output_steps].copy(), dict(
                edit_rmse_rad=0.0,
                edit_max_abs_rad=0.0,
                history_mask_max_error=0.0,
                zero_edit_exact=True,
            )

        controller = self.controller
        device = controller.device
        h = torch.as_tensor(history, device=device, dtype=self.policy.dtype)
        known = h[:, 1:, 22:44]
        clean_rad = torch.cat(
            [known, torch.as_tensor(future, device=device, dtype=self.policy.dtype)],
            dim=1,
        )
        clean = self.policy.normalizer["action"].normalize(clean_rad)
        global_cond = self.policy.normalizer["obs"].normalize(h).reshape(len(h), -1)
        if seeds is not None:
            controller.set_fixed_noise_from_seeds(seeds)
        noise = controller._noise(len(h), h.dtype)
        sample = sample_reference_edit(
            model=controller._predict_epsilon,
            alphas_cumprod=controller.scheduler.alphas_cumprod,
            clean_reference=clean,
            noise=noise,
            global_cond=global_cond,
            timesteps=self.timesteps,
            known_history_steps=self.history_steps,
            clip_sample=bool(controller.scheduler.config.clip_sample),
        )
        initial_rad = self.policy.normalizer["action"].unnormalize(
            sample.initial_sample
        )[:, self.history_steps:]
        edited = self.policy.normalizer["action"].unnormalize(
            sample.trajectory
        )[:, self.history_steps:]
        delta = edited - clean_rad[:, self.history_steps:]
        stats = dict(
            edit_rmse_rad=float(delta.square().mean().sqrt()),
            edit_max_abs_rad=float(delta.abs().max()),
            executed_prefix_edit_rmse_rad=float(
                delta[:, :self.execution_steps].square().mean().sqrt()
            ),
            injected_rmse_rad=float(
                (initial_rad - clean_rad[:, self.history_steps:]).square().mean().sqrt()
            ),
            history_mask_max_error=sample.history_mask_max_error,
            predicted_x0_clip_fraction=sample.predicted_x0_clip_fraction,
        )
        return edited[:, :output_steps].cpu().numpy(), stats


__all__ = ["ReferenceActionEditor", "ddim_transition"]
