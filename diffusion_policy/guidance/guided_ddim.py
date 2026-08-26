from __future__ import annotations

import diffusers
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

EXPECTED_DIFFUSERS_VERSION = "0.11.1"


def assert_pinned_diffusers_version() -> None:
    if diffusers.__version__ != EXPECTED_DIFFUSERS_VERSION:
        raise RuntimeError(
            f"Guided DDIM requires diffusers=={EXPECTED_DIFFUSERS_VERSION}, "
            f"got {diffusers.__version__}"
        )


def create_ddim_scheduler(training_scheduler: DDPMScheduler) -> DDIMScheduler:
    assert_pinned_diffusers_version()
    config = training_scheduler.config
    if config.prediction_type != "epsilon":
        raise ValueError("Sim DDPM prediction_type must be epsilon")
    if bool(getattr(config, "thresholding", False)):
        raise ValueError("Dynamic thresholding is not supported")
    return DDIMScheduler.from_config(
        config,
        set_alpha_to_one=True,
        steps_offset=0,
    )


def mse_guidance_gradient(
    x0_base: torch.Tensor,
    reference: torch.Tensor,
    guidance_slice: slice,
) -> torch.Tensor:
    if x0_base.shape[0] != reference.shape[0]:
        raise ValueError(
            f"batch mismatch: x0_base batch {x0_base.shape[0]} != "
            f"reference batch {reference.shape[0]}"
        )
    if x0_base.shape[2] != reference.shape[2]:
        raise ValueError(
            f"action dim mismatch: x0_base dim {x0_base.shape[2]} != "
            f"reference dim {reference.shape[2]}"
        )
    sliced = x0_base[:, guidance_slice]
    if sliced.shape != reference.shape:
        raise ValueError(
            f"slice length mismatch: x0_base[{guidance_slice}] shape "
            f"{tuple(sliced.shape)} != reference shape {tuple(reference.shape)}"
        )
    if x0_base.dtype != reference.dtype:
        raise ValueError(
            f"dtype mismatch: x0_base {x0_base.dtype} != reference {reference.dtype}"
        )
    if x0_base.device != reference.device:
        raise ValueError(
            f"device mismatch: x0_base {x0_base.device} != reference {reference.device}"
        )
    if not (torch.isfinite(x0_base).all() and torch.isfinite(reference).all()):
        raise ValueError("non-finite inputs are not supported")

    gradient = torch.zeros_like(x0_base)
    scale = 2.0 / (reference.shape[1] * reference.shape[2])
    gradient[:, guidance_slice] = scale * (x0_base[:, guidance_slice] - reference)
    return gradient
