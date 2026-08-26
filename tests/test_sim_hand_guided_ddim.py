from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import diffusers
import pytest
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.guidance.guided_ddim import (
    EXPECTED_DIFFUSERS_VERSION,
    assert_pinned_diffusers_version,
    create_ddim_scheduler,
    mse_guidance_gradient,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDED_DTIM_PATH = REPO_ROOT / "diffusion_policy" / "guidance" / "guided_ddim.py"

EXPECTED_CURRENT_TIMESTEPS = (88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0)


def _training_scheduler(*, num_train_timesteps: int, prediction_type: str = "epsilon") -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type=prediction_type,
    )


def _assert_dynamic_scheduler(
    training: DDPMScheduler,
    *,
    inference_steps: int,
    expected_timesteps: tuple[int, ...] | None = None,
) -> None:
    ddim = create_ddim_scheduler(training)
    ddim.set_timesteps(inference_steps)
    torch.testing.assert_close(
        ddim.alphas_cumprod,
        training.alphas_cumprod,
        rtol=0.0,
        atol=0.0,
    )
    assert len(ddim.timesteps) == inference_steps
    assert int(ddim.timesteps.max()) < training.config.num_train_timesteps
    if expected_timesteps is not None:
        assert tuple(map(int, ddim.timesteps)) == expected_timesteps


def test_version_assert_pinned_diffusers_version_matches_expected_constant() -> None:
    assert diffusers.__version__ == EXPECTED_DIFFUSERS_VERSION
    assert_pinned_diffusers_version()


def test_version_rejects_wrong_diffusers_version() -> None:
    training = _training_scheduler(num_train_timesteps=100)
    with patch("diffusion_policy.guidance.guided_ddim.diffusers.__version__", "0.12.0"):
        with pytest.raises(RuntimeError, match="0.11.1"):
            assert_pinned_diffusers_version()
        with pytest.raises(RuntimeError, match="0.11.1"):
            create_ddim_scheduler(training)


def test_factory_copies_alphas_cumprod_for_100_step_training() -> None:
    _assert_dynamic_scheduler(
        _training_scheduler(num_train_timesteps=100),
        inference_steps=12,
        expected_timesteps=EXPECTED_CURRENT_TIMESTEPS,
    )


def test_factory_copies_alphas_cumprod_for_60_step_training() -> None:
    _assert_dynamic_scheduler(
        _training_scheduler(num_train_timesteps=60),
        inference_steps=10,
    )


def test_factory_accepts_non_100_train_step_count() -> None:
    training = _training_scheduler(num_train_timesteps=60)
    ddim = create_ddim_scheduler(training)
    assert ddim.config.num_train_timesteps == 60


def test_factory_rejects_non_epsilon_prediction() -> None:
    training = _training_scheduler(
        num_train_timesteps=100,
        prediction_type="sample",
    )
    with pytest.raises(ValueError, match="epsilon"):
        create_ddim_scheduler(training)


def test_factory_rejects_thresholding() -> None:
    training = _training_scheduler(num_train_timesteps=100)
    training.config.thresholding = True
    with pytest.raises(ValueError, match="thresholding"):
        create_ddim_scheduler(training)


def test_factory_production_has_no_hardcoded_timesteps() -> None:
    source = GUIDED_DTIM_PATH.read_text()
    assert "EXPECTED_CURRENT_TIMESTEPS" not in source
    assert str(EXPECTED_CURRENT_TIMESTEPS) not in source


def _assert_closed_form_gradient(
    *,
    batch: int,
    horizon: int,
    action_dim: int,
    guidance_slice: slice,
) -> None:
    x0_base = torch.randn(batch, horizon, action_dim)
    reference = torch.randn(batch, guidance_slice.stop - guidance_slice.start, action_dim)
    gradient = mse_guidance_gradient(x0_base, reference, guidance_slice)

    expected = torch.zeros_like(x0_base)
    expected[:, guidance_slice] = (
        2.0 / (reference.shape[1] * reference.shape[2])
    ) * (x0_base[:, guidance_slice] - reference)
    torch.testing.assert_close(gradient, expected)


def test_gradient_matches_closed_form_for_2_12_22_slice_3_8() -> None:
    _assert_closed_form_gradient(
        batch=2,
        horizon=12,
        action_dim=22,
        guidance_slice=slice(3, 8),
    )


def test_gradient_matches_closed_form_for_2_8_22_slice_1_4() -> None:
    _assert_closed_form_gradient(
        batch=2,
        horizon=8,
        action_dim=22,
        guidance_slice=slice(1, 4),
    )


def test_gradient_rejects_mismatched_batch() -> None:
    x0_base = torch.randn(2, 12, 22)
    reference = torch.randn(1, 5, 22)
    with pytest.raises(ValueError, match="batch"):
        mse_guidance_gradient(x0_base, reference, slice(3, 8))


def test_gradient_rejects_mismatched_action_dim() -> None:
    x0_base = torch.randn(2, 12, 22)
    reference = torch.randn(2, 5, 21)
    with pytest.raises(ValueError, match="action dim"):
        mse_guidance_gradient(x0_base, reference, slice(3, 8))


def test_gradient_rejects_mismatched_slice_length() -> None:
    x0_base = torch.randn(2, 12, 22)
    reference = torch.randn(2, 4, 22)
    with pytest.raises(ValueError, match="slice"):
        mse_guidance_gradient(x0_base, reference, slice(3, 8))


def test_gradient_rejects_dtype_mismatch() -> None:
    x0_base = torch.randn(2, 12, 22, dtype=torch.float32)
    reference = torch.randn(2, 5, 22, dtype=torch.float64)
    with pytest.raises(ValueError, match="dtype"):
        mse_guidance_gradient(x0_base, reference, slice(3, 8))


def test_gradient_rejects_device_mismatch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for device mismatch test")
    x0_base = torch.randn(2, 12, 22, device="cpu")
    reference = torch.randn(2, 5, 22, device="cuda")
    with pytest.raises(ValueError, match="device"):
        mse_guidance_gradient(x0_base, reference, slice(3, 8))


def test_gradient_rejects_non_finite_inputs() -> None:
    x0_base = torch.randn(2, 12, 22)
    reference = torch.randn(2, 5, 22)
    x0_base[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        mse_guidance_gradient(x0_base, reference, slice(3, 8))

    x0_base = torch.randn(2, 12, 22)
    reference[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        mse_guidance_gradient(x0_base, reference, slice(3, 8))
