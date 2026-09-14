import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from inference_dp_controller import GuidedDDIMController, parse_args  # noqa: E402


def test_guidance_cli_defaults_use_segmentwise_ddim_scale_100(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference_dp_controller.py",
            "--dp-checkpoint",
            "dummy_dp.ckpt",
            "--controller-checkpoint",
            "dummy_ctrl.ckpt",
        ],
    )

    args = parse_args()

    assert args.guidance_scale == 100.0
    assert args.ddim_inference_steps == 4
    assert args.execution_steps == 2
    assert args.guidance_steps == 9
    assert args.controller_calls_per_dp == 4
    assert args.eta == 0.0
    assert not hasattr(args, "guidance_clip")
    assert not hasattr(args, "smoothness_scale")


def test_controller_does_not_use_homemade_clip_or_smooth_guidance():
    assert not hasattr(GuidedDDIMController, "_energy_gradient")
    assert not hasattr(GuidedDDIMController, "_guided_step")


def test_controller_predict_uses_segmentwise_sample_guided_trajectory(monkeypatch):
    captured = {}
    fake_trajectory = torch.arange(1 * 12 * 22, dtype=torch.float32).reshape(1, 12, 22)
    fake_output = SimpleNamespace(
        trajectory=fake_trajectory,
        steps=(
            SimpleNamespace(
                guidance_loss_before=torch.tensor([0.50]),
                guidance_loss_after=torch.tensor([0.10]),
            ),
        ),
    )

    def fake_sample_guided_trajectory(**kwargs):
        captured.update(kwargs)
        return fake_output

    monkeypatch.setattr(
        "inference_dp_controller.sample_guided_trajectory",
        fake_sample_guided_trajectory,
    )

    controller = object.__new__(GuidedDDIMController)
    controller.device = torch.device("cpu")
    controller.policy = SimpleNamespace(
        dtype=torch.float32,
        normalizer={
            "obs": SimpleNamespace(normalize=lambda value: value),
            "action": SimpleNamespace(
                normalize=lambda value: value,
                unnormalize=lambda value: value,
            ),
        },
    )
    controller.spec = {
        "n_obs_steps": 4,
        "obs_dim": 22,
        "horizon": 12,
        "n_pred_action_steps": 9,
    }
    controller.action_start = 3
    controller.reference_steps = 9
    controller.reference_slice = slice(3, 12)
    controller.execution_steps = 5
    controller.inference_steps = 8
    controller.guidance_scale = 100.0
    controller.eta = 0.0
    controller.scheduler = object()
    controller._noise = lambda *_args, **_kwargs: torch.zeros(1, 12, 22)
    controller._predict_epsilon = object()

    history = np.zeros((4, 22), dtype=np.float32)
    reference = np.ones((9, 22), dtype=np.float32)
    action, stats = controller.predict(history, reference)

    assert captured["guidance_scale"] == 100.0
    assert captured["eta"] == 0.0
    assert captured["num_inference_steps"] == 8
    assert captured["guidance_slice"] == slice(3, 12)
    assert captured["model"] is controller._predict_epsilon
    np.testing.assert_allclose(action, fake_trajectory[:, 3:8, :].numpy())
    assert stats.mse_before == pytest.approx(0.50)
    assert stats.mse_after == pytest.approx(0.10)
    np.testing.assert_allclose(stats.mse_before_batch, [0.50])
    np.testing.assert_allclose(stats.mse_after_batch, [0.10])


def test_set_guidance_horizon_shortens_reference_slice():
    controller = object.__new__(GuidedDDIMController)
    controller.spec = {"horizon": 12, "n_obs_steps": 4, "n_pred_action_steps": 9}
    controller.action_start = 3
    controller.reference_steps = 9
    controller.reference_slice = slice(3, 12)
    controller.set_guidance_horizon(6)
    assert controller.reference_steps == 6
    assert controller.reference_slice == slice(3, 9)
