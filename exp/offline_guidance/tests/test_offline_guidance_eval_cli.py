import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_EXP_DIR = Path(__file__).resolve().parents[1]
_ROOT_DIR = _EXP_DIR.parents[1]
sys.path.insert(0, str(_EXP_DIR))
sys.path.insert(0, str(_ROOT_DIR / "eval"))

from inference_dp_controller import GuidedDDIMController  # noqa: E402
from offline_guidance_eval import (  # noqa: E402
    _collate_policy_obs,
    _iter_batches,
    _predict_controller,
    method_label,
    parse_args,
)
from offline_guidance_metrics import linear_mix  # noqa: E402


def test_offline_eval_cli_defaults_point_at_epoch0950_and_obs66(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["offline_guidance_eval.py"])
    args = parse_args()
    assert args.dp_checkpoint.name == "epoch=0950.ckpt"
    assert args.controller_checkpoint.name == "obs_4-66.ckpt"
    assert args.zarr_path.name == "bulb_tac_50_dp"
    assert args.stride == 20
    assert args.guidance_scales == (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    assert args.mix_weights == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert args.dp_inference_steps == 16
    assert args.ddim_inference_steps == 8
    assert args.execution_steps == 5
    assert args.comparison_lengths is None
    assert args.generation_windows is None
    assert args.fixed_noise == 1
    assert args.eta == 0.0
    assert args.batch_size == 512


def test_method_label_distinguishes_guided_ddim_from_linear_mix():
    assert method_label("dp", None) == "dp"
    assert method_label("guided", 0.0) == "guided_scale_0.0"
    assert method_label("guided", 1.0) == "guided_scale_1.0"
    assert method_label("linear_mix", 0.5) == "linear_mix_0.5"


def test_linear_mix_at_endpoints_recovers_sim_and_dp():
    sim = np.array([[0.0, 1.0]], dtype=np.float32)
    dp = np.array([[2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(linear_mix(sim, dp, 0.0), sim)
    np.testing.assert_allclose(linear_mix(sim, dp, 1.0), dp)


def test_controller_exposes_reset_fixed_noise_for_fair_scale_comparison():
    assert callable(getattr(GuidedDDIMController, "reset_fixed_noise"))
    assert callable(getattr(GuidedDDIMController, "set_fixed_noise_from_seeds"))
    assert callable(getattr(GuidedDDIMController, "set_guidance_horizon"))


def test_iter_batches_keeps_order_and_tail():
    assert list(_iter_batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_collate_policy_obs_stacks_sample_axis():
    obs_list = [
        {
            "front_image": np.zeros((2, 3, 4, 4), dtype=np.float32),
            "ee_pose": np.ones((2, 9), dtype=np.float32),
        },
        {
            "front_image": np.ones((2, 3, 4, 4), dtype=np.float32),
            "ee_pose": np.full((2, 9), 2.0, dtype=np.float32),
        },
    ]
    batched = _collate_policy_obs(obs_list, torch.device("cpu"))
    assert tuple(batched["front_image"].shape) == (2, 2, 3, 4, 4)
    assert tuple(batched["ee_pose"].shape) == (2, 2, 9)
    np.testing.assert_allclose(batched["ee_pose"][1, 0, 0].item(), 2.0)


def test_predict_controller_returns_one_row_per_batch_item(monkeypatch):
    from types import SimpleNamespace

    fake_trajectory = torch.arange(2 * 12 * 22, dtype=torch.float32).reshape(2, 12, 22)
    fake_output = SimpleNamespace(
        trajectory=fake_trajectory,
        steps=(
            SimpleNamespace(
                guidance_loss_before=torch.tensor([0.50, 0.25]),
                guidance_loss_after=torch.tensor([0.10, 0.05]),
            ),
        ),
    )
    monkeypatch.setattr(
        "inference_dp_controller.sample_guided_trajectory",
        lambda **_kwargs: fake_output,
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
    controller.guidance_scale = 1.0
    controller.eta = 0.0
    controller.scheduler = object()
    controller._noise = lambda batch_size, *_args, **_kwargs: torch.zeros(batch_size, 12, 22)
    controller._predict_epsilon = object()

    history = np.zeros((2, 4, 22), dtype=np.float32)
    reference = np.ones((2, 9, 22), dtype=np.float32)
    hand, mse_before, mse_after = _predict_controller(controller, history, reference, 1.0)
    assert hand.shape == (2, 5, 22)
    np.testing.assert_allclose(mse_before, [0.50, 0.25])
    np.testing.assert_allclose(mse_after, [0.10, 0.05])
