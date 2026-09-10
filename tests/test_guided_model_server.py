from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from guided_pair_policy import (  # noqa: E402
    bind_conditional_sample_seed,
    extract_strong_reference,
    validate_compatible_specs,
)
from xjz_eval_strong_prior import _parse_args  # noqa: E402


class TrackingStrongPolicy:
    def __init__(self):
        self.observation = None

    def predict_action(self, obs_dict):
        self.observation = obs_dict["obs"]
        action_pred = torch.arange(12 * 22, dtype=torch.float32).reshape(1, 12, 22)
        return {
            "action": action_pred[:, 3:8],
            "action_pred": action_pred,
        }


def test_extract_strong_reference_uses_all_nine_predicted_steps():
    policy = TrackingStrongPolicy()
    observation = np.zeros((1, 4, 66), dtype=np.float32)

    reference = extract_strong_reference(
        policy,
        observation,
        device=torch.device("cpu"),
        action_start=3,
        reference_steps=9,
    )

    assert policy.observation.shape == (1, 4, 66)
    expected = np.arange(12 * 22, dtype=np.float32).reshape(1, 12, 22)[:, 3:12]
    np.testing.assert_array_equal(reference, expected)


def test_extract_strong_reference_rejects_truncated_action_only_output():
    policy = SimpleNamespace(
        predict_action=lambda _obs: {"action": torch.zeros(1, 5, 22)}
    )
    with pytest.raises(RuntimeError, match="full action_pred"):
        extract_strong_reference(
            policy,
            np.zeros((1, 4, 66), dtype=np.float32),
            device=torch.device("cpu"),
            action_start=3,
            reference_steps=9,
        )


def test_validate_compatible_specs_rejects_different_observation_contracts():
    weak = {
        "obs_dim": 66,
        "action_dim": 22,
        "n_obs_steps": 4,
        "n_pred_action_steps": 9,
        "horizon": 12,
    }
    strong = dict(weak, obs_dim=22)

    with pytest.raises(ValueError, match="obs_dim"):
        validate_compatible_specs(weak, strong)


def test_xjz_eval_strong_prior_script_labels_roles_in_help():
    result = subprocess.run(
        [sys.executable, str(EVAL_DIR / "xjz_eval_strong_prior.py"), "--help"],
        cwd=EVAL_DIR,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "strong prior" in result.stdout
    assert "guide checkpoint" in result.stdout
    assert "--guide-seed" in result.stdout


def test_guide_seed_defaults_to_prior_seed_and_can_be_overridden(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xjz_eval_strong_prior.py",
            "--checkpoint",
            "weak.ckpt",
            "--guide-checkpoint",
            "strong.ckpt",
            "--socket",
            "policy.sock",
            "--seed",
            "42",
        ],
    )
    assert _parse_args().guide_seed == 42

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xjz_eval_strong_prior.py",
            "--checkpoint",
            "weak.ckpt",
            "--guide-checkpoint",
            "strong.ckpt",
            "--socket",
            "policy.sock",
            "--seed",
            "42",
            "--guide-seed",
            "7",
        ],
    )
    assert _parse_args().guide_seed == 7


def test_bind_conditional_sample_seed_keeps_fixed_noise_repeatable_and_seed_dependent():
    from guided_pair_policy import bind_conditional_sample_seed

    class FakePolicy:
        def conditional_sample(
            self, condition_data, condition_mask, global_cond, generator=None, **kwargs
        ):
            return torch.randn(
                condition_data.shape,
                device=condition_data.device,
                dtype=condition_data.dtype,
                generator=generator,
            )

    zeros = torch.zeros(1, 12, 22)
    mask = torch.zeros_like(zeros, dtype=torch.bool)
    cond = torch.zeros(1, 4)

    seeded_a = bind_conditional_sample_seed(
        FakePolicy(), seed=42, device=torch.device("cpu"), fixed_noise=True
    )
    seeded_b = bind_conditional_sample_seed(
        FakePolicy(), seed=42, device=torch.device("cpu"), fixed_noise=True
    )
    seeded_c = bind_conditional_sample_seed(
        FakePolicy(), seed=7, device=torch.device("cpu"), fixed_noise=True
    )

    first = seeded_a.conditional_sample(zeros, mask, cond)
    second = seeded_a.conditional_sample(zeros, mask, cond)
    other_same_seed = seeded_b.conditional_sample(zeros, mask, cond)
    different_seed = seeded_c.conditional_sample(zeros, mask, cond)

    assert torch.equal(first, second)
    assert torch.equal(first, other_same_seed)
    assert not torch.equal(first, different_seed)


def test_guidance_steps_can_be_selected_from_environment(monkeypatch):
    monkeypatch.setenv("GUIDANCE_STEPS", "2")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xjz_eval_strong_prior.py",
            "--checkpoint",
            "weak.ckpt",
            "--guide-checkpoint",
            "strong.ckpt",
            "--socket",
            "policy.sock",
        ],
    )

    args = _parse_args()

    assert args.guidance_steps == 2


def test_old_guided_server_name_remains_a_working_compatibility_entrypoint():
    result = subprocess.run(
        [sys.executable, str(EVAL_DIR / "guided_model_server.py"), "--help"],
        cwd=EVAL_DIR,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--guide-checkpoint" in result.stdout
