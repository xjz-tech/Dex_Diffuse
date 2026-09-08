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

from guided_model_server import (  # noqa: E402
    extract_strong_reference,
    validate_compatible_specs,
)


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


def test_guided_server_script_imports_from_eval_entrypoint():
    result = subprocess.run(
        [sys.executable, str(EVAL_DIR / "guided_model_server.py"), "--help"],
        cwd=EVAL_DIR,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--guide-checkpoint" in result.stdout
