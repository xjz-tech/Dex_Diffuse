import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from checkpoint_loader import configure_policy_execution_steps  # noqa: E402


def _spec(n_action_steps=5):
    return {
        "obs_dim": 66,
        "action_dim": 22,
        "n_obs_steps": 4,
        "n_pred_action_steps": 9,
        "n_action_steps": n_action_steps,
        "horizon": 12,
    }


def test_configure_policy_execution_steps_overrides_slice_length():
    policy = SimpleNamespace(n_action_steps=5)
    spec = _spec(5)

    updated = configure_policy_execution_steps(policy, spec, 1)

    assert policy.n_action_steps == 1
    assert updated["n_action_steps"] == 1
    assert updated["n_pred_action_steps"] == 9
    assert spec["n_action_steps"] == 1


def test_configure_policy_execution_steps_keeps_full_predicted_horizon():
    policy = SimpleNamespace(n_action_steps=5)
    spec = _spec(5)

    updated = configure_policy_execution_steps(policy, spec, 9)

    assert policy.n_action_steps == 9
    assert updated["n_action_steps"] == 9
    assert updated["horizon"] == 12


def test_configure_policy_execution_steps_rejects_beyond_predicted_window():
    policy = SimpleNamespace(n_action_steps=5)
    spec = _spec(5)

    with pytest.raises(ValueError, match="n_action_steps"):
        configure_policy_execution_steps(policy, spec, 10)
