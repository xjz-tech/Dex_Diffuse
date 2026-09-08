import sys
from pathlib import Path

import pytest


EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from policy_observation import (  # noqa: E402
    QPOS_OBSERVATION,
    QPOS_TARGET_RESIDUAL_OBSERVATION,
    expected_policy_spec,
    observation_mode_from_dim,
    validate_policy_spec,
)


def test_observation_mode_from_dim_maps_22_to_qpos():
    assert observation_mode_from_dim(22) == QPOS_OBSERVATION


def test_observation_mode_from_dim_maps_66_to_qpos_target_residual():
    assert observation_mode_from_dim(66) == QPOS_TARGET_RESIDUAL_OBSERVATION


def test_observation_mode_from_dim_rejects_unknown_width():
    with pytest.raises(ValueError, match="obs_dim"):
        observation_mode_from_dim(44)


def test_expected_policy_spec_keeps_shared_temporal_layout():
    spec = expected_policy_spec(66)
    assert spec == {
        "obs_dim": 66,
        "action_dim": 22,
        "n_obs_steps": 4,
        "n_pred_action_steps": 9,
        "n_action_steps": 5,
        "horizon": 12,
    }
    assert expected_policy_spec(22)["obs_dim"] == 22


def test_expected_policy_spec_allows_execution_length_up_to_pred_steps():
    assert expected_policy_spec(66, n_action_steps=1)["n_action_steps"] == 1
    assert expected_policy_spec(66, n_action_steps=9)["n_action_steps"] == 9
    with pytest.raises(ValueError, match="n_action_steps"):
        expected_policy_spec(66, n_action_steps=0)
    with pytest.raises(ValueError, match="n_action_steps"):
        expected_policy_spec(66, n_action_steps=10)


def test_validate_policy_spec_accepts_checkpoint_obs_dim():
    spec = expected_policy_spec(66)
    assert validate_policy_spec(spec) == spec


def test_validate_policy_spec_accepts_runtime_execution_length():
    spec = expected_policy_spec(66, n_action_steps=3)
    assert validate_policy_spec(spec) == spec


def test_validate_policy_spec_rejects_wrong_temporal_fields():
    spec = expected_policy_spec(22)
    spec["n_obs_steps"] = 8
    with pytest.raises(ValueError, match="spec mismatch"):
        validate_policy_spec(spec)
