import re

import pytest

from diffusion_policy.common.sim_hand_temporal_util import (
    format_sim_hand_temporal_config,
    validate_sim_hand_temporal_config,
)


DEFAULTS = {
    "n_obs_steps": 4,
    "n_pred_action_steps": 9,
    "n_action_steps": 5,
    "horizon": 12,
    "obs_dim": 22,
    "action_dim": 22,
    "oa_step_convention": True,
}


def test_default_temporal_config_has_expected_action_slices():
    config = validate_sim_hand_temporal_config(**DEFAULTS)

    assert config.usable_action_slice == slice(3, 12)
    assert config.execution_action_slice == slice(3, 8)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"n_obs_steps": 0}, "Observation steps must be positive"),
        (
            {
                "n_pred_action_steps": 4,
                "n_action_steps": 5,
                "horizon": 7,
            },
            "Execution action steps must not exceed prediction action steps",
        ),
        (
            {"horizon": 16},
            "must equal n_obs_steps + n_pred_action_steps - 1",
        ),
        (
            {"n_pred_action_steps": 8, "horizon": 11},
            "horizon must be a multiple of 4",
        ),
        ({"obs_dim": 21}, "Observation dimension must be 22"),
        ({"action_dim": 21}, "Action dimension must be 22"),
        (
            {"oa_step_convention": False},
            "OA step convention must be enabled",
        ),
    ],
)
def test_invalid_temporal_config_fails_fast(overrides, message):
    values = dict(DEFAULTS)
    values.update(overrides)

    with pytest.raises(ValueError, match=re.escape(message)):
        validate_sim_hand_temporal_config(**values)


def test_invalid_temporal_error_reports_all_config_values():
    with pytest.raises(ValueError) as exc_info:
        validate_sim_hand_temporal_config(
            n_obs_steps=4,
            n_pred_action_steps=8,
            n_action_steps=3,
            horizon=11,
            obs_dim=22,
            action_dim=22,
        )

    message = str(exc_info.value)
    assert "Observation steps       : 4" in message
    assert "Prediction action steps : 8" in message
    assert "Execution action steps  : 3" in message
    assert "Derived horizon         : 11" in message
    assert "Observation dimension   : 22" in message
    assert "Action dimension        : 22" in message


def test_startup_report_is_derived_from_default_config():
    report = format_sim_hand_temporal_config(
        validate_sim_hand_temporal_config(**DEFAULTS)
    )

    assert "Observation steps       : 4" in report
    assert "Prediction action steps : 9" in report
    assert "Execution action steps  : 5" in report
    assert "Diffusion horizon       : 12" in report
    assert "Action dimension        : 22" in report
    assert "Observation dimension   : 22" in report
    assert "OA step convention      : True" in report
    assert "obs condition : s[t-3:t+1]" in report
    assert "usable actions: a[t:t+9]" in report
    assert "execute       : a[t:t+5]" in report


def test_startup_report_changes_with_temporal_config():
    report = format_sim_hand_temporal_config(
        validate_sim_hand_temporal_config(
            n_obs_steps=2,
            n_pred_action_steps=7,
            n_action_steps=3,
            horizon=8,
            obs_dim=22,
            action_dim=22,
        )
    )

    assert "Observation steps       : 2" in report
    assert "Prediction action steps : 7" in report
    assert "Execution action steps  : 3" in report
    assert "Diffusion horizon       : 8" in report
    assert "obs condition : s[t-1:t+1]" in report
    assert "usable actions: a[t:t+7]" in report
    assert "execute       : a[t:t+3]" in report
