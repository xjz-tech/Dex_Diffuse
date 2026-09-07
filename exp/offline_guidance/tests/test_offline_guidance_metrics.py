import sys
from pathlib import Path

import numpy as np
import pytest

_EXP_DIR = Path(__file__).resolve().parents[1]
_ROOT_DIR = _EXP_DIR.parents[1]
sys.path.insert(0, str(_EXP_DIR))
sys.path.insert(0, str(_ROOT_DIR / "eval"))

from offline_guidance_metrics import (
    linear_mix,
    mae,
    max_abs_jump,
    mean_abs_jerk,
    mean_abs_velocity,
    out_of_range_fraction,
    recommend_guidance_scale,
    rmse,
)


def test_mae_and_rmse_are_zero_when_predictions_match_demo():
    pred = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)
    assert mae(pred, pred) == pytest.approx(0.0)
    assert rmse(pred, pred) == pytest.approx(0.0)


def test_mae_and_rmse_use_all_joints_and_timesteps():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    demo = np.zeros_like(pred)
    assert mae(pred, demo) == pytest.approx(2.5)
    assert rmse(pred, demo) == pytest.approx(np.sqrt(7.5))


def test_max_abs_jump_is_the_largest_adjacent_joint_step():
    chunk = np.array(
        [
            [0.0, 0.0],
            [0.1, -0.4],
            [0.2, -0.5],
        ],
        dtype=np.float32,
    )
    assert max_abs_jump(chunk) == pytest.approx(0.4)


def test_velocity_and_jerk_average_absolute_finite_differences():
    chunk = np.array([[0.0], [1.0], [3.0], [6.0]], dtype=np.float32)
    assert mean_abs_velocity(chunk) == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)
    assert mean_abs_jerk(chunk) == pytest.approx((1.0 + 1.0) / 2.0)


def test_single_step_chunk_has_zero_jump_velocity_and_jerk():
    chunk = np.array([[0.2, -0.1]], dtype=np.float32)
    assert max_abs_jump(chunk) == pytest.approx(0.0)
    assert mean_abs_velocity(chunk) == pytest.approx(0.0)
    assert mean_abs_jerk(chunk) == pytest.approx(0.0)


def test_out_of_range_fraction_counts_joints_outside_inclusive_bounds():
    chunk = np.array([[0.0, 1.5], [-2.0, 0.5]], dtype=np.float32)
    lo = np.array([-1.0, 0.0], dtype=np.float32)
    hi = np.array([1.0, 1.0], dtype=np.float32)
    assert out_of_range_fraction(chunk, lo, hi) == pytest.approx(0.5)


def test_linear_mix_interpolates_sim_and_dp_hand_chunks():
    sim = np.zeros((2, 3), dtype=np.float32)
    dp = np.ones((2, 3), dtype=np.float32)
    mixed = linear_mix(sim, dp, 0.25)
    np.testing.assert_allclose(mixed, np.full((2, 3), 0.25, dtype=np.float32))


def test_recommend_guidance_scale_picks_lowest_demo_mae_among_guided_rows():
    rows = [
        {"method": "guided", "scale": 0.0, "demo_mae": 0.40},
        {"method": "guided", "scale": 1.0, "demo_mae": 0.22},
        {"method": "guided", "scale": 4.0, "demo_mae": 0.30},
        {"method": "linear_mix", "scale": 0.5, "demo_mae": 0.10},
    ]
    assert recommend_guidance_scale(rows) == pytest.approx(1.0)
