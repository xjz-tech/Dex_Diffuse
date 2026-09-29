"""Online visual-DP reference routing for real SDEdit inference."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import inference_dp_controller as pipeline  # noqa: E402
from inference_dp_controller import _controller_windows, _replace_hand_actions  # noqa: E402
from direct_robot_env import (  # noqa: E402
    DirectRobotEnv, FrankaJointIkSolver, POLICY2REAL_DOF_INDICES,
)


def test_edit_window_preserves_dp_arm_and_replaces_only_hand():
    dp = np.arange(13 * 31, dtype=np.float32).reshape(13, 31)
    windows = _controller_windows(13, 2, 2, 9)
    assert windows == ((slice(0, 2), slice(0, 9)),
                       (slice(2, 4), slice(2, 11)))
    edited_hand = np.full((1, 2, 22), 0.25, dtype=np.float32)
    combined = _replace_hand_actions(dp, windows[1][0], edited_hand)
    np.testing.assert_array_equal(combined[:, :9], dp[2:4, :9])
    np.testing.assert_array_equal(combined[:, 9:], edited_hand[0])
    np.testing.assert_array_equal(dp, np.arange(13 * 31, dtype=np.float32).reshape(13, 31))


def test_edit_requires_full_future_window_from_visual_dp():
    with pytest.raises(ValueError, match="last controller call needs"):
        _controller_windows(10, 2, 2, 9)
    with pytest.raises(ValueError, match="does not match"):
        _replace_hand_actions(np.zeros((13, 31)), slice(0, 2), np.zeros((1, 1, 22)))


def test_edit_execution_sends_unclipped_hand_target():
    env = DirectRobotEnv.__new__(DirectRobotEnv)
    env.disable_clamp = True
    env.max_arm_xyz_step = 0.03
    env.max_hand_step = 0.03
    env.live = True
    env.franka_control_mode = "cartesian"
    env.franka = Mock()
    env.hand = Mock()
    env.hand_interpolate = False
    env.log_action_steps = False
    arm = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64)
    env.previous_action = np.concatenate((arm, np.zeros(22)))
    raw = np.concatenate((arm, np.full(22, 2.0)))

    executed = env._execute_action_step(raw, 0)

    np.testing.assert_array_equal(executed, raw)
    sent_in_real_order = env.hand.set_action.call_args.args[0]
    np.testing.assert_array_equal(sent_in_real_order, raw[9:][POLICY2REAL_DOF_INDICES])


def test_edit_ik_skips_software_joint_limits():
    solver = FrankaJointIkSolver.__new__(FrankaJointIkSolver)
    solver.sync_qpos_fn = None
    solver.ee_to_tcp_pose = np.eye(4)
    solver.q_current = np.zeros(7)
    solver.max_joint_step = 0.05
    solver.apply_joint_limits = False
    solver.model = SimpleNamespace(
        lowerPositionLimit=np.full(7, -1.0),
        upperPositionLimit=np.full(7, 1.0),
    )
    solver._ik_step = Mock(return_value=(np.full(7, 2.0), np.zeros(3), np.zeros(3)))

    result = solver.solve(np.eye(4))

    np.testing.assert_array_equal(result.joint_positions, np.full(7, 2.0))


def test_edit_check_only_uses_online_dp_and_never_creates_hardware(monkeypatch):
    import diffusion_policy.SDEdit.reference_action_editor as edit_module

    args = SimpleNamespace(
        method="edit", dp_checkpoint=Path("dp.ckpt"),
        controller_checkpoint=Path("hand.ckpt"), dp_inference_steps=16,
        ddim_inference_steps=4, execution_steps=2, edit_noise_ratio=0.15,
        fixed_noise=1, seed=42, no_salvage=False,
        controller_calls_per_dp=1, validate_only=False, check_only=True,
    )
    capture = {}

    class FakeEditor:
        def __init__(self, checkpoint, noise_ratio, **kwargs):
            capture["editor_args"] = (checkpoint, noise_ratio, kwargs)
            self.controller = SimpleNamespace(execution_steps=2, fixed_noise=True)
            self.future_steps = 9
            self.actual_noise_ratio = 0.149
            self.timesteps = (8, 5, 3, 0)

    monkeypatch.setattr(edit_module, "ReferenceActionEditor", FakeEditor)
    monkeypatch.setattr(pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(pipeline, "_validate_args", lambda _: torch.device("cpu"))
    monkeypatch.setattr(pipeline, "_load_real_policy", lambda *a: (object(), object()))
    monkeypatch.setattr(
        pipeline, "_real_policy_metadata",
        lambda *a: (4, 13, False, {}),
    )
    monkeypatch.setattr(
        pipeline, "_run_check",
        lambda *a, **kw: capture.update(windows=a[-1], editor=kw["editor"]),
    )
    assert pipeline.main() == 0
    assert capture["editor_args"][1] == 0.15
    assert capture["windows"] == ((slice(0, 2), slice(0, 9)),)
    assert isinstance(capture["editor"], FakeEditor)
