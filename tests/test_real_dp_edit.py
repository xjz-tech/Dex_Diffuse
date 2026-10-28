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


def test_large_hand_action_interpolation_preserves_original_endpoint():
    previous = np.zeros(31, dtype=np.float64)
    previous[3] = 1.0
    previous[7] = 1.0
    target = previous.copy()
    target[:3] = [0.2, 0.1, 0.3]
    target[9] = 0.25
    target[10] = -0.13

    commands = pipeline._interpolate_large_hand_action(previous, target, 0.12)

    assert commands.shape == (2, 31)
    np.testing.assert_array_equal(commands[-1], target)
    np.testing.assert_array_equal(commands[0, :9], previous[:9])
    np.testing.assert_allclose(commands[0, 9:], (previous[9:] + target[9:]) / 2)
    large = target.copy()
    large[9] = 1.0
    assert pipeline._interpolate_large_hand_action(previous, large, 0.12).shape == (2, 31)
    small = target.copy()
    small[9:] = previous[9:] + 0.12
    np.testing.assert_array_equal(
        pipeline._interpolate_large_hand_action(previous, small, 0.12), small[None]
    )


def test_large_action_interpolation_is_optional(monkeypatch):
    argv = ["inference_dp_edit.py", "--dp-checkpoint", "dp.ckpt",
            "--controller-checkpoint", "hand.ckpt"]
    monkeypatch.setattr(sys, "argv", argv)
    assert not pipeline.parse_args().edit_interpolate_large_actions
    monkeypatch.setattr(sys, "argv", argv + ["--edit-interpolate-large-actions"])
    enabled = pipeline.parse_args()
    assert enabled.edit_interpolate_large_actions
    assert enabled.edit_interpolation_threshold_rad == 0.12


def test_trt_adapter_uses_visual_encoder_feature_dimension(monkeypatch):
    from diffusers import DDIMScheduler
    import trt_unet

    seen = {}

    def fake_compile(model, **kwargs):
        seen.update(kwargs)
        return Mock()

    monkeypatch.setattr(trt_unet, "compile_unet", fake_compile)
    policy = SimpleNamespace(
        model=torch.nn.Linear(1, 1), n_obs_steps=2, obs_feature_dim=128,
        horizon=16, action_dim=31, obs_as_global_cond=True,
        noise_scheduler=DDIMScheduler(num_train_timesteps=100),
        num_inference_steps=16,
    )
    trt_unet.accelerate_policy_unet(policy, max_batch=1)
    assert seen["global_cond_dim"] == 256
    assert seen["max_batch"] == 1


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
        trt_fused=False,
        edit_interpolate_large_actions=False, edit_interpolation_threshold_rad=0.12,
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


def test_trt_fused_check_compiles_visual_and_hand_unets(monkeypatch):
    from diffusers import DDIMScheduler
    import diffusion_policy.SDEdit.reference_action_editor as edit_module
    import trt_unet

    args = SimpleNamespace(
        method="edit", dp_checkpoint=Path("dp.ckpt"),
        controller_checkpoint=Path("hand.ckpt"), dp_inference_steps=16,
        ddim_inference_steps=4, execution_steps=2, edit_noise_ratio=0.15,
        fixed_noise=1, seed=42, no_salvage=False, trt_fused=True,
        controller_calls_per_dp=1, validate_only=False, check_only=True,
        edit_interpolate_large_actions=False, edit_interpolation_threshold_rad=0.12,
    )
    visual = SimpleNamespace(noise_scheduler=DDIMScheduler(num_train_timesteps=100))
    hand = object()
    compiled = []

    class FakeEditor:
        def __init__(self, *args, **kwargs):
            self.controller = SimpleNamespace(execution_steps=2, policy=hand, fixed_noise=True)
            self.future_steps = 9
            self.actual_noise_ratio = 0.15
            self.timesteps = (9, 6, 3, 0)
            self.fused = False

        def enable_fused_sampling(self):
            self.fused = True

    monkeypatch.setattr(edit_module, "ReferenceActionEditor", FakeEditor)
    monkeypatch.setattr(pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(pipeline, "_validate_args", lambda _: torch.device("cuda"))
    monkeypatch.setattr(pipeline, "_load_real_policy", lambda *a: (object(), visual))
    monkeypatch.setattr(pipeline, "_real_policy_metadata", lambda *a: (4, 13, False, {}))
    monkeypatch.setattr(trt_unet, "accelerate_policy_unet", lambda policy, **kw: compiled.append((policy, kw)))
    monkeypatch.setattr(pipeline, "_run_check", lambda *a, **kw: compiled.append(kw["editor"].fused))

    assert pipeline.main() == 0
    assert compiled == [
        (visual, {"fp16": True, "max_batch": 1}),
        (hand, {"fp16": True, "max_batch": 1}),
        True,
    ]


@pytest.mark.parametrize("activation_key,on_key,preview_available", [
    (None, True, True), (ord("e"), True, True), (ord("E"), True, True),
    (None, False, True), (None, True, False),
])
def test_online_edit_waits_for_key_and_switches_at_chunk_boundary(
    monkeypatch, activation_key, on_key, preview_available,
):
    import inference_dp as real_dp
    import diffusion_policy.SDEdit.reference_action_editor as edit_module
    from policy_observation import compose_policy_observation, QPOS_TARGET_RESIDUAL_OBSERVATION

    monkeypatch.setattr(sys, "argv", [
        "inference_dp_controller.py", "--method", "edit", "--guidance-scale", "0",
        "--dp-checkpoint", "dp.ckpt", "--controller-checkpoint", "hand.ckpt",
        "--edit-on-key", "--show-camera-input", "--controller-calls-per-dp", "3",
        "--max-chunks", "3",
    ])
    args = pipeline.parse_args()
    args.edit_on_key = on_key
    dp_actions = np.arange(13 * 31, dtype=np.float32).reshape(13, 31) / 100
    issued = []
    limits = []
    editor_calls = []
    environments = []

    class FakeEnv:
        def __init__(self, **kwargs):
            self.show_camera_input = kwargs["show_camera_input"] and preview_available
            self.callback = kwargs["camera_key_callback"]
            self.disable_clamp = kwargs["disable_clamp"]
            self.ik_solver = SimpleNamespace(apply_joint_limits=not kwargs["disable_ik_limits"])
            self.previous_action = np.zeros(31)
            self.chunk_idx = 0
            self.closed = False
            environments.append(self)

        def reset(self):
            return {
                "/observe/vision/front/rgb": np.zeros((4, 6, 3), dtype=np.uint8),
                "/observe/vision/wrist/rgb": np.zeros((4, 6, 3), dtype=np.uint8),
                "/state/arm/eef_pose": np.zeros(9, dtype=np.float32),
                "/state/hand/joint_angle": np.full(22, len(issued) / 10, dtype=np.float32),
            }

        def step_single(self, action, step_idx):
            issued.append(action.copy())
            limits.append((self.disable_clamp, self.ik_solver.apply_joint_limits))
            self.previous_action = action.copy()
            # The first E arrives halfway through a two-step chunk. Repeated E
            # presses must leave edit enabled rather than toggle it off.
            if self.callback is not None:
                self.callback(activation_key if activation_key is not None else ord("x"))
            return self.reset()

        def close(self):
            self.closed = True

    class FakeEditor:
        def __init__(self, *args, **kwargs):
            self.controller = SimpleNamespace(
                execution_steps=2, fixed_noise=True, spec={"n_obs_steps": 4},
                compose_observation=lambda q, target: compose_policy_observation(
                    q, target, QPOS_TARGET_RESIDUAL_OBSERVATION,
                ),
            )
            self.future_steps = 9
            self.actual_noise_ratio = 0.15
            self.timesteps = (8, 5, 3, 0)

        def predict(self, history, reference, seeds):
            editor_calls.append((len(issued), history.copy(), reference.copy()))
            return np.full((1, 2, 22), 0.75, dtype=np.float32), {"edit_rmse_rad": 0.1}

    policy = SimpleNamespace(predict_action=lambda _: {"action": torch.from_numpy(dp_actions[None])})
    monkeypatch.setattr(pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(pipeline, "_validate_args", lambda _: torch.device("cpu"))
    monkeypatch.setattr(pipeline, "_load_real_policy", lambda *a: (object(), policy))
    monkeypatch.setattr(pipeline, "_real_policy_metadata", lambda *a: (
        4, 13, False, {"front_image": (3, 4, 6), "wrist_image": (3, 4, 6)},
    ))
    monkeypatch.setattr(edit_module, "ReferenceActionEditor", FakeEditor)
    monkeypatch.setattr(real_dp, "DiffusionDirectRobotEnv", FakeEnv)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)

    if not preview_available:
        with pytest.raises(RuntimeError, match="working OpenCV camera window"):
            pipeline.main()
        assert environments[0].closed
        assert issued == editor_calls == []
        return

    assert pipeline.main() == 0
    assert environments[0].closed
    issued = np.stack(issued)
    np.testing.assert_array_equal(issued[:, :9], dp_actions[:6, :9])
    if not on_key:
        np.testing.assert_array_equal(issued[:, 9:], 0.75)
        assert limits == [(True, False)] * 6
        assert [call[0] for call in editor_calls] == [0, 2, 4]
    elif activation_key is None:
        np.testing.assert_array_equal(issued, dp_actions[:6])
        assert editor_calls == []
        assert limits == [(False, True)] * 6
    else:
        np.testing.assert_array_equal(issued[:2], dp_actions[:2])
        np.testing.assert_array_equal(issued[2:, 9:], 0.75)
        assert limits == [(False, True)] * 2 + [(True, False)] * 4
        assert [call[0] for call in editor_calls] == [2, 4]
        # Warm the controller history with actual feedback and issued targets
        # while executing raw DP, rather than restarting history at activation.
        first_history = editor_calls[0][1]
        np.testing.assert_allclose(first_history[-2:, :22], [[0.1] * 22, [0.2] * 22])
        np.testing.assert_array_equal(first_history[-2:, 22:44], dp_actions[:2, 9:])
        np.testing.assert_array_equal(editor_calls[0][2], dp_actions[2:11, 9:])


def test_camera_preview_delivers_keyboard_events(monkeypatch):
    import direct_robot_env as robot

    arm, hand, cameras, callback = Mock(), Mock(), Mock(), Mock()
    arm.get_current_tcp_pose.return_value = np.eye(4)
    hand.get_state.return_value = np.zeros(22)
    frame = SimpleNamespace(color_rgb=np.zeros((4, 6, 3), dtype=np.uint8))
    cameras.read_pair.return_value = {"front": frame, "wrist": frame}
    monkeypatch.setattr(robot, "FrankaArmController", lambda **kw: arm)
    monkeypatch.setattr(robot, "SharpaWaveController", lambda **kw: hand)
    monkeypatch.setattr(robot.RealSensePair, "start", lambda **kw: cameras)
    for name in ("namedWindow", "resizeWindow", "imshow", "destroyWindow"):
        monkeypatch.setattr(robot.cv2, name, Mock())
    monkeypatch.setattr(robot.cv2, "waitKey", lambda _: ord("e"))

    env = robot.DirectRobotEnv(use_tactile=False, show_camera_input=True, camera_key_callback=callback)
    try:
        env.reset()
        callback.assert_called_once_with(ord("e"))
    finally:
        env.close()


def test_edit_key_validation_requires_edit_and_a_preview(monkeypatch, tmp_path):
    for name in ("dp.ckpt", "hand.ckpt"):
        (tmp_path / name).touch()
    monkeypatch.setattr(sys, "argv", [
        "inference_dp_controller.py", "--dp-checkpoint", str(tmp_path / "dp.ckpt"),
        "--controller-checkpoint", str(tmp_path / "hand.ckpt"), "--device", "cpu",
        "--edit-on-key",
    ])
    args = pipeline.parse_args()
    with pytest.raises(ValueError, match="requires --method edit"):
        pipeline._validate_args(args)
    args.method, args.guidance_scale = "edit", 0
    with pytest.raises(ValueError, match="requires --show-camera-input"):
        pipeline._validate_args(args)
    args.check_only = True
    assert pipeline._validate_args(args) == torch.device("cpu")
