"""Hardware-free coverage for the front-only deployment path."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

import direct_robot_env as original_env
import direct_robot_front_env as front_env
import inference_dp as original_inference
import inference_dp_front as inference


class FrontInferenceTests(unittest.TestCase):
    def test_front_history_and_relative_pose_round_trip(self):
        history = []
        for x in (1.0, 2.0):
            history.append(inference.capture_obs({
                "/observe/vision/front/rgb": np.full((8, 10, 3), 255, np.uint8),
                "/state/arm/eef_pose": np.array([x, 0, 0, 0, 1, 0, -1, 0, 0]),
                "/state/hand/joint_angle": np.arange(22, dtype=np.float32),
            }, {"front_image": (3, 4, 5)}))
        for relative in (False, True):
            obs, base = inference.build_policy_obs(history, torch.device("cpu"), relative)
            self.assertEqual(set(obs), {"front_image", "ee_pose", "hand_joint"})
            self.assertEqual(tuple(obs["front_image"].shape), (1, 2, 3, 4, 5))
            self.assertTrue(bool((obs["front_image"] == 1).all()))
            actions = np.concatenate((obs["ee_pose"][0], obs["hand_joint"][0]), axis=-1)
            if relative:
                np.testing.assert_allclose(obs["ee_pose"][0, -1, :3], 0)
                actions = inference.mixed_actions_to_absolute(actions, base)
            np.testing.assert_allclose(actions[:, :9], [o["ee_pose"] for o in history])
            np.testing.assert_allclose(actions[:, 9:], [o["hand_joint"] for o in history])

    def test_camera_discovery_ignores_wrist_and_honors_serial(self):
        devices = [
            {"name": "Intel RealSense D435", "serial_number": "front1", "product_id": "0b3a"},
            {"name": "Intel RealSense D405", "serial_number": "wrist1", "product_id": "0b5b"},
            {"name": "Intel RealSense D405", "serial_number": "wrist2", "product_id": "0b5b"},
        ]
        rs = SimpleNamespace(
            context=lambda: SimpleNamespace(devices=devices),
            camera_info=SimpleNamespace(name="name", serial_number="serial_number", product_id="product_id"),
        )
        with patch.object(original_env, "_import_realsense", return_value=rs), \
             patch.object(original_env, "get_device_info", side_effect=lambda d, key: d[key]), \
             patch.object(original_env, "iter_color_profiles", return_value=[Mock()]), \
             patch.object(original_env, "select_color_profile"), \
             patch.object(original_env, "RealSenseColorCamera") as camera:
            front_env.start_front_camera(30, None, (640, 480))
            self.assertEqual(camera.call_args.args[0].serial, "front1")
            front_env.start_front_camera(30, "wrist1", (640, 480))
            self.assertEqual(camera.call_args.args[0].serial, "wrist1")
            with self.assertRaisesRegex(RuntimeError, "not found"):
                front_env.start_front_camera(30, "missing", (640, 480))
            devices.append(dict(devices[0], serial_number="front2"))
            with self.assertRaisesRegex(RuntimeError, "Multiple front"):
                front_env.start_front_camera(30, None, (640, 480))
            # No wrist camera is needed when only a front camera is connected.
            devices[:] = devices[:1]
            front_env.start_front_camera(30, None, (640, 480))

    def test_environment_reset_step_and_cleanup_without_wrist(self):
        with patch.object(sys, "argv", ["inference_dp_front.py", "--checkpoint", "unused"]):
            args = inference.parse_args()
        camera = Mock()
        camera.read.return_value.color_rgb = np.zeros((8, 10, 3), dtype=np.uint8)
        with patch.object(front_env, "start_front_camera", return_value=camera), \
             patch.object(original_env, "FrankaArmController") as arm, \
             patch.object(original_env, "SharpaWaveController") as hand, \
             patch.object(original_env.RealSensePair, "start", side_effect=AssertionError("pair started")):
            arm.return_value.get_current_tcp_pose.return_value = np.eye(4)
            hand.return_value.get_state.return_value = np.arange(22, dtype=np.float32)
            env = front_env.FrontDirectRobotEnv(args)
            try:
                obs = env.reset()
                self.assertNotIn("/observe/vision/wrist/rgb", obs)
                self.assertTrue(obs["clear"])
                np.testing.assert_array_equal(
                    obs["/state/hand/joint_angle"], original_env.REAL2POLICY_DOF_INDICES
                )
                next_obs = env.step_single(env.previous_action.copy(), 0)
                self.assertNotIn("clear", next_obs)
                self.assertEqual(camera.read.call_count, 2)
                arm.return_value.send_move.assert_not_called()
                arm.return_value.send_move_joints.assert_not_called()
                hand.return_value.set_action.assert_not_called()
            finally:
                env.close()
            camera.stop.assert_called_once()
            arm.return_value.close.assert_called_once()
            hand.return_value.close.assert_called_once()
        self.assertEqual(original_env.CAMERA_LABELS, ("front", "wrist"))
        self.assertIs(original_inference.DirectRobotEnv, original_env.DirectRobotEnv)


if __name__ == "__main__":
    unittest.main()
