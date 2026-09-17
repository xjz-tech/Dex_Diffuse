"""Exercise the real control loop without connecting to hardware."""

import contextlib
import io
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "real"))
import inference_real as runner


class SimulatedHand:
    def __init__(self, events):
        self.events = events
        self.reads = 0
        self.closed = False
        self.fail_command = False

    def get_state(self):
        qpos = np.zeros(22)
        qpos[0] = self.reads * 0.01
        self.reads += 1
        self.events.append(("read", qpos.copy()))
        return runner.policy_to_real(qpos)

    def set_action(self, angles, interpolate=False):
        if self.fail_command:
            raise RuntimeError("hand command failed")
        self.events.append(("command", runner.real_to_policy(angles), interpolate))

    def close(self):
        self.closed = True


class InferenceHoldTest(unittest.TestCase):
    def test_hold_and_no_hold_share_sampling_and_full_observation_history(self):
        # Fixed sensor readings isolate sampling from closed-loop motion.
        # Disable clipping so differing command references cannot change policy targets.
        for chunk_steps in (1, 2, 5):
            with self.subTest(chunk_steps=chunk_steps):
                off = self.run_loop(hold=False, chunk_steps=chunk_steps, limits_disabled=True)
                on = self.run_loop(hold=True, chunk_steps=chunk_steps, limits_disabled=True)
                off_reads = [e[1] for e in off if e[0] == "read"]
                on_reads = [e[1] for e in on if e[0] == "read"]
                self.assertEqual(len(on_reads), 1 + 2 * chunk_steps)
                np.testing.assert_array_equal(on_reads, off_reads)
                np.testing.assert_array_equal(
                    [e[1] for e in on if e[0] == "predict"],
                    [e[1] for e in off if e[0] == "predict"],
                )
                for i, event in enumerate(on):
                    if event[0] == "predict":
                        hold = on[i - 1]
                        self.assertEqual(hold[0], "command")
                        np.testing.assert_array_equal(hold[1], event[1][0, -1, :22])

    def test_start_delay_refreshes_shared_sample_before_first_observation(self):
        observations = []
        for hold in (False, True):
            events = self.run_loop(hold=hold, start_delay=0.5, limits_disabled=True)
            self.assertEqual(len([e for e in events if e[0] == "read"]), 4)
            first = next(e[1] for e in events if e[0] == "predict")
            np.testing.assert_allclose(first[0, :, 0], 0.01)
            np.testing.assert_allclose(first[0, :, 22], 0.01)
            np.testing.assert_array_equal(first[0, :, 44], 0.0)
            observations.append(first)
            if hold:
                command = next(e[1] for e in events if e[0] == "command")
                np.testing.assert_array_equal(command, first[0, -1, :22])
        np.testing.assert_array_equal(*observations)

    def run_loop(self, *, live=True, hold=True, hand=None, margin=0.0, chunk_steps=1,
                 limits_disabled=False, target_position=0.5, log_commands=False,
                 hold_on_exit=False, debug_path=None, start_delay=0):
        with patch.object(sys, "argv", ["inference_real.py", "--checkpoint", __file__]):
            args = runner.parse_args()
        args.live = live
        args.hold_during_inference = hold
        args.move_initial_pose = False
        args.wait_for_enter = False
        args.hold_current_on_exit = hold_on_exit
        args.log_action_steps = log_commands
        args.max_steps = 2 * chunk_steps
        args.action_chunk_steps = chunk_steps
        args.max_hand_step = 0.09
        args.joint_limit_margin = margin
        args.start_delay = start_delay
        args.disable_joint_limits = limits_disabled
        if limits_disabled:
            args.disable_step_clamp = True
            args.max_tracking_error = 0
        hand = hand or SimulatedHand([])
        policy = SimpleNamespace(n_action_steps=5, n_obs_steps=4, obs_dim=66, action_dim=22)

        def predict(policy, observation, device):
            hand.events.append(("predict", observation.copy()))
            action = np.zeros((5, 22))
            action[:, 0] = target_position
            return action, 0.01

        output = io.StringIO()
        with patch.object(runner, "SharpaWaveController", return_value=hand), \
             patch.object(runner, "predict", side_effect=predict), \
             patch.object(runner.time, "sleep"), \
             patch.dict(os.environ, {"REAL_DEBUG_LOG": str(debug_path) if debug_path else ""}), \
             contextlib.redirect_stdout(output):
            self.assertEqual(runner.run_hardware(args, policy, torch.device("cpu")), 0)
        self.loop_output = output.getvalue()
        self.assertTrue(hand.closed)
        return hand.events

    def test_structured_run_records_observation_action_state_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "debug.jsonl"
            events = self.run_loop(debug_path=path, chunk_steps=2, hold_on_exit=True)
            self.assertTrue(path.exists())
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            commands = [row for row in rows if row["event"] == "command"]
            states = [row for row in rows if row["event"] == "state"]
            self.assertEqual(len(commands), len([e for e in events if e[0] == "command"]))
            self.assertEqual(len(states), len([e for e in events if e[0] == "read"]))
            self.assertEqual(commands[-1]["kind"], "exit_hold")
            predictions = [r for r in rows if r["event"] == "inference_output"]
            inputs = [r for r in rows if r["event"] == "inference_input"]
            self.assertEqual(len(predictions), 2)
            for input_row in inputs:
                hold_row = next(r for r in commands if r["kind"] == "inference_hold"
                                and r["chunk"] == input_row["chunk"])
                self.assertLess(rows.index(input_row), rows.index(hold_row))
                np.testing.assert_array_equal(
                    hold_row["target_policy_rad"], input_row["observation"][0][-1][:22],
                )
            np.testing.assert_allclose(inputs[1]["observation"],
                                       [e[1] for e in events if e[0] == "predict"][1])
            self.assertEqual(np.asarray(predictions[0]["returned_actions_rad"]).shape, (5, 22))
            self.assertEqual(np.asarray(predictions[0]["selected_actions_rad"]).shape, (2, 22))
            policy_command = next(r for r in commands if r["kind"] == "policy")
            self.assertAlmostEqual(policy_command["raw_target_policy_rad"][0], 0.5)
            self.assertAlmostEqual(policy_command["target_policy_rad"][0], 0.09, places=6)
            self.assertIn("sha256", next(r for r in rows if r["event"] == "model")["checkpoints"]["prior"])
            self.assertEqual(rows[-1]["event"], "session_end")

    def test_command_logs_compare_actual_sends_including_holds_and_cleanup(self):
        events = self.run_loop(chunk_steps=2, log_commands=True, hold_on_exit=True)
        commands = [event[1] for event in events if event[0] == "command"]
        lines = [line for line in self.loop_output.splitlines() if line.startswith("[cmd ")]
        self.assertEqual(len(lines), len(commands))
        self.assertEqual(len(lines), 7)
        self.assertIn("delta_max_rad=nan delta_max_deg=nan reference=unknown", lines[0])
        for previous, target, line in zip(commands, commands[1:], lines[1:]):
            index = int(np.argmax(np.abs(target - previous)))
            expected = abs(target[index] - previous[index])
            values = dict(re.findall(r"(\w+)=([^\s\]]+)", line))
            self.assertAlmostEqual(float(values["delta_max_rad"]), expected, places=5)
            self.assertAlmostEqual(float(values["delta_max_deg"]), np.degrees(expected), places=3)
            self.assertEqual(values["joint"], runner.POLICY_SHARPA_DOF_NAMES[index])
            self.assertEqual(int(values["joint_index"]), index)
            self.assertAlmostEqual(float(values["from_rad"]), previous[index], places=5)
            self.assertAlmostEqual(float(values["to_rad"]), target[index], places=5)
        self.assertIn("kind=inference_hold", lines[3])
        self.assertIn("delta_max_rad=0.160000", lines[3])
        self.assertIn("kind=exit_hold", lines[-1])

    def test_command_logs_are_live_only_and_respect_logging_flag(self):
        for live, log_commands in ((False, True), (True, False)):
            with self.subTest(live=live, log_commands=log_commands):
                self.run_loop(live=live, log_commands=log_commands)
                self.assertNotIn("[cmd ", self.loop_output)

    def test_holds_shared_sample_before_each_prediction_and_rebases_step_limit(self):
        events = self.run_loop()
        predictions = [i for i, event in enumerate(events) if event[0] == "predict"]
        for index, position in zip(predictions, (0.0, 0.01)):
            self.assertEqual(events[index - 1][0], "command")
            self.assertFalse(events[index - 1][2])
            self.assertAlmostEqual(events[index - 1][1][0], position)
        commands = [event[1][0] for event in events if event[0] == "command"]
        np.testing.assert_allclose(commands, [0.0, 0.09, 0.01, 0.10], atol=1e-7)

    def test_observation_uses_period_end_qpos_and_retains_previous_policy_target(self):
        observations = [event[1] for event in self.run_loop() if event[0] == "predict"]
        self.assertEqual(observations[1].shape, (1, 4, 66))
        # Second inference: measured=.01, previous policy target=.09, residual=.08.
        np.testing.assert_allclose(observations[1][0, -1, [0, 22, 44]],
                                   [0.01, 0.09, 0.08], atol=1e-7)
        # The action adds one history frame; holding does not add or refresh one.
        np.testing.assert_allclose(observations[1][0, -2], observations[0][0, -1])

    def test_two_step_chunks_hold_only_at_inference_boundaries(self):
        events = self.run_loop(chunk_steps=2)
        commands = [event[1][0] for event in events if event[0] == "command"]
        np.testing.assert_allclose(commands, [0.0, 0.09, 0.18, 0.02, 0.11, 0.20], atol=1e-7)
        observations = [event[1] for event in events if event[0] == "predict"]
        self.assertEqual(len(observations), 2)
        np.testing.assert_allclose(observations[1][0, -1, [0, 22, 44]],
                                   [0.02, 0.18, 0.16], atol=1e-7)

    def test_cli_accepts_hold_flag_without_changing_default(self):
        argv = ["inference_real.py", "--checkpoint", __file__]
        with patch.object(sys, "argv", argv):
            self.assertFalse(runner.parse_args().hold_during_inference)
        with patch.object(sys, "argv", argv + ["--hold-during-inference"]):
            self.assertTrue(runner.parse_args().hold_during_inference)

    def test_disabled_mode_keeps_original_command_sequence(self):
        commands = [event[1][0] for event in self.run_loop(hold=False) if event[0] == "command"]
        np.testing.assert_allclose(commands, [0.09, 0.18])

    def test_dry_run_never_sends_hold_or_action_commands(self):
        self.assertFalse(any(event[0] == "command" for event in self.run_loop(live=False)))

    def test_disabled_limits_pass_through_policy_targets(self):
        commands = [event[1][0] for event in self.run_loop(
            limits_disabled=True, target_position=3.0) if event[0] == "command"]
        np.testing.assert_allclose(commands, [0.0, 3.0, 0.01, 3.0], atol=1e-7)

    def test_disabled_limits_hold_measured_out_of_range_position(self):
        class OutOfRangeHand(SimulatedHand):
            def get_state(self):
                qpos = np.zeros(22)
                qpos[0] = runner.POLICY_UPPER_LIMITS[0] + 0.0625
                self.events.append(("read", qpos.copy()))
                return runner.policy_to_real(qpos)

        hand = OutOfRangeHand([])
        events = self.run_loop(hand=hand, limits_disabled=True, target_position=3.0)
        commands = [event[1][0] for event in events if event[0] == "command"]
        np.testing.assert_allclose(commands, [1.6333, 3.0, 1.6333, 3.0], atol=1e-7)
        observations = [event[1] for event in events if event[0] == "predict"]
        np.testing.assert_allclose(observations[0][0, -1, [0, 22, 44]],
                                   [1.6333, 1.6333, 0.0], atol=1e-7)

    def test_disabled_limits_still_reject_nonfinite_policy_output(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
                self.run_loop(limits_disabled=True, target_position=value)

    def test_enabled_limits_still_reject_measured_overflow(self):
        hand = SimulatedHand([])
        hand.reads = 170  # First read is 1.7 rad, above index MCP's upper limit.
        with self.assertRaisesRegex(RuntimeError, "exceeds policy URDF limits"):
            self.run_loop(hand=hand)
        self.assertTrue(hand.closed)
        self.assertFalse(any(event[0] == "predict" for event in hand.events))

    def test_position_and_step_limits_can_be_controlled_independently(self):
        raw = np.zeros(22)
        raw[0] = 3.0
        for disable_joint, disable_step, expected in (
            (False, False, 0.09), (False, True, 1.5708),
            (True, False, 0.09), (True, True, 3.0),
        ):
            with self.subTest(joint=disable_joint, step=disable_step):
                args = SimpleNamespace(disable_joint_limits=disable_joint,
                                       disable_step_clamp=disable_step,
                                       joint_limit_margin=0.0, max_hand_step=0.09)
                target, _, _ = runner.safe_target(raw, np.zeros(22), args)
                self.assertAlmostEqual(target[0], expected)

    def test_initial_pose_limit_can_be_disabled_without_skipping_shape_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.json"
            qpos = np.zeros(22)
            qpos[0] = 3.0
            payload = {"policy_checkpoint_order": {
                "joint_names": list(runner.POLICY_SHARPA_DOF_NAMES),
                "qpos_rad": qpos.tolist(),
            }}
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "exceeds policy URDF limits"):
                runner.load_initial_pose(str(path))
            with contextlib.redirect_stdout(io.StringIO()):
                np.testing.assert_array_equal(
                    runner.load_initial_pose(str(path), disable_joint_limits=True), qpos)
            payload["policy_checkpoint_order"]["qpos_rad"] = [0.0]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "22 finite"):
                runner.load_initial_pose(str(path), disable_joint_limits=True)

    def test_both_real_launchers_disable_limits_and_allow_reenabling(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "python"
            python.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
            python.chmod(0o755)
            env = {"PATH": os.environ["PATH"], "MODEL_PYTHON": str(python),
                   "CKPT_PATH": __file__, "PRIOR_CKPT_PATH": __file__,
                   "GUIDE_CKPT_PATH": __file__, "CHECK_ONLY": "1",
                   "RUN_LOG": str(Path(directory) / "latest.txt")}
            for script in ("eval_para_obs66_real.sh", "xjz_eval_strong_prior_real.sh"):
                for enabled in (False, True):
                    with self.subTest(script=script, enabled=enabled):
                        settings = dict(env)
                        if enabled:
                            settings.update(DISABLE_JOINT_LIMITS="0", DISABLE_STEP_CLAMP="0",
                                            MAX_TRACKING_ERROR="0.35")
                        result = subprocess.run(["bash", str(ROOT / "eval" / script)],
                                                env=settings, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        argv = json.loads(result.stdout.splitlines()[-1])
                        self.assertEqual("--disable-joint-limits" in argv, not enabled)
                        self.assertEqual("--disable-step-clamp" in argv, not enabled)
                        self.assertEqual(float(argv[argv.index("--max-tracking-error") + 1]),
                                         0.35 if enabled else 0.0)

    def test_hold_respects_joint_limit_margin(self):
        events = self.run_loop(margin=0.02)
        hold = next(event[1] for event in events if event[0] == "command")
        self.assertAlmostEqual(hold[2], 0.02)  # PIP lower limit is zero.

    def test_failed_hold_aborts_before_inference_and_closes_connection(self):
        hand = SimulatedHand([])
        hand.fail_command = True
        with self.assertRaisesRegex(RuntimeError, "hand command failed"):
            self.run_loop(hand=hand)
        self.assertFalse(any(event[0] == "predict" for event in hand.events))
        self.assertTrue(hand.closed)

    def test_launcher_enables_hold_by_default_and_accepts_override(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "python"
            python.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
            python.chmod(0o755)
            for value in (None, "0", "1", "invalid"):
                with self.subTest(value=value):
                    env = {"PATH": os.environ["PATH"], "MODEL_PYTHON": str(python),
                           "CKPT_PATH": __file__, "CHECK_ONLY": "1",
                           "RUN_LOG": str(Path(directory) / "latest.txt")}
                    if value is not None:
                        env["HOLD_DURING_INFERENCE"] = value
                    result = subprocess.run(["bash", str(ROOT / "eval/eval_para_obs66_real.sh")],
                                            env=env, capture_output=True, text=True)
                    if value == "invalid":
                        self.assertEqual(result.returncode, 2)
                        self.assertIn("HOLD_DURING_INFERENCE must be 0 or 1", result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        argv = json.loads(result.stdout.splitlines()[-1])
                        self.assertEqual("--hold-during-inference" in argv, value != "0")


if __name__ == "__main__":
    unittest.main()
