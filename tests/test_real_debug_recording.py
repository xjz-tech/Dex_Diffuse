"""Portable real-run records, using synthetic state only."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "real"))
from debug_recording import DebugRecorder, RecordedHand, checkpoint_identity
from hardware import policy_to_real


class FakeHand:
    def __init__(self):
        self.reads = 0
        self.commands = []

    def get_state(self):
        self.reads += 1
        return policy_to_real(np.arange(22) * 0.01 * self.reads)

    def set_action(self, angles, interpolate=False):
        self.commands.append(np.asarray(angles).copy())

    def close(self):
        pass


class DebugRecordingTest(unittest.TestCase):
    def test_snapshots_and_append_survive_close_and_include_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            recorder = DebugRecorder(path, "inference")
            values = np.arange(22, dtype=np.float32)
            recorder.emit("sample", values=values)
            values[:] = 100
            recorder.close()
            other = DebugRecorder(path, "cleanup")
            other.emit("sample", optional=None)
            other.close()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0]["values"], list(range(22)))
            self.assertEqual(rows[1]["stage"], "cleanup")
            self.assertGreater(rows[0]["monotonic_ns"], 0)
            self.assertGreater(rows[0]["wall_time_ns"], 0)

    def test_wrapper_records_exact_commands_and_states_without_extra_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            recorder = DebugRecorder(path, "inference")
            raw = FakeHand()
            hand = RecordedHand(raw, recorder)
            first = hand.get_state()
            target = np.arange(22) * 0.02
            hand.context = dict(kind="policy", chunk=1, chunk_step=0, raw_target_policy_rad=target + 0.01)
            hand.set_action(policy_to_real(target), interpolate=True)
            hand.get_state()
            recorder.close()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(raw.reads, 2)
            self.assertEqual(len(raw.commands), 1)
            self.assertEqual([r["event"] for r in rows], ["state", "command", "state"])
            np.testing.assert_allclose(rows[1]["target_policy_rad"], target)
            np.testing.assert_allclose(rows[1]["target_real_rad"], policy_to_real(target))
            np.testing.assert_allclose(rows[1]["tracking_before_rad"], target / 2)
            np.testing.assert_allclose(rows[2]["tracking_error_rad"], np.zeros(22), atol=1e-7)
            self.assertTrue(rows[1]["interpolate"])
            self.assertIsNone(rows[0]["velocity_policy_rad_s"])
            self.assertGreaterEqual(rows[1]["send_end_ns"], rows[1]["send_start_ns"])
            self.assertGreaterEqual(rows[2]["read_end_ns"], rows[1]["send_end_ns"])

    def test_checkpoint_hash_is_content_based(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            path.write_bytes(b"checkpoint")
            identity = checkpoint_identity(path)
            self.assertEqual(identity["sha256"], hashlib.sha256(b"checkpoint").hexdigest())
            self.assertEqual(identity["size_bytes"], 10)

    def test_predict_capture_serializes_complete_sampling_data(self):
        import torch
        from types import SimpleNamespace
        from test_real_sampling_capture import policy
        from guided_policy import GuidedRealPolicy
        import inference_real
        from debug_recording import record_model

        for guided in (False, True):
            with self.subTest(guided=guided), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "run.jsonl"
                model = policy()
                args = SimpleNamespace(checkpoint=__file__, guide_checkpoint=None)
                if guided:
                    args = SimpleNamespace(checkpoint=__file__, guide_checkpoint=__file__,
                                           action_chunk_steps=2, inference_steps=4,
                                           guide_inference_steps=4, guidance_steps=2,
                                           guidance_scale=25., seed=42, guide_seed=43,
                                           fixed_noise=1, fused_ddim=True)
                    model = GuidedRealPolicy(model, policy(scale=1.5), args, "cpu")
                recorder = DebugRecorder(path, "inference")
                record_model(recorder, model, args, torch.device("cpu"))
                model._debug_capture = {}
                actions, duration = inference_real.predict(model, np.ones((1, 2, 3)), torch.device("cpu"))
                recorder.emit("prediction", details=model._last_debug_prediction)
                recorder.close()
                self.assertIsNone(recorder.error)
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                capture = rows[-1]["details"]["sampling"]
                self.assertEqual(np.asarray(capture["action_pred"]).shape, (1, 8, 2))
                key = "prior_initial_noise" if guided else "initial_noise"
                self.assertEqual(np.asarray(capture[key]).shape, (1, 8, 2))
                if guided:
                    self.assertEqual(np.asarray(capture["guide_initial_noise"]).shape, (1, 8, 2))
                    self.assertEqual(np.asarray(capture["guide_reference_physical"]).shape, (1, 2, 2))
                np.testing.assert_allclose(rows[-1]["details"]["output"]["action"][0], actions)
                self.assertGreaterEqual(rows[-1]["details"]["capture_copy_ns"], 0)

    def test_initialization_process_appends_commands_and_feedback(self):
        import robot_init

        class ReadyHand(FakeHand):
            def get_state(self):
                self.reads += 1
                return self.commands[-1].copy() if self.commands else np.zeros(22)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "init.jsonl"
            raw = ReadyHand()
            with patch.dict(os.environ, {"REAL_DEBUG_LOG": str(path)}), \
                 patch.object(sys, "argv", ["robot_init", "--skip-franka", "--hand-steps", "2"]), \
                 patch.object(robot_init, "SharpaWaveController", return_value=raw), \
                 patch.object(robot_init.time, "sleep"):
                self.assertEqual(robot_init.main(), 0)
            self.assertTrue(path.exists())
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(sum(r["event"] == "command" for r in rows), 3)
            self.assertEqual(sum(r["event"] == "state" for r in rows), raw.reads)
            self.assertTrue(all(r["stage"] == "initialization" for r in rows))
            self.assertEqual(rows[-1]["event"], "session_end")


if __name__ == "__main__":
    unittest.main()
