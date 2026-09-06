"""Hardware-free regression tests for the shared real-robot initializer."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "eval" / "real"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


local_env = load_module("_reset_direct_robot_env", REAL / "direct_robot_env.py")
with patch.dict(sys.modules, {"direct_robot_env": local_env}):
    init = load_module("_local_robot_init", REAL / "robot_init.py")


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class RobotResetTests(unittest.TestCase):
    def run_init(self, flags=(), hand_fail=False, arm_fail=False):
        events = []
        clock = Clock()

        class Arm:
            def __init__(self, *args):
                self.state = np.full(7, 0.2)

            def get_joint_positions(self):
                return self.state.copy()

            def send_move_joints(self, target):
                # The copied msgpack transport requires a list, not an ndarray.
                assert isinstance(target, list)
                events.append(("arm", clock.now, np.asarray(target)))
                if arm_fail and sum(e[0] == "arm" for e in events) == 1:
                    return {"status": "error"}
                self.state = np.asarray(target)
                return {"status": "ok"}

            def send_stop(self):
                events.append(("stop", clock.now, None))
                return {"status": "ok"}

            def close(self):
                events.append(("arm_close", clock.now, None))

        class Hand:
            def __init__(self, *args):
                self.state = np.full(22, 0.3)

            def get_state(self):
                return self.state.copy()

            def set_action(self, target, interpolate):
                assert not interpolate
                events.append(("hand", clock.now, target.copy()))
                if hand_fail:
                    raise RuntimeError("mock hand rejection")
                self.state = target.copy()

            def close(self):
                events.append(("hand_close", clock.now, None))

        with patch.object(init, "FrankaArmController", Arm), \
             patch.object(init, "SharpaWaveController", Hand), \
             patch.object(init, "time", clock), \
             patch.object(sys, "argv", [str(REAL / "robot_init.py"), *flags]), \
             contextlib.redirect_stdout(io.StringIO()):
            if hand_fail or arm_fail:
                with self.assertRaises(RuntimeError):
                    init.main()
            else:
                self.assertEqual(init.main(), 0)
        return events

    def test_pose_and_local_transport(self):
        np.testing.assert_array_equal(init.HAND_READY_JOINTS, np.zeros(22))
        self.assertIs(init.FrankaArmController, local_env.FrankaArmController)
        self.assertIs(init.SharpaWaveController, local_env.SharpaWaveController)
        text = (REAL / "robot_init.py").read_text()
        self.assertIn("# HAND_READY_JOINTS = np.asarray(", text)

    def test_hand_then_arm_interpolation(self):
        events = self.run_init()
        hand = [e for e in events if e[0] == "hand"]
        arm = [e for e in events if e[0] == "arm"]
        self.assertEqual(len(hand), 51)
        self.assertEqual(len(arm), 30)
        self.assertLess(events.index(hand[-1]), events.index(arm[0]))
        for step, (_, at, target) in enumerate(hand[:50], 1):
            np.testing.assert_allclose(target, 0.3 * (1 - step / 50), atol=1e-15)
            self.assertAlmostEqual(at, (step - 1) * 0.04)
        for step, (_, at, target) in enumerate(arm, 1):
            np.testing.assert_allclose(target, 0.2 + step / 30 * (init.DEFAULT_READY_JOINTS - 0.2))
            self.assertAlmostEqual(at, 2.0 + (step - 1) * 0.1)
        self.assertAlmostEqual(events[-1][1], 5.2)

    def test_check_only_and_hand_only(self):
        self.assertFalse(any(e[0] in ("arm", "hand", "stop") for e in self.run_init(["--check-only"])))
        self.assertFalse(any(e[0] in ("arm", "stop", "arm_close") for e in self.run_init(["--skip-franka"])))

    def test_failures_stop_progress(self):
        self.assertFalse(any(e[0] == "arm" for e in self.run_init(hand_fail=True)))
        events = self.run_init(arm_fail=True)
        self.assertEqual(sum(e[0] == "arm" for e in events), 2)  # rejected command + hold
        self.assertEqual(sum(e[0] == "stop" for e in events), 1)

    def test_launchers(self):
        for script in list(ROOT.glob("*.sh")) + list((ROOT / "eval").glob("*.sh")):
            subprocess.run(["bash", "-n", str(script)], check=True)
        for script in [ROOT / "inference_dp_dino.sh", ROOT / "inference_dp_dino_front.sh",
                       ROOT / "eval/eval_dp_controller_obs22.sh", ROOT / "eval/eval_dp_controller_obs66.sh"]:
            text = script.read_text()
            self.assertNotIn("TacMP/scripts/robot_init.py", text)
            self.assertIn("/real/robot_init.py}", text)
            self.assertIn('"${ROBOT_INIT_SCRIPT}"', text)
        for mode in ("obs22", "obs66"):
            text = (ROOT / f"eval/eval_para_{mode}_real.sh").read_text()
            self.assertIn('env "${COMMON_ENV[@]}" "${MODEL_PYTHON}" -u "${REAL_DIR}/robot_init.py"', text)
            self.assertIn("--skip-franka", text)
            self.assertIn("--no-move-initial-pose", text)
            self.assertNotIn("--initial-pose-file", text)


if __name__ == "__main__":
    unittest.main()
