"""Check launcher logging with a stub interpreter; never contact hardware."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherLoggingTest(unittest.TestCase):
    def test_without_flag_leaves_records_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "latest.txt"
            sidecar = log.with_suffix(".jsonl")
            stub = root / "python"
            stub.write_text(
                f"#!{sys.executable}\nimport os\n"
                "assert not os.environ.get('REAL_DEBUG_LOG')\n"
                "print('stub output')\n"
            )
            stub.chmod(0o755)
            env = dict(PATH=os.environ["PATH"], MODEL_PYTHON=str(stub),
                       CKPT_PATH=__file__, PRIOR_CKPT_PATH=__file__,
                       GUIDE_CKPT_PATH=__file__, CHECK_ONLY="1", RUN_LOG=str(log),
                       REAL_DEBUG_LOG=str(sidecar), DEBUG_RECORDING="1")
            for script in ("eval_para_obs66_real.sh", "xjz_eval_strong_prior_real.sh"):
                for existing in (False, True):
                    with self.subTest(script=script, existing=existing):
                        if existing:
                            log.write_text("previous output\n")
                            sidecar.write_text("previous data\n")
                        else:
                            log.unlink(missing_ok=True)
                            sidecar.unlink(missing_ok=True)
                        before = {p.name: p.read_bytes() for p in root.iterdir()}
                        result = subprocess.run(["bash", str(ROOT / "eval" / script)],
                                                env=env, capture_output=True, text=True,
                                                timeout=10)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn("stub output", result.stdout)
                        self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)

    def test_preserves_all_runs_across_both_launchers(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "latest.txt"
            log.write_text("existing run\n")
            log.with_suffix(".jsonl").write_text("existing structured run\n")
            stub = Path(directory) / "python"
            stub.write_text(
                f"#!{sys.executable}\nimport os\n"
                "print('RUN=' + os.environ['RUN_NUMBER'])\n"
                "with open(os.environ['REAL_DEBUG_LOG'], 'a') as f:\n"
                "    f.write(os.environ['RUN_NUMBER'] + '\\n')\n"
            )
            stub.chmod(0o755)
            env = dict(PATH=os.environ["PATH"], MODEL_PYTHON=str(stub),
                       CKPT_PATH=__file__, PRIOR_CKPT_PATH=__file__,
                       GUIDE_CKPT_PATH=__file__, CHECK_ONLY="1", RUN_LOG=str(log))
            for number in range(1, 13):
                script = ("eval_para_obs66_real.sh" if number % 2
                          else "xjz_eval_strong_prior_real.sh")
                result = subprocess.run(["bash", str(ROOT / "eval" / script), "--record"],
                                        env=dict(env, RUN_NUMBER=str(number)),
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if number == 1:
                    self.assertTrue((log.parent / "latest.1.txt").exists())
                    self.assertEqual((log.parent / "latest.1.txt").read_text(), "existing run\n")
                    self.assertEqual((log.parent / "latest.1.jsonl").read_text(), "existing structured run\n")
            self.assertEqual(len(list(log.parent.glob("latest*.txt"))), 13)
            for age in range(12):
                path = log if age == 0 else log.parent / f"latest.{age}.txt"
                content = path.read_text()
                self.assertEqual(content.splitlines()[-1], f"RUN={12 - age}")
                self.assertEqual(content.count("[real] started:"), 1)
                self.assertEqual(path.with_suffix(".jsonl").read_text(), f"{12 - age}\n")
            self.assertEqual((log.parent / "latest.12.txt").read_text(), "existing run\n")
            self.assertEqual((log.parent / "latest.12.jsonl").read_text(), "existing structured run\n")
            self.assertEqual(len(list(log.parent.glob("latest*.jsonl"))), 13)

    def test_overwrite_stream_errors_and_interrupt_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "logs with spaces" / "latest.txt"
            stub = Path(directory) / "python"
            stub.write_text(
                f"#!{sys.executable}\n"
                "import os, signal, sys, time\n"
                "def stop(*args):\n"
                "    print('[cleanup] stub held', flush=True)\n"
                "    sys.exit(130)\n"
                "signal.signal(signal.SIGINT, stop)\n"
                "print('stub stdout', flush=True)\n"
                "print('stub stderr', file=sys.stderr, flush=True)\n"
                "if os.environ.get('WAIT_STUB') == '1':\n"
                "    while True: time.sleep(0.02)\n"
                "sys.exit(7)\n"
            )
            stub.chmod(0o755)
            env = dict(PATH=os.environ["PATH"], MODEL_PYTHON=str(stub),
                       CKPT_PATH=__file__, PRIOR_CKPT_PATH=__file__,
                       GUIDE_CKPT_PATH=__file__, CHECK_ONLY="1", RUN_LOG=str(log))
            for script in ("eval_para_obs66_real.sh", "xjz_eval_strong_prior_real.sh"):
                with self.subTest(script=script):
                    log.parent.mkdir(exist_ok=True)
                    log.write_text("old run\n")
                    command = ["bash", str(ROOT / "eval" / script), "--record"]
                    result = subprocess.run(command, env=env, capture_output=True,
                                            text=True, timeout=10)
                    self.assertEqual(result.returncode, 7)
                    content = log.read_text()
                    self.assertNotIn("old run", content)
                    self.assertIn("stub stdout", content)
                    self.assertIn("stub stderr", content)
                    self.assertIn("stub stdout", result.stdout)
                    if script.startswith("xjz"):
                        self.assertIn("[real-guidance]", content)
                    previous_mtime = log.stat().st_mtime_ns
                    process = subprocess.Popen(command, env=dict(env, WAIT_STUB="1"),
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                               text=True, start_new_session=True)
                    try:
                        # Wait for a fresh run's data to reach disk BEFORE it exits.
                        deadline = time.monotonic() + 5
                        while time.monotonic() < deadline:
                            try:
                                if log.stat().st_mtime_ns != previous_mtime and "stub stderr" in log.read_text():
                                    break
                            except FileNotFoundError:
                                pass  # Rotation can briefly move the previous latest file.
                            time.sleep(0.02)
                        else:
                            self.fail("running process did not stream output to log")
                        self.assertIsNone(process.poll())
                        os.killpg(process.pid, signal.SIGINT)
                        output, errors = process.communicate(timeout=5)
                        self.assertEqual(process.returncode, 130, errors)
                        self.assertIn("[cleanup] stub held", log.read_text())
                        self.assertIn("[cleanup] stub held", output)
                    finally:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.communicate()


if __name__ == "__main__":
    unittest.main()
