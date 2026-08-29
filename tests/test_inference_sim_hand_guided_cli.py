from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FakeCheckReport:
    real_action_shape: tuple[int, ...] = (1, 50, 31)
    real_hand_reference_shape: tuple[int, ...] = (1, 54, 22)
    segment_count: int = 10
    sim_horizon: int = 12
    sim_obs_steps: int = 4
    sim_pred_action_steps: int = 9
    guidance_slice: tuple[int, int] = (3, 12)
    execution_slice: tuple[int, int] = (3, 8)
    timesteps: tuple[int, ...] = (88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0)
    max_x0_error: float = 0.0
    max_prev_error: float = 0.0


@dataclass(frozen=True)
class FakeDryRunReport:
    check: FakeCheckReport
    guided: SimpleNamespace


def _base_argv(*, mode: str = "check") -> list[str]:
    return [
        "--mode",
        mode,
        "--real-checkpoint",
        "/tmp/fake_real.ckpt",
        "--sim-checkpoint",
        "/tmp/fake_sim.ckpt",
        "--device",
        "cpu",
    ]


def test_parse_args_defaults_and_required():
    from inference_sim_hand_guided import parse_args

    args = parse_args(_base_argv())
    assert args.mode == "check"
    assert args.execution_mode == "closed-loop-5"
    assert args.execution_steps == 5
    assert args.guidance_scale == 1.0
    assert args.num_inference_steps == 12
    assert args.eta == 0.0
    assert args.seed == 0
    assert args.device == "cpu"


def test_parse_args_open_loop_mode_executes_fifty_steps():
    from inference_sim_hand_guided import parse_args

    args = parse_args(
        _base_argv() + ["--execution-mode", "open-loop-50"]
    )

    assert args.execution_mode == "open-loop-50"
    assert args.execution_steps == 50


def test_main_check_dispatches_and_prints_report(capsys):
    from inference_sim_hand_guided import main

    loaded = MagicMock(name="loaded")
    report = FakeCheckReport()
    with (
        patch("inference_sim_hand_guided.load_guided_policies", return_value=loaded) as load,
        patch("inference_sim_hand_guided.run_check", return_value=report) as check,
        patch("inference_sim_hand_guided.run_dry_run") as dry_run,
    ):
        code = main(_base_argv(mode="check") + ["--seed", "7", "--guidance-scale", "0.0"])

    assert code == 0
    load.assert_called_once()
    check.assert_called_once_with(loaded, seed=7)
    dry_run.assert_not_called()
    out = capsys.readouterr().out
    assert "real_action_shape=(1, 50, 31)" in out
    assert "real_hand_reference_shape=(1, 54, 22)" in out
    assert "segment_count=10" in out
    assert "sim_horizon=12" in out
    assert "sim_obs_steps=4" in out
    assert "sim_pred_action_steps=9" in out
    assert "guidance_slice=(3, 12)" in out
    assert "execution_slice=(3, 8)" in out
    assert "timesteps=(88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0)" in out


def test_main_open_loop_reports_mode_and_passes_fifty_step_config(capsys):
    from inference_sim_hand_guided import main

    loaded = MagicMock(name="loaded")
    report = FakeCheckReport(
        real_hand_reference_shape=(1, 68, 22),
        segment_count=1,
        sim_horizon=68,
        sim_pred_action_steps=65,
        guidance_slice=(3, 68),
        execution_slice=(3, 53),
    )
    with (
        patch(
            "inference_sim_hand_guided.load_guided_policies",
            return_value=loaded,
        ) as load,
        patch("inference_sim_hand_guided.run_check", return_value=report),
    ):
        code = main(
            _base_argv()
            + ["--execution-mode", "open-loop-50"]
        )

    assert code == 0
    guidance_config = load.call_args.args[3]
    assert guidance_config.execution_steps == 50
    out = capsys.readouterr().out
    assert "execution_mode=open-loop-50" in out
    assert "execution_steps=50" in out
    assert "segment_count=1" in out


def test_main_dry_run_dispatches_and_prints_report(capsys):
    from inference_sim_hand_guided import main

    loaded = MagicMock(name="loaded")
    report = FakeDryRunReport(
        check=FakeCheckReport(),
        guided=SimpleNamespace(records=tuple(range(10))),
    )
    with (
        patch("inference_sim_hand_guided.load_guided_policies", return_value=loaded) as load,
        patch("inference_sim_hand_guided.run_check") as check,
        patch("inference_sim_hand_guided.run_dry_run", return_value=report) as dry_run,
    ):
        code = main(_base_argv(mode="dry-run") + ["--seed", "3"])

    assert code == 0
    load.assert_called_once()
    dry_run.assert_called_once_with(loaded, seed=3)
    check.assert_not_called()
    out = capsys.readouterr().out
    assert "real_action_shape=(1, 50, 31)" in out
    assert "segment_count=10" in out
    assert "completed_segments=10" in out


def test_missing_checkpoint_returns_1(capsys):
    from inference_sim_hand_guided import main

    with patch(
        "inference_sim_hand_guided.load_guided_policies",
        side_effect=FileNotFoundError("Checkpoint not found: /tmp/fake_real.ckpt"),
    ):
        code = main(_base_argv())

    assert code == 1
    err = capsys.readouterr().err
    assert "Checkpoint not found" in err


def test_negative_guidance_scale_returns_1(capsys):
    from inference_sim_hand_guided import main

    code = main(_base_argv() + ["--guidance-scale", "-0.1"])
    assert code == 1
    assert "guidance_scale" in capsys.readouterr().err


def test_non_positive_execution_steps_returns_1(capsys):
    from inference_sim_hand_guided import main

    code = main(_base_argv() + ["--execution-steps", "0"])
    assert code == 1
    assert "execution_steps" in capsys.readouterr().err


def test_execution_mode_rejects_conflicting_execution_steps_before_load(capsys):
    from inference_sim_hand_guided import main

    with patch("inference_sim_hand_guided.load_guided_policies") as load:
        code = main(
            _base_argv()
            + [
                "--execution-mode",
                "open-loop-50",
                "--execution-steps",
                "5",
            ]
        )

    assert code == 1
    assert "open-loop-50" in capsys.readouterr().err
    load.assert_not_called()


def test_non_positive_inference_steps_returns_1(capsys):
    from inference_sim_hand_guided import main

    code = main(_base_argv() + ["--num-inference-steps", "0"])
    assert code == 1
    assert "num_inference_steps" in capsys.readouterr().err


def test_nonzero_eta_returns_1(capsys):
    from inference_sim_hand_guided import main

    code = main(_base_argv() + ["--eta", "0.1"])
    assert code == 1
    assert "eta" in capsys.readouterr().err


def test_unavailable_cuda_returns_1(capsys):
    from inference_sim_hand_guided import main

    with (
        patch("inference_sim_hand_guided.torch.cuda.is_available", return_value=False),
        patch("inference_sim_hand_guided.load_guided_policies") as load,
    ):
        code = main(_base_argv() + ["--device", "cuda:0"])

    assert code == 1
    assert "CUDA" in capsys.readouterr().err
    load.assert_not_called()


def test_runtime_failure_returns_1(capsys):
    from inference_sim_hand_guided import main

    with (
        patch("inference_sim_hand_guided.load_guided_policies", return_value=MagicMock()),
        patch(
            "inference_sim_hand_guided.run_check",
            side_effect=RuntimeError("oracle exploded"),
        ),
    ):
        code = main(_base_argv(mode="check"))

    assert code == 1
    assert "oracle exploded" in capsys.readouterr().err


def test_assertion_error_returns_1(capsys):
    from inference_sim_hand_guided import main

    with (
        patch("inference_sim_hand_guided.load_guided_policies", return_value=MagicMock()),
        patch(
            "inference_sim_hand_guided.run_check",
            side_effect=AssertionError("shape mismatch"),
        ),
    ):
        code = main(_base_argv(mode="check"))

    assert code == 1
    assert "shape mismatch" in capsys.readouterr().err


def test_import_and_help_have_no_hardware_side_effects():
    code = """
import sys
import inference_sim_hand_guided
assert "inference_dp" not in sys.modules
assert "direct_robot_env" not in sys.modules
assert "diffusion_policy.real_world" not in sys.modules
assert "multi_realsense" not in sys.modules
assert "pyrealsense2" not in sys.modules
rc = inference_sim_hand_guided.main(["--help"])
raise SystemExit(rc)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout.lower() or "usage:" in completed.stderr.lower()
    joined = completed.stdout + completed.stderr
    assert "dino" not in joined.lower()
    assert "DINOv2" not in joined


def test_cli_does_not_expose_dino_options():
    from inference_sim_hand_guided import parse_args

    with pytest.raises(SystemExit):
        parse_args(_base_argv() + ["--dino-env", "x"])
