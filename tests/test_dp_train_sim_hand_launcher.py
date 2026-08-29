from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "dp_train_sim_hand.sh"


@pytest.mark.parametrize(
    ("num_gpus", "expected_prefix"),
    [
        (1, ["train.py"]),
        (
            4,
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=4",
                "train.py",
            ],
        ),
    ],
)
def test_launcher_selects_single_or_multi_gpu_command(
    tmp_path,
    num_gpus,
    expected_prefix,
):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.json").write_text("{}", encoding="utf-8")

    capture_path = tmp_path / "argv.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$@" > "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "CAPTURE_PATH": str(capture_path),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "DATASET_PATH": str(dataset_dir),
            "NUM_GPUS": str(num_gpus),
            "PYTHON": str(fake_python),
            "WANDB_API_KEY": "test-only",
            "WANDB_MODE": "disabled",
        }
    )
    subprocess.run(
        [str(LAUNCHER), "training.num_epochs=2"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    argv = capture_path.read_text(encoding="utf-8").splitlines()
    assert argv[: len(expected_prefix)] == expected_prefix
    assert "task.dataset_path=" + str(dataset_dir) in argv
    assert "training.num_epochs=2" in argv
    if num_gpus > 1:
        assert "hydra/job_logging=disabled" in argv
        assert "hydra.output_subdir=null" in argv
    else:
        assert "hydra/job_logging=disabled" not in argv
        assert "hydra.output_subdir=null" not in argv


def test_launcher_exposes_requested_gpus_when_visibility_is_unset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.json").write_text("{}", encoding="utf-8")

    capture_path = tmp_path / "visible-devices.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\n\' "$CUDA_VISIBLE_DEVICES" > "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(
        {
            "CAPTURE_PATH": str(capture_path),
            "DATASET_PATH": str(dataset_dir),
            "NUM_GPUS": "4",
            "PYTHON": str(fake_python),
            "WANDB_API_KEY": "test-only",
            "WANDB_MODE": "disabled",
        }
    )
    subprocess.run([str(LAUNCHER)], cwd=REPO_ROOT, env=env, check=True)

    assert capture_path.read_text(encoding="utf-8").strip() == "0,1,2,3"


@pytest.mark.parametrize("num_gpus", ["0", "-1", "many"])
def test_launcher_rejects_invalid_gpu_count(tmp_path, num_gpus):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.json").write_text("{}", encoding="utf-8")
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "DATASET_PATH": str(dataset_dir),
            "NUM_GPUS": num_gpus,
            "PYTHON": str(fake_python),
            "WANDB_API_KEY": "test-only",
            "WANDB_MODE": "disabled",
        }
    )
    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "NUM_GPUS must be a positive integer" in result.stderr
