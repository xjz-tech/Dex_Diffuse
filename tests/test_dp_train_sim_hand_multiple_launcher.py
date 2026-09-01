from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "dp_train_sim_hand_multiple.sh"
CACHE_CLI = REPO_ROOT / "build_sim_hand_multiple_cache.py"


def _nested_data_root(tmp_path: Path) -> Path:
    root = tmp_path / "sim_data"
    rollout = root / "brush" / "20260831204042"
    rollout.mkdir(parents=True)
    (rollout / "manifest.json").write_text("{}", encoding="utf-8")
    return root


def _fake_python(tmp_path: Path, capture: Path) -> Path:
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1##*/}\" == \"build_sim_hand_multiple_cache.py\" ]]; then\n"
        "  printf '%s\\n' \"$@\" > \"$CACHE_CAPTURE_PATH\"\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return fake_python


@pytest.mark.parametrize("num_gpus", [1, 4])
def test_multiple_launcher_selects_lazy_non_mixed_mode(tmp_path, num_gpus):
    root = _nested_data_root(tmp_path)
    capture = tmp_path / "argv.txt"
    cache_capture = tmp_path / "cache_argv.txt"
    fake_python = _fake_python(tmp_path, capture)
    cache_path = tmp_path / "cache"
    env = os.environ.copy()
    env.update({
        "CAPTURE_PATH": str(capture),
        "CACHE_CAPTURE_PATH": str(cache_capture),
        "DATASET_PATH": str(root),
        "CACHE_PATH": str(cache_path),
        "NUM_GPUS": str(num_gpus),
        "PYTHON": str(fake_python),
        "WANDB_MODE": "disabled",
        "WANDB_API_KEY": "test-only",
    })

    subprocess.run(
        [str(LAUNCHER), "training.num_epochs=2"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    cache_argv = cache_capture.read_text(encoding="utf-8").splitlines()
    assert str(CACHE_CLI) in cache_argv or cache_argv[0].endswith(
        "build_sim_hand_multiple_cache.py"
    )
    assert "--dataset-path" in cache_argv
    assert str(root) in cache_argv
    assert str(cache_path) in cache_argv

    argv = capture.read_text(encoding="utf-8").splitlines()
    assert "task=sim_hand_multiple" in argv
    assert "task.dataset_path=" + str(root) in argv
    assert "task.cache_path=" + str(cache_path) in argv
    assert "exp_name=multiple" in argv
    assert "logging.mode=disabled" in argv
    assert "training.num_epochs=2" in argv
    if num_gpus == 4:
        assert "--nproc_per_node=4" in argv


def test_multiple_launcher_rejects_root_without_nested_manifests(tmp_path):
    root = tmp_path / "empty_sim_data"
    root.mkdir()
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "DATASET_PATH": str(root),
        "PYTHON": str(fake_python),
        "WANDB_API_KEY": "test-only",
    })

    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "No manifest.json found below DATASET_PATH" in result.stderr
