from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "dp_train_sim_hand_mixed.sh"


def _data_root(tmp_path):
    root = tmp_path / "Data"
    (root / "exp_data").mkdir(parents=True)
    (root / "exp_data" / "manifest.json").write_text("{}", encoding="utf-8")
    episode = root / "bulb_tac_80" / "episode_0"
    episode.mkdir(parents=True)
    return root


@pytest.mark.parametrize("num_gpus", [1, 4])
def test_mixed_launcher_labels_mode_and_passes_data_root(tmp_path, num_gpus):
    root = _data_root(tmp_path)
    capture = tmp_path / "argv.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n" 'printf \'%s\\n\' "$@" > "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "CAPTURE_PATH": str(capture),
        "DATA_ROOT": str(root),
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
    argv = capture.read_text(encoding="utf-8").splitlines()
    assert "task=sim_hand_mixed" in argv
    assert "task.data_root=" + str(root) in argv
    assert "exp_name=mixed" in argv
    assert "logging.mode=disabled" in argv
    assert "training.num_epochs=2" in argv
    if num_gpus == 4:
        assert "--nproc_per_node=4" in argv
