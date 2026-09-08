from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_xjz_guidance_launcher_runs_paired_scales_with_same_checkpoints(tmp_path):
    weak = tmp_path / "weak.ckpt"
    strong = tmp_path / "strong.ckpt"
    weak.touch()
    strong.touch()
    calls = tmp_path / "calls.txt"
    fake_xjz = tmp_path / "fake_xjz.sh"
    fake_xjz.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s|%s|%s|%s\\n' \"$GUIDANCE_SCALE\" \"$CKPT_PATH\" "
        "\"$GUIDE_CKPT_PATH\" \"$MODEL_SERVER\" >> \"$CALLS_FILE\"\n",
        encoding="utf-8",
    )
    fake_xjz.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "WEAK_CKPT_PATH": str(weak),
            "GUIDE_CKPT_PATH": str(strong),
            "GUIDANCE_SCALES": "0,100",
            "XJZ_TEST_SCRIPT": str(fake_xjz),
            "CALLS_FILE": str(calls),
            "RUN_DIR": str(tmp_path / "runs"),
        }
    )
    subprocess.run(
        ["bash", str(ROOT / "eval" / "xjz_guidance_test.sh")],
        cwd=ROOT,
        env=env,
        check=True,
    )

    lines = calls.read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"0|{weak}|{strong}|{ROOT / 'eval' / 'guided_model_server.py'}",
        f"100|{weak}|{strong}|{ROOT / 'eval' / 'guided_model_server.py'}",
    ]
