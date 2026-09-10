from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_xjz_guidance_test_runs_guide_by_exec_grid_and_skips_exec_beyond_guide(
    tmp_path,
):
    weak = tmp_path / "weak.ckpt"
    strong = tmp_path / "strong.ckpt"
    weak.touch()
    strong.touch()
    calls = tmp_path / "calls.txt"
    fake_xjz = tmp_path / "fake_xjz.sh"
    fake_xjz.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s|%s|%s|%s|%s|%s\\n' "
        "\"$GUIDANCE_STEPS\" \"$EXECUTION_STEPS\" \"$GUIDANCE_SCALE\" "
        "\"$INFERENCE_STEPS\" \"$GUIDE_INFERENCE_STEPS\" \"$RUN_NAME\" "
        ">> \"$CALLS_FILE\"\n",
        encoding="utf-8",
    )
    fake_xjz.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PRIOR_CKPT_PATH": str(strong),
            "GUIDE_CKPT_PATH": str(weak),
            "XJZ_TEST_SCRIPT": str(fake_xjz),
            "CALLS_FILE": str(calls),
            "RUN_DIR": str(tmp_path / "runs"),
            "MODEL_PYTHON": "true",
        }
    )
    subprocess.run(
        ["bash", str(ROOT / "eval" / "xjz_guidance_test.sh")],
        cwd=ROOT,
        env=env,
        check=True,
    )

    lines = calls.read_text(encoding="utf-8").splitlines()
    expected = []
    for scale in ("50", "100", "150"):
        for guide, exec_steps in (
            (2, 2),
            (4, 2),
            (4, 4),
            (6, 2),
            (6, 4),
            (6, 6),
            (8, 2),
            (8, 4),
            (8, 6),
            (8, 8),
        ):
            expected.append(
                f"{guide}|{exec_steps}|{scale}|4|4|scale{scale}_guide{guide}_exec{exec_steps}"
            )
    assert lines == expected


def test_xjz_eval_strong_prior_launcher_uses_strong_ckpt_as_prior(tmp_path):
    weak = tmp_path / "weak.ckpt"
    strong = tmp_path / "strong.ckpt"
    weak.touch()
    strong.touch()
    calls = tmp_path / "calls.txt"
    fake_xjz = tmp_path / "fake_xjz.sh"
    fake_xjz.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s|%s|%s|%s|%s|%s\\n' \"$GUIDANCE_SCALE\" \"$CKPT_PATH\" "
        "\"$GUIDE_CKPT_PATH\" \"$MODEL_SERVER\" "
        "\"$FIRST_EPISODE_ONLY\" \"$CENSOR_UNFINISHED_AT_CAP\" "
        ">> \"$CALLS_FILE\"\n",
        encoding="utf-8",
    )
    fake_xjz.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PRIOR_CKPT_PATH": str(strong),
            "GUIDE_CKPT_PATH": str(weak),
            "GUIDANCE_SCALES": "0,25",
            "XJZ_TEST_SCRIPT": str(fake_xjz),
            "CALLS_FILE": str(calls),
            "RUN_DIR": str(tmp_path / "runs"),
        }
    )
    subprocess.run(
        ["bash", str(ROOT / "eval" / "xjz_eval_strong_prior.sh")],
        cwd=ROOT,
        env=env,
        check=True,
    )

    lines = calls.read_text(encoding="utf-8").splitlines()
    server = str(ROOT / "eval" / "xjz_eval_strong_prior.py")
    assert lines == [
        f"0|{strong}|{weak}|{server}|1|1",
        f"25|{strong}|{weak}|{server}|1|1",
    ]
