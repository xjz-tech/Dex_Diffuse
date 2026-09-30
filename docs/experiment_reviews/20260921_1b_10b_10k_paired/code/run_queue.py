#!/usr/bin/env python3
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/carus/Program/Dexterous_Manipulation/Dex_diffuse")
EXP = ROOT / "docs/experiment_reviews/20260921_1b_10b_10k_paired"
RUNNER = EXP / "code/run_arm.sh"
SEEDS = (42, 8, 19, 25)
MODELS = ("1b", "10b")
NUM_ENVS = 2500
MAX_STEPS = 12000


def timestamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def read_records(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate(output):
    episodes = output / "episodes.jsonl"
    initial = output / "initial_state.npz"
    if not episodes.is_file() or not initial.is_file():
        return False, "missing episodes.jsonl or initial_state.npz"
    records = read_records(episodes)
    envs = [row["env"] for row in records]
    if len(records) != NUM_ENVS or sorted(envs) != list(range(NUM_ENVS)):
        return False, "expected one first-episode record for every environment"
    if any(row["length"] <= 0 or row["length"] > MAX_STEPS for row in records):
        return False, "invalid episode length"
    return True, "complete"


def write_status(payload):
    temporary = EXP / "status.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, EXP / "status.json")


def main():
    queue = [(seed, model) for seed in SEEDS for model in MODELS]
    status = {
        "state": "running",
        "started_at": timestamp(),
        "protocol": "native xjz; ordinary DDIM4; exec2; first episode; 400 s cap",
        "num_paired_initial_states": len(SEEDS) * NUM_ENVS,
        "seeds": list(SEEDS),
        "num_envs_per_seed": NUM_ENVS,
        "completed": [],
    }
    write_status(status)
    for seed, model in queue:
        name = f"seed{seed}_{model}"
        output = EXP / "runs" / name
        output.mkdir(parents=True, exist_ok=True)
        valid, detail = validate(output) if output.exists() else (False, "new")
        if valid:
            status["completed"].append({"run": name, "reused": True})
            write_status(status)
            continue
        for stale in (output / "episodes.jsonl", output / "initial_state.npz"):
            stale.unlink(missing_ok=True)
        status["current"] = name
        status["current_started_at"] = timestamp()
        write_status(status)
        with (output / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                [str(RUNNER), model, str(seed), str(NUM_ENVS), str(MAX_STEPS), str(output)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        valid, detail = validate(output)
        if not valid:
            status.update(
                state="failed",
                failed_run=name,
                return_code=process.returncode,
                failure=detail,
                finished_at=timestamp(),
            )
            write_status(status)
            return 1
        status["completed"].append(
            {"run": name, "return_code": process.returncode, "finished_at": timestamp()}
        )
        status.pop("current", None)
        status.pop("current_started_at", None)
        write_status(status)
    status.update(state="runs_complete", finished_at=timestamp())
    write_status(status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
