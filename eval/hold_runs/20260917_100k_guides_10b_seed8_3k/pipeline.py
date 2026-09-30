from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
DEX = ROOT.parents[2]
RUN = ROOT / "seed8_10B_100k_scale25"
EPISODES = RUN / "seed8_10B_100k_scale25.jsonl"
OLD_100K = DEX / "eval/hold_runs/20260914_serial_data_scaling_guide_scale_seed8_3k/serial/100k/seed8/episodes.jsonl"
OLD_10K_10B = DEX / "eval/hold_runs/20260916_10b_vs_obs466_3k/seed8_10B_scale25/seed8_10B_fresh_scale25.jsonl"
OLD_BASE_10B = DEX / "eval/hold_runs/20260916_10b_vs_obs466_3k/seed8_10B_scale0/seed8_10B_fresh_scale0.jsonl"
CHECKPOINTS = (
    Path("/home/carus/data_usb/10B_obs_4-66.ckpt"),
    DEX / "runs/sim_hand_100k_ownnorm_seed42_equal_epochs/checkpoints/latest.ckpt",
)
SOURCES = (
    DEX / "eval/xjz_eval_strong_prior.sh",
    DEX / "eval/xjz_test.sh",
    DEX / "eval/eval.sh",
    DEX / "eval/guided_pair_policy.py",
    DEX / "eval/inference_dp_controller.py",
    DEX / "eval/sim_eval.py",
    DEX / "eval/episode_stats.py",
    DEX / "eval/checkpoint_loader.py",
    DEX / "eval/model_server.py",
    DEX / "diffusion_policy/guidance/guided_ddim.py",
)


def state(status, **kwargs):
    data = {"status": status, "updated_at": time.time(), **kwargs}
    temporary = ROOT / "state.tmp"
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(ROOT / "state.json")
    print(json.dumps(data), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 3000 or {row["env"] for row in rows} != set(range(3000)):
        raise RuntimeError(f"incomplete environment coverage: {path}")
    if not all(row["episode"] == 0 and row["reason"] in ("failure", "timeout", "success") for row in rows):
        raise RuntimeError(f"invalid first-episode records: {path}")
    lengths = np.asarray([row["length"] for row in rows], dtype=np.float64)
    if not np.all((lengths > 0) & (lengths <= 12000)):
        raise RuntimeError(f"invalid episode lengths: {path}")
    if not all(row["length"] == 12000 for row in rows if row["reason"] == "timeout"):
        raise RuntimeError(f"invalid timeout records: {path}")
    seconds = lengths / 30
    return {
        "episodes": 3000,
        "survived_400s": int(np.sum(lengths >= 12000)),
        "survival_percent": float(100 * np.mean(lengths >= 12000)),
        "mean_capped_hold_seconds": float(seconds.mean()),
        "median_capped_hold_seconds": float(np.median(seconds)),
    }


def main():
    if (ROOT / "state.json").exists() or RUN.exists():
        raise RuntimeError("existing run state; refusing overwrite")
    paths = CHECKPOINTS + SOURCES
    manifest = {str(path): sha256(path) for path in paths}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    previous = {
        "obs_4-66 + 100k guide": summarize(OLD_100K),
        "10B baseline": summarize(OLD_BASE_10B),
        "10B + 10k guide": summarize(OLD_10K_10B),
    }
    (ROOT / "previous.json").write_text(json.dumps(previous, indent=2) + "\n")
    while True:
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        )
        if not processes.stdout.strip():
            break
        state("waiting_gpu")
        time.sleep(30)
    for name, expected in manifest.items():
        if sha256(Path(name)) != expected:
            raise RuntimeError(f"input changed while waiting: {name}")
    RUN.mkdir()
    with (RUN / "console.log").open("w") as output:
        process = subprocess.Popen(["bash", str(ROOT / "run.sh")], stdout=output, stderr=subprocess.STDOUT)
        state("running", pid=process.pid, log=str(RUN / "console.log"))
        return_code = process.wait()
    if return_code not in (0, 139):
        raise RuntimeError(f"evaluation exited with {return_code}")
    result = summarize(EPISODES)
    comparison = {"previous": previous, "10B + 100k guide": result,
                  "caveat": "Single seed 8; historical initial states not bitwise-verifiable."}
    (ROOT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    lines = ["# 100k guide → 10B_obs_4-66", "",
             "Seed 8, 3000 first episodes, 400-second cap; guide2 / scale25 / DDIM4 / execute2.", "",
             "| Model | 400-s survival | Mean capped hold (s) | Median capped hold (s) |",
             "|---|---:|---:|---:|"]
    for label, metrics in list(previous.items()) + [("10B + 100k guide", result)]:
        lines.append(f"| {label} | {metrics['survival_percent']:.2f}% | "
                     f"{metrics['mean_capped_hold_seconds']:.2f} | "
                     f"{metrics['median_capped_hold_seconds']:.2f} |")
    lines.extend(["", comparison["caveat"], ""])
    (ROOT / "comparison.md").write_text("\n".join(lines))
    state("complete", comparison=str(ROOT / "comparison.md"), result=result)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        state("failed", error=repr(exc))
        raise
