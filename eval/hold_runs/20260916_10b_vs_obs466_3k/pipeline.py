from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
DEX = ROOT.parents[2]
HISTORICAL = ROOT.parent / "20260912_fresh_noise_3000_seed42_8_19_25"
PREDECESSOR = ROOT.parent / "20260916_10k_obs466_20_25hz_3k" / "state.json"
SEEDS = (42, 8, 19, 25)
SCALES = (0, 25)
N_ENVS = 3000
CAP_STEPS = 12000
CONTROL_HZ = 30
SOURCE_FILES = (
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


def write_state(status: str, **kwargs) -> None:
    payload = {"status": status, "updated_at": time.time(), **kwargs}
    temporary = ROOT / "state.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(ROOT / "state.json")
    print(json.dumps(payload), flush=True)


def gpu_idle() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_integrity(manifest: dict) -> None:
    for relative_path, expected in manifest["source_hashes"].items():
        path = DEX / relative_path
        actual = checkpoint_sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"evaluation source changed while queued: {path} "
                f"expected={expected} actual={actual}"
            )


def verify_checkpoint_integrity(manifest: dict) -> None:
    for name, expected in manifest["checkpoints"].items():
        path = Path(name)
        actual = checkpoint_sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"checkpoint changed while queued: {path} "
                f"expected={expected} actual={actual}"
            )


def read_rows(path: Path, expected: int, cap_steps: int = CAP_STEPS) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected or {row["env"] for row in rows} != set(range(expected)):
        raise RuntimeError(f"invalid environment coverage in {path}: {len(rows)}")
    if not all(row["episode"] == 0 for row in rows):
        raise RuntimeError(f"non-first episode found in {path}")
    if not all(row["reason"] in ("failure", "timeout", "success") for row in rows):
        raise RuntimeError(f"unexpected episode reason in {path}")
    if not all(0 < int(row["length"]) <= cap_steps for row in rows):
        raise RuntimeError(f"invalid episode length in {path}")
    if not all(int(row["length"]) == cap_steps for row in rows if row["reason"] == "timeout"):
        raise RuntimeError(f"invalid timeout length in {path}")
    return rows


def episode_path(checkpoint: str, seed: int, scale: int) -> Path:
    if checkpoint == "10B":
        arm = ROOT / f"seed{seed}_10B_scale{scale}"
        return arm / f"seed{seed}_10B_fresh_scale{scale}.jsonl"
    arm = HISTORICAL / f"seed{seed}_1B_scale{scale}"
    return arm / f"seed{seed}_1B_fresh_scale{scale}.jsonl"


def summarize_rows(rows: list[dict]) -> dict:
    lengths = np.asarray([int(row["length"]) for row in rows], dtype=np.float64)
    seconds = lengths / CONTROL_HZ
    return {
        "environments": len(rows),
        "failures": sum(row["reason"] == "failure" for row in rows),
        "successes": sum(row["reason"] == "success" for row in rows),
        "survived_cap": sum(
            row["reason"] == "timeout" and int(row["length"]) == CAP_STEPS
            for row in rows
        ),
        "survival_percent": 100.0 * float(np.mean(lengths >= CAP_STEPS)),
        "mean_capped_hold_seconds": float(seconds.mean()),
        "median_capped_hold_seconds": float(np.median(seconds)),
        "survival_at_seconds": {
            str(second): float(np.mean(seconds >= second))
            for second in (30, 60, 120, 240, 400)
        },
    }


def generate_comparison(require_all: bool = False) -> dict:
    results: dict[str, dict] = {}
    for checkpoint in ("obs_4-66", "10B_obs_4-66"):
        key = "obs" if checkpoint == "obs_4-66" else "10B"
        results[checkpoint] = {}
        for scale in SCALES:
            combined = []
            seed_results = {}
            for seed in SEEDS:
                path = episode_path(key, seed, scale)
                if not path.exists():
                    if require_all:
                        raise RuntimeError(f"missing result: {path}")
                    continue
                rows = read_rows(path, N_ENVS)
                combined.extend(rows)
                seed_results[str(seed)] = summarize_rows(rows)
            if seed_results:
                results[checkpoint][str(scale)] = {
                    "per_seed": seed_results,
                    "aggregate": summarize_rows(combined),
                }

    for checkpoint in results:
        if "0" in results[checkpoint] and "25" in results[checkpoint]:
            base = results[checkpoint]["0"]["aggregate"]
            guided = results[checkpoint]["25"]["aggregate"]
            results[checkpoint]["guidance_delta"] = {
                "survival_percentage_points": (
                    guided["survival_percent"] - base["survival_percent"]
                ),
                "mean_capped_hold_seconds": (
                    guided["mean_capped_hold_seconds"]
                    - base["mean_capped_hold_seconds"]
                ),
            }

    payload = {
        "protocol": {
            "seeds": SEEDS,
            "environments_per_seed": N_ENVS,
            "cap_steps": CAP_STEPS,
            "control_hz": CONTROL_HZ,
            "sampler": "DDIM4/4",
            "execute_steps": 2,
            "guidance_steps": 2,
            "guidance_scale": 25,
            "fixed_noise": False,
            "guide": str(DEX / "runs/sim_hand_10k_seed42/checkpoints/latest.ckpt"),
        },
        "results": results,
        "caveat": (
            "Historical protocol did not save complete initial-state arrays. "
            "Matching seeds, simulator settings, data indices, and evaluator are used, "
            "but bitwise initial-state equality cannot be checked retroactively."
        ),
    }
    (ROOT / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# 10B_obs_4-66 versus obs_4-66",
        "",
        "Four seeds, 3000 first episodes per seed, 400-second cap.",
        "",
        "| Checkpoint | 10k guidance | Episodes | 400-s survival | Mean capped hold (s) | Median capped hold (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in ("obs_4-66", "10B_obs_4-66"):
        for scale in SCALES:
            entry = results[checkpoint].get(str(scale))
            if entry is None:
                lines.append(f"| {checkpoint} | scale {scale} | pending | — | — | — |")
                continue
            aggregate = entry["aggregate"]
            lines.append(
                f"| {checkpoint} | scale {scale} | {aggregate['environments']} | "
                f"{aggregate['survival_percent']:.2f}% | "
                f"{aggregate['mean_capped_hold_seconds']:.3f} | "
                f"{aggregate['median_capped_hold_seconds']:.3f} |"
            )
    lines.extend(["", payload["caveat"], ""])
    (ROOT / "comparison.md").write_text("\n".join(lines))
    return payload


def run_arm(seed: int, scale: int, *, smoke: bool = False) -> None:
    verify_source_integrity(json.loads((ROOT / "manifest.json").read_text()))
    name = f"smoke_seed{seed}_scale{scale}" if smoke else f"seed{seed}_10B_scale{scale}"
    folder = ROOT / name
    if folder.exists():
        raise RuntimeError(f"refusing to overwrite {folder}")
    while not gpu_idle():
        write_state("waiting_gpu", seed=seed, scale=scale, smoke=smoke)
        time.sleep(30)
    folder.mkdir()
    env = dict(
        os.environ,
        ARM_DIR=str(folder),
        SEED_VALUE=str(seed),
        SCALE_VALUE=str(scale),
        NUM_ENV_VALUE="4" if smoke else str(N_ENVS),
        MAX_STEPS_VALUE="60" if smoke else str(CAP_STEPS),
        PYTHONDONTWRITEBYTECODE="1",
    )
    log = folder / "console.log"
    with log.open("w") as output:
        child = subprocess.Popen(
            ["bash", str(ROOT / "run_arm.sh")],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        write_state("running", seed=seed, scale=scale, smoke=smoke, pid=child.pid, log=str(log))
        return_code = child.wait()
    if return_code not in (0, 139):
        raise RuntimeError(f"{name} exited with {return_code}")
    path = folder / f"seed{seed}_10B_fresh_scale{scale}.jsonl"
    rows = read_rows(
        path,
        4 if smoke else N_ENVS,
        60 if smoke else CAP_STEPS,
    )
    write_state(
        "arm_complete",
        seed=seed,
        scale=scale,
        smoke=smoke,
        exit_code=return_code,
        summary=summarize_rows(rows),
    )


def wait_for_predecessor() -> None:
    while True:
        if not PREDECESSOR.exists():
            write_state("waiting_predecessor", predecessor=str(PREDECESSOR))
            time.sleep(45)
            continue
        previous = json.loads(PREDECESSOR.read_text())
        if previous.get("status") == "complete":
            return
        if previous.get("status") == "failed":
            raise RuntimeError("frequency predecessor failed; refusing to start comparison")
        write_state("waiting_predecessor", predecessor=str(PREDECESSOR))
        time.sleep(45)


if __name__ == "__main__":
    try:
        if (ROOT / "state.json").exists():
            raise RuntimeError("existing state; refusing duplicate pipeline")
        manifest = {
            "checkpoints": {
                str(Path("/home/carus/data_usb/obs_4-66.ckpt")): "ad0bf60d9fe743161916c55fee36a1b6b13840757baaca2136dcfc3b5b597302",
                str(Path("/home/carus/data_usb/10B_obs_4-66.ckpt")): "da56ef248a4ccb7755619630590c26c1f19f1c42c9c0cc541116ecd488b81d3a",
                str(DEX / "runs/sim_hand_10k_seed42/checkpoints/latest.ckpt"): "ebefb9b170575f41695bb9e17fec2f0602bd5be403a447304f7e6ba36b20c031",
            },
            "source_hashes": {
                str(path.relative_to(DEX)): checkpoint_sha256(path)
                for path in SOURCE_FILES
            },
        }
        (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        generate_comparison(require_all=False)
        wait_for_predecessor()
        verify_checkpoint_integrity(manifest)
        verify_source_integrity(manifest)
        run_arm(42, 0, smoke=True)
        run_arm(42, 25, smoke=True)
        for seed in SEEDS:
            for scale in SCALES:
                run_arm(seed, scale)
                generate_comparison(require_all=False)
        comparison = generate_comparison(require_all=True)
        write_state("complete", comparison=str(ROOT / "comparison.md"), results=comparison["results"])
    except Exception as exc:
        write_state("failed", error=repr(exc))
        raise
