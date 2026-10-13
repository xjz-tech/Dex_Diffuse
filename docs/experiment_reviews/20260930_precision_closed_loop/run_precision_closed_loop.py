"""Paired 10B FP32/fused versus TensorRT-FP16/fused bulb-turn rollouts."""

import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "docs/experiment_reviews/20260924_object_state_data"
sys.path.insert(0, str(SOURCE))

from run_h8_cross_model_comparison import C, EPISODES, PHYSICS  # noqa: E402

SEEDS = json.loads((HERE.parent / "20260930_h8_object_state_random10/manifest.json").read_text())["prior_noise_seeds"]
CHECKPOINT = Path("/home/carus/data_usb/aggresive_random_ckpt/10B_obs_4-66.ckpt")
METHODS = ("guidance4", "edit015")
PRECISIONS = ("fp32", "fp16")
FP32_BASELINE = HERE.parent / "20260930_h8_object_state_random10"


def destination(precision, seed, physics, episode, method):
    return HERE / precision / f"seed{seed}" / "10b" / physics / f"episode_{episode:02d}" / method


def run_group(group, pilot):
    precision, episode, method = group
    jobs = [(seed, physics) for seed in SEEDS for physics in PHYSICS]
    if pilot:
        jobs = [(SEEDS[0], next(iter(PHYSICS)))]
    jobs = [(seed, physics) for seed, physics in jobs
            if not ((destination(precision, seed, physics, episode, method) / "summary.json").exists()
                    and (destination(precision, seed, physics, episode, method) / "predictions.json").exists())]
    if not jobs:
        return dict(group=group, completed=0, skipped=True)
    sock = Path(f"/tmp/precision_{os.getpid()}_{precision}_{episode}_{method}.sock")
    sock.unlink(missing_ok=True)
    server_log = HERE / "servers" / f"{precision}_ep{episode}_{method}.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    server_env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                      LD_LIBRARY_PATH="/home/carus/miniforge3/envs/dp/lib")
    sim_env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                   PATH="/home/carus/miniforge3/envs/decv2/bin:" + os.environ["PATH"],
                   LD_LIBRARY_PATH="/home/carus/miniforge3/envs/decv2/lib")
    sim_env["PYTHONPATH"] = ":".join(("/home/carus/opt/isaacgym/python",
        "/home/carus/Program/dex-controller", str(ROOT / "eval")))
    ref = C / f"episode_{episode:02d}" / "reference_full.npz"
    command = ["/home/carus/miniforge3/envs/dp/bin/python", str(HERE / "precision_server.py"),
               "--socket", str(sock), "--checkpoint", str(CHECKPOINT),
               "--reference", str(ref), "--method", method,
               "--precision", precision, "--sessions", str(len(jobs))]
    server = None
    try:
        with server_log.open("a") as log:
            server = subprocess.Popen(command, env=server_env, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(300):
            if sock.exists():
                break
            if server.poll() is not None:
                raise RuntimeError(f"server exited early; see {server_log}")
            time.sleep(.2)
        else:
            raise TimeoutError(f"server not ready: {server_log}")
        for seed, physics in jobs:
            mass, friction = PHYSICS[physics]
            out = destination(precision, seed, physics, episode, method)
            out.mkdir(parents=True, exist_ok=True)
            steps = 9 if method == "edit015" else 4
            scale = 0 if method == "edit015" else 50
            sim = ["/home/carus/miniforge3/envs/decv2/bin/python",
                str(SOURCE / "compare_reference_edit_rollouts.py"),
                "--mode", "guided", "--out", str(out), "--reference", str(ref),
                "--reference-id", "0", "--source-episode", str(episode),
                "--front-only", "--audit-recording", "--grasp-evidence", "--no-video",
                "--object-mass-kg", str(mass), "--friction", str(friction),
                "--guidance-steps", str(steps), "--guidance-scale", str(scale),
                "--execution-steps", "2", "--prior-noise-seed", str(seed),
                "--reference-interpolation-threshold", ".1", "--socket", str(sock)]
            with (out / "sim.log").open("w") as log:
                subprocess.run(sim, env=sim_env, stdout=log, stderr=subprocess.STDOUT, check=True)
            for _ in range(100):
                if (out / "predictions.json").exists():
                    break
                time.sleep(.1)
            else:
                raise TimeoutError(f"predictions were not saved: {out}")
            print("DONE", group, seed, physics, flush=True)
        if server.wait(timeout=30) != 0:
            raise RuntimeError(f"server exited nonzero: {server_log}")
        return dict(group=group, completed=len(jobs), skipped=False)
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            server.wait(timeout=30)
        sock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / "batch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = dict(checkpoint=str(CHECKPOINT), prior_noise_seeds=SEEDS,
            environment_seed=42, episodes=EPISODES, physics=PHYSICS,
            precisions=PRECISIONS, methods=METHODS,
            fp32_baseline=str(FP32_BASELINE),
            fp32_reuse_validation="pilot seed486266, ep76, 44g/mu1.1: full emitted action sequences exactly equal for guidance4 and edit015",
            edit=dict(noise_ratio=.15, ddim_steps=4, exec_steps=2),
            guidance=dict(scale=50, future_reference_steps=4, ddim_steps=4, exec_steps=2),
            reference_interpolation_threshold_rad=.1,
            initial_state="per-episode qualified lateral grasp; fixed wrist",
            settle_steps=60, terminal_hold_steps=60,
            unet=dict(fp32="PyTorch FP32", fp16="TensorRT FP16 with FP32 inputs and outputs"),
            ddim="same GPU precomputed-coefficient arithmetic, FP32 for both",
            native_protocol="eval/xjz_test.sh thresholds retained; geometry analyzed separately",
            video=False, workers=args.workers)
        (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        episodes = (EPISODES[0],) if args.pilot else EPISODES
        run_precisions = PRECISIONS if args.pilot else ("fp16",)
        groups = [(p, ep, method) for p in run_precisions for ep in episodes for method in METHODS]
        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_group, group, args.pilot): group for group in groups}
            for future in concurrent.futures.as_completed(futures):
                try:
                    print("GROUP", future.result(), flush=True)
                except Exception as exc:
                    failures.append((futures[future], repr(exc)))
                    print("FAILED", futures[future], repr(exc), flush=True)
        status = dict(pilot=args.pilot, groups=len(groups), failures=failures)
        (HERE / ("pilot_status.json" if args.pilot else "batch_status.json")).write_text(
            json.dumps(status, indent=2) + "\n")
        if failures:
            raise RuntimeError(f"{len(failures)} server groups failed")
        print("PILOT COMPLETE" if args.pilot else "BATCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
