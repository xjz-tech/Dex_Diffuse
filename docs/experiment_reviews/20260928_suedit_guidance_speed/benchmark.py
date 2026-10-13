"""Paired, warmed inference latency for reference editing and gradient guidance.

Run with the dp environment after other GPU compute jobs have finished.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "docs/experiment_reviews/20260924_object_state_data"
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(EXPERIMENT))

from reference_action_editor import ReferenceActionEditor
from policy_observation import compose_policy_observation


def measured_input(controller):
    case = EXPERIMENT / "reference_turn_baseline_20260926"
    trace_path = case / "episode54_reference_edit_20260928/edit020_seed44/trace.json"
    ref_path = case / "qualified_comparison/episode_54/reference_full.npz"
    trace = json.loads(trace_path.read_text())
    settled = [row for row in trace if row["phase"] == "settle"][-4:]
    q = np.asarray([row["q"] for row in settled], dtype=np.float32)
    target = np.asarray([row["executed_target"] for row in settled], dtype=np.float32)
    history = compose_policy_observation(q, target, controller.observation_mode)[None]
    future = np.load(ref_path, allow_pickle=False)["hand_target_rad"][:, :9]
    assert history.shape == (1, 4, 66) and future.shape == (1, 9, 22)
    return history, future, str(trace_path), str(ref_path)


def describe(values):
    ms = np.asarray(values, dtype=np.float64) * 1000
    return {"median_ms": float(np.median(ms)), "mean_ms": float(ms.mean()),
            "p95_ms": float(np.percentile(ms, 95)), "std_ms": float(ms.std()),
            "min_ms": float(ms.min()), "max_ms": float(ms.max()), "samples_ms": ms.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.repeats > 0 and args.warmup >= 0
    torch.set_num_threads(2)
    checkpoint = Path("/home/carus/data_usb/10B_obs_4-66.ckpt")
    editor = ReferenceActionEditor(checkpoint, 0.2, steps=4, execution_steps=2)
    controller = editor.controller
    history, future, trace_path, ref_path = measured_input(controller)
    seed = 44

    def editing():
        return editor.predict(history, future, [seed])

    def guidance(guide_steps, scale):
        controller.set_guidance_horizon(guide_steps)
        controller.guidance_scale = scale
        controller.set_fixed_noise_from_seeds([seed])
        return controller.predict(history, future[:, :guide_steps])

    methods = {
        "sdedit_ratio020_ddim4": editing,
        "guidance_g2_scale25_ddim4": lambda: guidance(2, 25.0),
        "guidance_g2_scale50_ddim4": lambda: guidance(2, 50.0),
        "guidance_g4_scale50_ddim4": lambda: guidance(4, 50.0),
    }
    for _ in range(args.warmup):
        for call in methods.values():
            call()
    torch.cuda.synchronize()
    samples = {name: [] for name in methods}
    names = list(methods)
    for rep in range(args.repeats):
        # Rotate starting method so clock and thermal drift affect each equally.
        order = names[rep % len(names):] + names[:rep % len(names)]
        if rep % 2:
            order.reverse()
        for name in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            methods[name]()
            torch.cuda.synchronize()
            samples[name].append(time.perf_counter() - start)
    results = {
        "checkpoint": str(checkpoint), "history": trace_path, "reference": ref_path,
        "hardware": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "batch": 1, "execution_steps": 2, "prior_ddim_steps": 4,
        "seed": seed, "warmup_per_method": args.warmup, "repeats_per_method": args.repeats,
        "measurement": "CPU perf_counter bracketing predict and explicit CUDA synchronization; model load excluded; shared loaded prior; no simulator or IPC",
        "results": {name: describe(values) for name, values in samples.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for name, values in results["results"].items():
        print(f"{name}: median {values['median_ms']:.3f} ms, mean {values['mean_ms']:.3f} ms, p95 {values['p95_ms']:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
