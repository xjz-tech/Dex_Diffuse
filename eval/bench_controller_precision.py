"""Offline, matched-noise accuracy and warmed latency for the real controller."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL / "real"))
from policy_loader import load_policy
from hardware import POLICY_LOWER_LIMITS, POLICY_UPPER_LIMITS
from trt_unet import (TensorRTUnetAdapter, accelerate_policy_unet,
                      default_engine_path, fused_conditional_sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/obs_4-66.ckpt")
    parser.add_argument("--output", default="reports/controller_precision.json")
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    loaded, policy, spec = load_policy(args.checkpoint, device, "ddim", 4)
    del loaded
    original_model = policy.model
    original_sample = policy.conditional_sample
    default_tf32 = (torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32)
    stats = policy.normalizer.get_input_stats()["obs"]
    mean = stats["mean"].detach().cpu().numpy()
    std = stats["std"].detach().cpu().numpy()
    rng = np.random.default_rng(20260915)
    histories = []
    # Synthetic near-training-mean joint states; target residual is consistent.
    # The first input reproduces the previous all-zero comparison.
    histories.append(torch.zeros(1, 4, 66, device=device))
    for index in range(1, args.cases):
        scale = (0.05, 0.25, 0.5)[index % 3]
        base = mean[:22] + rng.normal(size=22) * std[:22] * scale
        qpos = np.clip(base + rng.normal(size=(4, 22)) * 0.005,
                       POLICY_LOWER_LIMITS, POLICY_UPPER_LIMITS)
        target = np.clip(qpos + rng.normal(size=(4, 22)) * 0.02,
                         POLICY_LOWER_LIMITS, POLICY_UPPER_LIMITS)
        obs = np.concatenate((qpos, target, target - qpos), axis=-1)
        histories.append(torch.as_tensor(obs[None], device=device, dtype=torch.float32))

    report = {"checkpoint": str(Path(args.checkpoint).resolve()), "spec": spec,
              "cases": args.cases, "repeats": args.repeats, "warmup": 12,
              "inputs": "zero + synthetic joint histories near training mean; no hardware",
              "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
              "default_tf32": default_tf32, "results": {}}
    outputs = {}
    modes = {}

    def evaluate(name):
        modes[name] = (policy.model, policy.conditional_sample,
                       torch.backends.cuda.matmul.allow_tf32,
                       torch.backends.cudnn.allow_tf32)
        predictions = []
        for i, obs in enumerate(histories):
            torch.manual_seed(123 + i)
            prediction = policy.predict_action({"obs": obs})["action"]
            assert torch.isfinite(prediction).all(), name
            predictions.append(prediction.cpu())
        outputs[name] = torch.cat(predictions)
        for i in range(12):
            policy.predict_action({"obs": histories[i % len(histories)]})
        torch.cuda.synchronize()
        times = []
        for i in range(args.repeats):
            started = time.perf_counter()
            action = policy.predict_action({"obs": histories[i % len(histories)]})["action"]
            # Include CPU action transfer, as in the real runner.
            action.cpu()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000)
        result = {"mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
                  "p95_ms": float(np.percentile(times, 95))}
        for reference in ("eager", "eager_strict_fp32"):
            if reference not in outputs:
                continue
            delta = (outputs[name] - outputs[reference]).abs()
            result["vs_" + reference] = {
                "mae_rad": delta.mean().item(), "max_rad": delta.max().item(),
                "exec2_mae_rad": delta[:, :2].mean().item(),
                "exec2_max_rad": delta[:, :2].max().item(),
            }
        report["results"][name] = result
        print(name, json.dumps(result), flush=True)

    with torch.inference_mode():
        evaluate("eager")
        policy.conditional_sample = MethodType(fused_conditional_sample, policy)
        evaluate("fused")
        old_path = default_engine_path(args.checkpoint)
        if old_path.exists():
            import torch_tensorrt
            policy.model = TensorRTUnetAdapter(torch.jit.load(str(old_path), map_location=device))
            evaluate("fp16_at_start")
            policy.model = original_model

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        policy.conditional_sample = original_sample
        evaluate("eager_strict_fp32")
        for precision in ("fp16", "fp32"):
            fp16 = precision == "fp16"
            policy.model = original_model
            accelerate_policy_unet(
                policy, fp16=fp16, max_batch=1,
                engine_path=default_engine_path(args.checkpoint, fp16=fp16),
                source_path=args.checkpoint,
            )
            evaluate("corrected_" + precision)
        policy.model = original_model
        policy.conditional_sample = original_sample
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = default_tf32
        evaluate("eager_repeat")
        # Interleave warmed modes to reduce drift from background load/clocks.
        names = ("eager", "fused", "corrected_fp16", "corrected_fp32")
        times = {name: [] for name in names}
        for index in range(args.repeats + 12):
            for name in rng.permutation(names):
                (policy.model, policy.conditional_sample,
                 torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cudnn.allow_tf32) = modes[name]
                torch.cuda.synchronize()
                started = time.perf_counter()
                action = policy.predict_action({"obs": histories[index % len(histories)]})["action"]
                action.cpu()
                torch.cuda.synchronize()
                if index >= 12:
                    times[name].append((time.perf_counter() - started) * 1000)
        report["interleaved_latency"] = {
            name: {"mean_ms": float(np.mean(values)),
                   "median_ms": float(np.median(values)),
                   "p95_ms": float(np.percentile(values, 95))}
            for name, values in times.items()
        }
        print("interleaved_latency", json.dumps(report["interleaved_latency"]), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()
