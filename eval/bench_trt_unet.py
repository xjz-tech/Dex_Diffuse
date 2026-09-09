#!/usr/bin/env python3
"""Benchmark eager vs TensorRT FP16 UNet on batch-1 DDIM."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


EVAL_DIR = Path(__file__).resolve().parent
DEX_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(DEX_ROOT))

from checkpoint_loader import (  # noqa: E402
    build_policy,
    configure_policy_sampler,
    load_checkpoint,
)
from trt_unet import accelerate_policy_unet  # noqa: E402


def _bench(fn, n=40, warm=20):
    with torch.inference_mode():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(n):
            torch.cuda.synchronize()
            started = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000)
    times.sort()
    return sum(times) / len(times), times[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/carus/data_usb/obs_4-66.ckpt")
    parser.add_argument("--inference-steps", type=int, default=4)
    args = parser.parse_args()

    loaded = load_checkpoint(args.checkpoint, allow_salvage=True)
    policy, _spec = build_policy(loaded)
    configure_policy_sampler(policy, "ddim", args.inference_steps)
    policy = policy.to("cuda").eval()
    obs = torch.zeros(1, policy.n_obs_steps, policy.obs_dim, device="cuda")

    def predict():
        return policy.predict_action({"obs": obs})["action"]

    eager_mean, eager_med = _bench(predict)
    print(
        "eager DDIM%s batch1: mean=%.2f ms median=%.2f ms"
        % (args.inference_steps, eager_mean, eager_med)
    )

    torch.manual_seed(0)
    with torch.inference_mode():
        eager_action = predict().clone()

    print("compiling TensorRT FP16 UNet...")
    started = time.perf_counter()
    accelerate_policy_unet(policy, fp16=True)
    print("compile %.1fs" % (time.perf_counter() - started))

    trt_mean, trt_med = _bench(predict)
    torch.manual_seed(0)
    with torch.inference_mode():
        trt_action = predict()
    mae = (eager_action - trt_action).abs().mean().item()
    mx = (eager_action - trt_action).abs().max().item()
    print(
        "trt   DDIM%s batch1: mean=%.2f ms median=%.2f ms  action mae=%.4f max=%.4f rad"
        % (args.inference_steps, trt_mean, trt_med, mae, mx)
    )


if __name__ == "__main__":
    main()
