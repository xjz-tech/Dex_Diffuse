#!/usr/bin/env python3
"""Offline real-policy eager/fused/optional TensorRT latency and action parity."""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR / "real"))

from guided_policy import load_guided_policy
from trt_unet import accelerate_policy_unet, default_engine_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(EVAL_DIR.parent / "runs/obs_4-66.ckpt"))
    parser.add_argument("--guide-checkpoint", default=str(EVAL_DIR.parent / "runs/epoch_0200.ckpt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--guide-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=25.0)
    parser.add_argument("--guidance-steps", type=int, default=2)
    parser.add_argument("--action-chunk-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guide-seed", type=int, default=42)
    parser.add_argument("--tensorrt", action="store_true")
    parser.add_argument("--trt-precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--cache-engines", action="store_true",
                        help="Use the same validated engine cache as the real launcher.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup non-negative")
    args.fixed_noise = 1
    args.fused_ddim = False
    device = torch.device(args.device)
    use_trt = args.tensorrt
    args.tensorrt = False
    if use_trt:
        if device.type != "cuda":
            parser.error("TensorRT requires a CUDA device")
        import torch_tensorrt
    loaded, policy, _ = load_guided_policy(args, device)
    del loaded
    mean = policy.normalizer.get_input_stats()["obs"]["mean"]
    observation = {"obs": mean.reshape(1, 1, -1).repeat(1, policy.n_obs_steps, 1)}

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def bench(name):
        policy.reset_noise()
        times = []
        with torch.inference_mode():
            for index in range(args.warmup + args.repeats):
                synchronize()
                started = time.perf_counter()
                action = policy.predict_action(observation)["action"]
                synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                if index >= args.warmup:
                    times.append(elapsed)
        if not torch.isfinite(action).all():
            raise RuntimeError(f"{name} produced non-finite actions")
        print(f"{name}: mean={statistics.mean(times):.3f} ms "
              f"median={statistics.median(times):.3f} ms", flush=True)
        return action.clone()

    baseline = bench("eager")
    policy.fused_ddim = True
    fused = bench("fused")
    torch.testing.assert_close(fused, baseline, atol=1e-4, rtol=1e-4)
    print(f"fused action max_error={(fused - baseline).abs().max().item():.7f} rad", flush=True)
    if use_trt:
        started = time.perf_counter()
        for component, checkpoint in zip(
            (policy.prior, policy.guide), (args.checkpoint, args.guide_checkpoint)
        ):
            fp16 = args.trt_precision == "fp16"
            cache_options = dict(engine_path=default_engine_path(checkpoint, fp16=fp16),
                                 source_path=checkpoint) if args.cache_engines else {}
            accelerate_policy_unet(component, fp16=fp16, max_batch=1, **cache_options)
        print(f"TRT prepare (compile/cache): {time.perf_counter() - started:.1f}s", flush=True)
        actual = bench(f"TRT {args.trt_precision.upper()} + fused")
        error = (actual - baseline).abs()
        print(f"TRT action mae={error.mean().item():.7f} max_error={error.max().item():.7f} rad", flush=True)


if __name__ == "__main__":
    main()
