#!/usr/bin/env python3
"""Diffusion Policy inference server for the Isaac Gym evaluator."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


EVAL_DIR = Path(__file__).resolve().parent
DEX_ROOT = EVAL_DIR.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(DEX_ROOT) not in sys.path:
    sys.path.insert(0, str(DEX_ROOT))

from checkpoint_loader import (  # noqa: E402
    build_policy,
    configure_policy_execution_steps,
    configure_policy_sampler,
    load_checkpoint,
)
from ipc import recv_message, send_message  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Serve Sim-Hand Diffusion Policy predictions over a UNIX socket."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--socket", required=True, dest="socket_path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampler",
        choices=("ddpm", "ddim"),
        default="ddpm",
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=None,
        help="Override inference steps (default: the checkpoint value, 100).",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="How many predicted actions to execute before replanning (1-9).",
    )
    parser.add_argument(
        "--no-salvage",
        action="store_true",
        help="Reject an incomplete checkpoint instead of recovering its intact base model.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip one startup inference used to validate and warm the model.",
    )
    return parser.parse_args()


def _device_or_raise(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for inference but is unavailable")
    return device


def _normalizer_summary(policy):
    result = {}
    stats = policy.normalizer.get_input_stats()
    for field in ("obs", "action"):
        result[field] = {
            key: value.detach().cpu().tolist()
            for key, value in stats[field].items()
            if key in ("min", "max", "mean", "std")
        }
    return result


def _predict(policy, array, device):
    if array is None:
        raise ValueError("predict request has no observation array")
    if array.dtype != np.float32:
        raise TypeError("observation array must be float32, got %s" % array.dtype)
    expected_observation_shape = (
        int(policy.n_obs_steps),
        int(policy.obs_dim),
    )
    if array.ndim != 3 or array.shape[1:] != expected_observation_shape:
        raise ValueError(
            "observation must have shape (batch, %d, %d), got %s"
            % (
                expected_observation_shape[0],
                expected_observation_shape[1],
                array.shape,
            )
        )
    if not np.isfinite(array).all():
        raise ValueError("observation contains NaN or Inf")

    observation = torch.from_numpy(array).to(device=device, dtype=torch.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        result = policy.predict_action({"obs": observation})
        action = result["action"]
    expected_action_shape = (
        array.shape[0],
        int(policy.n_action_steps),
        int(policy.action_dim),
    )
    if tuple(action.shape) != expected_action_shape:
        raise RuntimeError(
            "policy returned unexpected action shape %s, expected %s"
            % (tuple(action.shape), expected_action_shape)
        )
    if not torch.isfinite(action).all():
        raise RuntimeError("policy returned NaN or Inf")
    action_np = action.detach().to(device="cpu", dtype=torch.float32).numpy()
    return action_np, time.perf_counter() - started


def _warm_up(policy, device, seed, obs_mean):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    mean = np.asarray(obs_mean, dtype=np.float32)
    expected_mean_shape = (int(policy.obs_dim),)
    if mean.shape != expected_mean_shape:
        raise ValueError(
            "observation normalizer mean has shape %s, expected %s"
            % (mean.shape, expected_mean_shape)
        )
    mean = mean.reshape(1, 1, expected_mean_shape[0])
    observation = np.repeat(mean, int(policy.n_obs_steps), axis=1)
    _, elapsed = _predict(policy, observation, device)
    # Make the first real request reproducible independently of the warm-up.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    print("[model] warm-up inference: %.3f s" % elapsed, flush=True)


def _serve(args, policy, device, checkpoint_info, spec, normalizer_summary):
    socket_path = Path(args.socket_path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists() or socket_path.is_symlink():
        socket_path.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client = None
    try:
        server.bind(str(socket_path))
        os.chmod(str(socket_path), 0o600)
        server.listen(1)
        print("[model] ready: %s" % socket_path, flush=True)
        client, _ = server.accept()
        client.settimeout(600.0)
        print("[model] simulator connected", flush=True)

        while True:
            try:
                message, array = recv_message(client)
            except ConnectionError:
                print("[model] simulator disconnected", flush=True)
                break
            message_type = message.get("type")
            request_id = message.get("request_id")

            if message_type == "hello":
                send_message(
                    client,
                    {
                        "type": "hello",
                        "request_id": request_id,
                        "ok": True,
                        "spec": spec,
                        "checkpoint": {
                            "weight_source": checkpoint_info.weight_source,
                            "global_step": checkpoint_info.global_step,
                            "epoch": checkpoint_info.epoch,
                            "salvaged": checkpoint_info.salvaged,
                        },
                        "normalizer": normalizer_summary,
                    },
                )
                continue
            if message_type == "shutdown":
                send_message(
                    client,
                    {"type": "shutdown", "request_id": request_id, "ok": True},
                )
                break
            if message_type != "predict":
                send_message(
                    client,
                    {
                        "type": "error",
                        "request_id": request_id,
                        "ok": False,
                        "error": "unsupported request type: %r" % message_type,
                    },
                )
                continue

            try:
                action, elapsed = _predict(policy, array, device)
                send_message(
                    client,
                    {
                        "type": "prediction",
                        "request_id": request_id,
                        "ok": True,
                        "inference_seconds": elapsed,
                    },
                    action,
                )
            except Exception as exc:  # Return context before terminating the eval.
                traceback.print_exc()
                send_message(
                    client,
                    {
                        "type": "error",
                        "request_id": request_id,
                        "ok": False,
                        "error": "%s: %s" % (type(exc).__name__, exc),
                    },
                )
    finally:
        if client is not None:
            client.close()
        server.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def main():
    args = _parse_args()
    device = _device_or_raise(args.device)
    print("[model] loading checkpoint: %s" % args.checkpoint, flush=True)
    loaded = load_checkpoint(args.checkpoint, allow_salvage=not args.no_salvage)
    policy, spec = build_policy(loaded)
    configure_policy_sampler(policy, args.sampler, args.inference_steps)
    if args.n_action_steps is not None:
        spec = configure_policy_execution_steps(policy, spec, args.n_action_steps)
    policy = policy.to(device)
    policy.eval()
    summary = _normalizer_summary(policy)

    print(
        "[model] loaded %s | step=%s epoch=%s | sampler=%s steps=%d "
        "exec=%d pred=%s"
        % (
            loaded.weight_source,
            loaded.global_step,
            loaded.epoch,
            type(policy.noise_scheduler).__name__,
            policy.num_inference_steps,
            int(policy.n_action_steps),
            spec["n_pred_action_steps"],
        ),
        flush=True,
    )
    if loaded.salvaged:
        print(
            "[model] WARNING: EMA is damaged; inference is using the complete "
            "non-EMA model from the checkpoint.",
            flush=True,
        )
    if not args.no_warmup:
        _warm_up(policy, device, args.seed, summary["obs"]["mean"])
    _serve(args, policy, device, loaded, spec, summary)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[model] stopped", flush=True)
