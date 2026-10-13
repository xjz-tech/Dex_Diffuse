#!/usr/bin/env python3
"""Offline DDIM 0.09-boundary check on stored training windows. No simulator."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from checkpoint_loader import (  # noqa: E402
    build_policy,
    configure_policy_sampler,
    load_checkpoint,
)
from diffusion_policy.common.sampler import get_val_mask  # noqa: E402

OUT = Path(__file__).resolve().parent
CHECKPOINT = Path("/home/carus/data_usb/obs_4-66.ckpt")
DATASET = ROOT / "data" / "sim_hand_10k_seed42" / "replay_buffer.zarr"
N_OBS_STEPS = 4
HORIZON = 12
ACTION_START = N_OBS_STEPS - 1  # oa_step convention: first executable is a[t]
N_PRED = HORIZON - ACTION_START  # 9 usable actions
THRESHOLD = 0.09
EPS = 1e-6
DDIM_STEPS = (4, 8, 16, 32)
BATCH_SIZE = 64
SEED = 20260918
NOISE_SEED = 42


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    over = values > THRESHOLD + EPS
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "over_0.09_count": int(over.sum()),
        "over_0.09_fraction": float(over.mean()),
    }


def collect_complete_windows(obs: np.ndarray, action: np.ndarray, ends: np.ndarray):
    val_mask = get_val_mask(n_episodes=len(ends), val_ratio=0.1, seed=42)
    starts = np.r_[0, ends[:-1]]
    obs_windows = []
    action_windows = []
    records = []
    for episode, (start, end) in enumerate(zip(starts, ends)):
        if val_mask[episode]:
            continue
        last_start = end - HORIZON
        for buffer_start in range(start, last_start + 1):
            obs_windows.append(obs[buffer_start : buffer_start + N_OBS_STEPS])
            action_windows.append(action[buffer_start : buffer_start + HORIZON])
            records.append(
                {
                    "episode": int(episode),
                    "buffer_start": int(buffer_start),
                    "local_start": int(buffer_start - start),
                }
            )
    return (
        np.stack(obs_windows).astype(np.float32),
        np.stack(action_windows).astype(np.float32),
        records,
        val_mask,
    )


@torch.inference_mode()
def sample_from_noise(policy, nobs: torch.Tensor, noise: torch.Tensor, steps: int):
    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(int(steps), device=noise.device)
    global_cond = nobs[:, :N_OBS_STEPS].reshape(nobs.shape[0], -1)
    trajectory = noise.clone()
    for timestep in scheduler.timesteps:
        model_output = policy.model(trajectory, timestep, global_cond=global_cond)
        trajectory = scheduler.step(
            model_output,
            timestep,
            trajectory,
        ).prev_sample
    return policy.normalizer["action"].unnormalize(trajectory)


def main() -> None:
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = zarr.open_group(str(DATASET), mode="r")
    obs = np.asarray(root["data/obs"][:], dtype=np.float32)
    action = np.asarray(root["data/action"][:], dtype=np.float32)
    ends = np.asarray(root["meta/episode_ends"][:])
    obs_win, act_win, records, val_mask = collect_complete_windows(obs, action, ends)

    previous_target = obs_win[:, -1, 22:44]
    gt_first = act_win[:, ACTION_START]
    gt_chunk = act_win[:, ACTION_START:]
    gt_d_first = np.max(np.abs(gt_first - previous_target), axis=-1)
    gt_d_inside = np.max(np.abs(np.diff(gt_chunk, axis=1)), axis=-1)

    loaded = load_checkpoint(CHECKPOINT, allow_salvage=False)
    policy, spec = build_policy(loaded)
    if int(spec["n_obs_steps"]) != N_OBS_STEPS or int(spec["horizon"]) != HORIZON:
        raise ValueError("checkpoint temporal layout does not match this check")
    policy = policy.to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    configure_policy_sampler(policy, "ddim")
    scheduler_cfg = dict(policy.noise_scheduler.config)

    obs_t = torch.as_tensor(obs_win, device=device, dtype=policy.dtype)
    nobs = policy.normalizer["obs"].normalize(obs_t)
    generator = torch.Generator(device=device)
    generator.manual_seed(NOISE_SEED)
    noise = torch.randn(
        (obs_win.shape[0], HORIZON, spec["action_dim"]),
        device=device,
        dtype=policy.dtype,
        generator=generator,
    )

    result = {
        "scope": (
            "No simulator. Complete unpadded 4-frame training windows from 10k zarr. "
            "Same observations and the same initial noise tensor for every DDIM step count. "
            "First executable action is action_pred[:, n_obs_steps-1], not frame 0. "
            "Chunk is the 9 usable future actions a[t:t+9]. Threshold is >0.09+1e-6 rad."
        ),
        "checkpoint": str(CHECKPOINT),
        "weight_source": loaded.weight_source,
        "salvaged": loaded.salvaged,
        "dataset": str(DATASET),
        "n_windows": int(obs_win.shape[0]),
        "n_train_episodes": int((~val_mask).sum()),
        "n_val_episodes_excluded": int(val_mask.sum()),
        "seed_window_split": 42,
        "noise_seed": NOISE_SEED,
        "spec": spec,
        "action_start": ACTION_START,
        "n_pred_action_steps": N_PRED,
        "scheduler_after_ddim_conversion": {
            key: (value if isinstance(value, (int, float, str, bool, type(None))) else str(value))
            for key, value in scheduler_cfg.items()
        },
        "normalizer_obs_scale_sha256": hashlib.sha256(
            policy.normalizer["obs"].params_dict["scale"].detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "normalizer_action_scale_sha256": hashlib.sha256(
            policy.normalizer["action"].params_dict["scale"].detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "gt": {
            "d_first": summarize(gt_d_first),
            "d_inside": summarize(gt_d_inside),
            "d_inside_by_k": {
                str(k): summarize(gt_d_inside[:, k - 1]) for k in range(1, N_PRED)
            },
        },
        "ddim": {},
    }

    arrays = {
        "gt_d_first": gt_d_first,
        "gt_d_inside": gt_d_inside,
    }

    for steps in DDIM_STEPS:
        preds = []
        unused_frame0 = []
        for start in range(0, obs_win.shape[0], BATCH_SIZE):
            stop = min(start + BATCH_SIZE, obs_win.shape[0])
            pred = sample_from_noise(
                policy,
                nobs[start:stop],
                noise[start:stop],
                steps,
            )
            unused_frame0.append(pred[:, 0].detach().cpu().numpy())
            preds.append(pred[:, ACTION_START:].detach().cpu().numpy())
        chunk = np.concatenate(preds, axis=0)
        frame0 = np.concatenate(unused_frame0, axis=0)
        first = chunk[:, 0]
        d_first = np.max(np.abs(first - previous_target), axis=-1)
        d_inside = np.max(np.abs(np.diff(chunk, axis=1)), axis=-1)
        frame0_vs_first = np.max(np.abs(frame0 - first), axis=-1)
        row = {
            "d_first": summarize(d_first),
            "d_inside": summarize(d_inside),
            "d_inside_by_k": {
                str(k): summarize(d_inside[:, k - 1]) for k in range(1, N_PRED)
            },
            "unused_action_pred_frame0_vs_first_executable": summarize(frame0_vs_first),
            "mae_first_to_gt": float(np.abs(first - gt_first).mean()),
            "mae_chunk_to_gt": float(np.abs(chunk - gt_chunk).mean()),
        }
        result["ddim"][str(steps)] = row
        arrays[f"ddim{steps}_d_first"] = d_first
        arrays[f"ddim{steps}_d_inside"] = d_inside
        print(f"DDIM{steps}", json.dumps(row["d_first"]), flush=True)
        print(f"DDIM{steps} inside", json.dumps(row["d_inside"]), flush=True)

    (OUT / "statistics.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "deltas.npz", **arrays)
    (OUT / "windows.json").write_text(
        json.dumps({"records": records, "n": len(records)}, indent=2) + "\n"
    )
    print("wrote", OUT / "statistics.json", flush=True)


if __name__ == "__main__":
    main()
