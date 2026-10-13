#!/usr/bin/env python3
"""Offline analysis: guided prior output vs ground-truth actions.

Quantifies, on held-out Sim-Hand samples, how far the guided prior output is
from the dataset GT action prefix:

  1. prior alone (DDIM, fixed seed) exec prefix vs GT
  2. prior + guide A (10k sim)  with scale S vs GT
  3. prior + guide B (real ep200) with scale S vs GT
  4. guide A / guide B references themselves vs GT
  5. ||guided - unguided prior||: how far guidance pulls the prior

Everything runs on CPU by default and reuses GuidedDDIMController /
build_policy / extract_guide_reference from the eval servers, so the sampling
path is identical to the online eval. All inputs/outputs are in the raw
(unnormalized) space; normalizers live inside the policies.

Alignment (oa_step_convention): a dataset sample is a (horizon=12) window where
obs[:n_obs_steps=4] is the conditioning history and action[3:12] are the 9
future actions. The compared exec prefix is action[3:3+exec_steps] (default 2).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

EVAL_DIR = Path(__file__).resolve().parent
DEX_ROOT = EVAL_DIR.parent
for import_path in (EVAL_DIR, DEX_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from checkpoint_loader import (  # noqa: E402
    build_policy,
    configure_policy_sampler,
    load_checkpoint,
)
from guided_pair_policy import (  # noqa: E402
    bind_conditional_sample_seed,
    extract_guide_reference,
    validate_compatible_specs,
)
from inference_dp_controller import GuidedDDIMController  # noqa: E402
from diffusion_policy.dataset.sim_hand_lowdim_dataset import (  # noqa: E402
    SimHandLowdimDataset,
)

DEFAULT_PRIOR = Path("/home/carus/data_usb/obs_4-66.ckpt")
DEFAULT_GUIDE_A = DEX_ROOT / "runs/sim_hand_10k_seed42/checkpoints/latest.ckpt"
DEFAULT_GUIDE_B = DEX_ROOT / (
    "data/outputs/2026.09.10/"
    "18.09.41_train_diffusion_unet_sim_hand_sim_hand_lowdim/"
    "checkpoints/epoch_0200.ckpt"
)
DEFAULT_DATASET = DEX_ROOT / "data/sim_hand_10k_seed42"

HAND_DIM = 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-checkpoint", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--guide-a-checkpoint", type=Path, default=DEFAULT_GUIDE_A)
    parser.add_argument("--guide-b-checkpoint", type=Path, default=DEFAULT_GUIDE_B)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--n-samples", type=int, default=96)
    parser.add_argument("--scale", type=float, default=25.0, help="guidance scale")
    parser.add_argument(
        "--guidance-steps",
        type=int,
        default=2,
        help="leading predicted actions constrained by guidance",
    )
    parser.add_argument(
        "--exec-steps",
        type=int,
        default=2,
        help="length of the executed/compared action prefix",
    )
    parser.add_argument("--inference-steps", type=int, default=4, help="prior DDIM steps")
    parser.add_argument(
        "--guide-inference-steps", type=int, default=4, help="guide DDIM steps"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0, help="0: leave torch default")
    parser.add_argument("--no-salvage", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON path for the full result payload",
    )
    args = parser.parse_args()
    if args.device != "cpu":
        print(
            "[warn] this script is meant for CPU offline analysis; using %s"
            % args.device,
            flush=True,
        )
    if args.n_samples <= 0 or args.batch_size <= 0:
        raise ValueError("n-samples and batch-size must be positive")
    if args.scale < 0:
        raise ValueError("scale must be non-negative")
    if args.exec_steps <= 0 or args.guidance_steps <= 0:
        raise ValueError("exec-steps and guidance-steps must be positive")
    if args.guidance_steps < args.exec_steps:
        raise ValueError(
            "guidance-steps (%d) must cover exec-steps (%d)"
            % (args.guidance_steps, args.exec_steps)
        )
    return args


def load_guide_policy(checkpoint: Path, device, inference_steps: int, seed: int, allow_salvage: bool):
    loaded = load_checkpoint(checkpoint.expanduser().resolve(), allow_salvage=allow_salvage)
    policy, spec = build_policy(loaded)
    configure_policy_sampler(policy, "ddim", inference_steps=inference_steps)
    policy = policy.to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    bind_conditional_sample_seed(policy, seed=seed, device=device, fixed_noise=True)
    return policy, spec, loaded


def select_sample_indices(n_total: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = min(int(n_samples), int(n_total))
    indices = rng.choice(n_total, size=n, replace=False)
    return np.sort(indices)


def prefix_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Per-sample metrics on the exec prefix. pred/gt: (B, T, 22)."""
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64)
    flat = diff.reshape(diff.shape[0], -1)
    return {
        "l2": np.linalg.norm(flat, axis=1),               # Frobenius norm
        "rmse": np.sqrt(np.square(flat).mean(axis=1)),    # per-entry RMSE
        "max_abs": np.abs(flat).max(axis=1),              # per-dim max error
    }


def mean_std(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def main() -> int:
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    allow_salvage = not args.no_salvage

    print("[data] loading %s (split=%s)" % (args.dataset_path, args.split), flush=True)
    dataset = SimHandLowdimDataset(
        dataset_path=str(args.dataset_path.expanduser().resolve()),
        horizon=12,
        pad_before=3,
        pad_after=8,
        seed=42,
        val_ratio=0.1,
    )
    eval_dataset = (
        dataset.get_validation_dataset() if args.split == "val" else dataset
    )
    indices = select_sample_indices(len(eval_dataset), args.n_samples, args.seed)
    print(
        "[data] %s split has %d windows; evaluating %d"
        % (args.split, len(eval_dataset), len(indices)),
        flush=True,
    )

    n_obs_steps = 4
    action_start = n_obs_steps - 1  # oa_step_convention
    histories = []
    gt_prefixes = []
    for index in indices:
        sample = eval_dataset[int(index)]
        obs = sample["obs"].numpy().astype(np.float32)      # (12, 66)
        action = sample["action"].numpy().astype(np.float32)  # (12, 22)
        histories.append(obs[:n_obs_steps])
        gt_prefixes.append(action[action_start : action_start + args.exec_steps])
    histories = np.stack(histories, axis=0)
    gt_prefixes = np.stack(gt_prefixes, axis=0)

    print("[prior] loading %s" % args.prior_checkpoint, flush=True)
    controller = GuidedDDIMController(
        args.prior_checkpoint.expanduser().resolve(),
        device,
        inference_steps=args.inference_steps,
        execution_steps=args.exec_steps,
        guidance_scale=args.scale,
        eta=0.0,
        fixed_noise=True,
        seed=args.seed,
        allow_salvage=allow_salvage,
    )
    controller.set_guidance_horizon(args.guidance_steps)

    guides = {}
    for label, path in (
        ("guideA_sim10k", args.guide_a_checkpoint),
        ("guideB_real", args.guide_b_checkpoint),
    ):
        print("[guide] loading %s (%s)" % (label, path), flush=True)
        policy, spec, loaded = load_guide_policy(
            path, device, args.guide_inference_steps, args.seed, allow_salvage
        )
        validate_compatible_specs(controller.spec, spec)
        guides[label] = policy
        print(
            "[guide] %s weights=%s epoch=%s step=%s"
            % (label, loaded.weight_source, loaded.epoch, loaded.global_step),
            flush=True,
        )

    n = len(indices)
    exec_steps = args.exec_steps
    outputs = {
        "prior_unguided": np.zeros((n, exec_steps, HAND_DIM), dtype=np.float32),
        "guided_guideA_sim10k": np.zeros((n, exec_steps, HAND_DIM), dtype=np.float32),
        "guided_guideB_real": np.zeros((n, exec_steps, HAND_DIM), dtype=np.float32),
        "ref_guideA_sim10k": np.zeros((n, args.guidance_steps, HAND_DIM), dtype=np.float32),
        "ref_guideB_real": np.zeros((n, args.guidance_steps, HAND_DIM), dtype=np.float32),
    }
    guidance_stats = {  # normalized-space slice MSE, from the last DDIM step
        "guideA_sim10k": {"mse_before": [], "mse_after": []},
        "guideB_real": {"mse_before": [], "mse_after": []},
    }

    started = time.perf_counter()
    for begin in range(0, n, args.batch_size):
        end = min(begin + args.batch_size, n)
        history = histories[begin:end]
        history_tensor = torch.from_numpy(history).to(device)

        refs = {}
        for label, policy in guides.items():
            refs[label] = extract_guide_reference(
                policy,
                history_tensor,
                device=device,
                action_start=controller.action_start,
                reference_steps=controller.reference_steps,
            )
            outputs["ref_" + label][begin:end] = refs[label]

        # Unguided prior and both guided variants share the same fixed initial
        # noise per batch (fixed_noise=True, same seed), so differences are
        # purely due to guidance.
        controller.guidance_scale = 0.0
        unguided, _ = controller.predict(history, refs["guideA_sim10k"])
        outputs["prior_unguided"][begin:end] = unguided

        for label in ("guideA_sim10k", "guideB_real"):
            controller.guidance_scale = float(args.scale)
            guided, stats = controller.predict(history, refs[label])
            outputs["guided_" + label][begin:end] = guided
            guidance_stats[label]["mse_before"].append(stats.mse_before_batch)
            guidance_stats[label]["mse_after"].append(stats.mse_after_batch)

        print(
            "[eval] %d/%d (%.1fs)" % (end, n, time.perf_counter() - started),
            flush=True,
        )

    gt = gt_prefixes
    unguided = outputs["prior_unguided"]
    rows = []

    def add_row(name: str, pred: np.ndarray) -> None:
        metrics = prefix_metrics(pred, gt)
        pull = np.linalg.norm(
            (pred - unguided).reshape(n, -1), axis=1
        )
        row = {
            "name": name,
            "l2_vs_gt": mean_std(metrics["l2"]),
            "rmse_vs_gt": mean_std(metrics["rmse"]),
            "max_abs_vs_gt": mean_std(metrics["max_abs"]),
            "l2_vs_unguided_prior": mean_std(pull),
        }
        rows.append(row)

    add_row("prior_unguided", unguided)
    add_row("guided_by_A_sim10k", outputs["guided_guideA_sim10k"])
    add_row("guided_by_B_real", outputs["guided_guideB_real"])
    add_row("ref_A_sim10k", outputs["ref_guideA_sim10k"][:, :exec_steps])
    add_row("ref_B_real", outputs["ref_guideB_real"][:, :exec_steps])

    def fmt(cell: dict[str, float]) -> str:
        return "%.4f ± %.4f" % (cell["mean"], cell["std"])

    print()
    print(
        "Config: split=%s n=%d ddim=%d guide_ddim=%d scale=%g guidance_steps=%d "
        "exec_prefix=%d seed=%d device=%s"
        % (
            args.split,
            n,
            args.inference_steps,
            args.guide_inference_steps,
            args.scale,
            args.guidance_steps,
            exec_steps,
            args.seed,
            device,
        )
    )
    header = "%-20s | %-19s | %-19s | %-19s | %-19s" % (
        "output",
        "L2 vs GT",
        "RMSE vs GT",
        "max|err| vs GT",
        "L2 vs unguided prior",
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            "%-20s | %-19s | %-19s | %-19s | %-19s"
            % (
                row["name"],
                fmt(row["l2_vs_gt"]),
                fmt(row["rmse_vs_gt"]),
                fmt(row["max_abs_vs_gt"]),
                fmt(row["l2_vs_unguided_prior"]),
            )
        )
    print()
    for label, stat in guidance_stats.items():
        before = np.concatenate(stat["mse_before"])
        after = np.concatenate(stat["mse_after"])
        print(
            "[guidance:%s] normalized-slice MSE %.6f -> %.6f (last DDIM step)"
            % (label, float(before.mean()), float(after.mean()))
        )

    payload = {
        "config": {
            "prior_checkpoint": str(args.prior_checkpoint),
            "guide_a_checkpoint": str(args.guide_a_checkpoint),
            "guide_b_checkpoint": str(args.guide_b_checkpoint),
            "dataset_path": str(args.dataset_path),
            "split": args.split,
            "n_samples": n,
            "scale": args.scale,
            "guidance_steps": args.guidance_steps,
            "exec_steps": exec_steps,
            "inference_steps": args.inference_steps,
            "guide_inference_steps": args.guide_inference_steps,
            "seed": args.seed,
            "device": str(device),
        },
        "rows": rows,
        "guidance_stats": {
            label: {
                "mse_before": mean_std(np.concatenate(stat["mse_before"])),
                "mse_after": mean_std(np.concatenate(stat["mse_after"])),
            }
            for label, stat in guidance_stats.items()
        },
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
        print("[done] wrote %s" % output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
