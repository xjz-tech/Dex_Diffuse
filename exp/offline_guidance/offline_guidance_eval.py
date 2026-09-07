#!/usr/bin/env python3
"""Offline replay eval for Real DP + Sim-Hand guided DDIM.

Reads the DP training zarr, never starts Isaac Gym or real hardware, and
compares DP proposals, unguided/guided sim-hand actions, and linear mixing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

EXP_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXP_DIR.parents[1]
EVAL_DIR = ROOT_DIR / "eval"
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(ROOT_DIR))

from inference_dp_controller import (  # noqa: E402
    ARM_DIM,
    HAND_DIM,
    GuidedDDIMController,
    _load_real_policy,
    _real_policy_metadata,
)
from offline_guidance_data import (  # noqa: E402
    ACTION_DIM,
    ReplayIndex,
    build_controller_history,
    build_dp_observation,
    episode_bounds,
    hand_chunk,
    select_eval_indices,
)
from offline_guidance_metrics import (  # noqa: E402
    linear_mix,
    mae,
    max_abs_jump,
    mean_abs_jerk,
    mean_abs_velocity,
    out_of_range_fraction,
    recommend_guidance_scale,
    rmse,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)

DEFAULT_DP = Path(
    "/home/carus/Program/diffusion_policy/data/outputs/2026.09.05/"
    "22.35.37_train_diffusion_unet_dino_image_bulb_image/checkpoints/epoch=0950.ckpt"
)
DEFAULT_CONTROLLER = Path("/home/carus/data_usb/obs_4-66.ckpt")
DEFAULT_ZARR = Path("/home/carus/Data/bulb_tac_50_dp")
DEFAULT_DINOV2 = Path("/home/carus/Program/diffusion_policy/dinov2_assets")
DEFAULT_OUTPUT = EXP_DIR / "outputs" / "epoch0950_obs66"


def _parse_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated float list")
    return values


def _parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated int list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp-checkpoint", type=Path, default=DEFAULT_DP)
    parser.add_argument("--controller-checkpoint", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--zarr-path", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all selected frames")
    parser.add_argument(
        "--guidance-scales",
        type=_parse_floats,
        default=_parse_floats("0,0.5,1,2,4,8"),
    )
    parser.add_argument(
        "--mix-weights",
        type=_parse_floats,
        default=_parse_floats("0,0.25,0.5,0.75,1"),
    )
    parser.add_argument("--dp-inference-steps", type=int, default=16)
    parser.add_argument("--ddim-inference-steps", type=int, default=8)
    parser.add_argument("--execution-steps", type=int, default=5)
    parser.add_argument(
        "--comparison-lengths",
        type=_parse_ints,
        default=None,
        help="Score prefixes of the predicted chunk, e.g. 2,5,8. Default: execution-steps only.",
    )
    parser.add_argument(
        "--generation-windows",
        type=_parse_ints,
        default=None,
        help="Guidance horizons used to generate the chunk, e.g. 5,6,7,8. Score still uses comparison-lengths.",
    )
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--fixed-noise", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-salvage", action="store_true")
    parser.add_argument("--perturb-samples", type=int, default=50)
    parser.add_argument("--image-noise", type=float, default=0.02)
    parser.add_argument("--qpos-noise", type=float, default=0.01)
    parser.add_argument("--ref-noise", type=float, default=0.02)
    parser.add_argument("--perturb-scale", type=float, default=1.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="DP and guided DDIM batch size (default: 512)",
    )
    return parser.parse_args()


def method_label(method: str, value: float | None) -> str:
    if method == "dp":
        return "dp"
    if method == "guided":
        return "guided_scale_%s" % (float(value),)
    if method == "linear_mix":
        return "linear_mix_%s" % (float(value),)
    raise ValueError("unknown method: %r" % (method,))


def _configure_dinov2() -> None:
    os.environ.setdefault(
        "DINOV2_REPO_OR_DIR",
        str(DEFAULT_DINOV2 / "facebookresearch_dinov2_main"),
    )
    os.environ.setdefault(
        "DINOV2_WEIGHTS",
        str(DEFAULT_DINOV2 / "dinov2_vits14_pretrain.pth"),
    )
    os.environ.setdefault("DINOV2_SOURCE", "local")


def resolve_zarr_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name != "replay_buffer.zarr" and (path / "replay_buffer.zarr").is_dir():
        path = path / "replay_buffer.zarr"
    if not path.is_dir():
        raise FileNotFoundError("replay buffer not found: %s" % path)
    return path


def _iter_batches(items: list, batch_size: int) -> list:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _collate_policy_obs(
    obs_list: list[dict[str, np.ndarray]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not obs_list:
        raise ValueError("obs_list must be non-empty")
    keys = obs_list[0].keys()
    return {
        key: torch.from_numpy(
            np.stack([np.ascontiguousarray(obs[key]) for obs in obs_list], axis=0)
        ).to(device=device, dtype=torch.float32)
        for key in keys
    }


def _as_policy_obs(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return _collate_policy_obs([obs], device)


def _metric_dict(
    pred: np.ndarray,
    demo: np.ndarray,
    reference: np.ndarray,
    unguided: np.ndarray | None,
    lower: np.ndarray,
    upper: np.ndarray,
    mse_before: float | None = None,
    mse_after: float | None = None,
) -> dict[str, float]:
    prior_mae = 0.0 if unguided is None else mae(pred, unguided)
    prior_rmse = 0.0 if unguided is None else rmse(pred, unguided)
    return {
        "demo_mae": mae(pred, demo),
        "demo_rmse": rmse(pred, demo),
        "prior_mae": prior_mae,
        "ref_mse": rmse(pred, reference) ** 2,
        "prior_mse": prior_rmse ** 2,
        "max_jump": max_abs_jump(pred),
        "mean_vel": mean_abs_velocity(pred),
        "mean_jerk": mean_abs_jerk(pred),
        "oob_frac": out_of_range_fraction(pred, lower, upper),
        "mse_before": float("nan") if mse_before is None else float(mse_before),
        "mse_after": float("nan") if mse_after is None else float(mse_after),
    }


def _append_window_rows(
    rows: list[dict[str, object]],
    index: ReplayIndex,
    method: str,
    value: float | None,
    pred: np.ndarray,
    demo: np.ndarray,
    reference: np.ndarray,
    unguided: np.ndarray | None,
    lower: np.ndarray,
    upper: np.ndarray,
    windows: tuple[int, ...],
    mse_before: float | None = None,
    mse_after: float | None = None,
    stored_window: int | None = None,
) -> None:
    for window in windows:
        rows.append(
            _row(
                index,
                method,
                value,
                _metric_dict(
                    pred[:window],
                    demo[:window],
                    reference[:window],
                    None if unguided is None else unguided[:window],
                    lower,
                    upper,
                    mse_before,
                    mse_after,
                ),
                window,
                gen_window=stored_window if stored_window is not None else window,
            )
        )


def _row(
    index: ReplayIndex,
    method: str,
    value: float | None,
    metrics: dict[str, float],
    window: int,
    gen_window: int | None = None,
) -> dict[str, object]:
    return {
        "episode_idx": index.episode_idx,
        "t_in_episode": index.t_in_episode,
        "global_idx": index.global_idx,
        "method": method,
        "label": method_label(method, value),
        "scale": float("nan") if value is None else float(value),
        "window": int(window),
        "gen_window": int(window if gen_window is None else gen_window),
        **metrics,
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(array.mean()), "std": float(array.std())}


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped["%s_g%s_e%s" % (row["label"], int(row["gen_window"]), int(row["window"]))].append(row)

    per_label = {}
    metric_keys = (
        "demo_mae",
        "demo_rmse",
        "prior_mae",
        "ref_mse",
        "prior_mse",
        "max_jump",
        "mean_vel",
        "mean_jerk",
        "oob_frac",
        "mse_before",
        "mse_after",
    )
    for label, group in grouped.items():
        per_label[label] = {
            key: _mean_std([float(item[key]) for item in group])
            for key in metric_keys
        }
        per_label[label]["n"] = len(group)
        per_label[label]["method"] = group[0]["method"]
        per_label[label]["scale"] = float(group[0]["scale"])
        per_label[label]["window"] = int(group[0]["window"])
        per_label[label]["gen_window"] = int(group[0]["gen_window"])

    guided_rows = [
        {"method": "guided", "scale": float(row["scale"]), "demo_mae": float(row["demo_mae"])}
        for row in rows
        if row["method"] == "guided"
    ]
    recommended = recommend_guidance_scale(guided_rows) if guided_rows else float("nan")
    return {
        "per_label": per_label,
        "recommended_guidance_scale": recommended,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no metric rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def collect(method: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
        windows = [int(row["window"]) for row in rows if "window" in row]
        plot_window = 5 if 5 in windows else (sorted(set(windows))[0] if windows else None)
        grouped: dict[float, list[float]] = defaultdict(list)
        for row in rows:
            if row["method"] != method:
                continue
            if plot_window is not None and int(row["window"]) != plot_window:
                continue
            grouped[float(row["scale"])].append(float(row[metric]))
        xs = np.array(sorted(grouped), dtype=np.float64)
        ys = np.array([np.mean(grouped[x]) for x in xs], dtype=np.float64)
        return xs, ys

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = (
        (axes[0, 0], "demo_mae", "Demo MAE"),
        (axes[0, 1], "max_jump", "Max joint jump"),
        (axes[1, 0], "ref_mse", "DP reference MSE"),
        (axes[1, 1], "prior_mse", "Deviation from unguided prior"),
    )
    for axis, metric, title in panels:
        gx, gy = collect("guided", metric)
        mx, my = collect("linear_mix", metric)
        if gx.size:
            axis.plot(gx, gy, marker="o", label="guided DDIM")
        if mx.size:
            axis.plot(mx, my, marker="s", linestyle="--", label="linear mix")
        axis.set_title(title)
        axis.set_xlabel("guidance scale / mix weight")
        axis.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_trajectories(
    output_dir: Path,
    examples: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    joint_ids = (0, 4, 9, 17)
    for example in examples:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        time = np.arange(example["demo"].shape[0])
        for axis, joint in zip(axes.ravel(), joint_ids):
            axis.plot(time, example["demo"][:, joint], label="demo", color="black")
            axis.plot(time, example["dp"][:, joint], label="DP", linestyle=":")
            axis.plot(time, example["unguided"][:, joint], label="unguided")
            axis.plot(time, example["guided"][:, joint], label="guided")
            axis.plot(time, example["mix"][:, joint], label="linear mix", linestyle="--")
            axis.set_title("joint %d" % joint)
            axis.grid(True, alpha=0.3)
        fig.suptitle(
            "episode %d t=%d global=%d"
            % (example["episode_idx"], example["t_in_episode"], example["global_idx"])
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.tight_layout()
        fig.savefig(
            output_dir
            / ("episode%02d_t%04d.png" % (example["episode_idx"], example["t_in_episode"])),
            dpi=130,
        )
        plt.close(fig)


def _write_failure_cases(path: Path, rows: list[dict[str, object]]) -> None:
    windows = [int(row["window"]) for row in rows if "window" in row]
    use_window = 5 if 5 in windows else None
    by_key: dict[tuple[int, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        if use_window is not None and int(row["window"]) != use_window:
            continue
        key = (int(row["episode_idx"]), int(row["global_idx"]))
        by_key[key][str(row["label"])] = row

    records = []
    for (episode_idx, global_idx), methods in by_key.items():
        guided = methods.get("guided_scale_1.0")
        unguided = methods.get("guided_scale_0.0")
        if guided is None or unguided is None:
            continue
        delta = float(guided["demo_mae"]) - float(unguided["demo_mae"])
        records.append(
            {
                "episode_idx": episode_idx,
                "global_idx": global_idx,
                "t_in_episode": int(guided["t_in_episode"]),
                "guided_mae": float(guided["demo_mae"]),
                "unguided_mae": float(unguided["demo_mae"]),
                "delta_mae": delta,
                "guided_max_jump": float(guided["max_jump"]),
                "unguided_max_jump": float(unguided["max_jump"]),
            }
        )
    records.sort(key=lambda item: item["delta_mae"], reverse=True)
    payload = {
        "guidance_hurt_most": records[:15],
        "guidance_helped_most": list(reversed(records[-15:])) if records else [],
    }
    path.write_text(json.dumps(payload, indent=2))


def _predict_dp(policy, policy_obs: dict[str, torch.Tensor]) -> np.ndarray:
    with torch.inference_mode():
        action = policy.predict_action(policy_obs)["action"].detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 3 or action.shape[2] != ACTION_DIM:
        raise RuntimeError("DP returned unexpected shape %s" % (action.shape,))
    if not np.isfinite(action).all():
        raise RuntimeError("DP proposal contains NaN or Inf")
    return action


def _predict_controller(
    controller: GuidedDDIMController,
    history: np.ndarray,
    reference: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    controller.guidance_scale = float(scale)
    guided, stats = controller.predict(history, reference)
    hand = np.asarray(guided, dtype=np.float32)
    if hand.ndim != 3:
        raise RuntimeError("controller returned unexpected shape %s" % (hand.shape,))
    if not np.isfinite(hand).all():
        raise RuntimeError("controller produced NaN or Inf")
    mse_before = np.asarray(stats.mse_before_batch, dtype=np.float64)
    mse_after = np.asarray(stats.mse_after_batch, dtype=np.float64)
    if mse_before.shape != (hand.shape[0],) or mse_after.shape != (hand.shape[0],):
        raise RuntimeError(
            "controller stats batch %s/%s != action batch %d"
            % (mse_before.shape, mse_after.shape, hand.shape[0])
        )
    return hand, mse_before, mse_after


def _perturb_images(obs: dict[str, np.ndarray], noise: float, rng: np.random.Generator):
    perturbed = dict(obs)
    for key in ("front_image", "wrist_image"):
        delta = rng.normal(0.0, noise, size=obs[key].shape).astype(np.float32)
        perturbed[key] = np.clip(obs[key] + delta, 0.0, 1.0)
    return perturbed


def _perturb_history(history: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    history = np.array(history, dtype=np.float32, copy=True)
    history[..., :HAND_DIM] += rng.normal(0.0, noise, size=history[..., :HAND_DIM].shape).astype(
        np.float32
    )
    qpos = history[..., :HAND_DIM]
    target = history[..., HAND_DIM : 2 * HAND_DIM]
    history[..., 2 * HAND_DIM :] = target - qpos
    return history


def main() -> int:
    args = parse_args()
    _configure_dinov2()

    if args.stride <= 0:
        raise ValueError("stride must be positive")
    if args.max_samples < 0:
        raise ValueError("max_samples cannot be negative")
    if args.execution_steps <= 0:
        raise ValueError("execution_steps must be positive")
    if any(scale < 0 for scale in args.guidance_scales):
        raise ValueError("guidance scales must be non-negative")
    if args.eta != 0.0:
        raise ValueError("eta must be 0.0")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    comparison_lengths = args.comparison_lengths or (args.execution_steps,)
    if any(length <= 0 for length in comparison_lengths):
        raise ValueError("comparison lengths must be positive")
    args.execution_steps = max(args.execution_steps, max(comparison_lengths))
    generation_windows = args.generation_windows
    if generation_windows is not None and any(length <= 0 for length in generation_windows):
        raise ValueError("generation windows must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    zarr_path = resolve_zarr_path(args.zarr_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[data] opening %s" % zarr_path, flush=True)
    replay = zarr.open(str(zarr_path), mode="r")
    state = np.asarray(replay["data/state"][:], dtype=np.float32)
    action = np.asarray(replay["data/action"][:], dtype=np.float32)
    episode_ends = np.asarray(replay["meta/episode_ends"][:], dtype=np.int64)
    front = replay["data/front_image"]
    wrist = replay["data/wrist_image"]
    if state.shape != action.shape or state.shape[1] != ACTION_DIM:
        raise ValueError("state/action must have shape (T, %d)" % ACTION_DIM)

    max_samples = None if args.max_samples == 0 else args.max_samples
    indices = select_eval_indices(
        episode_ends,
        n_obs_steps=4,
        chunk_steps=args.execution_steps,
        stride=args.stride,
        max_samples=max_samples,
    )
    if not indices:
        raise RuntimeError("no evaluation frames selected")
    bounds = episode_bounds(episode_ends)
    demo_hand = action[:, ARM_DIM:]
    lower = demo_hand.min(axis=0)
    upper = demo_hand.max(axis=0)

    print("[dp] loading %s" % args.dp_checkpoint, flush=True)
    real_cfg, real_policy = _load_real_policy(
        args.dp_checkpoint.expanduser().resolve(),
        device,
        args.dp_inference_steps,
    )
    n_obs_steps, action_steps, relative_ee, _rgb_shapes = _real_policy_metadata(
        real_cfg, real_policy
    )
    if relative_ee:
        raise ValueError(
            "offline eval expects an absolute-EE Real DP checkpoint "
            "(task.dataset.relative=false)"
        )
    if n_obs_steps != 2:
        print("[dp] warning: n_obs_steps=%d, expected 2" % n_obs_steps, flush=True)

    print("[controller] loading %s" % args.controller_checkpoint, flush=True)
    controller = GuidedDDIMController(
        args.controller_checkpoint.expanduser().resolve(),
        device,
        inference_steps=args.ddim_inference_steps,
        execution_steps=args.execution_steps,
        guidance_scale=0.0,
        eta=args.eta,
        fixed_noise=bool(args.fixed_noise),
        seed=args.seed,
        allow_salvage=not args.no_salvage,
    )
    if action_steps < controller.reference_steps:
        raise ValueError(
            "DP action chunk %d is shorter than controller guidance %d"
            % (action_steps, controller.reference_steps)
        )
    active_generation_windows = generation_windows or (controller.reference_steps,)
    max_gen = max(active_generation_windows)
    if max_gen > controller.spec["horizon"] - controller.action_start:
        raise ValueError("generation window %d exceeds diffusion horizon" % max_gen)
    if max_gen > action_steps:
        raise ValueError(
            "generation window %d exceeds DP action chunk %d" % (max_gen, action_steps)
        )

    print(
        "[eval] samples=%d batch_size=%d stride=%d exec=%d score=%s gen=%s guide_steps=%d scales=%s mix=%s"
        % (
            len(indices),
            args.batch_size,
            args.stride,
            args.execution_steps,
            comparison_lengths,
            active_generation_windows,
            controller.reference_steps,
            args.guidance_scales,
            args.mix_weights,
        ),
        flush=True,
    )

    rows: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    example_targets = {
        indices[0].global_idx,
        indices[len(indices) // 2].global_idx,
        indices[-1].global_idx,
    }
    perturb_rng = np.random.default_rng(args.seed + 7)
    perturb_every = max(1, len(indices) // max(args.perturb_samples, 1)) if args.perturb_samples else 0
    perturb_stats: dict[str, list[float]] = defaultdict(list)

    ordered_scales = (0.0,) + tuple(
        scale for scale in args.guidance_scales if float(scale) != 0.0
    )
    memory_logged = False
    processed = 0

    def _process_batch(batch: list[ReplayIndex], start_i: int) -> None:
        nonlocal memory_logged
        dp_obs_list = []
        histories = []
        demos = []
        for index in batch:
            episode_start, _episode_end = bounds[index.episode_idx]
            dp_obs_list.append(
                build_dp_observation(
                    front, wrist, state, index.global_idx, n_obs_steps=n_obs_steps
                )
            )
            histories.append(
                build_controller_history(
                    state,
                    action,
                    episode_start=episode_start,
                    global_idx=index.global_idx,
                    n_obs_steps=controller.spec["n_obs_steps"],
                )
            )
            demos.append(hand_chunk(action, index.global_idx, args.execution_steps))

        history = np.stack(histories, axis=0)
        demo = np.stack(demos, axis=0)
        seeds = [int(args.seed + index.global_idx) for index in batch]
        torch.manual_seed(seeds[0])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seeds[0])
        dp_action = _predict_dp(real_policy, _collate_policy_obs(dp_obs_list, device))
        dp_hand_full = dp_action[:, : max(active_generation_windows), ARM_DIM:]
        dp_hand_exec = dp_action[:, : args.execution_steps, ARM_DIM:]
        if dp_hand_exec.shape != demo.shape:
            raise RuntimeError(
                "DP exec chunk %s != demo chunk %s" % (dp_hand_exec.shape, demo.shape)
            )

        if bool(args.fixed_noise):
            controller.set_fixed_noise_from_seeds(seeds)
        else:
            controller.reset_fixed_noise(seed=seeds[0])

        guided_by_scale: dict[float, np.ndarray] = {}
        last_unguided = None
        for gen_w in active_generation_windows:
            controller.set_guidance_horizon(gen_w)
            unguided = None
            for scale in ordered_scales:
                hand, mse_before, mse_after = _predict_controller(
                    controller, history, dp_hand_full, scale
                )
                if float(scale) == 0.0:
                    unguided = hand
                    last_unguided = hand
                if gen_w == active_generation_windows[-1]:
                    guided_by_scale[float(scale)] = hand
                for i, index in enumerate(batch):
                    _append_window_rows(
                        rows,
                        index,
                        "guided",
                        scale,
                        hand[i],
                        demo[i],
                        dp_hand_exec[i],
                        None if unguided is None or float(scale) == 0.0 else unguided[i],
                        lower,
                        upper,
                        comparison_lengths,
                        float(mse_before[i]),
                        float(mse_after[i]),
                        stored_window=gen_w,
                    )
            if unguided is None:
                raise RuntimeError("guidance scales must include 0.0 for the unguided baseline")
            for i, index in enumerate(batch):
                _append_window_rows(
                    rows,
                    index,
                    "dp",
                    None,
                    dp_hand_exec[i],
                    demo[i],
                    dp_hand_exec[i],
                    unguided[i],
                    lower,
                    upper,
                    comparison_lengths,
                    stored_window=gen_w,
                )
                for weight in args.mix_weights:
                    mixed = linear_mix(unguided[i], dp_hand_exec[i], weight)
                    _append_window_rows(
                        rows,
                        index,
                        "linear_mix",
                        weight,
                        mixed,
                        demo[i],
                        dp_hand_exec[i],
                        unguided[i],
                        lower,
                        upper,
                        comparison_lengths,
                        stored_window=gen_w,
                    )
                if gen_w == active_generation_windows[-1] and index.global_idx in example_targets:
                    guided_example = guided_by_scale.get(1.0, next(iter(guided_by_scale.values())))
                    examples.append(
                        {
                            "episode_idx": index.episode_idx,
                            "t_in_episode": index.t_in_episode,
                            "global_idx": index.global_idx,
                            "demo": demo[i],
                            "dp": dp_hand_exec[i],
                            "unguided": unguided[i],
                            "guided": guided_example[i],
                            "mix": linear_mix(unguided[i], dp_hand_exec[i], 0.5),
                        }
                    )
        unguided = last_unguided
        if unguided is None:
            raise RuntimeError("guidance scales must include 0.0 for the unguided baseline")

        if device.type == "cuda" and not memory_logged:
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(
                "[eval] cuda allocated=%.2fGB reserved=%.2fGB peak=%.2fGB batch=%d"
                % (allocated, reserved, peak, len(batch)),
                flush=True,
            )
            memory_logged = True

        for i, index in enumerate(batch):
            sample_i = start_i + i
            if (
                args.perturb_samples > 0
                and sample_i % perturb_every == 0
                and len(perturb_stats.get("image", [])) < args.perturb_samples
            ):
                sample_seed = seeds[i]
                dp_hand_guide = dp_hand_full[i, : controller.reference_steps]
                controller.reset_fixed_noise(seed=sample_seed)
                base_guided, _, _ = _predict_controller(
                    controller, history[i], dp_hand_guide, args.perturb_scale
                )
                noisy_obs = _perturb_images(dp_obs_list[i], args.image_noise, perturb_rng)
                torch.manual_seed(sample_seed)
                noisy_dp = _predict_dp(real_policy, _as_policy_obs(noisy_obs, device))
                noisy_guide = noisy_dp[0, : controller.reference_steps, ARM_DIM:]
                controller.reset_fixed_noise(seed=sample_seed)
                image_guided, _, _ = _predict_controller(
                    controller, history[i], noisy_guide, args.perturb_scale
                )
                noisy_history = _perturb_history(history[i], args.qpos_noise, perturb_rng)
                controller.reset_fixed_noise(seed=sample_seed)
                qpos_guided, _, _ = _predict_controller(
                    controller, noisy_history, dp_hand_guide, args.perturb_scale
                )
                noisy_ref = dp_hand_guide + perturb_rng.normal(
                    0.0, args.ref_noise, size=dp_hand_guide.shape
                ).astype(np.float32)
                controller.reset_fixed_noise(seed=sample_seed)
                ref_guided, _, _ = _predict_controller(
                    controller, history[i], noisy_ref, args.perturb_scale
                )
                perturb_stats["image"].append(rmse(image_guided[0], base_guided[0]))
                perturb_stats["qpos"].append(rmse(qpos_guided[0], base_guided[0]))
                perturb_stats["reference"].append(rmse(ref_guided[0], base_guided[0]))

    def _process_batch_with_oom_split(batch: list[ReplayIndex], start_i: int) -> None:
        try:
            _process_batch(batch, start_i)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            mid = max(1, len(batch) // 2)
            print(
                "[eval] CUDA OOM at batch=%d, splitting into %d + %d"
                % (len(batch), mid, len(batch) - mid),
                flush=True,
            )
            _process_batch_with_oom_split(batch[:mid], start_i)
            _process_batch_with_oom_split(batch[mid:], start_i + mid)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    batches = _iter_batches(indices, args.batch_size)
    iterator = tqdm(batches, desc="offline-guidance") if tqdm is not None else batches
    for batch in iterator:
        _process_batch_with_oom_split(batch, processed)
        processed += len(batch)
        if tqdm is None:
            print("[eval] %d/%d" % (processed, len(indices)), flush=True)

    summary = _summarize(rows)
    summary.update(
        {
            "n_samples": len(indices),
            "n_episodes": int(len(episode_ends)),
            "n_frames": int(state.shape[0]),
            "stride": args.stride,
            "batch_size": args.batch_size,
            "execution_steps": args.execution_steps,
            "comparison_lengths": list(comparison_lengths),
            "generation_windows": list(active_generation_windows),
            "guidance_scales": list(args.guidance_scales),
            "mix_weights": list(args.mix_weights),
            "dp_checkpoint": str(args.dp_checkpoint),
            "controller_checkpoint": str(args.controller_checkpoint),
            "zarr_path": str(zarr_path),
            "relative_ee": False,
            "caveat": (
                "All 50 episodes were used to train the Real DP checkpoint. "
                "This is an in-distribution sanity check, not a generalization or real-robot success proof."
            ),
            "perturbation": {
                key: _mean_std(values) for key, values in perturb_stats.items()
            },
        }
    )
    per_label = summary["per_label"]
    rec = summary["recommended_guidance_scale"]
    beat_windows = sorted({int(item["window"]) for item in per_label.values()})
    beat_window = 5 if 5 in beat_windows else (beat_windows[-1] if beat_windows else 5)
    rec_key = (
        "%s_w%s" % (method_label("guided", rec), beat_window) if np.isfinite(rec) else None
    )
    unguided_key = "%s_w%s" % (method_label("guided", 0.0), beat_window)
    dp_key = "dp_w%s" % beat_window
    best_mix = min(
        (
            item
            for item in per_label.values()
            if item["method"] == "linear_mix" and int(item["window"]) == beat_window
        ),
        key=lambda item: item["demo_mae"]["mean"],
        default=None,
    )
    if rec_key in per_label and unguided_key in per_label:
        summary["beats_unguided"] = (
            per_label[rec_key]["demo_mae"]["mean"] < per_label[unguided_key]["demo_mae"]["mean"]
        )
        summary["beats_dp"] = (
            dp_key in per_label
            and per_label[rec_key]["demo_mae"]["mean"] < per_label[dp_key]["demo_mae"]["mean"]
        )
        summary["beats_best_linear_mix"] = (
            best_mix is not None
            and per_label[rec_key]["demo_mae"]["mean"] < best_mix["demo_mae"]["mean"]
        )
        if best_mix is not None:
            summary["best_linear_mix_weight"] = best_mix["scale"]

    _write_csv(output_dir / "per_sample_metrics.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot_curves(output_dir / "scale_comparison.png", rows)
    _plot_trajectories(output_dir / "joint_trajectory_examples", examples)
    _write_failure_cases(output_dir / "failure_cases.json", rows)

    print("[done] wrote %s" % output_dir, flush=True)
    print(json.dumps({k: summary[k] for k in (
        "n_samples",
        "recommended_guidance_scale",
        "beats_unguided",
        "beats_dp",
        "beats_best_linear_mix",
        "caveat",
    ) if k in summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
