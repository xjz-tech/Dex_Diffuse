"""Summarize 3000 independent first episodes by actual turn direction."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "seed42"
FIELDS = ("target_abs_joint_delta_mean_rad", "measured_abs_joint_delta_mean_rad")


def main():
    assert (RUN / "RUN_COMPLETE").exists(), "run is not complete"
    paths = sorted(RUN.glob("trajectory_*.npz"))
    assert paths, "no trajectories"
    chunks = {direction: {field: [] for field in FIELDS} for direction in ("left", "right")}
    env_sum = {direction: {field: np.zeros(3000, np.float64) for field in FIELDS} for direction in chunks}
    env_count = {direction: np.zeros(3000, np.int64) for direction in chunks}
    total_first_steps = 0
    last_global_step = 0

    for path in paths:
        with np.load(path) as z:
            ep = z["episode"]
            ep_step = z["episode_step"]
            speed = z["right_speed_deg_s"]
            step = z["global_step"]
            assert ep.shape[1] == 3000 and np.isfinite(speed).all()
            assert int(step[0]) == last_global_step + 1
            last_global_step = int(step[-1])
            valid = (ep == 0) & (ep_step >= 2) & (ep_step <= 900)
            total_first_steps += int(valid.sum())
            for direction, mask in (("right", valid & (speed > 5)), ("left", valid & (speed < -5))):
                env_count[direction] += mask.sum(axis=0)
                for field in FIELDS:
                    values = z[field]
                    assert np.isfinite(values).all() and np.all(values >= 0)
                    chunks[direction][field].append(values[mask].astype(np.float64))
                    env_sum[direction][field] += np.where(mask, values, 0).sum(axis=0)

    assert last_global_step == 900, last_global_step
    records = [json.loads(line) for line in (RUN / "episodes.jsonl").read_text().splitlines()]
    assert len(records) == 3000 and {r["env"] for r in records} == set(range(3000))
    result = {
        "complete": True,
        "environments": 3000,
        "global_steps": last_global_step,
        "first_episode_valid_control_steps": total_first_steps,
        "excluded_initial_action_step": True,
        "native_end_reasons": {key: sum(r["reason"] == key for r in records) for key in sorted({r["reason"] for r in records})},
        "directions": {},
    }
    for direction in ("left", "right"):
        counts = env_count[direction]
        item = {"motion_frames": int(counts.sum()), "environments_with_motion": int((counts > 0).sum()), "metrics": {}}
        for field in FIELDS:
            values = np.concatenate(chunks[direction][field])
            assert len(values) == item["motion_frames"]
            per_env = env_sum[direction][field][counts > 0] / counts[counts > 0]
            item["metrics"][field] = {
                "frame_weighted_mean_rad": float(values.mean()),
                "frame_median_rad": float(np.median(values)),
                "frame_p95_rad": float(np.percentile(values, 95)),
                "equal_environment_mean_rad": float(per_env.mean()),
            }
        result["directions"][direction] = item
    paired = (env_count["left"] > 0) & (env_count["right"] > 0)
    rng = np.random.default_rng(42)
    result["paired_environments_with_both_directions"] = int(paired.sum())
    result["paired_left_minus_right_rad"] = {}
    for field in FIELDS:
        left_mean = env_sum["left"][field][paired] / env_count["left"][paired]
        right_mean = env_sum["right"][field][paired] / env_count["right"][paired]
        differences = left_mean - right_mean
        boot = np.empty(2000, np.float64)
        for i in range(len(boot)):
            boot[i] = differences[rng.integers(0, len(differences), len(differences))].mean()
        result["paired_left_minus_right_rad"][field] = {
            "mean": float(differences.mean()),
            "bootstrap_95pct_interval": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "environments_with_left_greater": int((differences > 0).sum()),
        }
    result["slow_frames"] = total_first_steps - sum(v["motion_frames"] for v in result["directions"].values())
    (ROOT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
