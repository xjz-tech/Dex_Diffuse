"""Record per-episode eval outcomes and summarize hold-before-failure stats."""

from __future__ import annotations

import json
from pathlib import Path


class EpisodeRecorder:
    def __init__(self, log_path, max_failure_episodes=0):
        self.log_path = Path(log_path) if log_path else None
        self.max_failure_episodes = int(max_failure_episodes)
        self.failure_count = 0
        self.records = []
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")

    def record(self, episode):
        record = dict(episode)
        self.records.append(record)
        if record.get("reason") == "failure":
            self.failure_count += 1
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return self.should_stop()

    def should_stop(self):
        return (
            self.max_failure_episodes > 0
            and self.failure_count >= self.max_failure_episodes
        )


def load_episode_records(log_path):
    path = Path(log_path)
    records = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def censored_timeout_records(open_envs, cap, run_name=""):
    """Record still-running episodes at the step cap as completed holds."""
    records = []
    for env_id in open_envs:
        records.append(
            {
                "run": run_name,
                "env": int(env_id),
                "episode": 0,
                "length": int(cap),
                "reason": "timeout",
                "reward": 0.0,
            }
        )
    return records


# Isaac Gym hydra_config: sim.dt = 1/60, controlFrequencyInv = 2 → one eval step.
DEFAULT_STEP_DT = 2.0 / 60.0


def _mean_median(lengths):
    if not lengths:
        return None, None
    ordered = sorted(lengths)
    mean = float(sum(ordered)) / float(len(ordered))
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = float(ordered[mid])
    else:
        median = float(ordered[mid - 1] + ordered[mid]) / 2.0
    return mean, median


def summarize_episode_records(records, step_dt=DEFAULT_STEP_DT):
    reasons = [str(item.get("reason", "")) for item in records]
    failure_lengths = [
        int(item["length"])
        for item in records
        if item.get("reason") == "failure"
    ]
    hold_lengths = [
        int(item["length"])
        for item in records
        if item.get("reason") in ("failure", "timeout", "success")
    ]
    n_episodes = len(records)
    n_timeout = sum(reason == "timeout" for reason in reasons)
    n_success = sum(reason == "success" for reason in reasons)
    mean_failure, median_failure = _mean_median(failure_lengths)
    mean_hold, median_hold = _mean_median(hold_lengths)
    median_hold_s = (
        None if median_hold is None else float(median_hold) * float(step_dt)
    )
    summary = {
        "n_episodes": n_episodes,
        "n_failure": sum(reason == "failure" for reason in reasons),
        "n_timeout": n_timeout,
        "n_success": n_success,
        "mean_failure_length": mean_failure,
        "median_failure_length": median_failure,
        "mean_hold_length": mean_hold,
        "median_hold_length": median_hold,
        "median_hold_s": median_hold_s,
        "completion_rate": (
            float(n_timeout + n_success) / float(n_episodes) if n_episodes else None
        ),
    }
    return summary


def format_hold_summary(label, summary):
    median_hold_s = summary.get("median_hold_s")
    completion = summary.get("completion_rate")
    return (
        "%s | episodes=%d failures=%d timeouts=%d successes=%d "
        "completion_rate=%s median_hold=%s"
        % (
            label,
            summary["n_episodes"],
            summary["n_failure"],
            summary["n_timeout"],
            summary["n_success"],
            "na" if completion is None else "%.3f" % completion,
            "na" if median_hold_s is None else "%.1fs" % median_hold_s,
        )
    )


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize hold-before-failure episode JSONL logs."
    )
    parser.add_argument("logs", nargs="+", help="Episode JSONL files")
    args = parser.parse_args(argv)
    for log_path in args.logs:
        summary = summarize_episode_records(load_episode_records(log_path))
        print(format_hold_summary(log_path, summary))


if __name__ == "__main__":
    main()

