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


def summarize_episode_records(records):
    reasons = [str(item.get("reason", "")) for item in records]
    failure_lengths = [
        int(item["length"])
        for item in records
        if item.get("reason") == "failure"
    ]
    summary = {
        "n_episodes": len(records),
        "n_failure": sum(reason == "failure" for reason in reasons),
        "n_timeout": sum(reason == "timeout" for reason in reasons),
        "n_success": sum(reason == "success" for reason in reasons),
        "mean_failure_length": None,
        "median_failure_length": None,
    }
    if failure_lengths:
        ordered = sorted(failure_lengths)
        summary["mean_failure_length"] = float(sum(ordered)) / float(len(ordered))
        mid = len(ordered) // 2
        if len(ordered) % 2:
            summary["median_failure_length"] = float(ordered[mid])
        else:
            summary["median_failure_length"] = (
                float(ordered[mid - 1] + ordered[mid]) / 2.0
            )
    return summary


def format_hold_summary(label, summary):
    mean_length = summary["mean_failure_length"]
    median_length = summary["median_failure_length"]
    return (
        "%s | episodes=%d failures=%d timeouts=%d successes=%d "
        "mean_failure_length=%s median_failure_length=%s"
        % (
            label,
            summary["n_episodes"],
            summary["n_failure"],
            summary["n_timeout"],
            summary["n_success"],
            "na" if mean_length is None else "%.1f" % mean_length,
            "na" if median_length is None else "%.1f" % median_length,
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

