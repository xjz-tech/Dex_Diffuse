import json
import sys
from pathlib import Path

import pytest


EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from episode_stats import (  # noqa: E402
    EpisodeRecorder,
    censored_timeout_records,
    format_hold_summary,
    summarize_episode_records,
)


def test_recorder_stops_after_configured_failure_episodes(tmp_path):
    log_path = tmp_path / "episodes.jsonl"
    recorder = EpisodeRecorder(log_path, max_failure_episodes=2)

    assert recorder.record(_episode(length=10, reason="timeout")) is False
    assert recorder.record(_episode(length=40, reason="failure")) is False
    assert recorder.record(_episode(length=55, reason="failure")) is True
    assert recorder.failure_count == 2

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[-1])["length"] == 55


def test_recorder_without_failure_limit_never_requests_stop(tmp_path):
    recorder = EpisodeRecorder(tmp_path / "episodes.jsonl", max_failure_episodes=0)
    for _ in range(5):
        assert recorder.record(_episode(length=12, reason="failure")) is False
    assert recorder.failure_count == 5


def test_summarize_uses_failure_lengths_only():
    summary = summarize_episode_records(
        [
            _episode(length=10, reason="timeout"),
            _episode(length=20, reason="failure"),
            _episode(length=40, reason="failure"),
            _episode(length=30, reason="success"),
        ]
    )

    assert summary["n_episodes"] == 4
    assert summary["n_failure"] == 2
    assert summary["n_timeout"] == 1
    assert summary["n_success"] == 1
    assert summary["mean_failure_length"] == pytest.approx(30.0)
    assert summary["median_failure_length"] == pytest.approx(30.0)


def test_summarize_empty_failures_is_none():
    summary = summarize_episode_records([_episode(length=10, reason="timeout")])
    assert summary["n_failure"] == 0
    assert summary["mean_failure_length"] is None
    assert summary["median_failure_length"] is None
    assert summary["mean_hold_length"] == pytest.approx(10.0)
    assert summary["median_hold_length"] == pytest.approx(10.0)
    assert summary["completion_rate"] == pytest.approx(1.0)


def test_summarize_hold_includes_cap_timeouts_in_mean_and_median():
    summary = summarize_episode_records(
        [
            _episode(length=100, reason="failure"),
            _episode(length=200, reason="failure"),
            _episode(length=12000, reason="timeout"),
            _episode(length=12000, reason="timeout"),
        ]
    )

    assert summary["n_episodes"] == 4
    assert summary["n_failure"] == 2
    assert summary["n_timeout"] == 2
    assert summary["mean_failure_length"] == pytest.approx(150.0)
    assert summary["mean_hold_length"] == pytest.approx(6075.0)
    assert summary["median_hold_length"] == pytest.approx(6100.0)
    assert summary["median_hold_s"] == pytest.approx(6100.0 / 30.0)
    assert summary["completion_rate"] == pytest.approx(0.5)


def test_format_hold_summary_reports_median_seconds_not_mean_hold():
    summary = summarize_episode_records(
        [
            _episode(length=100, reason="failure"),
            _episode(length=200, reason="failure"),
            _episode(length=12000, reason="timeout"),
            _episode(length=12000, reason="timeout"),
        ]
    )
    text = format_hold_summary("hold", summary)

    assert "median_hold=203.3s" in text
    assert "mean_hold_length" not in text
    assert "median_hold_length" not in text


def test_censored_timeout_records_use_cap_length():
    records = censored_timeout_records(
        open_envs=(0, 3, 7),
        cap=12000,
        run_name="scale25",
    )
    assert [item["env"] for item in records] == [0, 3, 7]
    assert {item["reason"] for item in records} == {"timeout"}
    assert {item["length"] for item in records} == {12000}
    assert format_hold_summary("hold", summarize_episode_records(records)).startswith(
        "hold |"
    )


def _episode(length, reason):
    return {
        "env": 0,
        "episode": 0,
        "length": length,
        "reason": reason,
        "reward": 0.0,
    }
