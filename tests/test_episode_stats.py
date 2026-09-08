import json
import sys
from pathlib import Path

import pytest


EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from episode_stats import (  # noqa: E402
    EpisodeRecorder,
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


def _episode(length, reason):
    return {
        "env": 0,
        "episode": 0,
        "length": length,
        "reason": reason,
        "reward": 0.0,
    }
