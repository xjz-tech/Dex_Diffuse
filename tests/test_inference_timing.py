import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import inference_timing


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_timer_waits_for_gpu_and_reports_milliseconds(device, monkeypatch, capsys):
    events = []
    ticks = iter((10.0, 10.125))

    def clock():
        events.append("clock")
        return next(ticks)

    monkeypatch.setattr(inference_timing.time, "perf_counter", clock)
    monkeypatch.setattr(inference_timing.torch.cuda, "synchronize", lambda d: events.append("sync"))
    with inference_timing.InferenceTimer(device, "dp") as timer:
        events.append("predict")
    assert timer.elapsed == 0.125
    assert "[latency] dp inference_ms=125.000" in capsys.readouterr().out
    assert events == (["sync", "clock", "predict", "sync", "clock"]
                      if device.startswith("cuda") else ["clock", "predict", "clock"])


def test_failed_inference_does_not_print_success_timing(capsys):
    with pytest.raises(ValueError, match="failed"):
        with inference_timing.InferenceTimer("cpu", "dp"):
            raise ValueError("failed")
    assert capsys.readouterr().out == ""
