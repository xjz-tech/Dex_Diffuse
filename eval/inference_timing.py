"""Wall-clock inference timing, including completion of queued CUDA work."""

import time

import torch


class InferenceTimer:
    def __init__(self, device, label):
        self.device = torch.device(device)
        self.label = label
        self.elapsed = 0.0

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __enter__(self):
        self._synchronize()
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self._synchronize()
            self.elapsed = time.perf_counter() - self.started
            print(f"[latency] {self.label} inference_ms={self.elapsed * 1000:.3f}", flush=True)
        return False
