import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from trt_unet import TensorRTUnetAdapter, _cuda_timestep  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for TensorRT adapter tests"
)


class _FakeCompiled(nn.Module):
    def forward(self, sample, timestep, global_cond):
        assert sample.is_cuda
        assert timestep.is_cuda
        assert timestep.dtype == torch.int64
        assert timestep.shape == (sample.shape[0],)
        return sample * 0.0 + timestep.float().view(-1, 1, 1)


def test_cuda_timestep_promotes_python_int():
    sample = torch.zeros(1, 12, 22, device="cuda")
    out = _cuda_timestep(50, sample)
    assert tuple(out.shape) == (1,)
    assert out.device.type == "cuda"
    assert int(out.item()) == 50


def test_cuda_timestep_moves_cpu_scalar_tensor():
    sample = torch.zeros(1, 12, 22, device="cuda")
    cpu_t = torch.tensor(80)
    out = _cuda_timestep(cpu_t, sample)
    assert out.device.type == "cuda"
    assert int(out.item()) == 80


def test_adapter_accepts_scheduler_cpu_timestep():
    adapter = TensorRTUnetAdapter(_FakeCompiled().cuda())
    sample = torch.ones(1, 12, 22, device="cuda")
    cond = torch.zeros(1, 264, device="cuda")
    out = adapter(sample, torch.tensor(12), local_cond=None, global_cond=cond)
    assert out.shape == sample.shape
    assert torch.allclose(out, torch.full_like(sample, 12.0))
