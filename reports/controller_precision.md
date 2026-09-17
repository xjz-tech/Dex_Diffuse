# obs_4-66 controller: timing and numerical accuracy

Checkpoint: `runs/obs_4-66.ckpt` (EMA). RTX 4090, Torch 2.4.0+cu121,
Torch-TensorRT 2.4.0, TensorRT 10.1.0. DDIM 4, batch 1, execute 2.

The benchmark uses 64 matched observation/noise pairs: one zero input and
63 synthetic histories near training joint means, clipped to joint limits.
Each history has consistent `[qpos, target, target-qpos]` fields. These are
synthetic observations, not recorded hardware episodes. Action errors are
measured before the runtime safety clamps.

Latency is synchronized wall time including copying actions to CPU, after
12 warmup rounds and 100 measured rounds per mode. Modes are randomly
interleaved to reduce drift from background load and GPU clocks. Compilation,
checkpoint loading, hardware communication, and the two 30 Hz execution
periods are excluded. Times represent this machine's load during this test;
earlier sequential runs measured eager inference near 11 ms.

| Mode | Mean / median / p95 (ms) | First 2 actions MAE (rad) | First 2 actions max (rad) | All 5 actions max (rad) |
|---|---|---|---|---|
| Original PyTorch DDIM | 24.01 / 23.87 / 26.12 | 0 | 0 | 0 |
| PyTorch + fused DDIM | 20.96 / 20.88 / 22.03 | 0.00003130 | 0.0002860 | 0.0002894 |
| TensorRT FP16 + fused | 2.87 / 2.82 / 3.62 | 0.0005052 | 0.006158 | 0.008123 |
| TensorRT FP32 + fused | 9.82 / 9.76 / 11.18 | 0.00004091 | 0.0002647 | 0.0006772 |

Errors above use the original PyTorch settings as reference (cuDNN TF32
enabled). Against strict FP32 PyTorch with TF32 disabled, TensorRT FP32
first-two-action MAE is 1.03e-7 rad, max is 6.78e-7 rad; all-five max is
1.85e-6 rad. This separates export error from the original backend's own
precision differences.

Both cached engines include the existing GroupNorm export correction
(`groupnorm-layernorm-v1`). The FP16 cache present at benchmark start already
had this correction, so this run does not measure the uncorrected old engine.
FP32 was compiled and cached during this task. The obs66 shell launcher's
TensorRT precision now defaults to FP32; FP16 remains selectable explicitly.
TensorRT itself still requires `TENSORRT=1`.

Reproduce offline (does not connect to hardware):

```bash
env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  LD_LIBRARY_PATH=/home/frankagvl/anaconda3/envs/dexIL/lib \
  /home/frankagvl/anaconda3/envs/dexIL/bin/python \
  eval/bench_controller_precision.py
```

Full numerical results: `reports/controller_precision.json`.
