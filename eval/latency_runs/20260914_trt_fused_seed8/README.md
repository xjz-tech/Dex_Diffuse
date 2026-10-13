# Isolated TRT + fused benchmark, 2026-09-14

User requested pausing simulation to compare optimized inference latency.
The candidate service and watchdog timer/service were stopped; the watchdog
PAUSED flag was installed. No automatic resume after these measurements.
Completed simulation results and the interrupted dual_x0_ramp directory remain
in their original locations. Its state reports `suite stopped` due to this
intentional stop, not a new experiment crash. Resume must preserve/archive
partial data using the existing queue mechanism.

Source: parallel-guidance-1b-base worktree, eval/bench_optimized_guidance.py.
Frozen simulation worktrees and their policies were not changed.

Protocol: one otherwise idle RTX 4090, batch 1, DDIM 4 per model, fresh noise,
full 9-action guidance, 2 executed actions. Base obs_4-66.ckpt and shared-base-
normalizer 10k checkpoint. Both UNets use TRT FP16 with batch-1 optimization;
IO and update arithmetic are FP32. DDIM coefficients are precomputed, and the
serial frozen-epsilon MSE gradient is evaluated analytically. This preserves
the algorithm, but FP16 is not claimed to preserve task outcomes without eval.
No diagnostics or device synchronization inside the sampling loop; stream
dependencies remain. Inputs stay on GPU, outputs are copied to CPU for timing.
No simulator, socket round trip, compilation or warmup in the measured interval.
Methods run individually, 20 warmups then 100 measured calls per round, 3 rounds.
GPU ownership is checked before and after each method's block.

`fused` uses the existing repo meaning: precomputed PyTorch DDIM update loop,
not a whole-trajectory CUDA Graph or a single fused CUDA kernel.

run01: retained failed strict whole-trajectory FP32 comparison. Identical-input
single-step differences were only ~2.4e-7 but amplified to ~3.6e-4 after UNets.
run02: strict local formula tests plus separately recorded full-trajectory drift,
TRT precision drift and ordered/streams equivalence, followed by isolated timing.
See the JSON status before treating a run as completed evidence.
