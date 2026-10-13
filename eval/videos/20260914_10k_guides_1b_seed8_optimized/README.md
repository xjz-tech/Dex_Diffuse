# Optimized original 10k -> 1B serial visualization

Same original checkpoints, separate original normalizers, guide2/execute2,
predict9, scale25, DDIM4 per model, seed8/guide seed100008, fresh noise, and
historical hold thresholds as the FP32 recording. One environment, 1800
control steps, 1280x720 H.264 at 30 fps, wall-clock paced, no success selection.
This is a visualization, not a 3000-environment effectiveness measurement.

Opt-in server: `.worktrees/parallel-guidance-1b-base/eval/optimized_serial_server.py`.
Uses existing TRT UNet and precomputed DDIM implementation. Removes intermediate
CPU transfers and per-step diagnostic/autograd overhead using the same analytical
frozen-epsilon guidance gradient. It preserves guide-to-base action normalizer
conversion. Only this recording uses it; frozen experiment entry points unchanged.

Startup gate against the original serial controller on three synthetic
observations with identical initial noise: FP32 maximum action deviation
0.000133; TRT FP16 maximum action deviation 0.004570 (physical joint coordinates).
These gates do not establish equal rollout success rate. Existing update-loop
tests: 16 passed. Exact numerical checks and timings: optimization_validation.json.

Completed video: 20260914_171153_env0.mp4, 1800 steps / 900 requests,
122.667 seconds, 3680 encoded frames. Entire MP4 decoded with ffmpeg -xerror
successfully; preview inspected. No drops or resets during this clip. Actual
request latency mean 6.541 ms, median 6.500 ms; unoptimized recording mean
22.710 ms, median 22.338 ms. Control loop 14.68 Hz vs about 13.1 Hz previously.
After finalizing the video, Isaac Gym again segfaulted during teardown (exit139).
Do not classify the process as a clean exit; the complete artifact is valid.

Isolated timing before simulation startup, 20 warmups then 100 requests,
GPU observation to ready CPU action: mean 5.908 ms, p50 5.744 ms, p95 6.606 ms.
Use requests.jsonl for actual in-simulation request latency; that includes the
shared server's request-side output validation but not renderer/physics time.

The preceding FP32 video completed all 1800 steps and finalized its MP4, then
Isaac Gym segfaulted during teardown. The finalized video was successfully
decoded and previewed. Do not treat its process exit as a successful clean exit.
