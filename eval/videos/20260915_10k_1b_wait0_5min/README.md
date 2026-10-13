# 10k guide → 1B, WAIT=0, five-minute recording

Queued after the scale sweep, earlier-epoch sweep, and 1k-guide train/evaluation queue have all completed, and only with an idle GPU. No GPU execution during preparation.

Reference: ../20260914_10k_1b_until_failure/record.sh. Same model files, original TensorRT FP16/fused DDIM implementation and validation gates, guide2/execute2, scale25, DDIM4/4, seeds8/100008, fresh noise, camera, physics and failure settings. No added joint step clamp. Independent source snapshot avoids changes to running experiments.

Only behavioral change: model requests execute on a worker while the simulator continues stepping with its most recently issued absolute target. Responses are consumed in order, executing both predicted steps before the next request. A pending request uses an immutable observation history. Physics updates the history on every control step, including hold steps. The existing once-per-control-step rendering/pacing remains unchanged.

Recording automatically stops at 300 seconds from the recorder start or on the first failure, whichever occurs first. Original MP4 writer is capped at 9000 frames at 30 FPS. A short failure clip is retained rather than replaced by another episode. No speed-up or added overlays.

Run `/usr/bin/python3 -B run_queued.py` to inspect readiness. The task heartbeat calls it with `--start` once all predecessors are complete; this launches one dedicated transient systemd service. The worker independently rechecks readiness and immutable source/checkpoint fingerprints. State and console logs are saved here. Known post-finalization Isaac Gym exit139 is accepted only with a valid stop summary and full successful MP4 decode. Failed or running states are never automatically rerun.

CPU validation: test_wait0.py verifies nonblocking holds, observation snapshots, ordered two-action execution, and the exact video frame cap. GPU integration remains queued for idle resources.

## Completed 2026-09-15

Video: 20260915_175834_env0.mp4, 124.000 seconds, 3720 frames. First environment failure at control step 3262 (108.733 seconds simulated time), before the 300-second wall-clock cap. 2174 predicted-action steps and 1088 hold-last-target steps verify WAIT=0 was exercised. Approximate control throughput 26.34 Hz; model-reported average inference 7 ms. Full MP4 decode passed and overview/near-terminal frames visually checked. Isaac Gym exit139 occurred after MP4 finalization, as in the reference recording. Single trial and environment failure flag: this is not proof of a physical drop or proof of the real-hardware inward-drift cause. Heartbeat paused after delivery.
