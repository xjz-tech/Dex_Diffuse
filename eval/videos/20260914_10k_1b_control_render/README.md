# One viewer render per control step

Original 10k -> 1B, TRT FP16 + fused, DDIM4 per model, predict9 / guide2 /
execute2, scale25, fresh noise seed8 / guide seed100008. Original normalizers
and hold thresholds preserved. 1 environment, 1800 control steps. This is a
visualization only, not a task-success-rate measurement.

RecordingRuntime now deduplicates viewer rendering and frame pacing by
control_steps. Quit events are still polled every physics step. Physics remains
dt=0.0166667 and controlFrequencyInv=2; both simulate calls are untouched.
Full control-period sleep is applied only once per control step. Video camera
capture remains once per control step and retains wall-clock pacing.

An intermediate recording (../20260914_10k_1b_fixed_pacing) only changed the
sleep interval per physics step and still drew the viewer twice; that was
insufficient to achieve 25 Hz. Both intermediate and original videos retained.

22 tests passed for viewer pacing and DDIM formulas. Startup validates optimized
actions against the original serial controller. FP16 numerical drift does not
establish rollout equivalence. Experiments and watchdog remain paused.

Completed: 20260914_172601_env0.mp4, 75.267 seconds / 2258 frames, full
1800 control steps, no drops/resets in this clip. Full-video ffmpeg -xerror
decode passed and preview inspected. Actual loop mean 23.94 Hz (not a claim
of sustained 25 Hz), versus 14.68 Hz before the fix. 900 requests, mean
6.536 ms and median 6.506 ms. The physics/control timestep is unchanged.
Isaac Gym again segfaulted during teardown AFTER saving the video; process
exit139 is not a clean exit even though the complete MP4 is valid.
