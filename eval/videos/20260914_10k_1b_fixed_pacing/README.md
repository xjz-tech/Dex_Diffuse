# TRT original 10k -> 1B, corrected viewer pacing

Same configuration as the previous optimized recording: original 10k checkpoint
and normalizer; original 1B base; predict9, guide2, execute2, scale25, DDIM4 per
model, fresh noise, seed8 / guide seed100008, one environment, 1800 control steps.
Not a quantitative evaluation of task success. Exact environment: record.sh.

Only recording viewer pacing changed: VecTask renders once per physics step
(two per control step), so the default sleep interval is now physics dt instead
of control dt. Physics dt=0.0166667 and controlFrequencyInv=2 remain unchanged.
The nominal simulated control rate remains ~30 Hz. Wall-clock rate is measured,
not forced to 25 Hz. Video retains wall-clock timestamps (not post-hoc sped up).

Prior artifacts are retained. Experiments/watchdog remain paused at user request.
Known unrelated Isaac Gym teardown crash may occur after MP4 finalization; check
the full decode and control-step count separately from process exit status.
