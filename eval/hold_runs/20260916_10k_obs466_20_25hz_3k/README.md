# Exact 25-Hz and 20-Hz add-on

User requested 25 Hz and 20 Hz after adding 40 Hz. This independent pipeline waits for `20260916_10k_obs466_40hz_3k` to complete and GPU compute to be idle. No existing evaluator/results are modified. Order: matching 30-Hz bridge, 25 Hz, 20 Hz. Each has 3000 first episodes and a 400-s simulated cap (12000/10000/8000 actions respectively).

Same obs_4-66.ckpt prior, original 10k guide, original normalizers, TensorRT FP16/fused updates, DDIM4/4, guide2/execute2, scale25, seed8/100008, WAIT=1. No added real-hand step clamp. No hardware commands.

Exact 25 Hz cannot use integer decimation on either preceding 180-Hz or 360-Hz outer physics clocks. All three arms here use 300-Hz outer physics with one substep, decimations 10/12/15 for 30/25/20 Hz. Integration step is 1/300 s, different from earlier 1/360 s. The added matched 30-Hz bridge is necessary to separate the timestep change from frequency effects. Report these results as a separate matched comparison, not as identical physics to the previous sweeps.

World-time perturbation schedule is unchanged: 30-Hz force updates, first five of every ten physics ticks loaded, maintaining the 1/60-s force pulse. Failure accumulation and stable dwell use 300-Hz ticks with ten times old 30-Hz thresholds. History4 and execute2 remain in policy steps. Four-environment/two-second smoke checks precede full runs; force-clock CPU checks and syntax/integer-duration checks already passed. Validate equal initial states against this bridge AND the original sweep, unique episode0 records, complete physical duration, zero WAIT holds, exact force tick counts and existing numerical gates. Teardown139 accepted only with complete validated output; no automatic retry/overwrite.

State and reports: `state.json`, `pipeline.log`, per-arm directories, `comparison.md/json`. Service: `frequency-10k-obs466-20-25hz-20260916.service`. Shared 45-minute heartbeat must remain active until all three frequency pipelines finish. Final report should identify the matched 30-Hz bridge for each physics grouping and avoid treating simulation control Hz as measured real-hardware throughput.
