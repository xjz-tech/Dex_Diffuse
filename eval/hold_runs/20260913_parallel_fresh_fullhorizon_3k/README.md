# Fresh-noise full-horizon DDIM guidance suite

STOPPED intentionally at the user's request: do not rerun existing serial
controls. Any seed8 records here are interrupted and must not enter comparisons.
Replacement: 20260913_parallel_fresh_only_3k (fixed/dynamic only).

Source commit: 5f682e7 (feature development worktree parallel-guidance-1b-base).
Frozen execution worktree: /home/carus/Program/Dexterous_Manipulation/Dex_diffuse/.worktrees/parallel-fresh-noise-run-20260913
Do not edit this execution copy while its jobs are running.

All evaluation uses FIXED_NOISE=0; seeds 8,19,25; 3000 first episodes per run;
four DDIM steps; full nine-action guidance; two executed actions. Serial guide
seed is base seed + 100000. Shared parallel branches use one fresh noise draw
per request. Base: /home/carus/data_usb/obs_4-66.ckpt.
Guide: runs/sim_hand_10k_fullnorm_seed42/checkpoints/latest.ckpt in the main repo.
Reuse the completed 200-epoch shared-normalizer retraining, not the original
10k checkpoint. Hold thresholds are explicitly pinned by the launcher.

Order: numerical check, serial scale25 (3 seeds), fixed base coefficients
.95/.90/.70/.60 (3 seeds each), dynamic scale25 (3 seeds), summary.
Total 18 evaluation runs; old fixed-noise seed8 is NOT reused.

User service: parallel-guidance-fresh-fullhorizon-3k.service
Initial supervisor PID: 1707797. The two viewer PIDs 1333322/1333323 had already
exited when rechecked after the user's approval; no kill was necessary.
Status: state.env. Log: pipeline.log. Each run saves parameters.env.
The queue waits for GPU compute jobs before starting and does not kill unrelated
jobs. Failures stop the queue and attempt a desktop notification.

Old fixed-noise results remain in 20260912_parallel_1Bbase_matched_hold_fullhorizon_3k.
The separate 45.88% reference used guide2 and the original checkpoint, so it
is not an exactly matched control even after this fresh-noise update.
