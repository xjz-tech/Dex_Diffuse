# Active experiment supervision, 2026-09-14

Monitors parallel-guidance-refine-seed8.service and
parallel-guidance-candidates-refine-seed8.service every minute.
Only 0.85, 0.80, dynamic and five new candidates are scheduled, all seed8.
Cancelled 0.60 is not recovered. Previously completed 0.95/0.90/0.70 are not rerun.
Fresh noise, 3000 first episodes, full9 guidance, DDIM4 and execution2 remain.

watch-state.json and incidents/ contain current monitoring/diagnosis/retry state.
Previous 0.60 segmentation-fault incident is preserved in experiment_watchdog_20260913.
Integrity was deliberately recaptured after this user-authorized matrix update.
User-authorized startup exception: verified Isaac Gym segfaults before task
creation and with zero request/episode/fusion data retry unchanged at most twice
per stage, even if the native root cause is unknown. This deterministic rule
does not depend on a model call; it records evidence and an attributed local
decision in incidents/. Other unknown/code/numerical faults still stop for
review. GPU-idle, integrity, pause and intentional-stop guards remain in force.
The watchdog integrity baseline was deliberately refreshed for this update;
running frozen experiment code and parameters were not modified.

Current source documentation: docs/experiment_watchdog.md in parallel-guidance-1b-base.

2026-09-14 14:52 user explicitly requested resuming the remaining candidates.
The temporary latency-benchmark PAUSED flag was removed. Original transient
units had been garbage-collected after stopping; persistent (not boot-enabled)
user unit definitions now retain candidate restartability. The first-queue unit
is a completion guard only, since all its experiments already finished. Candidate
arguments, frozen code, checkpoints and manifest remain unchanged. All 408
protected file/checkpoint entries were verified before reviewing the new unit
definitions and deliberately refreshing integrity. The previous integrity and
batch-1 benchmark JSON were retained as before_resume_20260914_1452 audit copies.
The frozen suite reruns its numerical/timing gate, skips both completed shared_x0
groups, archives interrupted dual_x0_ramp data, then restarts that group and runs
late_x0_w10 and late_x0_w30. No TRT implementation is substituted into simulation.
