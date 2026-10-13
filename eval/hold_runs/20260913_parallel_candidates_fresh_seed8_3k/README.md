# Active candidate queue: seed8 only

Predecessor: parallel-guidance-fresh-only.service, PID1711253/start217427719.
Its current fixed_c0.95_seed8 run continues uninterrupted, then only .90/.70/.60
and dynamic seed8 run. Seed19/25 entries are skipped without GPU work.

This queue runs five candidates, seed8 only, 3000 environments each:
shared_x0_w10, shared_x0_w30, dual_x0_ramp, late_x0_w10, late_x0_w30.
All task evaluation uses fresh noise, full nine-action guidance, two executed
actions, four DDIM steps, and shared-normalizer retrained 10k correction.
No training or serial simulation is scheduled.

Effect reference: seed8 from 20260912_fresh_noise_3000_seed42_8_19_25,
46.50% hold-to-cap / median332.6s. Its guide2/original checkpoint differs, so
screening is cross-configuration, single-seed evidence only. Other completed
reference seeds remain untouched; their averages are context, not the comparator.
Serial timing only uses original 10k/guide2/fresh noise, no 3000-env simulation.

Service: parallel-guidance-candidates-fresh-seed8.service.
Frozen execution copy: .worktrees/parallel-candidates-seed8-run-20260913.
Status: state.json; log: pipeline.log; numerical/timing log: benchmark.log.
Reports: comparison.md and comparison.json. Passing effect/latency checks
attempts a desktop notification, without waiting for other seeds.
