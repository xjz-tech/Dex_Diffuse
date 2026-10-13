# Active candidate queue: reuse user's completed reference

SUPERSEDED before any candidate ran: user requested seed8 ONLY.
Replacement: 20260913_parallel_candidates_fresh_seed8_3k (five candidate runs).

Wait for parallel-guidance-fresh-only.service to finish 15 fixed/dynamic runs,
then perform candidate numerical checks and short batch-1 latency measurements,
then 15 candidate simulations (5 configs x seeds 8/19/25 x 3000 environments).
All new parallel simulations use fresh noise, full nine-action guidance, four
DDIM steps, two executed actions, and the shared-normalizer retrained 10k.

No serial simulation is run. Effect reference is read from the completed
20260912_fresh_noise_3000_seed42_8_19_25 suite: 1B+10k scale25, four seeds,
45.88% mean hold-to-cap and 319.0 s mean of seed hold-time medians.
Timing-only serial reference uses the original 10k checkpoint, fresh noise,
guide2 and guide seed100008, matching its protocol. This is a cross-configuration
effect comparison, not a causal parallelization ablation. Reports also show
pooled medians, which differ from means of per-seed medians.

Service: parallel-guidance-candidates-fresh-only.service.
Frozen execution worktree: .worktrees/parallel-fresh-only-run-20260913.
Status: state.json. Logs: pipeline.log and benchmark.log. Reports:
comparison.md/comparison.json. Promising results attempt desktop notification.
