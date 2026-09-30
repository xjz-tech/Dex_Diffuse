# Active matrix: 0.85, 0.80, dynamic — seed8

User cancelled 0.60. It will not be evaluated or automatically recovered.
Run 0.85 then 0.80 then dynamic, sequentially. All: seed8, 3000 first episodes,
fresh noise, full9 guidance, DDIM4, execution2, unchanged shared-normalizer
10k checkpoint and pinned hold thresholds. No serial simulation or training.

Service: parallel-guidance-refine-seed8.service.
Frozen source: .worktrees/parallel-refine-seed8-run-20260914.
State: state.env; pipeline.log; per-run parameters.env and episode/latency logs.

fixed/c0.95, c0.90, c0.70 are links to completed results in
20260913_parallel_fresh_only_3k. They are comparisons only and not scheduled.
Old 0.60 crash data remains in that old root; it is excluded from this suite.
