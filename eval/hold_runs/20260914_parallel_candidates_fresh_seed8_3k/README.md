# Candidate continuation — seed8

Wait for 20260914_parallel_refine_fresh_seed8_3k to finish 0.85/0.80/dynamic,
then numerical gates + batch1 timing, then shared_x0_w10, shared_x0_w30,
dual_x0_ramp, late_x0_w10, late_x0_w30. Only seed8; each 3000 first episodes,
fresh noise, full9 guidance, DDIM4, execution2. Five candidate simulations.

Existing seed8 serial reference is reused (46.50%, 332.6s); no serial simulation.
Serial timing only uses the reference original checkpoint and guide2.
Completed 0.95/0.90/0.70 are included in reports; 0.60 is excluded.

Service: parallel-guidance-candidates-refine-seed8.service.
Frozen source: .worktrees/parallel-refine-seed8-run-20260914.
State: state.json; logs: pipeline.log/benchmark.log; comparison.md/json.
