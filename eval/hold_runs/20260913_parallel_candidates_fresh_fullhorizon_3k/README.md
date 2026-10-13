# Fresh-noise follow-on parallel candidates

STOPPED before candidates ran. Replaced by
20260913_parallel_candidates_fresh_only_3k, which uses the user's existing
four-seed serial reference instead of waiting for new serial simulations.

Source commit: 5f682e7. Frozen execution copy:
/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/.worktrees/parallel-fresh-noise-run-20260913

User service: parallel-guidance-candidates-fresh.service.
Initial supervisor PID: 1708211. Predecessor PID: 1707797, start token 217388866.
Wait for the fresh-noise 18-run predecessor suite to complete successfully,
then for GPU availability, then run real-checkpoint numerical gates and a
fresh-noise batch-1 latency benchmark. Numerical equivalence tests deliberately
reuse identical test inputs between implementations; task requests and timed
requests draw fresh noise every call.

Evaluate shared_x0_w10, shared_x0_w30, dual_x0_ramp, late_x0_w10, late_x0_w30,
each with seeds 8/19/25 and 3000 first episodes: 15 runs. All guide the complete
nine-action horizon and execute two actions. Late correction skips early DDIM
steps, not action-horizon elements.

state.json and pipeline.log contain live queue status. Results populate
comparison.md and comparison.json. Passing the matched-control effect/latency
screen attempts a desktop notification; failures also attempt notification.
No future chat message is scheduled. Old 1000-env reference data is excluded.
