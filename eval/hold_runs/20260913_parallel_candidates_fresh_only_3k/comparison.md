# Parallel guidance comparison

Batch-1 timing is a warmed synthetic-observation microbenchmark; effectiveness comes from 3000 episodes per seed.

| Method | Complete seeds | Completion | Pooled median hold (s) | Batch-1 p50 (ms) | Promising |
|---|---:|---:|---:|---:|---|
| serial | 4 | 45.88% | 320.6 | pending | False |
| fixed_c0.95 | pending | pending | pending | pending | False |
| fixed_c0.90 | pending | pending | pending | pending | False |
| fixed_c0.70 | pending | pending | pending | pending | False |
| fixed_c0.60 | pending | pending | pending | pending | False |
| dynamic | pending | pending | pending | pending | False |
| shared_x0_w10 | pending | pending | pending | pending | False |
| shared_x0_w30 | pending | pending | pending | pending | False |
| dual_x0_ramp | pending | pending | pending | pending | False |
| late_x0_w10 | pending | pending | pending | pending | False |
| late_x0_w30 | pending | pending | pending | pending | False |

Promising against the selected serial reference: within 1 pp of completion, >=95% of pooled hold median, no shared seed worse by >3 pp, >=20% lower batch-1 p50, and no p95 increase.

Screening only, not statistical significance. When using the external reference, guide horizon/checkpoint differ, so this is not a controlled parallelization ablation. Retired 1000-env data excluded.

Existing reference (no new serial simulation): /home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/hold_runs/20260912_fresh_noise_3000_seed42_8_19_25
Four-seed mean completion: 45.88%; mean of seed hold medians: 319.0 s. The table above uses pooled medians, not the mean of seed medians.
