# Parallel guidance comparison

Batch-1 timing is a warmed synthetic-observation microbenchmark; effectiveness comes from 3000 episodes per seed.

| Method | Complete seeds | Completion | Pooled median hold (s) | Batch-1 p50 (ms) | Promising |
|---|---:|---:|---:|---:|---|
| serial | 1 | 46.50% | 332.6 | 20.23 | False |
| fixed_c0.95 | 1 | 22.60% | 118.1 | 18.52 | False |
| fixed_c0.90 | 1 | 31.50% | 161.5 | 18.80 | False |
| fixed_c0.85 | 1 | 37.07% | 212.9 | 18.70 | False |
| fixed_c0.80 | 1 | 36.60% | 207.4 | 18.37 | False |
| fixed_c0.70 | 1 | 16.20% | 85.4 | 18.53 | False |
| dynamic | 1 | 43.17% | 265.0 | 18.72 | False |
| shared_x0_w10 | 1 | 29.83% | 158.2 | 18.22 | False |
| shared_x0_w30 | 1 | 23.23% | 106.0 | 17.84 | False |
| dual_x0_ramp | 1 | 0.00% | 4.5 | 18.12 | False |
| late_x0_w10 | 1 | 27.93% | 134.7 | 14.08 | False |
| late_x0_w30 | 1 | 30.00% | 155.2 | 14.39 | False |

Promising against the selected serial reference: within 1 pp of completion, >=95% of pooled hold median, no shared seed worse by >3 pp, >=20% lower batch-1 p50, and no p95 increase.

Seed8-only screening, not multi-seed evidence or statistical significance. External reference differs in guide horizon/checkpoint, so this is not a controlled parallelization ablation. Retired 1000-env data excluded.

Existing reference (no new serial simulation): /home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/hold_runs/20260912_fresh_noise_3000_seed42_8_19_25
Four-seed mean completion: 45.88%; mean of seed hold medians: 319.0 s. The table above uses pooled medians, not the mean of seed medians.
