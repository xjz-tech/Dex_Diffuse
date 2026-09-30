# Parallel guidance comparison

Batch-1 timing is a warmed synthetic-observation microbenchmark; effectiveness comes from 3000 episodes per seed.

| Method | Complete seeds | Completion | Median hold (s) | Batch-1 p50 (ms) | Promising |
|---|---:|---:|---:|---:|---|
| serial | pending | pending | pending | pending | False |
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

Promising against the matched 3000-env serial control only: within 1 pp of completion, >=95% of hold median, no seed worse by >3 pp, >=20% lower batch-1 p50, and no p95 increase.

Retired 1000-env results are excluded. This screen does not establish retention of the newer guide-2 result; its guidance horizon and guide checkpoint differ from this full-horizon suite.
