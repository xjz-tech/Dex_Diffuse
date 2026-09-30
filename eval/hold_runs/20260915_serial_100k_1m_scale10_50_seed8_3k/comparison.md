# 100k/1M guide scale sweep

FP32 serial, seed8, 3000 envs, fresh noise, predict9 guide2 execute2 DDIM4/4

| Guide | Scale | Completion | Median hold (s) | Source |
|---|---:|---:|---:|---|
| 100k | 10 | 18.03% | 99.0 | new |
| 100k | 25 | 20.63% | 114.0 | reused |
| 100k | 50 | 14.00% | 90.9 | new |
| 1m | 10 | 16.83% | 91.9 | new |
| 1m | 25 | 19.70% | 100.0 | reused |
| 1m | 50 | 17.00% | 91.6 | new |
