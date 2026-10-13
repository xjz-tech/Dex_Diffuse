# Guide checkpoint epoch comparison

FP32 serial guide2 scale25 predict9 execute2 DDIM4/4 seed8 3000 envs fresh noise

| Guide | Epoch | Completion | Median hold (s) | Source |
|---|---:|---:|---:|---|
| 100k | 20 | 5.07% | 51.7 | new |
| 100k | 100 | 14.50% | 90.0 | new |
| 100k | 200 | 20.63% | 114.0 | reused |
| 1m | 20 | 9.87% | 72.5 | new |
| 1m | 100 | 17.47% | 97.3 | new |
| 1m | 200 | 19.70% | 100.0 | reused |
