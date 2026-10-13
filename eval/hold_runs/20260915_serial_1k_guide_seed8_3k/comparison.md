# 1k guide comparison

FP32 serial guide2 scale25, predict9 execute2, DDIM4/4, seed8 3000 envs fresh noise

200 epochs, own normalizer, warmup56 (not historical warmup500)

| Guide | Completion | Median hold (s) |
|---|---:|---:|
| 1k | 0.00% | 1.1 |
| 10k (reused) | 46.50% | 332.6 |
| 100k (reused) | 20.63% | 114.0 |
| 1m (reused) | 19.70% | 100.0 |
