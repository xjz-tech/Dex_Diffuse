# 100k guide → 10B_obs_4-66

Seed 8, 3000 first episodes, 400-second cap; guide2 / scale25 / DDIM4 / execute2.

| Model | 400-s survival | Mean capped hold (s) | Median capped hold (s) |
|---|---:|---:|---:|
| obs_4-66 + 100k guide | 20.63% | 167.37 | 113.95 |
| 10B baseline | 13.07% | 137.35 | 77.90 |
| 10B + 10k guide | 43.17% | 230.94 | 276.23 |
| 10B + 100k guide | 19.13% | 158.42 | 95.93 |

Single seed 8; historical initial states not bitwise-verifiable.
