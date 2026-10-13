# 10B_obs_4-66 versus obs_4-66

Four seeds, 3000 first episodes per seed, 400-second cap.

| Checkpoint | 10k guidance | Episodes | 400-s survival | Mean capped hold (s) | Median capped hold (s) |
|---|---:|---:|---:|---:|---:|
| obs_4-66 | scale 0 | 12000 | 11.21% | 129.625 | 73.833 |
| obs_4-66 | scale 25 | 12000 | 45.90% | 238.703 | 320.600 |
| 10B_obs_4-66 | scale 0 | 12000 | 13.51% | 136.632 | 76.800 |
| 10B_obs_4-66 | scale 25 | 12000 | 42.85% | 229.021 | 267.400 |

Historical protocol did not save complete initial-state arrays. Matching seeds, simulator settings, data indices, and evaluator are used, but bitwise initial-state equality cannot be checked retroactively.
