# 10k guide + obs_4-66: control-frequency comparison

| Control Hz | Finished | 400-s survivors | Survival | Mean capped hold (s) | Median capped hold (s) |
|---|---|---:|---:|---:|---:|
| 30 | yes | 1514/3000 | 50.47% | 269.330 | 400.000 |
| 45 | yes | 928/3000 | 30.93% | 212.483 | 197.111 |
| 60 | yes | 519/3000 | 17.30% | 156.642 | 103.392 |
| 90 | yes | 147/3000 | 4.90% | 88.587 | 43.794 |

Single seed8, 3000 first episodes, 400 simulated seconds. WAIT=1 freezes physics during inference. Same 180-Hz outer physics and 2 substeps for all new arms. Policy history4 and execute2 remain in action steps, intentionally changing their physical duration. Force updates and loaded-force duty cycle retain the old 30-Hz world-time schedule; stability/failure timers use common 180-Hz ticks. Random realizations and event-detection granularity can diverge. Historical 30-Hz physics was 60 Hz with 2 substeps, so compare primarily against the new 30-Hz control; historical result is context. No true-hardware throughput/latency claim. Failure is the existing environment predicate, not an independent drop detector.

Historical optimized WAIT=1 30 Hz: 1388/3000 survivors, mean 240.396s, median 324.267s.
