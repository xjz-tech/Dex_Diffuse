# 10k guide + obs_4-66: exact 40-Hz add-on

| Control Hz | Finished | 400-s survivors | Survival | Mean capped hold (s) | Median capped hold (s) |
|---|---|---:|---:|---:|---:|
| 30 | yes | 1489/3000 | 49.63% | 266.099 | 394.967 |
| 40 | yes | 1120/3000 | 37.33% | 232.844 | 250.738 |

Single seed8, 3000 first episodes, 400 simulated seconds. WAIT=1 freezes physics during inference. Same 360-Hz outer physics and 1 substep for both new arms; integration dt equals the earlier 180-Hz/2-substep sweep. Policy history4 and execute2 remain in action steps, intentionally changing their physical duration. Force updates and loaded-force duty cycle retain the old 30-Hz world-time schedule; stability/failure timers use common 360-Hz ticks. Random realizations and event-detection granularity can diverge. Historical 30-Hz physics was 60 Hz with 2 substeps, so compare primarily against the new 30-Hz control; historical result is context. No true-hardware throughput/latency claim. Failure is the existing environment predicate, not an independent drop detector.

Historical optimized WAIT=1 30 Hz: 1388/3000 survivors, mean 240.396s, median 324.267s.
