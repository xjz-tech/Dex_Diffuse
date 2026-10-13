# 10k guide + obs_4-66: exact 20/25-Hz add-on

| Control Hz | Finished | 400-s survivors | Survival | Mean capped hold (s) | Median capped hold (s) |
|---|---|---:|---:|---:|---:|
| 30 | yes | 1669/3000 | 55.63% | 284.554 | 400.000 |
| 25 | yes | 1797/3000 | 59.90% | 295.293 | 400.000 |
| 20 | yes | 1978/3000 | 65.93% | 306.335 | 400.000 |

Single seed8, 3000 first episodes, 400 simulated seconds. WAIT=1 freezes physics during inference. Same 300-Hz outer physics and 1 substep for these three arms; integration dt is 1/300 s, different from preceding sweeps. Compare primarily against this matched 30-Hz bridge. Policy history4 and execute2 remain in action steps, intentionally changing their physical duration. Force updates and loaded-force duty cycle retain the old 30-Hz world-time schedule; stability/failure timers use common 300-Hz ticks. Random realizations and event-detection granularity can diverge. Historical 30-Hz physics was 60 Hz with 2 substeps, so compare primarily against the new 30-Hz control; historical result is context. No true-hardware throughput/latency claim. Failure is the existing environment predicate, not an independent drop detector.

Historical optimized WAIT=1 30 Hz: 1388/3000 survivors, mean 240.396s, median 324.267s.
