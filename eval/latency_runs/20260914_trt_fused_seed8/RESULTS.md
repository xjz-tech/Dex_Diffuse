# TRT + fused, isolated RTX 4090 / batch 1 / DDIM4

Completed run: run02/results.json. Three blocks of 100 measured requests per
method, each preceded by 20 warmups; methods never execute concurrently with
other benchmark methods or simulation. Within a streams method, its two model
forwards intentionally overlap. All methods have GPU observations and ready
CPU action outputs, fresh noise, execution2. Compilation excluded.

| Method | p50 (ms) | p95 (ms) | mean (ms) |
|---|---:|---:|---:|
| Single 1B, existing TRT + fused implementation | 2.837 | 3.339 | 2.864 |
| Single 1B, lean precomputed loop | 2.804 | 3.247 | 2.834 |
| Serial guide -> base, full9 | 5.682 | 6.482 | 5.832 |
| Fixed0.85, ordered forwards | 6.121 | 6.767 | 6.211 |
| Fixed0.85, CUDA streams | 4.000 | 4.402 | 4.046 |
| Dynamic, ordered forwards | 6.186 | 6.818 | 6.256 |
| Dynamic, CUDA streams | 4.051 | 4.500 | 4.111 |

The fixed0.85 streams method has ~29.6% lower p50 than full9 serial; dynamic
streams has ~28.7% lower p50. The single-model ~2.8ms measurement corroborates
the expected low-millisecond optimized regime, but is not an exact reproduction
of an unidentified historical 4ms run (batch-1 TRT optimization profile here).

All paired methods use the shared-normalizer retrained 10k corrector, full9,
scale25 for serial/dynamic and base coefficient0.85 for fixed. Historical serial
task completion46.50% uses ORIGINAL 10k/guide2 and must not be paired with this
full9 shared-normalizer latency as if it were the same configuration.

Validation: 21 unit tests passed. Real-checkpoint FP32 identical-input local
update comparisons pass (max3.58e-7); complete trajectory drift for single and
serial is <=4.46e-4 normalized units on three synthetic observations. Fixed and
dynamic match exactly on those probes. TRT ordered/streams trajectory gate
passes at rtol1e-4/atol1e-5. These tests are not rollout success-rate evidence.

One fixed-noise numerical probe (timing itself uses fresh noise) comparing the
TRT result to the FP32 lean loop gives future-action absolute differences:

| Method | mean abs | max abs |
|---|---:|---:|
| Single | 0.000588 | 0.004474 |
| Serial | 0.001002 | 0.004907 |
| Fixed0.85 | 0.000603 | 0.002603 |
| Dynamic | 0.002241 | 0.010162 |

These are physical action units (joint-angle coordinates); dynamic shows the
largest FP16 drift. Task effectiveness after FP16 optimization is untested.
No simulation was restarted or modified. Queues and watchdog remain paused.
