# 10k guide → 1B: WAIT=0 vs WAIT=1, 3000 environments each

| Metric | WAIT=0 | WAIT=1 |
|---|---:|---:|
| environments | 3000 | 3000 |
| failures | 2085 | 1612 |
| survived_cap | 915 | 1388 |
| mean_capped_hold_seconds | 190.7066 | 240.3964 |
| median_capped_hold_seconds | 144.1833 | 324.2667 |
| action_steps | 7503 | 12000 |
| hold_steps | 4497 | 0 |
| mean_inference_seconds | 0.1554 | 0.1307 |
| wall_seconds | 1683.5856 | 1966.1862 |

All saved initial-state arrays are identical.

Single seed, 3000 environments per arm. Native batched GPU inference latency; not a reproduction of single-hand hardware latency. Failure is the existing environment predicate, not an independent physical-drop detector. Capped hold includes right-censored survivors at 400 seconds.

## 中文结果

两组各3000个首次试验，初始观测、示范编号、初始帧和物体/手根状态逐项一致；数值检查通过。

| 指标 | WAIT=0 | WAIT=1 |
|---|---:|---:|
| 400秒存活率 | 30.50% (915/3000) | 46.27% (1388/3000) |
| 截断平均保持时间 | 190.71秒 | 240.40秒 |
| 截断中位保持时间 | 144.18秒 | 324.27秒 |
| 平均批量推理耗时 | 155.42毫秒 | 130.66毫秒 |
| 保持上一目标的额外物理步占比 | 37.475% (4497/12000) | 0% |

WAIT=0存活率低15.77个百分点。统计时间为仿真时间；存活指未触发环境失败判据。保持时间在400秒截断，不是无限时长寿命估计。单seed对比，原生批量推理加GPU竞争/物理吞吐的时序不能直接视为真机单手延迟；本结果支持检查执行时序，但不证明真机向内回收的根因。
