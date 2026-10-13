# SDEdit 与梯度 guidance 推理速度

后续已完成[TRT FP16 + fused复测](TRT_FUSED_RESULTS.md)：两边都优化后约3.2ms，未看到SDEdit稳定更快。下文保留原始未优化实现及并发交叉验证的历史结果。

结论：在本仓库当前 10B EMA 手部 prior、RTX 4090、batch=1、DDIM4、每次执行2步的设置下，reference initialized DDIM（SDEdit）比梯度 guidance 快。下面比较的是**单次策略推理延迟**，不含模型加载、IPC、PhysX 仿真或录像。

## 已完成仿真的实测日志（主结果）

取 episode54 相同原始 reference、seed 44/45/46 的既有运行日志。每个方法每个 seed 有231次预测；剔除每次运行的首个调用，汇总余下690次。两种方法均用 `/home/carus/data_usb/10B_obs_4-66.ckpt`，同一 66维观测/22维动作、DDIM4、exec2；SDEdit 使用9步未来参考，guidance 使用前4步参考、scale50。两种方法各自按实际执行结果形成后续观测，因此这不是逐调用相同状态的配对实验。

| 方法 | 调用数 | 中位数 | 均值 | P95 | 每个执行控制步分摊的中位推理时间 |
|---|---:|---:|---:|---:|---:|
| SDEdit，noise ratio 0.10 | 690 | 15.93 ms | 17.03 ms | 25.93 ms | 7.96 ms |
| SDEdit，noise ratio 0.20 | 690 | 17.81 ms | 19.69 ms | 29.19 ms | 8.90 ms |
| SDEdit，noise ratio 0.35 | 690 | 17.30 ms | 18.17 ms | 27.47 ms | 8.65 ms |
| Guidance，guide4/scale50 | 690 | 27.23 ms | 26.62 ms | 34.46 ms | 13.61 ms |

以noise ratio 0.20为例，中位数快 `27.23 / 17.81 = 1.53` 倍，即延迟低约34.6%。三个 seed 单独比较，中位加速比为1.39、1.59、1.56倍。SDEdit的不同噪声比仍是4次UNet前向，耗时差异主要反映运行噪声，而非UNet次数变化。

原始数据：[`episode54_reference_edit_20260928`](../20260924_object_state_data/reference_turn_baseline_20260926/episode54_reference_edit_20260928/) 中各组的 `predictions.json` 与 `summary.json`。`inference_seconds` 在模型服务进程内包住`predict`；两种实现最终都将动作传到CPU，因此该调用完成时GPU计算已同步。首调用剔除，仍可能有其他运行时资源争用。

## 同进程交替计时（并发负载下的交叉验证）

[`benchmark.py`](benchmark.py) 从 episode54 的实际静置观测和原始 reference 构造同一输入，复用一个已加载的 prior。每方法预热12次、交替调用80次，用 `perf_counter` 和显式 CUDA 同步。运行时另一个 DDIM6 仿真扫描占用相同 GPU，故绝对毫秒数**不作为空闲 GPU 延迟**；原始样本见 [`results_contended.json`](results_contended.json)。

| 方法 | 中位数 | P95 | 相对 SDEdit 0.20 的中位耗时 |
|---|---:|---:|---:|
| SDEdit 0.20 / DDIM4 | 21.86 ms | 40.29 ms | 1.00× |
| Guidance guide2/scale25 / DDIM4 | 30.36 ms | 48.12 ms | 1.39× |
| Guidance guide2/scale50 / DDIM4 | 30.47 ms | 46.60 ms | 1.39× |
| Guidance guide4/scale50 / DDIM4 | 33.03 ms | 47.60 ms | 1.51× |

两个计时来源一致地表明当前 SDEdit 实现推理更快。原因与实现相符：两者每次都有4次UNet前向，guidance 每个DDIM步另外计算参考损失梯度；SDEdit只执行前向更新和已知历史动作遮罩。速度结论不代表策略效果相同，也不能推断更换DDIM步数、硬件、batch或实现后的加速比。
