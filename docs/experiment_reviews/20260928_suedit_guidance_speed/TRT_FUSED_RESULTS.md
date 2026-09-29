# TRT FP16 + fused DDIM 速度复测

2026-09-28。**两边都优化后，SDEdit 与 guidance 的单次推理延迟基本相同，约3.2ms。之前未优化实现约1.5倍的差距不能用于TRT+fused版本。**

## 当前结果

同一RTX 4090、10B EMA checkpoint、batch=1、DDIM4、exec2。每种方法每个计时边界预热20次，3轮×100次，随机交替顺序；计时区间前后均检查没有其他GPU计算进程。桌面图形进程仍正常运行。

主表是CPU NumPy观测/reference输入，经GPU推理，再取回CPU动作的墙钟耗时；包含归一化、输入输出传输及CUDA同步。

| 方法 | TRT+fused 中位数 | TRT+fused P95 | 同样fused但UNet使用PyTorch FP32：中位数 |
|---|---:|---:|---:|
| SDEdit，noise ratio 0.20 | **3.219 ms** | 4.087 ms | 7.981 ms |
| Guidance，guide2 / scale25 | **3.188 ms** | 4.144 ms | 8.153 ms |
| Guidance，guide4 / scale50 | **3.185 ms** | 4.325 ms | 8.096 ms |

输入输出传输口径下，SDEdit比两组guidance约慢0.03ms（约1%）；该差距小于轮次波动，不能据此声称guidance有稳定优势。各轮中位数为：

| 方法 | 第1轮 | 第2轮 | 第3轮 |
|---|---:|---:|---:|
| SDEdit | 3.139 ms | 3.262 ms | 3.208 ms |
| Guide2/scale25 | 3.220 ms | 3.215 ms | 3.101 ms |
| Guide4/scale50 | 3.227 ms | 3.062 ms | 3.271 ms |

输入常驻GPU、输出留在GPU时：

| 方法 | 墙钟中位数（含主机发射和同步） | CUDA Event区间中位数 |
|---|---:|---:|
| SDEdit | 2.947 ms | 2.879 ms |
| Guide2/scale25 | 2.985 ms | 2.893 ms |
| Guide4/scale50 | 3.120 ms | 2.908 ms |

CUDA Event记录同一stream上起止事件之间的区间，可能包含kernel之间的空隙，不是将每个kernel的执行时长单独求和。每次调用输出两个动作；3.2ms/次约为1.6ms/执行步的推理摊销，不代表改变了实际30Hz控制频率。

## 优化与数值核验

- 共享同一个TensorRT FP16 UNet，batch-1优化；IO和DDIM算术为FP32。实际编译图只有一个`TorchTensorRTModule`调用，无PyTorch算子回退，见[后端审计](trt_backend_audit.json)。TensorRT 10.7.0.post1，PyTorch 2.6.0+cu124。
- 沿用仓库对fused的含义：GPU预计算DDIM系数，采样循环不做诊断或CPU同步。DDIM更新仍是PyTorch张量运算；没有使用CUDA Graph或把全轨迹融合成单个kernel。
- SDEdit保留9步未来reference、3步已执行历史mask、本地时间表`[11,7,4,0]`。Guidance保留时间表`[75,50,25,0]`及每步引导。两者都为4次UNet前向。
- Guidance将现有“冻结epsilon”的关节MSE梯度改为等价解析式，保留基础x0裁剪、重新计算epsilon、guided x0不裁剪等原语义。固定seed44/45/46噪声预生成并缓存；两种方法按同一seed配对，计时包含采样中的必要复制/初始化。
- 3个输入窗口×3个seed、三种方法：FP32优化前后的最终动作逐值一致；全部单步更新核验也逐值一致。输入窗口包含episode54仿真末4个静置状态，以及其原始reference的第40、100窗口。后两者是离线源数据观测，不伪称新的仿真rollout。
- TRT FP16相对优化FP32的动作误差如下。它说明数值漂移大小，不等于已验证闭环任务效果。

| 方法 | 平均绝对误差 | 最大绝对误差 |
|---|---:|---:|
| SDEdit | 0.000312 rad | 0.002086 rad |
| Guide2/scale25 | 0.000458 rad | 0.003149 rad |
| Guide4/scale50 | 0.000426 rad | 0.002310 rad |

本次是独立速度基准，未启动仿真。计时不含模型加载、TRT编译、预热、参考生成、IPC、录像或诊断统计；未将这些加速改动接入生产推理服务。旧实跑17.81/27.23ms含不同的诊断与主机开销且使用各自闭环观测，不能与本表作为单变量TRT加速比直接相除。当前结果说明：当两者都去掉这些开销并用TRT时，主要成本都是4次相同UNet前向，guidance解析更新的额外成本很小。

## 复测材料

- [完整原始计时与验证数据](trt_fused_results.json)
- [基准脚本](benchmark_trt_fused.py)
- [执行日志](trt_fused_run.log)
- [TRT后端审计](trt_backend_audit.json)

在仓库根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/carus/miniforge3/envs/dp/bin/python \
  docs/experiment_reviews/20260928_suedit_guidance_speed/benchmark_trt_fused.py
```

脚本发现其他GPU计算进程会失败并标记结果，不会将争用下的数字当作本次空闲计算GPU基准。默认会覆盖本目录`trt_fused_results.json`，保留新一轮时请使用`--output`指定新文件。
