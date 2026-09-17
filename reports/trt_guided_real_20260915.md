# Strong prior real：TensorRT 离线编译与验证

环境：RTX 4090，驱动 535.183.01，Python 3.8，torch 2.4.0+cu121，
torch_tensorrt 2.4.0，TensorRT 10.1.0。

模型：`runs/obs_4-66.ckpt` + `runs/epoch_0200.ckpt`，均为 EMA。
参数：batch=1，prior/guide 采样 8/4 步，guidance scale=25，
引导/执行窗口各 2 步，seed=guide_seed=42，固定噪声。
输入为 normalizer 平均观测，预热 10 次、计时 50 次；每次计时前后同步 CUDA。
未连接机械手。这是单个合成观测的数值与延迟对照，不是任务成功率评估。

## 修正后的测量

| 测量轮次 | 路径 | 平均耗时 ms | 中位数 ms | 对本轮原始 PyTorch 的最大动作差 rad |
| --- | --- | ---: | ---: | ---: |
| FP16 | 原始 PyTorch | 73.427 | 77.920 | 0 |
| FP16 | PyTorch + fused | 25.876 | 25.745 | 0.0001239 |
| FP16 | TensorRT FP16 + fused | 9.286 | 9.282 | 0.0049985 |
| FP32 | 原始 PyTorch | 49.916 | 38.874 | 0 |
| FP32 | PyTorch + fused | 59.642 | 59.627 | 0.0001239 |
| FP32 | TensorRT FP32 + fused（禁用 TF32） | 34.005 | 34.275 | 0.0005490 |

各轮有系统负载波动，不应跨轮计算严格加速比。
FP16 轮中，TRT 相对同轮 fused 提速 2.79 倍，耗时减少 64.1%。
FP16 平均动作绝对误差为 0.0011824 rad，最大约 0.286°。
FP32 平均动作绝对误差为 0.0001227 rad。

### 启动脚本生成的缓存重载验证

启动脚本重编译并保存两个 FP16 引擎后，用 `--cache-engines` 重新启动基准进程。
两个缓存加载耗时 1.9 秒（不包含 checkpoint 加载）。这一轮实测：

| 路径 | 平均耗时 ms | 中位数 ms | 最大动作差 rad |
| --- | ---: | ---: | ---: |
| 原始 PyTorch | 76.607 | 76.258 | 0 |
| PyTorch + fused | 63.010 | 62.275 | 0.0001239 |
| 缓存 TensorRT FP16 + fused | 11.768 | 12.066 | 0.0078722 |

该缓存的平均动作绝对误差为 0.0018300 rad，最大约 0.451°。
不同编译构建选择的 FP16 执行内核可能不同，误差也有所变化；实际使用当前
落盘引擎应以这组数据为准。缓存位于 `runs/obs_4-66.unet.fp16.trt.pt` 和
`runs/epoch_0200.unet.fp16.trt.pt`，旁边各有元数据 JSON。
启动脚本的 CHECK_ONLY 合成推理及缓存重载对照均通过。

## 修复的转换问题

初版 TensorRT 的最大动作差约 0.04 rad，FP32 同样存在。
逐层对照发现第一层卷积一致，但第一处 GroupNorm 已出现大误差。
现仅在导出副本上用逐组 LayerNorm 加原始通道仿射替换 GroupNorm。
修复后，严格 FP32 UNet 对严格 FP32 PyTorch 的三个测试时间步最大误差不超过
5.13e-6；对原始默认允许 cuDNN TF32 的 PyTorch 仍有小幅精度差异。
缓存元数据包含导出修订号，旧转换缓存不再复用。

## 运行

```bash
CHECK_ONLY=1 TENSORRT=1 bash eval/xjz_eval_strong_prior_real.sh
CHECK_ONLY=1 TENSORRT=1 TRT_PRECISION=fp32 bash eval/xjz_eval_strong_prior_real.sh

LD_LIBRARY_PATH=/home/frankagvl/anaconda3/envs/dexIL/lib OMP_NUM_THREADS=4 \
  /home/frankagvl/anaconda3/envs/dexIL/bin/python eval/bench_guided_real.py \
  --tensorrt --warmup 10 --repeats 50
```

基准脚本可额外传 `--trt-precision fp32` 或 `--cache-engines`。
启动脚本保存并验证引擎缓存；基准脚本默认直接编译以评估当前源码，
指定 `--cache-engines` 后与启动脚本共享缓存。

日志：[FP16](trt_guided_real_fp16_fixed_20260915.log)、
[FP32](trt_guided_real_fp32_fixed_20260915.log)、
[UNet 修复后对照](trt_unet_fixed_20260915.log)、
[缓存重载对照](trt_guided_real_cached_20260915.log)、
[启动脚本](trt_real_launcher_build_20260915.log)。
