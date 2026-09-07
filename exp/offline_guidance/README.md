# Offline guided DDIM eval

在 DP 训练用的 `replay_buffer.zarr` 上对比：

- Real DP 手部动作
- unguided sim-hand DDIM（从高斯噪声出发）
- 用 DP 手部做 guidance 的 sim-hand DDIM
- DP 与 unguided 的线性混合

不启动真机，也不启动 Isaac Gym。全部 50 个 episode 都进过 Real DP 训练，只是 in-distribution sanity check。

## 两个长度

- `--generation-windows`：用多长的 DP 手部做 guidance（`set_guidance_horizon`）。
- `--comparison-lengths`：评分时只取预测轨迹的前几步（exec）。
- `exec > guidance` 的格子没有物理意义，表格里应记为 `/`。

起点是 sim policy 的 `randn`，DP 只作为 `||x0 - DP||²` 的拉力。`scale=0` 就是普通 sim DDIM。

## 运行

仓库根目录：

```bash
./exp/offline_guidance/run.sh \
  --guidance-scales 0,100 \
  --generation-windows 1,2,3,4,5,6,7,8,9 \
  --comparison-lengths 1,2,3,4,5,6,7,8,9 \
  --execution-steps 9 \
  --ddim-inference-steps 8 \
  --mix-weights 1 \
  --perturb-samples 0 \
  --output-dir exp/offline_guidance/outputs/scale100_ddim8_g1to9_e1to9
```

常用扫描：

```bash
# DDIM 步数 × guidance × exec 网格（scale=100）
./exp/offline_guidance/run.sh \
  --guidance-scales 0,100 \
  --generation-windows 1,2,3,4,5,6,7,8,9 \
  --comparison-lengths 1,2,3,4,5,6,7,8,9 \
  --execution-steps 9 \
  --ddim-inference-steps 4 \
  --mix-weights 1 --perturb-samples 0 \
  --output-dir exp/offline_guidance/outputs/scale100_ddim4_g1to9_e1to9

# 固定 guidance=9、exec=5，扫 scale
./exp/offline_guidance/run.sh \
  --guidance-scales 0,100,150,175,200,225,250,275,300,325,350,500 \
  --generation-windows 9 --comparison-lengths 5 --execution-steps 5 \
  --ddim-inference-steps 15 \
  --mix-weights 1 --perturb-samples 0 \
  --output-dir exp/offline_guidance/outputs/ddim15_g9_e5_scale
```

CSV 里 `gen_window` 是 guidance 步数，`window` 是 exec 步数。相对改善用同一 exec 上 `guided_scale_0.0` 的 `demo_rmse**2`。

## 测试

```bash
python -m pytest exp/offline_guidance/tests tests/test_inference_dp_controller.py -q
```
