# Bulb2 Isaac Gym evaluation

这个目录中的 evaluator 会加载 Sim-Hand Diffusion Policy checkpoint，在
`dex-controller` 的 `SingleDexHandRH` bulb2 Isaac Gym 任务中持续执行：

- 输入：最近 4 个 30 Hz control step 的 SharpA `qpos`，形状 `(B, 4, 22)`；
- 输出：5 个 control step 的绝对关节目标，形状 `(B, 5, 22)`；
- 每执行完 5 步重新推理，形成闭环控制；
- episode 失败后随机选择一条已配置的 demo，再由任务随机选择其前 90%
  中的初始帧，并随机 wrist orientation；
- 不限制 episode 数；录制模式可按 `Q` 保存并退出，也可用 `Ctrl-C`、`Esc`
  或关闭 viewer 结束。

## 运行

```bash
cd /mnt/work/dexIL/Dex_Diffuse
bash eval/eval.sh
```

默认显示一个仿真环境、自动录制，并在 bulb2 的 `000`、`001` 两条轨迹间随机
reset。视频写入 `eval/record/`。`NUM_ENV` 可以增加并行推理环境数，`RECORD_ENV`
选择其中一个环境显示坐标轴并录制（从 `0` 开始编号）。
常用覆盖方式：

```bash
# 不录制，使用 24 个并行环境和 000..149 全部轨迹
RECORDING=0 NUM_ENV=24 DATA_INDICES=000-149 bash eval/eval.sh

# 同时推理 8 个环境，并录制 env 3
NUM_ENV=8 RECORD_ENV=3 bash eval/eval.sh

# 无 viewer 的有限 smoke run
RECORDING=0 HEADLESS=1 MAX_STEPS=20 PRINT_EVERY=1 bash eval/eval.sh

# 指定其他 checkpoint / GPU
CKPT_PATH=/path/to/checkpoint.ckpt CUDA_VISIBLE_DEVICES=1 bash eval/eval.sh
```

主要环境变量：

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `NUM_ENV` | `1` | 并行 Isaac Gym 环境数（兼容旧变量 `NUM_ENVS`） |
| `RECORD_ENV` | `0` | 录制及显示物体坐标轴的环境索引，范围 `0..NUM_ENV-1` |
| `DATA_INDICES` | `000,001` | 支持逗号列表和闭区间，如 `000-149` |
| `HEADLESS` | `0` | 设为 `1` 关闭 viewer |
| `RECORDING` | `1` | 自动录制；设为 `0` 时不创建相机、不写视频 |
| `MAX_STEPS` | `0` | `0` 表示无限运行 |
| `INFERENCE_STEPS` | checkpoint 值 | DDPM 推理步数，当前训练值为 100 |
| `RANDOMIZE_DEMO_ON_FAILURE` | `1` | 失败 reset 时是否重新随机 demo |
| `ALLOW_SALVAGE` | `1` | 是否允许安全恢复缺少 ZIP 尾部的 checkpoint |
| `MODEL_WARMUP` | `1` | 启动时是否先执行一次模型推理校验 |

路径和 Python 环境也都可覆盖：`CONTROLLER_ROOT`、`SIM_DATASET`、
`SIM_CONFIG`、`NOKOV3_DATA_DIR`、`NOKOV3_RETARGET_DIR`、`SIM_PYTHON`、
`MODEL_PYTHON`。

## 带物体坐标轴的 MP4 录制

```bash
cd /mnt/work/dexIL/Dex_Diffuse
bash eval/record_eval.sh
```

录制模式只为 `RECORD_ENV` 创建一台相机，并在该环境的真实物体上显示三根长方体：
红色 X、绿色 Y、蓝色 Z。坐标轴跟随物体的实时位置和旋转。它们不创建 PhysX
actor，viewer 中使用 Isaac Gym debug lines，因此不会改变 actor 索引、碰撞、
动力学或 reset。MP4 直接写入相机 tensor 的原始 RGB，不投影坐标轴，也不做图像
后处理。相机固定在物体的 +Y 侧；viewer 与 MP4 camera 使用同一组位置和 look-at
参数。

- 仿真启动后自动开始录制；
- 按 `Q`、`Esc`、关闭 viewer 或在终端按 `Ctrl-C` 都会结束程序，并先正常封装 MP4；
- MP4 写入 `eval/record/`，文件名为 `YYYYMMDD_HHMMSS_env{RECORD_ENV}.mp4`；
- MP4 默认编码为 30 FPS，并按墙钟时间补重复帧；模型串行推理和 viewer 等待
  会表现为画面短暂停顿，因此保存视频中的动作速度与运行时 viewer 一致。

第一版视角和坐标轴尺寸可通过环境变量调整：

```bash
RECORD_CAMERA_POSITION='-0.10,0.55,0.10' \
RECORD_CAMERA_TARGET='-0.10,0.00,-0.14' \
RECORD_CAMERA_FOV=60 \
RECORD_AXIS_LENGTH=0.20 \
RECORD_AXIS_THICKNESS=0.008 \
bash eval/record_eval.sh
```

还可设置 `RECORD_WIDTH`、`RECORD_HEIGHT` 和 `RECORD_FPS`，默认分别为
`1280`、`720` 和 `30`。

## 为什么有两个 Python 进程

本机 Isaac Gym 绑定是 CPython 3.8 (`gym_38.so`)，位于 `dec_sapg` 环境；
Diffusion Policy 和 `diffusers` 位于 Python 3.10 的 `rdp` 环境。`eval.sh`
分别启动仿真进程和模型进程，通过私有 UNIX socket 交换 float32 数组，
不修改也不混装现有 Conda 环境。

当前 `runs/step_01700000.ckpt` 缺少 ZIP central directory，且 EMA/optimizer
部分没有写完；base model 的 162 个 state entries（包含 normalizer）是完整的。
加载器会先尝试标准 `torch.load`：完整 checkpoint 使用 EMA；只有遇到该特定
ZIP 尾部错误时，才逐条校验并严格加载完整 base model，绝不会混用残缺 EMA。
建议后续若有完整 checkpoint，直接替换 `CKPT_PATH`，程序会自动切回 EMA。
