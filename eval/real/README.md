# 66-D controller 真机推理

这个目录包含 `eval_para_obs66_real.sh` 所需的全部 Python 代码，包括 checkpoint
兼容的最小 Diffusion Policy 网络实现。它不会导入 `eval/` 下的仿真入口、
Dex_Diffuse 根目录的 Python 模块、TacMP 或 dex-controller 的真机实现。

模型和真机接口如下：

- observation：连续 4 帧，每帧
  `[qpos, target_before, target_before - qpos]`，共 66 维；
- action：5 个 22 维 SharpA 绝对关节目标；
- checkpoint/Isaac Gym 顺序：index、middle、pinky、ring、thumb；
- SharpA server 顺序：thumb、index、middle、ring、pinky。

`hardware.py` 在收发时完成两种关节顺序的转换。每个预测动作还会经过 URDF
关节限位和默认 `0.03 rad` 的单步变化限幅。执行每个动作一个控制周期后读取一次
真机 qpos，更新四帧历史；一个 5-step chunk 完成后再进行下一次推理。

## 推荐运行顺序

先只检查 checkpoint，不连接硬件：

```bash
CHECK_ONLY=1 ./eval/eval_para_obs66_real.sh
```

再检查 SharpA 协议和状态，不发送运动命令：

```bash
PREFLIGHT_ONLY=1 ./eval/eval_para_obs66_real.sh
```

真机第一次测试建议只执行一个 chunk：

```bash
MAX_STEPS=5 ./eval/eval_para_obs66_real.sh
```

确认行为后持续运行：

```bash
./eval/eval_para_obs66_real.sh
```

两个 `eval_para_obs*_real.sh` 默认不连接或控制 Franka，先调用本地
`robot_init.py --skip-franka`，用 50 个线性小步、默认 2 秒 reset SharpA。
确认到位后等待 Enter（可关闭），再加载模型、预热并开始推理。
推理程序不会再次移动到 JSON pose。

所有真机推理启动脚本的 reset 均使用本目录 `robot_init.py`，通信类来自复制到
本目录的 `direct_robot_env.py`，不再导入外部 TacMP 文件。手部目标只由
`robot_init.py` 中的 `HAND_READY_JOINTS` 定义：当前启用全零，记录姿态的
`np.asarray(...)` 整块注释保留。切换时确保只有一个赋值块生效。
`sharpa_initial_pose.json` 仍保留，但这些启动脚本不再读取它或 `INITIAL_POSE_FILE`。
移动速度和到位检查可使用 `INITIAL_POSE_STEPS`、
`INITIAL_POSE_SECONDS`、`INITIAL_POSE_TOLERANCE` 和
`INITIAL_POSE_TIMEOUT_SECONDS` 配置。

根目录两个 `inference_dp_dino*.sh` 和两个 `eval_dp_controller_obs*.sh` 则
先 reset 手部，确认到位后再将机械臂分 30 步、3 秒线性插值到 ready joints，
然后加载模型并推理。模型相关校验错误会在 reset 后报告。

`LIVE=0` 只读取真实手状态并运行 policy，不移动 SharpA、不等待 Enter，也不发送
action。需要临时跳过初始 pose 移动或 Enter 门控时，可分别设置
`MOVE_TO_INITIAL_POSE=0` 或 `WAIT_FOR_ENTER=0`。

常用参数均通过环境变量配置：`CKPT_PATH`、`MODEL_PYTHON`、`HAND_HOST`、
`HAND_PORT`、`INFERENCE_STEPS`、`ACTION_CHUNK_STEPS`、
`CONTROL_HZ`、`MAX_HAND_STEP`、`MAX_TRACKING_ERROR`、`JOINT_LIMIT_MARGIN` 和
`MAX_STEPS`。

按 Ctrl-C 或发生异常时，runner 会尽力读取 SharpA 当前角度并将其设为保持目标，
然后断开连接。正常运行只需要提前启动 SharpA server，不需要 Franka joints server。

启动脚本默认使用 `/home/frankagvl/anaconda3/envs/dexIL/bin/python`，无需依赖
当前 shell 是否已经执行 `conda activate dexIL`；仍可通过 `MODEL_PYTHON` 覆盖。

## 在线视觉 DP → SDEdit

`eval/eval_dp_edit_obs66.sh` 使用在线视觉 DP 的 31 维动作窗口：机械臂的 9 维
动作仍由视觉 DP 给出，22 维手部参考动作交给
`diffusion_policy/SDEdit/reference_action_editor.py` 编辑。每个 edit 调用执行两步，
读取新的手部状态后继续闭环。这个入口复用
`eval/inference_dp_controller.py` 的相机、Franka、SharpA 执行和安全限制。

执行前，手部目标按 `hardware.py` 中的 22 维 SharpA URDF 关节范围截断；
默认还按每步 `0.03 rad` 限制手部目标变化。Franka 的关节模式通过 IK
限制每步关节变化为 `0.05 rad`，并按 FR3 URDF 的绝对关节范围截断。
机械臂末端位置目标每轴每步最多变化 `0.03 m`。这些是软件命令限幅；
此入口尚未实现实测关节跟踪误差、接触力或碰撞监控。

先设置 `DP_CKPT_PATH` 和 `CONTROLLER_CKPT_PATH`，然后只做离线模型检查：

```bash
CHECK_ONLY=1 ./eval/eval_dp_edit_obs66.sh
```

脚本默认就是 `CHECK_ONLY=1`，不会连接硬件。确认模型、动作窗口长度和参数后，
可显式设置 `CHECK_ONLY=0 LIVE=1` 运行；默认 `MAX_CHUNKS=1`，即先执行一个
两步 chunk。`EDIT_NOISE_RATIO` 默认 `0.15`、`DDIM_INFERENCE_STEPS` 默认 `4`、
`FIXED_NOISE` 默认 `1`。脚本不读取离线 reference 文件，也不更改视觉 DP 的机械臂动作。
