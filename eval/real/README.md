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

`hardware.py` 在收发时完成两种关节顺序的转换。obs66 真机启动脚本默认关闭
软件 URDF 位置限位、单步目标变化限幅和跟踪误差退出检查。执行每个动作一个
控制周期后读取一次真机 qpos，更新四帧历史；默认执行 2 个动作后重新推理。

## 推荐运行顺序

`eval_para_obs66_real.sh` 和 `xjz_eval_strong_prior_real.sh` 默认不保存记录，只显示终端输出，
也不创建或轮转日志。测试时需在命令行显式加上 `--record` 开启 TXT 和 JSONL 记录：

```bash
bash eval/eval_para_obs66_real.sh --record
bash eval/xjz_eval_strong_prior_real.sh --record
# 只检查模型、不连接硬件，同时保存记录：
CHECK_ONLY=1 bash eval/eval_para_obs66_real.sh --record
```

开启后，终端输出和报错持续写入 `eval/real/latest_run.txt`，终端仍同步显示。
保留全部历史运行，不设数量上限：
当前记录为 `latest_run.txt`，上一次为 `latest_run.1.txt`，依次到
`latest_run.2.txt` 等；每次启动先轮转旧记录，不再淘汰更早的运行。
可用 `RUN_LOG=/path/to/run.txt` 改变保存路径。两个脚本共用最新日志；同时运行时
应指定不同路径。日志包含启动时间、推理/动作输出以及 Ctrl-C 后的清理信息。
可用 `EXPERIMENT_DESCRIPTION='实验条件与轮次'` 在 TXT 开头标明实验内容。

默认 `LOG_ACTION_STEPS=1` 时，推理执行期间每条已发送指令还会打印 `[cmd ...]`：
`kind=policy` 是模型动作，`inference_hold` 是推理前保持，`exit_hold` 是退出保持。
`delta_max_rad` / `delta_max_deg` 表示与上一条实际发送指令逐关节比较后的最大绝对差，
同时记录 `joint`、从 0 开始的 policy 顺序 `joint_index` 和该关节的
`from_rad` / `to_rad`。它描述目标变化，不是实测运动幅度。第一条记录没有已知的
上一条指令，标记 `reference=unknown` 和 `nan`；启动初始化动作不在此序列中。
`[action ...]` 中的 `raw` / `sent` 仍是单条指令所有关节的最小值和最大值。

### 拷贝到另一台仿真机器的记录

两个 obs66 真机入口加上 `--record` 后同时记录 `latest_run.jsonl`。它和 TXT 一起轮转为
`latest_run.1.jsonl`、`latest_run.2.jsonl` 等，不设数量上限；编号相同表示同一轮运行。
不需要另外安装软件；`RUN_LOG=/path/run.txt` 对应
`/path/run.jsonl`。JSONL 每行是一条独立 JSON，数组完整保留，不是 min/max 摘要。
启动初始化、正式推理、退出保持写入同一文件，通过 `stage` 和 `pid` 区分进程。

| event | 保存内容 |
| --- | --- |
| `session` | 启动参数、关节顺序、单位、时钟说明；初始化进程也保存姿态及插值配置 |
| `model` | 主模型/guide 文件 SHA256、选用权重及训练步数、策略配置、normalizer 参数、scheduler 配置、软件版本、设备、精度相关设置、推理源文件哈希 |
| `inference_input` | 本次送入模型的完整 observation（通常 `1×4×66`，float32）及上一 policy target |
| `inference_output` | 完整返回动作、选取执行的动作、推理起止时间和模型计时；`details.output.action_pred` 是完整物理角度轨迹（通常 `1×12×22`） |
| `command` | 原始模型目标（policy 命令）、限幅计数、实际发送的完整 22 维目标（policy 和硬件顺序）、上一条目标、最近实测状态、发送起止时间及确认状态 |
| `state` | 每次原有读取的完整实测 qpos、读取起止时间、上一条已确认目标和逐关节 `target-qpos` 误差；误差记录不依赖退出阈值开关 |
| `exception` / `command_error` / `state_error` / `session_end` | 异常、通信失败及退出信息；通信失败标记确认状态未知，不能当作成功命令 |

所有事件包含主机 `monotonic_ns`（计算时间间隔）和 `wall_time_ns`（对齐外部视频）。
通信事件的时间包围客户端调用，不是电机执行完成时间或传感器采样时刻。
`command.latest_qpos_policy_rad` 是最近一次原有读取，时间戳一并保存；不额外读取状态。
`state.tracking_error_rad` 为最近下发目标减实测位置；首次读取还没有已知目标时为 null。
当前 SharpA 位置接口不提供关节速度，因此 `velocity_policy_rad_s=null`，不伪造速度。
保持命令前后通常都有状态读取，连续动作的反馈由下一条 `state` 关联；中断时可能
只有退出保持前的读取，不补发额外读请求。`command_id` 在每个进程内递增，关联时
使用 `(pid, command_id)`，推理事件另外用 `(pid, chunk)` 关联。

采样数据放在 `inference_output.details.sampling`：普通模型记录 `initial_noise`、
`normalized_obs`、条件、`timesteps`、`normalized_trajectory`；guidance 记录
`prior_initial_noise`、`guide_initial_noise`、两边归一化观测与时间步、完整
`guide_action_pred`、指导窗口的 `guide_reference_physical` / `guide_reference_normalized`。
普通 DDPM / 随机 DDIM 还按去噪步保存实际 `scheduler_noise`；若 scheduler 实现无法
捕获其噪声调用，则 `scheduler_capture_method=pre_step_rng_state`，逐步保存 RNG 状态。
fused / guided DDIM 为 `deterministic_eta_zero`，初始噪声之后不再抽取调度器噪声。
记录不会额外抽取随机数。回放时把保存的实际噪声作为采样输入，不要只重设 seed。
RNG 状态回放需要匹配 PyTorch、scheduler 和设备后端；跨设备/精度不保证逐位一致。

将匹配的 `.txt`、`.jsonl` 和对应 checkpoint 拷到仿真机器，先按文件 SHA256 确认
模型相同，再重放 observation 和噪声，比较完整动作。物体位姿和接触状态不在关节
记录中，需要另行测量，不能只靠这些数据还原物体的物理过程。

JSON 序列化和写盘在有界后台队列中进行，每行及时刷新；退出时等待写完。
GPU 张量拷贝和采样快照仍会增加开销，`details.capture_copy_ns` 单独记录 CPU 拷贝耗时，
`model_inference_seconds` 的采样区间包含设备端快照开销；起止时间可用于分析总耗时。
磁盘慢时队列会施加等待，不丢帧；写入失败会在 TXT 中打印 `[debug] recording failed`。
测纯推理性能时省略 `--record` 即可；`--record` 配合 `DEBUG_RECORDING=0` 只保存 TXT。
`RUN_LOG` 或 `DEBUG_RECORDING=1` 本身不会开启记录。直接调用 Python 时通过
`REAL_DEBUG_LOG=/path/run.jsonl` 开启结构化记录（自动追加，轮转由 shell 入口负责）。
`CHECK_ONLY=1` 配合 `--record` 也会保存配置与合成推理记录，标记 `synthetic=true`，不连接硬件。

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
`robot_init.py` 在启动时按 `--init-pose` 设置 `HAND_READY_JOINTS`：`ZERO` 为全零，
`ROTATE` 使用 `HAND_ROTATE_JOINTS`，两个 obs66 launcher 默认选择 `ROTATE`。
当前 ROTATE 已恢复为原来的固定姿态。之前从采集数据随机选取的姿态仅作为
注释备选保留，不参与启动；采样来源与统计依据见
`reports/initial_pose_sample_20260917.json`。
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

obs66 及调用它的 strong-prior 真机入口默认使用 `DISABLE_JOINT_LIMITS=1`、
`DISABLE_STEP_CLAMP=1`、`MAX_TRACKING_ERROR=0`。因此不会因实测 qpos 超出
软件 URDF 范围而退出，也不会按 URDF 或 `MAX_HAND_STEP` 裁剪策略目标。
`MAX_HAND_STEP=0.09` 仅在重新开启单步限幅时生效。需要恢复之前的检查时设置
`DISABLE_JOINT_LIMITS=0 DISABLE_STEP_CLAMP=0 MAX_TRACKING_ERROR=0.35`。
NaN/Inf、向量维度检查、推理期间保持及退出保持仍生效；硬件服务或固件自身的
限制不受这些开关影响。直接运行 Python 或使用 obs22 入口时保留原有默认限制。

holding 与不 holding 统一使用每个动作控制周期结束时读取的 qpos，更新历史；
chunk 结束后先固定本次推理的 obs，再决定是否发送 hold。holding 分支不额外
读取 qpos，也不刷新或插入历史帧。首次推理使用启动阶段的实测位置；若设置
`START_DELAY`，两种模式都会在延时结束后重新采样并初始化历史。
`HOLD_DURING_INFERENCE=1` 时发送同一份采样位置作为保持目标（不插值；只有
启用软件关节限位时才应用 URDF 限位和 `JOINT_LIMIT_MARGIN`），收到通信确认后
开始推理，不等待机械停稳。66 维观测的目标仍为上一条实际下发的策略目标，
残差为“策略目标 − 实测位置”；保持命令不替换观测目标。若开启单步限幅，推理后的首个动作
以保持目标为限幅基准。
设置 `HOLD_DURING_INFERENCE=0` 可恢复推理期间保留上一运动目标的行为。
该功能仅在 LIVE 模式生效；直接运行 Python 时用 `--hold-during-inference` 启用。
位置保持不保证机械瞬间静止，实际停止过程仍受通信延迟和底层控制器影响。

按 Ctrl-C 或发生异常时，runner 会尽力读取 SharpA 当前角度并将其设为保持目标，
然后断开连接。正常运行只需要提前启动 SharpA server，不需要 Franka joints server。

启动脚本默认使用 `/home/frankagvl/anaconda3/envs/dexIL/bin/python`，无需依赖
当前 shell 是否已经执行 `conda activate dexIL`；仍可通过 `MODEL_PYTHON` 覆盖。

## Strong prior 的 fused / TensorRT 加速

`xjz_eval_strong_prior_real.sh` 默认启用 `FUSED_DDIM=1`。两个模型的 DDIM
系数会缓存到推理设备上，先验的轨迹 MSE 引导使用解析梯度，省去每个去噪步骤的
autograd 和诊断张量计算。这里的 fused 指预计算系数的 PyTorch 更新路径，
不是把整个采样循环编译成一个 TensorRT engine。

```bash
# 离线检查：使用当前 PyTorch 环境，不连接硬件
CHECK_ONLY=1 bash eval/xjz_eval_strong_prior_real.sh

# 原始实现，用于对照
CHECK_ONLY=1 FUSED_DDIM=0 TENSORRT=0 bash eval/xjz_eval_strong_prior_real.sh

# 两个 UNet 都使用 TensorRT FP16，自动启用 fused
# 本机 dexIL 已验证 torch 2.4.0+cu121 / torch_tensorrt 2.4.0 / TensorRT 10.1
CHECK_ONLY=1 TENSORRT=1 bash eval/xjz_eval_strong_prior_real.sh

# 更高精度：关闭 FP16 和 TF32
CHECK_ONLY=1 TENSORRT=1 TRT_PRECISION=fp32 bash eval/xjz_eval_strong_prior_real.sh
```

TRT 按 batch=1 编译，启动脚本把引擎保存在 checkpoint 旁的
`*.unet.fp16.trt.pt`（或 `fp32`）及 `.json` 元数据中。再次启动会重用匹配的缓存；
checkpoint、运行库、GPU、精度或导出版本变化会触发重建。
首次编译时间会明显增加。
运行时依赖会在手部初始化前检查。FP16 会引入数值误差，可用下面的离线基准脚本
比较动作误差和预热后的推理耗时；省略 `--tensorrt` 只比较原始实现与 fused：

```bash
python eval/bench_guided_real.py --tensorrt --warmup 5 --repeats 20
```

加 `--cache-engines` 可验证启动脚本使用的同一份落盘引擎。

基准使用 normalizer 的平均观测和固定噪声，不连接硬件；它不代表任务成功率。
正式执行仍使用同一启动脚本，把 `CHECK_ONLY` 改为 `0`。

导出时会在模型副本上将 GroupNorm 改写为逐组 LayerNorm 加通道仿射，
修复本机 Torch-TensorRT 2.4 原始 GroupNorm 转换的数值偏差。
依赖版本固定在 `eval/requirements-tensorrt-cu121.txt`，安装时使用
`NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=1`，防止旧 TensorRT 元包拉取不同版本的子包。
