# Sim-Hand Guided Inference Design

## 1. 目标与结论

新增一条独立的推理链路：目标 Bulb/Real Diffusion Policy 先生成 50 步动作，
Sim-Hand Diffusion Policy 再以闭环方式引导其中 22 维手部动作。方案借鉴
DexGen 的思想，但不声称完整复现 DexGen。

本设计合并此前方案和最新确认：

- Real DP 的 diffusion horizon 是 64，公开执行输出是 50 步；
- Sim DP 的 horizon 是 12，观测历史为 4 步，可预测动作数为 9；
- 每段实际执行长度改为 5，不再是 4；
- 50 步分为 10 段，每段用 Sim DP 独立引导；
- Sim 上直接引导的 x0 切片是 [3:8]；
- 自定义 DDIM 在 guidance_scale=0 时必须逐步等价于官方 scheduler.step。

这里的 64/50 是由 run.sh 中 horizon=64、n_obs_steps=1、n_action_steps=50
overrides 训练出的目标 Real checkpoint contract，不是所有仓库 Bulb checkpoint
的通用默认值。基础 DINO 配置仍可能是 16/2/8；加载到非目标 checkpoint 必须
拒绝继续，而不是假定它会返回 50 步。

## 2. 架构边界

新功能由四部分组成：

1. 分别加载 Real 与 Sim checkpoint、EMA（若存在）及各自 normalizer；
2. 独立的 custom guided DDIM sampler；
3. 10 段、每段 5 步的闭环 coordinator；
4. 新的命令行入口，提供 check 与 dry-run 模式。

现有 inference_dp.py 保持为单独的 Real DP inference，不复用成新入口。
本阶段不改 Real/Sim 训练逻辑、数据集格式、U-Net 主体或 Diffusers 源码。
唯一共享兼容性改动是删除 temporal utility 中硬编码的 horizon modulo 检查；
训练 loss、模型结构、checkpoint 格式与原 policy sampling 均不改变。
新入口及 loader 不得 import inference_dp.py、DirectRobotEnv 或任何真机包；
它们只依赖 torch、dill、Hydra、policy/workspace 和数据表示工具，确保 check 与
dry-run 在没有 ViTacFormer/机器人环境时仍可运行。

执行器使用抽象接口。测试和 dry-run 使用 fake executor；真机通信、时间戳、
控制频率、安全限制和故障恢复暂不在范围内。

hand-only sampler API 与 full-action coordinator 严格分层：

~~~python
guided_hand = guidance.guide_segment(
    hand_state_history,   # (1, 4, 22)
    hand_reference,       # (1, 5, 22)
)                         # -> (1, 5, 22)
~~~

guidance 模块不知道 arm 或 31-D action 的存在；coordinator 独占 Real 分段、
[9:31] 提取/替换、executor 调用和 state-history 更新职责。

## 3. 维度语义

Bulb/Real action 共 31 维：

- [0:9]：末端执行器动作；
- [9:31]：22 维 absolute hand joint action。

Sim-Hand action 正好 22 维。已确认两者的关节顺序、单位和 absolute 语义一致，
因此直接提取：

~~~python
real_hand_reference = real_segment[..., 9:31]
~~~

不增加 permutation、符号变换、relative/absolute 转换或学习映射。运行时必须
验证 Real action_dim=31、Sim action_dim=22，否则直接抛错。

引导完成后只替换 [9:31]；[0:9] 必须与 Real proposal 保持一致。

## 4. 时间语义

### 4.1 Real

- diffusion horizon：64；
- 可执行输出：50；
- segment length：5；
- segment count：10。

加载目标 checkpoint 后必须断言 saved cfg 的 horizon=64、n_obs_steps=1、
n_action_steps=50，并以实际 prediction["action"] 的 shape 再次验证 50 步。

分段边界固定覆盖：

~~~text
[0:5], [5:10], [10:15], [15:20], [20:25],
[25:30], [30:35], [35:40], [40:45], [45:50]
~~~

64 是扩散轨迹长度，50 是 policy 返回给执行侧的动作长度；只对 50 做分段，
不能把 64 当作 execution length。

### 4.2 Sim

- n_obs_steps：4；
- horizon：12；
- n_pred_action_steps：9；
- oa_start：3；
- guided execution_steps：5；
- guided slice：[3:8]；
- 全部 9 步 prediction slice：[3:12]。

execution_steps=5 是新 guided inference 模块自己的配置。不得从旧 checkpoint
中的 n_action_steps=4 或 execution_action_slice=[3:7] 推导它。

### 4.3 验证规则

只验证真正的语义约束：

~~~python
execution_steps > 0
execution_steps <= sim_n_pred_action_steps
sim_oa_start + execution_steps <= sim_horizon
real_execution_steps % execution_steps == 0
~~~

本方案对应 5>0、5<=9、3+5<=12、50%5==0。

明确不要求：

- 64、12 或 9 能被 5 整除；
- execution length 能被 4 整除。

sim_hand_temporal_util.py 中硬编码的 horizon % 4 检查是对 U-Net 下采样结构的
推测，不应成为执行语义。删除该 modulo 拒绝，保留 policy 已有的 no-grad
dummy U-Net forward shape probe，以真实网络输入输出形状作为兼容性依据。
若网络确实无法处理指定 horizon，probe 必须立刻给出明确错误。

这项改动会改变共享 Sim temporal validator 和它现有的测试，但不会改变训练
算法。实现时必须同步更新旧 Sim design 文档与 temporal tests：validator 不再
仅因 modulo 拒绝合法的 derived horizon；policy construction 仍可由真实 U-Net
probe 基于实际 shape 拒绝不兼容结构。guided checkpoint 本身仍保持 horizon=12，
且绝不把 checkpoint policy.n_action_steps 或 execution_action_slice 改成 5。

## 5. 旧 Sim Checkpoint 兼容性

仅把 guided execution length 从 4 改成 5 不需要重新训练。Sim 训练 loss 覆盖
完整的 12 步 action trajectory；旧 checkpoint 的 4 步只是公开 inference slice，
不代表位置 7、8 没训练。

新 sampler 必须生成完整 (B,12,22) x0，再取 [3:8]，不能依赖旧 policy 只返回
4 步的公共接口。若 checkpoint 的真实网络形状或维度不兼容，则抛错，禁止静默
padding、truncation 或重解释。

guided path 直接访问恢复后的 Sim policy.model、normalizer 与 scheduler config，
不得调用 Sim policy.predict_action()。Sim policy 没有独立 observation encoder；
global condition 的定义固定为：

~~~python
normalized_history = sim_normalizer["obs"].normalize(state_history)
global_cond = normalized_history[:, :4, :].reshape(B, 4 * 22)
~~~

## 6. Checkpoint 与 Normalizer

通过仓库现有 Workspace/Hydra payload 机制分别恢复两套 policy：

- 恢复模型结构和权重；
- checkpoint 有 EMA 且配置选择 EMA 时使用 EMA；
- 恢复各自 normalizer；
- 移到目标 device，设置 eval，关闭参数梯度。

Real 与 Sim normalizer 完全独立，不共享、不覆盖。

每个 segment 的数据流：

1. Real policy 输出已反归一化、但仍是 mixed policy representation 的 50 步
   proposal：前 9 维为相对 latest EE frame，手部 [9:31] 已是物理 absolute target；
2. 提取当前 5 步的 [9:31] hand reference；
3. 用 Sim action normalizer 归一化 reference；
4. 用 Sim observation normalizer 归一化当前 (B,4,22) state history；
5. 在 Sim normalized action space 内做 DDIM guidance；
6. 将 DDIM chain 的 terminal_sample[:,3:8,:] 用 Sim action normalizer
   反归一化；
7. 只替换当前 Real segment 的 [9:31]。

guidance loss 也在 Sim normalized action space 计算。两套 policy 数据相交处都要
检查 shape、dtype、device 和 finite。

## 7. DDIM Scheduler

Sim policy 原有 scheduler 是 DDPM。新模块从已加载 scheduler 的 config 构造
单独的 DDIMScheduler，不修改 policy 原 scheduler。

首版约束：

- 与仓库固定的 diffusers 0.11.1 行为一致；
- check mode 必须断言并打印 diffusers.__version__ == "0.11.1"；
- num_inference_steps=12；
- eta=0.0；
- eta 非零直接抛错；
- prediction_type 必须严格等于 epsilon，其他类型直接抛错；
- thresholding 必须为 False，首版不支持 dynamic thresholding；
- 从 DDPM config 构造 DDIM 时显式固定 set_alpha_to_one=True；
- 显式固定 steps_offset=0。

betas/alphas、clip_sample、num_train_timesteps 与 prediction_type 来自恢复后的
DDPM config。set_alpha_to_one 和 steps_offset 不存在于 DDPM config，因此属于
本 inference-DDIM 的显式版本契约。check mode 必须打印并验证这两个值及实际
timestep 序列 [88,80,72,64,56,48,40,32,24,16,8,0]，不能依赖库默认值。
factory regression 必须逐元素断言 DDIM alphas_cumprod 与恢复的训练 DDPM
alphas_cumprod 完全一致，避免 custom 与 official 同时使用一张错误 alpha 表而
产生“自洽但错误”的零 guidance 结果。

## 8. Custom Guided DDIM

### 8.1 输入输出

单步输入：

~~~text
sample x_t                 (B,12,22)
model_output               (B,12,22)
timestep                   scheduler scalar timestep
reference                  (B,5,22), Sim normalized
guidance_scale             non-negative scalar
scheduler state/config
~~~

输出至少包含实际执行的整数 timestep、prev_sample、pred_original_sample、
base_pred_original_sample、raw_pred_original_sample、raw variance 与 direction
coefficient。full-chain diagnostics 的 timestep 序列必须从这些 step outputs
提取，不能只回显 scheduler.timesteps。
pred_original_sample 表示 reverse update 真正使用的 x0_guided；scale=0 时它
等于官方 clipped/base x0。base_pred_original_sample 始终是 guidance 前的
x0_base，raw_pred_original_sample 始终是 clipping 前的 x0_raw。

### 8.2 官方基础语义

epsilon prediction 下：

~~~text
x0_raw = (x_t - sqrt(1-alpha_bar_t) * epsilon_pred) / sqrt(alpha_bar_t)
~~~

若 clip_sample=True：

~~~text
x0_base = clip(x0_raw, -1, 1)
~~~

scale=0 时 pred_original_sample 的回归比较使用官方 clipped/base 语义；日志
必须区分 x0_raw、x0_base 与 x0_guided。

在固定版本默认 use_clipped_model_output=False 时，direction term 仍使用原始
epsilon prediction。custom step 必须逐行复现当前安装版本的算术语义，不能意外
从 clipped 或 guided x0 重新推导 direction。

### 8.3 Guidance

只对 x0_base[:,3:8,:] 定义每个 batch sample 独立的 MSE：

~~~text
L_b = mean((x0_base[b,3:8,:] - reference[b,:,:]) ** 2)
grad[b,3:8,:] = 2/(5*22) * (x0_base[b,3:8,:] - reference[b,:,:])
grad outside [3:8] = 0
~~~

更新：

~~~text
x0_guided = x0_base - guidance_scale * raw_ddim_variance * grad
~~~

guidance 后不做第二次 clipping；这是显式的 x0-space energy update。实现需要
记录 x0_guided 超出 [-1,1] 的比例，供 guidance scale 实验诊断。scale=0 时
自然仍严格退化为 x0_base。

raw_ddim_variance 是 scheduler variance 在乘 eta 前的值，不是 eta=0 时为零的
随机标准差。首版固定 set_alpha_to_one=True 且最终 t=0，因此最后一步 raw
variance 必为零，direct guidance 在该步是 no-op；此前步骤的影响仍会传递。

不对 U-Net 反向传播。模型预测和解析梯度更新都在 torch.no_grad() 下执行。
梯度 reduction 只跨 5*22，不能跨 batch。

raw variance 与 eta=0 reverse update 明确为：

~~~text
v_t = (1-alpha_bar_prev)/(1-alpha_bar_t)
      * (1-alpha_bar_t/alpha_bar_prev)
x_prev = sqrt(alpha_bar_prev) * x0_guided
         + sqrt(1-alpha_bar_prev) * epsilon_pred
~~~

第二项必须使用原始 model_output epsilon，不能从 x0_base 或 x0_guided 反推。
eta=0 时不加入随机 variance noise。

## 9. 零 Guidance 等价性

核心 invariant：

~~~text
guidance_scale == 0
=> Custom DDIM == Diffusers DDIMScheduler.step
~~~

必须是同一 custom arithmetic path 的自然结果。禁止用 scale==0 时直接调用
scheduler.step 的分支掩盖实现错误。

### 9.1 逐 timestep regression

固定相同 x_t、model_output、timestep、scheduler config 和 eta=0，对每个 reverse
step 比较：

- custom.pred_original_sample 与 official.pred_original_sample；
- custom.prev_sample 与 official.prev_sample。

使用 torch.testing.assert_close。失败时输出第一个不一致 timestep，以及
alpha_bar_t、alpha_bar_prev、raw variance、direction coefficient、x0_raw、
x0_base 和 x_prev。
这里的“等价”指固定 rtol=1e-5、atol=1e-6 内的浮点数值等价，不要求跨设备或
不同算子写法 bitwise identical。

### 9.2 Full-chain regression

还要从同一个固定 initial noise、condition 和 timestep 序列出发，分别运行完整
官方链与 custom scale=0 链。每一步都比较，不只比较最终 action。这能捕获单步
fixture 看不到的 timestep/state propagation 错误。

## 10. 分段闭环流程

首版仅在流程开始时运行一次 Real DP，得到原始 50-step proposal。后续十段都
引用这份 proposal 中各自的 [5i:5i+5]；本阶段不在每执行 5 步后重跑 Real DP。

对 10 个 Real segment 逐一执行：

1. 提取并 Sim-normalize 当前 (B,5,22) reference；
2. normalize 当前 (B,4,22) hand state history；
3. flatten normalized history 得到 (B,88) global_cond，并直接调用
   sim_policy.model(sample, timestep, global_cond=global_cond)；
4. 为本 segment 新采样 (B,12,22) Gaussian initial noise；
5. 运行 12 个 custom DDIM reverse steps；
6. 反归一化 terminal_sample[:,3:8,:]；
7. 替换当前 Real segment 的 [9:31]；
8. 将 5 步 mixed policy action 交给 fake executor；它不是可直接下发真机的
   absolute controller command；
9. 接收 5 个 post-action hand states；
10. 更新下一段的 4-state history。

每段必须 fresh noise，不复用一份 initial noise。

若执行前状态为 s_t，5 步返回 s_{t+1} 到 s_{t+5}，下一段 condition 必须是：

~~~text
[s_{t+2}, s_{t+3}, s_{t+4}, s_{t+5}]
~~~

executor 必须正好返回 5 个 finite、22 维状态；数量或形状不符立即抛错。

本阶段 coordinator 与 executor 明确只支持 B=1，入口发现其他 batch size 时
fail fast。最小 executor protocol 是：

~~~text
reset() -> initial_hand_history: (1,4,22)
execute(segment: (1,5,31)) -> post_states: (1,5,22)
~~~

fake reset 根据 seed 生成四个有顺序、可区分的初始状态；禁止 coordinator 自行
重复或 padding 单帧。若第 k 个 action 执行时失败，executor 抛出包含 segment_id、
executed_count 和可选 partial states 的 ExecutionError。coordinator 必须终止，
不得更新成伪造的完整 history，也不得继续下一 segment。

## 11. 新推理入口

新增独立脚本，至少接受：

~~~text
real checkpoint path
sim checkpoint path
device
guidance scale
execution steps, default 5
sim inference steps, default 12
mode: check or dry-run
optional deterministic seed
~~~

底层 custom DDIM 必须支持 guidance_scale=0 以完成 oracle regression；面向
check/dry-run 的 SimHandGuidance 配置则要求 guidance_scale>0，确保实际服务链
与独立 zero-guidance oracle 是两条不同的验证路径。CLI 对 0 或负值直接失败。

check mode 加载 checkpoint 并检查：

- action dimensions；
- temporal bounds 与 segment count；
- normalizer；
- scheduler config、prediction_type 和 eta；
- 实际 U-Net temporal shape probe；
- 从 Real checkpoint 的 shape_meta 构造最小合法 synthetic observation，实际
  调用一次 Real predict_action 并验证 finite (1,50,31) 输出；
- 使用配置中的非零 guidance_scale 实际调用一次 guide_segment_detailed；报告的
  完整 (1,12,22) chain、12 个 timestep 与返回的物理空间 (1,5,22) hand action
  都必须 finite；
- 零 guidance 的逐步和 full-chain 等价性。

任何失败非零退出。

dry-run 从 Real checkpoint 的 shape_meta 生成全部必需 image/low-dimensional
observation，跑一次真实 Real inference，再接 seeded fake executor 跑完整 10 段。
第一版明确支持 Bulb schema 的 front_image、wrist_image、ee_pose、hand_joint；
每个 tensor 的 (B,n_obs_steps,...) shape、dtype 与 device 从 cfg.shape_meta 和
normalizer contract 构造并校验。
仅向 coordinator 注入 synthetic Real proposal 的测试属于单元测试，不代替此
CLI integration dry-run。全程输出 shape、segment boundary、state-window update
和 guidance diagnostics，不连接真机。

建议的独立配置默认值：

~~~yaml
real_horizon: 64
real_obs_steps: 1
real_execution_steps: 50
execution_steps: 5
real_hand_slice: [9, 31]
sim_obs_steps: 4
sim_horizon: 12
sim_n_pred_action_steps: 9
sim_oa_start: 3
sim_num_inference_steps: 12
eta: 0.0
guidance_scale: 1.0
distance: mse
distance_reduction: per_sample_mean
variance_weighting: ddim_raw_variance
fresh_noise_per_segment: true
~~~

guidance slice 必须由 slice(sim_oa_start, sim_oa_start + execution_steps) 派生。
segment count、slice end 等派生值不在多个模块重复硬编码。

## 12. 错误策略

下列情况直接抛出带上下文的 error：

- checkpoint 缺失或不兼容；
- normalizer 缺失；
- action/state dimension 错误；
- scheduler config 或 prediction_type 不支持；
- check/dry-run guidance_scale 不是正数；
- eta 非零；
- temporal slice 或 execution length 越界；
- Real 返回长度不能被 segment length 整除；
- U-Net dummy forward shape mismatch；
- 组合 tensor 的 dtype/device 不一致；
- observation、reference、gradient 或 sampler output 出现 NaN/Inf；
- executor 未返回正好 5 个合法状态；
- executor 在 segment 中途失败或只返回 partial states。

禁止静默 pad、truncate、repeat、reorder 或强制解释不兼容数据。

## 13. 测试计划

### 13.1 DDIM 算术

- 断言 runtime diffusers version 是 0.11.1；
- 从 Sim DDPM config 构造 DDIM 后断言 set_alpha_to_one=True、steps_offset=0，
  并断言 100 train steps / 12 inference steps 的实际 timesteps；
- DDIM 与训练 DDPM 的 alphas_cumprod 逐元素完全一致；
- 所有 reverse timestep 的 scale=0 x0 与 prev_sample 等价；
- 完整 scale=0 chain 每一步等价；
- fixture 主动触发 clip_sample=True clipping；
- thresholding=True 直接抛错；
- 分别验证 raw x0 和 clipped x0；
- 对全部 12 个 timestep 用闭式公式验证 raw variance，而不是标准差，并在一个
  非末步直接验证 x0_guided = x0_base - scale * raw_variance * gradient；
- 验证最终 zero-variance；
- eta 非零抛错。

### 13.2 Guidance

- 解析 gradient 与仅测试中使用的 autograd reference 一致；
- x0 guidance 只修改 [3:8]；
- reverse combination 前其他位置不变；
- batch sample 之间不串 gradient；
- scale=0 没有 official bypass；
- 在 raw variance 非零的受控 fixture 中，正 scale 降低 reference loss；
- reference 与 base x0 相同时 gradient 严格为零；
- loss-decrease fixture 使用满足 0 < scale * raw_variance < 110 的固定小 scale，
  不把任意正 scale 都会降低 loss 当作性质；
- 不做二次 clipping，并验证/记录越界比例。

### 13.3 Temporal

- 50 精确分成 10 个长度 5 的 segment；
- [0:50] 无 gap、无 overlap；
- guided output 正好是 [3:8] 的 (B,5,22)；
- 64、12、9 不被错误要求整除 5；
- execution 不再要求整除 4；
- U-Net probe 取代 modulo 规则判断网络兼容性；
- derived horizon=11 的 temporal validator 不因 modulo 本身失败；若当前 U-Net
  不能保持该长度，则由实际 policy probe 给出结构性 shape error；

### 13.4 Normalization 与组合

- Real/Sim normalizer 始终独立；
- 真实 Sim Workspace checkpoint round-trip 后，obs/action 两组非 identity
  normalizer state 逐项保持且位于 policy device；
- known hand action 可经恢复后的 Sim normalizer 数值 round-trip；
- guidance 收到 normalized reference 与 state；
- 只替换 [9:31]；
- [0:9] 保持不变。

### 13.5 Closed loop 与兼容性

- 10 个 segment 各自生成 fresh noise；
- 5 个返回状态后保留 latest 4；
- fake executor 每次正好收到 5 个 command；
- malformed executor output 立即失败；
- dry-run 无硬件完成全部 10 段；
- 第一段使用 fake reset 返回的四帧，不做隐式 repeat/padding；
- 第 i+1 段 history 精确等于第 i 段 post-states 的最后四帧；
- 每段 reference 始终来自最初 Real proposal 的 [5i:5i+5]；
- 任一 action 位置发生 ExecutionError 后不再调用后续 segment；
- 固定 seed 时全链确定，同时每段独立消费 RNG 获得 fresh noise；
- 旧 public action count=4 的 Sim checkpoint 仍可生成 full x0 并取 [3:8]；
- EMA、normalizer 恢复符合 checkpoint；
- import 新 package 与 CLI --help 时，inference_dp、DirectRobotEnv、
  diffusion_policy.real_world、ViTacFormer、pyrealsense2 与 ur_rtde 均未加载；
- 不兼容 dimension/shape 清晰报错。

## 14. Debug 信息

debug level 记录：

~~~text
segment_id, segment_start, segment_end
guidance_scale, num_inference_steps, timestep
alpha_bar_t, alpha_bar_prev, raw_variance, direction_coefficient
guidance_loss_mean, guidance_gradient_norm
x0_raw_min_max, x0_base_min_max, x0_guided_min_max
x0_guided_out_of_range_ratio
distance_before_guidance, distance_after_guidance
norm_guided_minus_reference, sim_inference_latency
prev_sample_min_max
~~~

零 guidance regression 额外报告第一个不一致 timestep 和 field。默认不打印完整
production tensor。

## 15. 非目标

本阶段不包含：

- 真机通信与安全接口；
- 重新训练任一 policy；
- 改变 22 维 joint semantics；
- 修改 dataset format；
- 修改 standalone inference_dp.py 行为；
- stochastic DDIM；
- 调参确定最优 guidance scale；
- 宣称任务成功率提升；
- 完整复现 DexGen。

## 16. 验收标准

1. 新入口独立加载 Real 与 Sim checkpoints。
   Real checkpoint 必须满足 saved 64/1/50 contract，实际输出 (1,50,31)。
2. Real 50 步严格按 10 个 5-step closed-loop segment 处理。
3. Sim 在四步 state history 下引导 22 维 [3:8]。
4. 只替换 Real [9:31]，保留 [0:9]。
5. 旧 4-step public slice 的 Sim checkpoint 无需重训。
6. guided path 不含错误的整除 4 或 horizon 整除 5 验证。
7. scale=0 时 custom sampler 逐步及全链等价于固定版本官方 DDIM。
8. check mode 同时通过 configured nonzero guidance 与独立 zero-guidance oracle，
   fake-executor dry-run 完成十段。
9. 所有 contract violation 都以可定位 error 失败。
10. 现有 standalone inference 与 training 测试不回归。
