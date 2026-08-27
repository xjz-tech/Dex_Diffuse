# Sim-Hand Guided Inference Design

## 1. 目标

新增一条独立、无真机依赖的 inference 链路：

1. 原 Bulb/Real Diffusion Policy 按现有方式加载并调用
   `predict_action()`；
2. 从 Real proposal 提取最后 22 维 absolute hand target；
3. 原 Sim-Hand checkpoint 提供模型、scheduler 与 normalizer；
4. custom DDIM 在完整 Sim trajectory 的 `x0` 上执行 guidance；
5. 每次输出 5 步 hand action，按闭环方式处理完整 Real proposal；
6. fake executor 完成 dry-run 与回归测试。

当前目标 checkpoint 的观测事实是：

- Real policy：`horizon=64`、`n_obs_steps=1`，实际 proposal 为
  `(1,50,31)`；
- Sim policy：`n_obs_steps=4`、`horizon=12`、
  `n_pred_action_steps=9`、`n_action_steps=4`；
- `execution_steps=5` 时，当前 Real proposal 形成 10 段，当前 Sim guided
  slice 为 `[3:8]`。

这些值用于当前 checkpoint 的 integration test 与运行日志，不在 guidance
配置或生产算法中重复硬编码。生产代码从恢复后的 policy、temporal metadata、
scheduler 和实际 Real proposal shape 读取它们。

## 2. 最小改动边界

下列现有体系保持不变：

- Real training、Real policy 和 DINOv2 加载逻辑；
- `inference_dp.py` 的行为；
- Sim training、Sim policy 和 `predict_action()`；
- `sim_hand_temporal_util.py`，包括现有 `horizon % 4` validator；
- 原有 U-Net temporal shape probe；
- checkpoint 格式、dataset 格式和 Diffusers 源码。

新入口不 import `inference_dp.py`，因为它同时引入真机环境；但薄 loader 使用
与其相同的 Workspace 恢复顺序：

~~~text
checkpoint
-> payload["cfg"]
-> Workspace(cfg)
-> workspace.load_payload(...)
-> workspace.ema_model or workspace.model
-> eval(), device
~~~

不新增 DINO preflight、offline contract、视觉依赖抽象或通用 inference
framework。Guidance 从以下边界之后介入：

~~~python
real_action = real_policy.predict_action(real_obs)["action"]
~~~

仅新增：

~~~text
thin checkpoint restore
-> policy-derived Real/Sim adapters
-> custom guided DDIM
-> 5-step hand guidance
-> segment coordinator
-> fake check/dry-run/tests
~~~

## 3. 动作语义

当前 Bulb action 为 31 维：

- `[0:9]`：Real policy 的 relative-EE 表示；
- `[9:31]`：22 维 absolute hand joint target。

Sim action 为同顺序、同单位、同 absolute 语义的 22 维，因此：

~~~python
hand_reference = real_segment[..., 9:31]
~~~

不增加 permutation、符号变换、relative/absolute 转换或学习映射。Guidance
完成后只替换 `[9:31]`，`[0:9]` 必须逐元素保持原 proposal。

运行时检查实际 Real proposal 的最后一维为 31，Sim policy 的
`obs_dim == action_dim == 22`。

## 4. 时间语义全部从 policy 派生

### 4.1 Real proposal

Real loader 不覆盖任何 temporal 字段。调用 `predict_action()` 后，以实际
proposal 为准：

~~~python
real_execution_steps = real_action.shape[1]
segment_count = real_execution_steps // execution_steps
~~~

只要求：

~~~python
real_action.ndim == 3
real_action.shape[0] == 1
real_action.shape[2] == 31
real_execution_steps % execution_steps == 0
~~~

当前 checkpoint 的 integration test 额外断言
`real_action.shape == (1,50,31)` 和 `segment_count == 10`，但 production
coordinator 不把 50 或 10 写成模型配置常量。

### 4.2 Sim policy

从恢复后的 policy 读取：

~~~python
sim_horizon = sim_policy.horizon
sim_obs_steps = sim_policy.n_obs_steps
sim_pred_action_steps = sim_policy.n_pred_action_steps
sim_obs_dim = sim_policy.obs_dim
sim_action_dim = sim_policy.action_dim
oa_start = sim_policy.temporal.usable_action_slice.start
guided_slice = slice(oa_start, oa_start + execution_steps)
~~~

当前实现中 `oa_start == sim_obs_steps - 1`；优先读取 policy 已保存的 temporal
metadata，避免 guidance 复制 temporal convention。

Guidance 只新增以下验证：

~~~python
execution_steps > 0
execution_steps <= sim_pred_action_steps
guided_slice.stop <= sim_horizon
sim_obs_dim == sim_action_dim == 22
~~~

不得修改 `sim_policy.n_action_steps`、`execution_action_slice` 或任何旧
checkpoint 字段。当前 checkpoint 仍公开返回 4 步，但 guidance 直接调用
`sim_policy.model` 生成完整 `(B,sim_horizon,22)` trajectory，再取动态
`guided_slice`。当前值自然得到 `[3:8]`。

## 5. Condition 与 Normalizer

Real 与 Sim normalizer 完全独立。每段数据流：

1. Real `predict_action()` 返回已反归一化的 mixed action；
2. 提取当前 segment 的 `[9:31]`；
3. 用 Sim action normalizer 归一化 hand reference；
4. 用 Sim obs normalizer 归一化 state history；
5. 在 Sim normalized action space 执行 guidance；
6. 用 Sim action normalizer 反归一化 guided hand；
7. 只替换 Real segment 的 `[9:31]`。

global condition 从恢复后的 Sim policy 派生：

~~~python
normalized_history = sim_normalizer["obs"].normalize(state_history)
global_cond = normalized_history[
    :, :sim_policy.n_obs_steps, :
].reshape(batch_size, -1)
~~~

目标 checkpoint 的 shape 是 `(1,4,22) -> (1,88)`，但生产代码不写死 4 或
88。所有交界 tensor 必须检查 shape、dtype、device 与 finite。

## 6. 动态 DDIM Scheduler

Sim policy 原 scheduler 为 DDPM。Guidance 从恢复后的 scheduler config 构造独立
`DDIMScheduler`，不修改 `sim_policy.noise_scheduler`：

~~~python
ddim = DDIMScheduler.from_config(
    sim_policy.noise_scheduler.config,
    set_alpha_to_one=True,
    steps_offset=0,
)
ddim.set_timesteps(num_inference_steps)
timesteps = ddim.timesteps
~~~

生产约束：

- 使用仓库环境固定的 Diffusers 0.11.1 算术；
- `prediction_type == "epsilon"`；
- `thresholding == False`；
- `eta == 0.0`；
- `num_inference_steps > 0` 且由 scheduler 判断是否超过训练步数；
- DDIM 的 alpha/beta schedule 来自恢复后的 DDPM config。

生产代码不得：

- 要求 `num_train_timesteps == 100`；
- 写死 `[88,80,72,64,56,48,40,32,24,16,8,0]`；
- 根据固定 timestep 数组驱动或验证 sampling。

当前 100-train-step checkpoint 的测试可以断言
`set_timesteps(12)` 得到上述序列。另一个非 100-step fixture 必须证明生产
factory 与 sampler 仍按 scheduler 动态工作。两个 scheduler 的
`alphas_cumprod` 必须逐元素一致。

## 7. Custom Guided DDIM

### 7.1 动态输入

单步输入 shape 由实际 sample/reference 决定：

~~~text
sample x_t       (B,H,A)
model_output     (B,H,A)
reference        (B,E,A)
guided_slice     length E
guidance_scale   non-negative scalar
~~~

当前目标值为 `H=12`、`A=22`、`E=5`。

### 7.2 官方基础算术

`epsilon` prediction 下：

~~~text
x0_raw = (x_t - sqrt(1-alpha_bar_t) * epsilon_pred)
         / sqrt(alpha_bar_t)
x0_base = clip(x0_raw, -1, 1)  # only when clip_sample=True
~~~

direction term 使用原始 `epsilon_pred`，不从 clipped/guided x0 重算。

### 7.3 Analytic guidance

每个 batch sample 独立计算：

~~~text
L_b = mean((x0_base[b,guided_slice,:] - reference[b,:,:]) ** 2)
grad = 2/(E*A) * (x0_base[guided_slice] - reference)
x0_guided = x0_base - guidance_scale * raw_ddim_variance * grad
~~~

guided slice 外 gradient 为零。Guidance 后不做第二次 clipping。不对 U-Net
反向传播；模型预测与 analytic update 都在 `torch.no_grad()` 下执行。

raw variance：

~~~text
v_t = (1-alpha_bar_prev)/(1-alpha_bar_t)
      * (1-alpha_bar_t/alpha_bar_prev)
~~~

`eta=0` reverse update：

~~~text
x_prev = sqrt(alpha_bar_prev) * x0_guided
         + sqrt(1-alpha_bar_prev) * epsilon_pred
~~~

每个 step output 记录实际执行的整数 timestep。full-chain diagnostics 从 step
outputs 提取实际序列，不用固定常量替代。

## 8. Zero-guidance 回归

核心 invariant：

~~~text
guidance_scale == 0
=> Custom DDIM == Diffusers DDIMScheduler.step
~~~

`guidance_scale=0` 必须走同一 custom arithmetic path，禁止分支调用官方
`scheduler.step()` 掩盖错误。

对 scheduler 动态产生的每个 timestep，固定相同 sample、model output、config
与 `eta=0`，比较：

- `pred_original_sample`；
- `prev_sample`。

还要从相同 initial noise 与 condition 分别运行完整 official/custom chain，
逐步比较 evolving state。容差固定为 `rtol=1e-5, atol=1e-6`。

`guidance_scale` 的公共配置允许 `>=0`。check mode 总是运行独立 zero oracle；
configured path 本身可以是 0，方便直接回归。

## 9. Hand guidance API

Guidance config 只包含真正新增的参数：

~~~yaml
execution_steps: 5
guidance_scale: 1.0
num_inference_steps: 12
eta: 0.0
~~~

稳定 API：

~~~python
guided_hand = guidance.guide_segment(
    hand_state_history,
    hand_reference,
    generator=generator,
)  # (1, execution_steps, 22)
~~~

内部：

1. 从 adapter 读取 Sim shape 和 `guided_slice`；
2. normalize history/reference；
3. 每次调用新采样 `(1,sim_horizon,22)` initial noise；
4. scheduler 动态生成 timesteps；
5. 运行完整 custom chain；
6. 取 terminal trajectory 的动态 `guided_slice`；
7. unnormalize 为物理 hand target。

不调用 `sim_policy.predict_action()`，不改变它原来的 4-step public output。

## 10. Closed-loop coordinator

coordinator 接收一次 Real proposal，并根据实际长度动态分段：

~~~python
for start in range(0, real_action.shape[1], execution_steps):
    stop = start + execution_steps
~~~

当前 proposal 为 50 步，因此得到 10 个 `[5i:5i+5]` segment。每段：

1. reference 始终来自最初 Real proposal；
2. 调 `guide_segment()`；
3. 只替换当前 segment 的 `[9:31]`；
4. fake executor 接收 `(1,E,31)`；
5. 返回 `(1,E,22)` post states；
6. 使用
   `concat(previous_history, post_states)[:, -sim_obs_steps:, :]`
   更新下一段 condition。

当前 `E=5`、`sim_obs_steps=4` 时，下一 history 是
`[s_{t+2},s_{t+3},s_{t+4},s_{t+5}]`。

每段必须消费 fresh noise。首版 coordinator/executor 仅支持 `B=1`。partial
execution 抛 `ExecutionError` 并立即停止，禁止构造伪 history。

## 11. Check、dry-run 与入口

新增独立脚本，不修改或 import `inference_dp.py`。参数至少包括：

~~~text
real checkpoint
sim checkpoint
device
execution_steps
guidance_scale
num_inference_steps
eta
mode: check | dry-run
seed
~~~

check：

- 薄 loader 恢复两套 policy/EMA/normalizer；
- 从 Real shape_meta 构造 synthetic observation；
- 实际调用一次原 Real `predict_action()`；
- 按实际 proposal shape 计算 segment count；
- 实际调用一次 configured `guide_segment()`；
- 运行动态 timestep 的 zero-guidance step/full-chain oracle；
- 输出恢复后的 temporal values、实际 proposal shape、guided shape 和实际
  timestep 序列。

dry-run：

- Real inference 只执行一次；
- 使用同一 original proposal；
- seeded fake executor 跑完动态 segment count；
- 当前 target gate 断言 10 段；
- 不连接真机。

## 12. 错误策略

以下情况直接抛出带上下文的 error：

- checkpoint/payload/workspace/model/normalizer 缺失；
- Real proposal 不是 finite `(1,T,31)`；
- `T % execution_steps != 0`；
- Sim `obs_dim/action_dim != 22`；
- `execution_steps <= 0` 或超过 Sim usable prediction；
- derived guided slice 超出 Sim horizon；
- `guidance_scale < 0`；
- `num_inference_steps <= 0`、`eta != 0`；
- scheduler prediction type/thresholding 不支持；
- tensor shape、dtype、device 不一致或出现 NaN/Inf；
- executor output 不合法或 partial failure。

禁止静默 pad、truncate、repeat、reorder、cast 或修改 checkpoint temporal 字段。

## 13. 测试

### 13.1 不回归边界

- `sim_hand_temporal_util.py` 与旧 temporal tests 不修改；
- Real/Sim training 和 public `predict_action()` 不修改；
- import 新入口不加载真机模块；
- thin loader 不覆盖任何 policy temporal/inference 字段。

### 13.2 当前 checkpoint contract

- 当前 Real proposal 是 `(1,50,31)`，动态得到 10 段；
- 当前 Sim 恢复值是 `4/12/9/4`，derived guided slice 是 `[3:8]`；
- 当前 100-step DDPM 配 12-step DDIM 得到
  `[88,80,72,64,56,48,40,32,24,16,8,0]`；
- 这些常量只存在于测试 expected，不被 production module 导入。

### 13.3 动态性

- 非 100 train-step scheduler 仍能构造 DDIM 并动态采样；
- adapter 使用 policy temporal metadata，而非 guidance 常量；
- coordinator 使用实际 proposal length；
- history window 使用实际 `sim_obs_steps`；
- 不要求 execution length 整除 4，也不修改现有 horizon validator。

### 13.4 DDIM 与 guidance

- alpha schedule 与训练 DDPM 一致；
- raw variance 用独立公式验证；
- scale=0 每步/full-chain 等价；
- analytic gradient 与测试 autograd reference 一致；
- gradient 只在 dynamic guided slice；
- no second clipping；
- U-Net 无反向图；
- 每段 fresh noise。

### 13.5 组合与闭环

- 只替换 `[9:31]`；
- `[0:9]` 完全不变；
- Real/Sim normalizer 独立；
- latest state window 正确；
- malformed/partial executor 立即停止；
- current fake dry-run 完成 10 段。

## 14. 非目标

- 真机通信与安全控制；
- DINO/offline/vision dependency 重构；
- 修改 Real 或 Sim training；
- 修改原 policy sampling；
- 修改 temporal validator；
- 支持新的 hand joint semantics；
- stochastic DDIM；
- guidance scale 调参或成功率声明；
- 完整复现 DexGen。

## 15. 验收标准

1. 原 Real/Sim/temporal 文件零行为改动。
2. Guidance 从原 Real `predict_action()["action"]` 后介入。
3. 模型 temporal 参数全部来自 checkpoint/policy/proposal。
4. guidance config 只含 execution steps、scale、DDIM inference steps 与 eta。
5. 当前 `(1,50,31)` proposal 动态分成 10 个 5-step segment。
6. 当前 Sim policy 动态派生 `[3:8]` 并输出 `(1,5,22)` hand action。
7. production DDIM 不要求 100 train steps、不写死 timestep 数组。
8. scale=0 custom step 与 full chain 等价于官方 Diffusers。
9. fake dry-run 完成，且没有真机 import/连接。
10. 所有 contract violation fail loudly。
