# Bulb Hand Diffusion Policy（无图 DP）

Date: 2026-08-24

## 目标

在现有 bulb 图像 Diffusion Policy 上增加一个 **只看本体感觉、只预测手** 的 DP，并让两个网络在 inference 时都容易单独调用，以便以后用图像 DP 做 guidance（score 混合或分层 inpaint）。**本期不实现 guidance。**

## 三个策略的职责

| | 图像 DP（已有，不改行为） | 无图 step | 无图 chunk |
|--|--|--|--|
| obs | `front_image`, `wrist_image`, `ee_pose`, `hand_joint` | **只有 `hand_joint`** | 同左 |
| action | 31 维：相对 EE + 绝对手 | **22 维：只有绝对 `hand_joint`** | 同左 |
| 扩散轨迹 / loss | 全 horizon 31 维 chunk | **`(B, 1, 22)`，当前这一步** | **`(B, Ta, 22)`，整段 chunk** |
| 输出 | 一定是 action chunk | `(B, 1, 22)` | `(B, Ta, 22)` |
| 权重 | 现有 ckpt | **独立 ckpt** | **独立 ckpt** |

单步不是从 chunk 模型切一刀。两种无图策略是 **两种训练目标、两套 loss、两个 checkpoint**。推理时加载哪个 ckpt，就是哪种模式。

图像 DP 不提供单步开关。

## 架构

三个网络都走现有 **image-policy 栈**（`DiffusionUnetImagePolicy` + `TrainDiffusionUnetImageWorkspace`），用 `shape_meta` 区分观测和动作维数，**不**走 `DiffusionUnetLowdimPolicy`（那条路是拼好的 `obs` 张量，和图像 DP 的 dict obs 对不齐，以后 guide 会别扭）。

```
真机 zarr (有图)          仿真 zarr (可以没有图)
        │                          │
        ▼                          ▼
   bulb_image.yaml            bulb_hand.yaml     各自 dataset_path
        │                          │
        └──────────┬───────────────┘
                   ▼
         BulbImageDataset           同一套代码；按 shape_meta 读键
                   │
                   ├─ 图像 DP：图 + ee + hand → action 31
                   └─ 无图 DP：只有 hand → action 22
                            ├─ hand_step   Ta=1
                            └─ hand_chunk  Ta=8
```

仿真没有 EE pose，无图 DP **只 condition on `hand_joint`，也只预测手**。相对 EE 变换只存在于真机图像数据路径。

## 数据来源（仿真 vs 真机）

- **图像 DP**：主要真机数据，沿用现有 `data2dp.py` 产物（有图、有 31 维 state/action）。
- **无图 DP**：主要仿真数据，**另一个 `dataset_path`**。仿真 **没有 ee_pose，只能提供手关节**。

本期 **不新写仿真转换器**，也不为仿真另开 dataset 类。约束只加在 zarr 边界：

- 仿真 replay buffer 至少有手关节和手 action，例如 `hand_joint` `(N, 22)`、`action` `(N, 22)`、`episode_ends`。**不要求** 31 维 `state`、不要求图像
- 真机仍是 `state`/`action` `(N, 31)`：`[:9]` EE，`[9:31]` 绝对手
- 只引导手，手是绝对值，没有坐标系分叉

若仿真控制频率和真机不同，只调对应 yaml 的 `horizon` / `n_action_steps` / 推理 `hz`，不改网络类。

## 归一化（手用仿真，EE 用真机，unnormalize 必须拆开）

仿真只覆盖手。两套统计 **禁止揉成一个 31 维 `create_fit`**：

| 字段 | 来源 | 谁用 |
|--|--|--|
| `hand_joint` (22) | **可选共享**：有 `normalizer_path` 则仿真冻结 `create_manual`；**未设则在本数据集 `create_fit`**（仿真训仿真、真机训真机） | 手 obs；无图全部 action；图像 `action[..., 9:31]` |
| `ee_pose` (9) | 始终在 **当前图像数据集** 上 `create_fit`（相对 EE） | 仅图像 DP 的 `ee_pose` obs 和 `action[..., :9]` |
| 图像 | 现成 `get_image_range_normalizer()` | 仅图像 DP |

两种训练模式靠 **有没有设路径** 区分，不要做成必填：

| | 未设 `normalizer_path` | 设了且文件有效 | 设了但缺文件 / npz 坏了 |
|--|--|--|--|
| 行为 | 各自在本数据上分段 fit（合法） | 手用仿真尺子；EE 仍真机 fit | **报错**，不回落 |
| 用途 | 两边独立训练、暂不混 score | 以后要 guide / 对齐手空间 | 你意图是共享却配错了 |

未设路径时打 **一次 INFO/WARNING**：本 run 在当前数据集上 fit `hand_joint`，与另一域的手 DP **不是同一把尺子**，不要混归一化空间。这不是失败。

`scripts/fit_bulb_sim_normalizer.py` 只在你要共享时跑。

`BulbImageDataset.get_normalizer()`：

- 永远 **不要**对整段 31 维 action 做一次 `create_fit`，也不写 `normalizer["action"]`
- `normalizer_path` 为空 / `null`：对当前数据 `create_fit` `hand_joint`（以及若有 `ee_pose` 则 fit EE）
- 路径非空：文件必须存在且含 min/max，否则 raise；手 `create_manual`，EE 仍对本数据 fit
- 无图任务没有 `ee_pose` 键，normalizer 里也就没有这项

**normalize / unnormalize 按 ckpt 里有没有 `action` 键分流：**

```python
# 新模型：没有 normalizer["action"]，只有 ee_pose + hand_joint
hand = normalizer["hand_joint"].unnormalize(naction)                 # 22
ee   = normalizer["ee_pose"].unnormalize(naction[..., :9])          # 31 的前 9
hand = normalizer["hand_joint"].unnormalize(naction[..., 9:31])     # 31 的后 22

# 旧模型：ckpt 里仍有 normalizer["action"]（当时整段 31 维 create_fit）
action = normalizer["action"].unnormalize(naction)                  # 禁止改走分段
```

旧 ckpt 必须继续用那把 31 维尺子，否则会解错。第一次走到 legacy 分支时打 **warning**（每个进程一次，避免逐步推理刷屏），说明正在用旧模型、归一化与新的仿真手尺子不一致，不能和新手 DP 混 score。

新训的图像 DP 才走拆开的 EE/手。旧图像 DP 仍可按原方式加载、推理，不用重训。

这样以后只在手上混 score 时，两边手的归一化空间一致；EE 始终只在真机图像 DP 里。

## 调用面（现在就要有，guidance 以后用）

在 `DiffusionUnetImagePolicy` 上抽出已有逻辑，不新开 Handle 类：

```python
cond = policy.encode_obs(obs_dict)                 # → global_cond
eps  = policy.predict_eps(x_t, t, cond)            # → UNet 噪声
out  = policy.predict_action(obs_dict)             # 高层 API，内部走上面两个
# out["action"] 形状由 ckpt 决定：
#   图像  (B, Ta, 31)
#   手 step (B, 1, 22)
#   手 chunk (B, Ta, 22)
```

以后 guidance（本期不做）只碰手的 22 维；EE 只来自图像 DP：

```python
full = image_dp.predict_action(obs_full)                 # (B, Ta, 31)
hand = hand_dp.predict_action({"hand_joint": ...})   # (B, 1, 22) 或 (B, Ta, 22)
# 或 score 混合：只混合 action[..., 9:31] 上的 eps
```

分层可以用图像 DP 的手段当 `condition_data` 做 inpaint，不必给无图网加新 cond 通道。

## 文件

### 新增

- `diffusion_policy/config/task/bulb_hand.yaml`
  - **自己的 `dataset_path`**，指向仿真 zarr
  - obs **只有** `hand_joint` `[22]`（没有 `ee_pose`）
  - action `[22]`
  - dataset 传入 `shape_meta` 和可选的 `normalizer_path`（`${oc.env:BULB_HAND_NORMALIZER,null}`）
  - `bulb_image.yaml` 同样：`shape_meta` + 可选 `normalizer_path`
- `diffusion_policy/config/train_diffusion_unet_bulb_hand_step_workspace.yaml`
- `diffusion_policy/config/train_diffusion_unet_bulb_hand_chunk_workspace.yaml`
- `train_bulb_hand_step.sh` / `train_bulb_hand_chunk.sh`（照 `train_bulb_dino.sh` 的写法）
- `scripts/fit_bulb_sim_normalizer.py`：从仿真 zarr 流式只写出 `hand_joint` 的 min-max

两个 train yaml 都：

- `_target_`: 现有 `TrainDiffusionUnetImageWorkspace`
- `policy._target_`: 现有 `DiffusionUnetImagePolicy`
- `task: bulb_hand`
- `obs_as_global_cond: True`
- `pred_action_steps_only: True`（只去噪要执行的 `Ta` 步）
- `n_obs_steps: 2`
- 噪声日程与图像 DP **对齐**（同样的 DDIM / epsilon / 100 train steps），否则以后无法在同一条噪声日程上混 score
- encoder 无 RGB：`MultiImageObsEncoder` 在没有 rgb 键时允许 `rgb_model: null`
- `freeze_encoder: False`（没有 DINO 可冻）
- UNet 可以更小（默认 `down_dims: [256, 512, 1024]`），因为 `global_cond` 只有 `22 * n_obs_steps`

默认超参：

| | step | chunk |
|--|--|--|
| `horizon` | `2`（`>= n_obs_steps-1+n_action_steps`） | `16`（与图像 DP 同一时间窗） |
| `n_action_steps` | `1` | `8` |
| 扩散 / loss 形状 | `(B, 1, 22)` | `(B, 8, 22)` |

chunk 的 dataset 仍按 `horizon=16` 采样，但 loss 只用 `action[:, To-1 : To-1+Ta]`（`pred_action_steps_only`）。step 的 `horizon=2`，loss 只用最新观测对齐的那一步手动作。

### 修改

- `diffusion_policy/dataset/bulb_image_dataset.py`
  - 接收 `shape_meta`
  - 无 rgb 键：不加载图像；无 `ee_pose` 键：不读 EE、不做相对变换
  - 仿真：按 22 维手读写；真机图像：仍从 31 维 `state`/`action` 拆相对 EE + 绝对手
  - `get_normalizer`：手从 `normalizer_path` `create_manual`；EE 仅在真机图像任务上 `create_fit`；禁止对 31 维 action 一次 fit
- `diffusion_policy/common/bulb_action_normalizer.py`（或同等小模块）：`normalize_action` / `unnormalize_action` 按 22 / 31 维拆手与 EE
- `diffusion_policy/config/task/bulb_image.yaml`：增加可选 `normalizer_path`
- `train_bulb_dino.sh`：若设置了 `BULB_HAND_NORMALIZER` 则传入并校验文件存在；未设则不传，走独立 fit
- `diffusion_policy/model/vision/multi_image_obs_encoder.py`
  - 没有 rgb 键时 `rgb_model` 可为 `None`
  - low_dim 键仍按排序后 concat（与现在一致）
- `diffusion_policy/policy/diffusion_unet_image_policy.py`
  - 增加 `pred_action_steps_only`（语义对齐 `diffusion_transformer_hybrid_image_policy` / `diffusion_unet_lowdim_policy`）
  - 抽出 `encode_obs`、`predict_eps`；`predict_action` / `compute_loss` 对 action 走拆开的 normalize/unnormalize
  - 默认 `pred_action_steps_only=False`，旧图像 ckpt（整段 `action` scaler）推理仍可用
- `inference_dp.py`
  - 从 ckpt 的 `shape_meta` 决定喂哪些 obs 键
  - 从 `action_dim` / `n_action_steps` 决定输出形状
  - 无图 DP 只出 22 维手：执行时把 **最新观测的绝对 EE pose** 与预测手关节拼成 31 维再下发（手 DP 不预测 EE，也不沿用上一拍命令 EE，避免漂移）
  - 机器人仍采集全量观测；无图策略 **只把 `hand_joint` 送进网络**
  - `--check_only` 按 ckpt 构造 dummy obs / 断言 action 形状

### 不改

- `data2dp.py`（真机转换保持原样；仿真如何进 zarr 本期不纳入，只要产物满足上面的键与 31 维约定）
- 相对 EE 数学、UNet 类、workspace 类
- 不上 `GuidedPolicy`、不复制 `inference_dp_lowdim.py`、不引入 `DiffusionUnetLowdimPolicy`

## 训练命令

```bash
# 已有
./train_bulb_dino.sh

# 新增
./train_bulb_hand_step.sh
./train_bulb_hand_chunk.sh
```

对应：

```bash
python train.py --config-name=train_diffusion_unet_bulb_hand_step_workspace
python train.py --config-name=train_diffusion_unet_bulb_hand_chunk_workspace
```

## 推理命令

```bash
python inference_dp.py --checkpoint image.ckpt          # (B, 8, 31)，EE+手 chunk
python inference_dp.py --checkpoint hand_step.ckpt      # (B, 1, 22)，当前手；EE 保持
python inference_dp.py --checkpoint hand_chunk.ckpt     # (B, 8, 22)，手 chunk；EE 保持
```

不要在 chunk ckpt 上用「只执行第 1 步」冒充 step 模型。

## 错误处理

- dataset：`shape_meta.action.shape` 只能是 `[31]` 或 `[22]`；否则直接报错
- dataset：缺少 `state`/`action`，或声明了 rgb 但 zarr 没有对应键 → 报错
- 无图任务：zarr **没有**图像键是合法的；有图像键也忽略，不读入
- 无图 encoder：若 `shape_meta` 里仍有 rgb 却 `rgb_model=None` → 报错
- `pred_action_steps_only=True` 时必须 `obs_as_global_cond=True`（与现有 lowdim 实现一致）
- inference：手 ckpt 的 `action_dim` 必须为 22；图像 ckpt 必须为 31
- dataset：`normalizer_path` **未设** 是合法独立 fit，不报错；**设了** 但文件不存在或 npz 缺 min/max → **报错**，不得静默改回对本数据 fit 手
- 未设路径时 warning 一次：正在对本数据集 fit `hand_joint`，勿与另一域手 DP 混尺子
- 加载旧 ckpt（存在 `normalizer['action']`）时 warning 一次：正在使用旧模型，action 按整段 31 维尺子 unnormalize
- 对 31 维 action 调用未拆开的 `normalizer['action']` 视为错误路径（新训代码禁止）
- 策略输出含 NaN/Inf 时拒绝下发（现有逻辑保留）

## 验证

- 无图 dataset：`obs` 只有 `hand_joint`，`action.shape[-1]==22`，不碰图像、不碰 EE
- 图像 dataset：keys、31 维 action、相对 EE 不变；有 `normalizer_path` 时手来自仿真、EE 真机 fit；无路径时手也在真机上 fit
- 给了有效 `normalizer_path` 时 **不会**对当前数据重新 fit 手
- 未设路径时 `get_normalizer` 成功，且不写 `action` 键
- 手-only 实机/dry-run：下发的 31 维里 EE 等于最新观测 EE，只有手在变
- 手 step policy smoke：`predict_action` → `(1, 1, 22)`
- 手 chunk policy smoke：`predict_action` → `(1, 8, 22)`
- 图像 policy：`pred_action_steps_only` 默认 False，现有 dummy 形状不变
- `inference_dp.py --check_only` 对三类 ckpt 都能过
- 同一份仿真手统计：无图 `action` 的 22 维 scaler 与图像手维 `unnormalize` 数值一致
- 图像 31 维 unnormalize：`[:9]` 走 `ee_pose`，`[9:31]` 走 `hand_joint`，互不套用

## 明确不做

- guidance 采样循环、`GuidedPolicy`、第三个推理入口
- 用一个 chunk 权重在推理时切成单步
- 改图像 DP 的 action 定义或相对 EE 公式
- 给无图网加「图像计划」额外 cond 通道（需要时用 inpaint）
