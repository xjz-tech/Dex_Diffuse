# 训练数据疑问：数据记录

日期：2026-09-18。对象：后续模型。本文只列用户原话和对应测量值、出处。未新跑训练或评估。

口径：

- 单关节 `|Δ|`：每个关节、每一步 `|a[t,j]−a[t−1,j]|`，22 关节全部摊开。
- `d[t]`：每步先取 `max_j |a[t,j]−a[t−1,j]|`，再沿时间统计。
- action = 绝对关节目标，rad。排除 episode / 初始化到首动作边界。超 0.09 用 1e-6 容差。

---

## 用户疑问 1

> 训练数据中基本都在 0.09 rads 以内

单关节 `|Δ|`，全量源抽样 1,046,886 步（seed42，2048×512 窗口，排除边界）：

| 指标 | 数值 |
|---|---:|
| 均值 | 0.029354 |
| 中位数 | 0.012406 |
| P90 | 0.090000 |
| P95 | 0.090000 |
| P99 | 0.090000 |
| max | 0.090000 |
| `\|Δ\| > 0.09` 关节更新比例 | 0.000% |
| `\|Δ\| > 0.09` 步比例（任关节） | 0.000% |
| 区间 `(0.06, 0.09]` | 23.703% |

出处：`docs/experiment_reviews/20260916_action_step_audit/statistics_1m.json`  
数据：`/home/carus/Data/exp_data/exp_data_mmap_obs4_h12`

10k 训练集 9979 有效步，单关节同样 max = 0.090000、超过 0.09 为 0。  
出处：`docs/experiment_reviews/20260916_action_step_audit/statistics.json`  
数据：`data/sim_hand_10k_seed42/replay_buffer.zarr`

同一批数据改用 `d[t]`（与推理报告相同）：

| | 10k（9979 步） | 全量抽样（130878 步） |
|---|---:|---:|
| `d` 均值 | 0.079683 | 0.080328 |
| `d` 中位数 | 0.090000 | 0.090000 |
| `d` max | 0.090000 | 0.090000 |
| `d > 0.09 + 1e-6` | 0 | 0 |
| `d` 落在 `(0.089, 0.0900001]` 的步 | 77.79% | 78.40% |

出处：2026-09-18 对上述两个文件复算。10k 公式：`d = abs(action − obs[:,22:44]).max(axis=1)`，且 `obs[1:,22:44] == action[:-1]` 已逐元素核对。

---

## 用户疑问 2

> 仿真推理是从训练数据选一帧出发对吧？

原生入口 `_reset_default`：`random_state_init` 时  
`seq_idx = floor(seq_len * 0.9 * rand)`，手和物体取该示范该帧。

出处：`maniptrans_envs/lib/envs/tasks/sindexhandmanip_sh.py` 约 2364–2411 行。

S3 实测初态（seed42，4 环境，22 关节与示范帧误差 = 0）：

| env | 示范 | 帧 |
|---|---|---:|
| 0 | v3:bulb2@000 | 1065 |
| 1 | v3:bulb2@001 | 139 |
| 2 | v3:bulb2@002 | 605 |
| 3 | v3:bulb2@003 | 563 |

出处：`.worktrees/sim-real-holding-comparison/reports/original_entry_init_check_20260918/ddim4_exec2/resets.jsonl`

S2 原生组 16 条演示编号：`docs/experiment_reviews/20260917_1b_initialization_audit/results.json` 字段 `native_demo_ids`。

---

## 用户疑问 3

> 为什么就变成了均值就会到 0.09 rads 了呢

从示范帧出发的仿真，指标 `d[t]`，无步长夹子，`obs_4-66.ckpt`，DDIM4，exec2。

S3：4 环境 × 300 步，1196 次相邻切换。

| | 均值 | 中位数 | P95 | max | `d>0.09` |
|---|---:|---:|---:|---:|---:|
| 全部 | 0.086507 | 0.093611 | 0.105464 | 0.137087 | 68.81% |
| env0 000/1065 | 0.084676 | 0.093311 | 0.105465 | 0.120321 | 66.56% |
| env1 001/139 | 0.083387 | 0.092687 | 0.105502 | 0.114292 | 60.87% |
| env2 002/605 | 0.088419 | 0.094090 | 0.106244 | 0.137087 | 69.57% |
| env3 003/563 | 0.089545 | 0.094477 | 0.104837 | 0.129550 | 78.26% |

S3 单关节均值（与训练单关节均值 0.029354 同一口径）：0.034504。

出处：`.worktrees/sim-real-holding-comparison/reports/original_entry_init_check_20260918/ddim4_exec2/statistics.json`  
配置：同目录 `README.md`。`step_clamp` 未启用。勿引用该目录上级误用 DDPM8/exec5 的那次。

S2：16 轮，前 20 秒，`d[t]`，`step_clamp: null`。

| 组 | `d` 均值 | `d` 中位数 | `d` P95 | `d` max | 步超 0.09 |
|---|---:|---:|---:|---:|---:|
| 原生姿态／有灯泡 | 0.081722 | 0.091270 | 0.106871 | 0.136112 | 54.83% |
| 原生姿态／空手 | 0.101101 | 0.102948 | 0.122013 | 0.168883 | 87.21% |

出处：`docs/experiment_reviews/20260917_1b_initialization_audit/results.json`、`report.md`

S2 前 100 步 / 后 500 步 `d` 均值：

| 组 | 前 100 步 | 后 500 步 |
|---|---:|---:|
| 原生有灯泡 | 0.085505 | 0.079434 |
| 原生空手 | 0.099869 | 0.101345 |

出处：`docs/experiment_reviews/20260917_1b_initialization_audit/early_late.json`

同口径并排：

| | 训练 10k `d` | 训练全量抽样 `d` | S3 示范帧 `d` |
|---|---:|---:|---:|
| 均值 | 0.079683 | 0.080328 | 0.086507 |
| 中位数 | 0.090000 | 0.090000 | 0.093611 |
| max | 0.090000 | 0.090000 | 0.137087 |
| `d>0.09` | 0% | 0% | 68.81% |

---

## 用户疑问 4

> 我的意思就是我们难道不是按照采集的数据去训练的吗？

10k 训练启动覆盖：

```
task.dataset_path=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/data/sim_hand_10k_seed42
```

出处：`runs/sim_hand_10k_fullnorm_seed42/.hydra/overrides.yaml`

数据集定义：观测 `[qpos(22), target_before(22), residual(22)]`，动作绝对 `target_after(22)`。  
出处：`diffusion_policy/dataset/sim_hand_lowdim_dataset.py`

对照用的 1B 权重：`obs_4-66.ckpt`（`/home/carus/data_usb/obs_4-66.ckpt`），SHA256 `ad0bf60d9fe743161916c55fee36a1b6b13840757baaca2136dcfc3b5b597302`。S3 / S2 均加载该 ckpt。

训练数据相邻目标差审计对象就是上述 RL 仿真采集轨迹，不是评估 rollout。  
出处：`docs/experiment_reviews/20260916_action_step_audit/report.md` 第 3 行。
