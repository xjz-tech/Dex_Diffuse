# Sim-Hand Guided Inference — Open Checks

执行 `2026-08-26-sim-hand-guided-inference` plan 时由 controller 拍板的 ruling。
每条都是**待核验风险**，不是已关闭的设计决策。勾选前请按「如何核验」跑一遍。

关联：

- Spec: `docs/superpowers/specs/2026-08-26-sim-hand-guided-inference-design.md`
- Plan: `docs/superpowers/plans/2026-08-26-sim-hand-guided-inference.md`
- Branch: `segmentwise-closedloop-guidance`

---

## Checklist

- [ ] **就地功能分支、未建 worktree**
  - Ruling: 在 `segmentwise-closedloop-guidance` 上直接实现，未隔离 worktree。
  - 风险: 污染当前 checkout / 与其它并行改动混在一起。
  - 如何核验: `git status`；确认无关改动未误入本分支；需要隔离时再建 worktree 迁出。

- [ ] **梯度只用闭式公式（不做 Spec 13.4 的 autograd reference）**
  - Ruling: `mse_guidance_gradient` 用
    `2/(E*A)*(base_prev_mean-ref)`，不与 autograd 对照。
  - 风险: 漏掉与 autograd 不一致的浮点边角。
  - 如何核验: 抽一组 `(base_prev_mean, reference, slice)`，令
    `base_prev_mean.requires_grad_()` 后对 MSE 求 `autograd.grad`，与闭式结果比对。

- [ ] **FakeExecutor 是 command-is-state**
  - Ruling: `post_states = segment[..., 9:31]`，命令即反馈状态。
  - 风险: dry-run 的闭环 history ≠ 真机手部状态；接真机后 history 语义会变。
  - 如何核验: 真机或仿真 executor 上对比「执行后实际 hand joint」与「刚发出的 hand command」；若不一致，改 `SegmentExecutor` 实现并重跑 coordinator 测试。

- [ ] **初始 history 用 seeded synthetic，不是 Real `hand_joint`**
  - Ruling: check/dry-run 的 `(1, n_obs_steps, 22)` history 由 seeded `torch.randn` 生成；`FakeSegmentExecutor.reset()` 持有该窗口。
  - 风险: 与 Real 观测里真实手部位姿脱节；条件分布与上线不一致。
  - 如何核验: 接真机前，用 Real obs 的 `hand_joint` 窗口替换 synthetic history，对比 `guide_segment` 输出差异；确认上线路径是否应改用真实 history。

- [ ] **Plan 里 bash 的 `+` 是软换行，不是参数**
  - Ruling: 实现时剥掉 plan 命令中的单独 `+`。
  - 风险: 照抄 plan 会得到坏命令；后人再改 plan 时可能又引入 `+`。
  - 如何核验: 打开 plan，搜索单独一行的 `+`；若仍有，改成正常续行或去掉。

- [ ] **测试名须展开成真实断言（plan 里不少只列了名字）**
  - Ruling: implementer 按 Interfaces + Global Constraints 自己补全断言。
  - 风险: 个别用例偏弱、只测「能 import / 不抛错」。
  - 如何核验: 抽查 `tests/test_sim_hand_guided_*.py` 与 `tests/test_inference_sim_hand_guided_cli.py`，确认每个 brief 命名用例都有 shape/数值/拒绝路径断言，而非空 `pass`。

- [ ] **冻结检查锚点 `2b40ee6` 可能过期**
  - Ruling: Task 9 仍用 `git diff --name-only 2b40ee6..HEAD -- <冻结文件列表>`；若锚点漂移，应改为直接对命名冻结列表做 diff。
  - 风险: 假绿（以为旧系统未动，实际锚点已不合适）。
  - 如何核验: 确认 `2b40ee6` 仍是「Sim-Hand training 落地、guidance 之前」的合理基线；否则改成 `git diff --name-only -- inference_dp.py train.py diffusion_policy/common/sim_hand_temporal_util.py ...`（相对工作树/上游）并更新 plan。

- [ ] **Step 7 真 checkpoint gate 未跑**
  - Ruling: `REAL_CKPT_PATH` / `SIM_CKPT_PATH` 未设时跳过；单元与 fake dry-run 视为可合并。
  - 风险: 真 ckpt 的 proposal shape、Sim temporal、DDIM timesteps 与测试假值漂移，要到第一次真机无关 check 才发现。
  - 如何核验:
    ```bash
    export REAL_CKPT_PATH=...   # Real bulb checkpoint
    export SIM_CKPT_PATH=...    # Sim-Hand checkpoint
    GUIDED_TEST_PYTHON=/tmp/dex-diffuse-guided-test/bin/python   # 或等价环境
    $GUIDED_TEST_PYTHON inference_sim_hand_guided.py \
      --mode check \
      --real-checkpoint "$REAL_CKPT_PATH" \
      --sim-checkpoint "$SIM_CKPT_PATH" \
      --device cuda:0 \
      --execution-steps 5 \
      --guidance-scale 0.0 \
      --num-inference-steps 12 \
      --eta 0.0 \
      --seed 7
    $GUIDED_TEST_PYTHON inference_sim_hand_guided.py \
      --mode dry-run \
      --real-checkpoint "$REAL_CKPT_PATH" \
      --sim-checkpoint "$SIM_CKPT_PATH" \
      --device cuda:0 \
      --execution-steps 5 \
      --guidance-scale 1.0 \
      --num-inference-steps 12 \
      --eta 0.0 \
      --seed 7
    ```
    期望当前 target：Real `(1,50,31)`、full hand reference 至少 54 步、
    10 段、Sim `4/12/9`、guidance slice `(3,12)`、execution slice
    `(3,8)`、12 个动态 timesteps、zero-oracle 误差在容差内。

---

## 已关闭（终审后）

- [x] CUDA 上 CPU `Generator` + CUDA `torch.randn` 崩溃 — 已在 `27bbb3a` 修复，并加 CUDA smoke。
