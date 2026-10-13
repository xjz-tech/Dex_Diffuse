# 1B prior + 10k guide：真机手型初始化的仿真对照

> 历史诊断，已停用此评估口径：本目录使用“相对初始化位置偏移5cm立即结束”，不符合用户于2026-09-18指定的统一 `xjz_test` DP仿真评估规则。不得将本目录的阈值时间作为标准抓持寿命或后续默认基线。按原生协议重测的记录在 `../20260918_xjz_realinit_3seeds/`；项目规则见根目录 `AGENTS.md`。

本目录保存一次独立诊断，不修改训练或真机控制代码。

## 实验范围

- prior：`/home/carus/data_usb/obs_4-66.ckpt`，EMA。
- guide：`runs/sim_hand_10k_seed42/checkpoints/latest.ckpt`，EMA；各组 `model_manifest.json` 保存实际文件哈希。
- guide scale=25、引导前2步、prior/guide各4次DDIM、每次执行2步、30Hz。
- 沿用 `../20260917_1b_initialization_audit/initial_state.npz` 中32–47号环境，4种实测手型/假定灯泡摆放，各4个扩散seed（50–65）。guide另用seed+100000。
- 三种策略各运行16轮，依次新建仿真进程，复用相同环境槽位和逐chunk的prior高斯噪声。
- 不holding、不加入推理等待、无随机外力或域随机化；保留关节机械范围，不做相邻目标限幅。
- 每轮首次物体位置相对初态偏移超过5cm时停止策略控制；最大观察400秒。**这是位置阈值事件，不等于确认掉落，也不是旋拧成功。** 部分向上/侧向位移会触发阈值而物体仍在指间。

## 文件

- `report.md`、`results.json`：三组完整结果、逐seed指标、共同有效前缀的动作差。
- `validation.json`：初始状态、静态接触响应、首批动作、阈值事件重算和关节裁剪检查。
- `ordinary_1b/`、`guided_scale0/`、`guide10k_scale25/`：逐步记录、配置、日志、checkpoint哈希与采样器数值检查。
- `comparison.png` / `.pdf`：阈值曲线与逐seed比较。
- `video_seed65/replay.mp4`：三策略并排的已保存状态回放；达到阈值的画面冻结。不是运行时实时录屏。
- `preview_guide10k_scale25_seed63/`：guide最短样本的状态回放与渲染误差检查。
- `pilot_different_slots/`：不同物理槽位的预检查，已排除，不能与正式结果混合。

## 复现

从仓库根目录依次运行（会覆盖本目录下对应实验数据）：

```bash
bash docs/experiment_reviews/20260918_10k_guide_initialization/code/launch.sh 2
bash docs/experiment_reviews/20260918_10k_guide_initialization/code/launch.sh 0
bash docs/experiment_reviews/20260918_10k_guide_initialization/code/launch.sh 1
/home/carus/miniforge3/envs/dp/bin/python docs/experiment_reviews/20260918_10k_guide_initialization/code/summarize_serial.py docs/experiment_reviews/20260918_10k_guide_initialization
/home/carus/miniforge3/envs/dp/bin/python docs/experiment_reviews/20260918_10k_guide_initialization/code/plot_results.py docs/experiment_reviews/20260918_10k_guide_initialization
```

`guided_scale0` 是必要的采样器控制：生产guided DDIM即使scale=0也会重算裁剪后的epsilon，而普通DDIM路径保留原epsilon。scale25对scale0的比较更接近guide本身的增量。

旧实验的活跃推理batch及其他同时运行的环境不同。初始状态一致不保证闭环轨迹逐位相同；旧统计仅作历史参考，正式结论使用本次控制组。16轮小样本、假定物体摆放、无真实推理延迟的结果不能直接外推为真机抓持时间。
