# 张手对照补一组 reference edit：噪声 0.15、DDIM4

2026-09-28。在 2026-09-22 同一份预先固定的全手渐开参考上，新增一组 reference initialized DDIM，其余四组不重跑。

同一 seed42、同一初态与物理随机化、同一 16 步提案时间表。10B EMA `/home/carus/data_usb/10B_obs_4-66.ckpt`。编辑请求噪声比 0.15，实际噪声比 0.153397，4 次时间步 8→5→3→0，eta 0，执行 2 步，未来 reference 9 步，过去 3 步已知动作在各扩散时刻约束。梯度 guidance scale 为 0。噪声 seed44，每个窗口复用该 seed 的噪声张量。旧 guidance 臂当时用的是 fresh noise，这一项和编辑方法同时不同，不能单独归因。

原生协议与原四组相同：demo000–149，位置 0.05 m，指尖 0.1 m，旋转 180°，立即位置 0.15 m，FailureToleranceScale 10000，fixedToleranceSteps 20000，traj_steps_limit 12000，resetOnReachGoal=false，跨轨迹 0.3。观察上限仍为 240 步。原生 failure 是失败代理。

| 执行方式 | 原生失败步 / 时间 |
|---|---|
| 直接执行 | 19 / 0.633 s |
| guidance scale25，guide 9 | 100 / 3.333 s |
| guidance scale50，guide 9 | 68 / 2.267 s |
| Prior-only scale0 | 240 步观察截止，未失败 |
| edit 0.15 DDIM4 | **58 / 1.933 s** |
| guidance scale50，guide 4，exec 2，DDIM 4 | **39 / 1.300 s** |

scale 50 的 guide 4 是 2026-09-28 补跑。同一张手时间表、同一初态、fresh noise；逐步 reference 与原来的 guide 9 scale 50 在共同的 39 步上一致。第 35 步灯泡仍在指间，第 37 步已离开手，第 39 步落在桌面。原来的 scale 50 是 guide 9，失败在第 68 步。两次 scale 50 的 prior 噪声不是同一次抽样。

edit 终帧灯泡已落在桌面。第 54 步画面仍在指间，第 56 步已与手分开。这是相机审阅区间，不是新的终止规则，也不把第 58 步称作精确脱手时刻。单一物理 seed、单一噪声 seed。

执行前缀相对 reference 的平均 RMSE 为 0.0777 rad。29 个窗口的 history mask 误差为 0。初态与 direct 逐字段一致，每步 reference 与冻结计划一致。

六组视频：`.worktrees/Astra-controller/outputs/astra_open_hand/comparison/open_hand_with_edit015_and_scale50_guide4.mp4`。上排 direct、scale25 guide9、scale50 guide9；下排 prior-only、edit 0.15、scale50 guide4。1 倍仿真时间，失败后面板冻结。之前的五组视频仍保留。
