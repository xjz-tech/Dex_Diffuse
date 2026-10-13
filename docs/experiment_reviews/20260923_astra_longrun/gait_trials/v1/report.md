# 小幅抬指右转尝试：v1

用户要求在右转reference中尝试换指、减小抬指幅度。本批单列于此前无主动换指对照；所有组最多900个控制步（约30秒），原生failure另行记录，脱手经实录审阅确认。

实际卸载幅度与reference需要分开核查。

动作阶段：右推24步 → 单指卸载8步 → 侧向回位8步 → 重新闭合16步，食指、中指、无名指依次尝试。换位期间其余手指的参考固定在本次换位前的支撑目标，拇指不抬起。卸载时PIP/DIP小幅伸展，Jacobian一阶估计的指垫位移不超过1mm；侧摆回位不超过0.06rad。1mm不是实测离物间隙，也不保证完全卸载。每8步读取自身状态，右推重新计算，16步reference、guide9/exec2，无尾部补齐。

物理配置沿用正式比较前冻结的seed50/19/25：分别约95.508g/摩擦2.572、95.012g/1.718、140.003g/3.258。独立Prior噪声101先跑5秒，排除5秒内原生失败或质量(g)/摩擦>100的候选。此筛选不保证正式噪声都能维持30秒；正式早期失败仍保留。每个物理配置的Prior和各scale使用配对噪声0、1，direct每配置仅一次，并在两份对比视频中复用，不算两次独立试验。

本版从原关节目标卸载；实际力审阅表明许多周期仍持续压住物体，不能把它称为完全抬离。

控制器是当前Astra编写的分阶段数值反馈规则，没有每窗口调用语言模型。接触力变化可包含自碰撞，不能单独证明物体接触或稳定力闭合。

| 环境/噪声 | 方法 | 原生failure/s | 脱手区间或截止/s | 截止净右转/° | 截止最大净右转/° |
|---|---|---:|---|---:|---:|
| 19/0 | [astra_direct](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/runs/astra_direct_e19_n0/comparison_clip.mp4) | 6.90 | 6.70–6.90 | 31.24 | 31.24 |
| 19/0 | [prior](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/runs/prior_e19_n0/comparison_clip.mp4) | — | ≥30，观察截止 | -681.32 | 0.00 |
| 19/0 | [guided38](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/runs/guided38_e19_n0/comparison_clip.mp4) | 9.80 | 9.60–9.80 | -41.58 | 15.89 |
| 19/0 | [guided50](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/runs/guided50_e19_n0/comparison_clip.mp4) | 8.87 | 8.67–8.87 | -62.05 | 18.42 |
| 19/1 | [prior](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/runs/prior_e19_n1/comparison_clip.mp4) | — | ≥30，观察截止 | -1031.77 | 174.13 |
| 19/1 | [guided38](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/runs/guided38_e19_n1/comparison_clip.mp4) | 7.77 | 7.57–7.77 | -7.25 | 7.99 |
| 19/1 | [guided50](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/runs/guided50_e19_n1/comparison_clip.mp4) | 9.03 | 8.83–9.03 | -52.79 | 10.67 |
| 25/0 | [astra_direct](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/runs/astra_direct_e25_n0/comparison_clip.mp4) | 24.87 | 24.67–24.87 | 19.75 | 29.03 |
| 50/0 | [astra_direct](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/runs/astra_direct_e50_n0/comparison_clip.mp4) | 10.50 | 10.37–10.43 | 57.10 | 57.10 |
| 50/0 | [prior](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/runs/prior_e50_n0/comparison_clip.mp4) | — | ≥30，观察截止 | -864.88 | 11.92 |
| 50/0 | [guided25](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/runs/guided25_e50_n0/comparison_clip.mp4) | 27.13 | 26.93–27.13 | 6.04 | 20.67 |
| 50/0 | [guided38](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/runs/guided38_e50_n0/comparison_clip.mp4) | 17.40 | 17.30–17.40 | 0.49 | 12.14 |
| 50/0 | [guided50](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/runs/guided50_e50_n0/comparison_clip.mp4) | — | ≥30，观察截止 | -6.83 | 10.01 |
| 50/1 | [prior](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/runs/prior_e50_n1/comparison_clip.mp4) | — | ≥30，观察截止 | 90.78 | 555.52 |
| 50/1 | [guided25](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/runs/guided25_e50_n1/comparison_clip.mp4) | — | ≥30，观察截止 | 32.58 | 34.02 |
| 50/1 | [guided38](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/runs/guided38_e50_n1/comparison_clip.mp4) | 24.07 | 23.87–24.07 | -0.93 | 15.62 |
| 50/1 | [guided50](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/runs/guided50_e50_n1/comparison_clip.mp4) | 24.67 | 24.47–24.67 | 11.85 | 24.35 |

角度在脱手样本中截到末个确认仍接触的帧，存在失稳滚动，不能称为稳定转动成功。30秒截止样本使用观察终点，不把未观察到的掉落时刻当30秒。

## 同步实录

- [comparison_env50_noise0](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/comparison_env50_noise0.mp4)
- [comparison_env50_noise1](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260923_astra_longrun/gait_trials/v1/comparison_env50_noise1.mp4)

## 与旧版的比较限制

旧direct参考每轮右扫8°、IK增量上限0.12rad；本版同时改为4°、0.06rad并加入换指阶段。不能把所有变化单独归因于换指。此次属于探索性尝试，不以筛选后的少数样本声称普遍提升。原生物理参数保持不变，初态27字段与预筛逐元素核对。所有失败保留。

前期v1探索中的CUDA初始化错误和560步pending文件读取竞争保留在v1/attempts；读取竞争已修复，本版使用修复后的调度器。软件或资源中断不算掉落。原v1批次因实际卸载不足而停止扩展，已完成的探索结果保留。

10B EMA checkpoint；Prior DDIM4，guidance使用Astra数值关节reference，没有独立学习型guide网络。三个guidance scale仅25、38、50。此前scale5/100不纳入本版。
