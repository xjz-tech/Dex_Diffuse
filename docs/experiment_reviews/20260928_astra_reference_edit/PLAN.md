# Astra 固定 reference 的加噪去噪编辑试跑

用户于 2026-09-28 要求复用之前 Astra 的 reference actions，先把近期使用的 reference editing 方法用于左右转，并确认先用已有固定 reference。

先验配置：左转 `astra_left_sequence/v2_s5_pause100`，右转 `astra_halfturn/seed42_10b_right_gait_s100_v2`；动作来自各自既有存档，不重写动作、不根据本轮结果挑选 reference、不按新状态重新规划。左转源包含预定暂停；本轮只运行其前 300 控制步，不能称完成完整三次左转任务。reference 原先经过人工调试，属于选定实例。

两个方向各测 direct、noise=0 验证、noise ratio 0.1/0.2/0.35。全部物理 seed42；编辑噪声 seed44，每个窗口复用该 seed 的噪声张量，沿用近期 ReferenceActionEditor 行为。10B EMA `/home/carus/data_usb/10B_obs_4-66.ckpt`，DDIM 4 次更新，执行2步，9步 future reference，加3步实际已执行 target history。归一化空间噪声比是 sqrt((1-alpha)/alpha)，不是关节弧度标准差；离散实际比例和 timesteps 写入 model.json。无 gradient guidance，无额外插值，无物理参数覆盖。每个窗口的观测来自本组实际状态。

保留原 Astra 原生初始化、随机化、30 Hz、demo000–149和失败终止逻辑。物体位置0.05m、指尖0.1m、旋转180°、立即位置0.15m、FailureToleranceScale10000、fixedToleranceSteps20000、traj_steps_limit12000、resetOnReachGoal=false、跨轨迹0.3。观察上限300步（约10秒），原生failure提前结束；failure不等于独立确认物理脱手。左右转均以请求方向为正报告，原始记录仍左正右负。

运行前先比较 direct 与 noise=0 的完整逐步物理轨迹；通过后才运行非零编辑。所有组27项初态与各自源存档逐元素核验，每份预测9步 reference 与源文件逐元素核验。真实相机逐控制步录制，包括终止帧；初始 reset frame0 图形尚未同步，第一实际控制帧另行检查，不以 frame0 证明接触正确。视频对齐物理步，提前终止的面板明确冻结并注明结束。

本轮是一个物理初态、一个噪声 seed 的探索，不报告总体成功率，也不将保持、指定转向、转角达到目标混为同一指标。局限：固定 reference 在状态分岔后可能不适配当前接触，试验不等价于闭环 Astra 规划。

源码冻结在 `source/`，基于20260923已有仿真快照，仅接入 reference editor 和录像方法标签。原工作树、原存档与同时运行的其它实验不修改。原始 reference 文件哈希保存在 provenance.json。
