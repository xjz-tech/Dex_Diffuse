# 170g、物体摩擦2.2：固定Astra reference编辑至物理脱手

2026-09-28用户要求在上一轮左右转对照基础上，改灯泡质量为170g、物体滑动摩擦为2.2，跑到所有灯泡实际掉落。仅这两项物理参数主动改变；手摩擦保持与原seed42相同的随机值。先前自然随机物理为262.262g/物体摩擦1.06，旧时长不与本轮混算。

方向与reference来源沿用上一轮：左 `astra_left_sequence/v2_s5_pause100`，右 `astra_halfturn/seed42_10b_right_gait_s100_v2`；direct、noise0验证、reference editing 0.1/0.2/0.35。物理seed42、编辑噪声seed44、10B EMA、DDIM4、exec2、9步future窗口、无插值和gradient guidance；其它原生评估阈值保持。每个run从同一原生reset态出发，先设置属性、simulate/fetch/refresh、恢复完整root/dof/目标状态，然后核验第一实际控制步。27字段中仅物体mass、inertia、friction及其缓存mass允许与存档不同。

native failure持续记录；如果录像确认仍握住，就抑制自动reset并继续原生任务的目标更新与误差计算。每次native首次出现或达到定期审阅点暂停，依据实录帧确定是否物理脱手；停止前保存最后确认接触和最早确认分离的帧。native failure不称为脱手。单次执行上限10000步是防止无界运行的安全界，不作为计划终点。若任何组到界仍在握，则必须检查并续跑/说明未完成，不能标成掉落。

源左转reference存档含至step424起的16步计划；右转至step456起。用完所有原始response后显式保持最后一个存档关节目标，记录开始step。这个尾段只检验握持至掉落，不代表Astra继续规划转动。所有组遵循相同规则；每份历史响应仍逐步原样重放。本轮条件性结论限这两份手工选出的固定reference、单初态和单噪声seed。

实现审计时发现第一版尾段在剩8步原始动作时提前使用计划终点，已停止并归档 `left_edit035_invalid_tail`，未将其纳入结果。修正后先执行完全部存档动作，再保持终点；左转预计从step440开始终点保持，右转从step472开始。所有越过存档尾段的正式组均用修正版重跑。
