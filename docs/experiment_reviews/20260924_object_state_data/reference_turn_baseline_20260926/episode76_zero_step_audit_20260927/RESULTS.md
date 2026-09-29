# Episode76：零物理步中指接触与 RL 坐标审计

使用源 state114 的原始 22 关节角、已保存的导入 wrist/object root、实际加载的手 URDF 和灯泡网格，离线计算正向运动学并渲染；零步图没有调用任何物理推进。中指标成青色，仅修改显示颜色，未改几何或位姿。拼图为 `zero_step_vs_first_step_vs_real.png`，左列零物理步，中列首个实际控制步的记录状态重建，右列原始 front114。两个仿真列使用与原录像相同相机，真机相机未做外参标定。

## 零步结果

| 阶段 | 中指指腹与灯泡碰撞网格的分离下界 |
|---|---:|
| 零物理步 | 13.573 mm |
| 第一个控制步后 | 11.202 mm |
| 第60控制步后 | 11.534 mm |

通过对每个中指碰撞链接与灯泡网格构建凸包、寻找候选分离轴，再将全部原始网格顶点投影到该轴，得到严格正的投影区间间隙。完整三角面都处在顶点投影范围内，因此正间隙证明几何表面分离。零步中指所有五个碰撞链接的分离下界均为正，最小为 MCP_VL 的12.609mm，指腹为13.573mm。数字是原始碰撞网格的分离证据，不是 PhysX 接触力；不将离线几何测试当作零步物理受力。详见 `geometry_audit.json`。

因此，中指在物理推进前就没有接触灯泡；首个控制步使指腹更靠近灯泡，而不是把原来接触的中指移开。静置和 target 设定不能解释这项已经存在的零步接触构型偏差。

正向运动学独立验证：用同一实现重建第1、60控制步全部33个可见手链接，与 Isaac Gym 保存的刚体位姿比较，最大位置误差分别为1.331e-7m、1.109e-7m，最大旋转误差均小于0.000064°。这验证了 URDF 根、关节顺序、固定关节变换及FK计算与本次仿真一致。零步图不依赖此前可能含过期刚体变换的导入预览。

## 按用户要求核查 RL rollout

实际任务是本项目私有环境副本的 `SinDexHandManipRHEnv`（`SingleDexHandRH`），不是直接导入 controller 目录下的环境。已核查以下路径：

- `maniptrans_envs/lib/envs/tasks/sindexhandmanip_sh.py:calculate_relative_pose` 使用 `T_wrist^-1 @ T_object`，位置为 `R_wrist^-1 (p_object-p_wrist)`。当前状态和目标奖励都使用这个手根相对坐标。
- `_update_states` 的 wrist 来自 actor 的 `base_state`；奖励中的目标 wrist 来自示范 `opt_wrist_pos/opt_wrist_rot`，不是直接使用 palm 链接原点。
- reset 中保留物体相对 wrist 的关系再应用随机 wrist 朝向；本次导入重新用各 episode 自己的 object-in-wrist 位姿和实际仿真 wrist 组成物体 world pose。
- 实际手资产是 `v3right_sharpa_wave-forhammer5.urdf`。其 `world→right_hand_C_MC` 固定变换包含平移 `[-.01,-.03,-.02]m` 和绕 Z 的−90°；这个变换已经在仿真、离线FK和现有真机 RL FK 的同一URDF里，不能再无依据地额外补一次。
- 当前真机读取代码的 `POLICY_SHARPA_DOF_NAMES` 与本次仿真22关节顺序一致；arm9 的旋转按前两行解码，也与保存的 reference wrist 矩阵一致。

直接提取并运行实际 RL 源码中的 `calculate_relative_pose`，对源数据与导入 root 进行数值比较：物体相对位置误差 **3.22e-8m**，旋转误差 **1.52e-6°**。未发现本次导入相对于 RL 约定漏乘变换、使用了不同的 wrist 根，或关节顺序不一致。细项见 `rl_coordinate_audit.json`。

当前灯泡 actor scale 为原生随机值0.987454。即使恢复到名义scale1，任意灯泡表面点最多移动0.859mm，无法独自消除13.573mm的指腹间隙。本机 `bulb2.obj` 与 `bulb1_col.obj` 虽然文件名不同，顶点和面数组完全相同；不能据名称差别认定视觉与碰撞几何错配。

## 结论与边界

用户关于检查RL相对坐标的建议有助于排除实现层面的猜测：当前导入与RL坐标约定一致，不应再把“可能漏了TCP固定变换”作为已发现的错误。与此同时，RL示范来自NOKOV-v3中配对的重定向 `opt_wrist`、`opt_dof_pos` 与 `obj_trajectory`；这些量的内部一致性，不等于独立验证了 Object_state_data 的真实编码器、TCP、物体追踪和图像在物理世界中的几何对应关系。

已经确认的是当前源数据按现有模型解释时，中指零步即离开灯泡。尚未独立确定真实图中接触位置差异来自传感器与图像时序、真实手与URDF几何/零位、物体追踪坐标对应、或真实外覆层形状。需要进一步对这些来源做可测量的标定/同步核验，不通过随意移灯泡或修改关节角来制造接触。之前的成对策略结果仍只适用于其具体仿真初态，不能作为准确复现真机逐指接触的证据。

本次没有修改 RL 环境、物理参数、reference 或策略结果，也未向 Downloads 写入文件。入口：`../../audit_episode76_zero_step.py` 和 `../../audit_episode76_rl_coordinates.py`。
