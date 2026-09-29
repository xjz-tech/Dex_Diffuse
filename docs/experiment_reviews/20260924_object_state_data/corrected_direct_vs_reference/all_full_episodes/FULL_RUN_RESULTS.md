# Object_state_data：80 条各自横抓起点至末帧的完整仿真

> **复审更正（2026-09-26）：本报告的数值只能描述这套仿真设置，不能用作真实横抓翻转/放置成功率，也不能据此比较 prior 的任务能力。** 旧录像遗漏了 2 秒静置过程，近景看不到已落在桌面的灯泡；37 条回退起点未保证离桌稳定抓握，真实桌面支撑关系未被导入，末段固定 wrist 且无灯座承接。已有初始化稳定性检查仅 5/80 条通过。详见 [过程复审](../../visibility_audit_20260926/PROCESS_AUDIT.md)。下表保留原始描述性数字，撤回其成功率含义。

2026-09-26。范围为 `episode_0`–`episode_79`。每条从自己选中的横抓帧导入 22 个手关节和灯泡相对 wrist 的 4×4 位姿，按该条自己的 reference 运行至最后一个动作，再保持 60 个控制步。80 条手关节起点均不同，80 条灯泡相对 wrist 的初始位置也均不同。43 条满足估计离桌面至少 1 cm 的优先条件；37 条是仍满足横向姿态条件的回退起点。所有起点值见 `horizontal_start_states.csv`，帧选择和筛选诊断见 `selection.json`。

每条分别运行原始动作直接仿真、10B prior `guide2/exec2`、10B prior `guide2/exec1`。两种 guidance 均使用 scale 50、每两个原始 reference action 中插入一个中点、prior DDIM 4、guide DDIM 4、固定 noise seed 44。灯泡质量 170 g，摩擦系数 2.2，仿真 wrist 固定。每组先静置 60 步；即使任务的原生 failure 信号出现，也继续执行该条全部 reference 和末端保持段。原始动作与插值 guidance 的物理时长不同，表中末端指各自执行完全部动作之后。

原生 failure 口径沿用 `eval/xjz_test.sh`：物体相对 wrist 的位置误差阈值 0.05 m、指尖 0.1 m、物体旋转 180°、立即无效位置误差 0.15 m、`FailureToleranceScale=10000`、`fixedToleranceSteps=20000`、`resetOnReachGoal=false`、跨轨迹目标概率 0.3、示范范围 000–149。failure 是环境评估信号，不等同于独立确认的物理掉落。

| 方法 | 全部动作及末态保持执行 | 首次原生 failure 在末帧前 | 失效前连续 15 步轴角 ≤45°（47 条静置后仍横向的子集） | 动作末端轴角 ≤45° | 动作末端灯泡相对 wrist 位置距真机末态 ≤10 cm | 末端相对位置误差中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始动作直接仿真 | 80/80 | 80/80 | 33/47 | 7/80 | 0/80 | 33.3 cm |
| 10B guide2/exec2 | 80/80 | 80/80 | 30/47 | 5/80 | 0/80 | 34.4 cm |
| 10B guide2/exec1 | 80/80 | 80/80 | 33/47 | 5/80 | 0/80 | 33.4 cm |

真机 reference 的末端灯泡轴在 79/80 条中距任一竖直方向 ≤45°。仿真中有些灯泡曾转到竖直附近，或末端轴角看起来也接近竖直，但对应末端位置离真机的 bulb-in-wrist 位姿至少 13.6 cm，常已不在正面画面。按“末端轴角 ≤45° 且相对位置误差 ≤10 cm”的描述性计数，三种方法都是 0/80；由于起点及完整尾段场景不等价，这个计数不具备任务成功率含义。23/80 条在任何参考动作前的 60 步静置阶段便触发原生 failure。初始导入后的第一个控制步，灯泡相对 wrist 的位置变化中位数为 1.41 cm，最大为 12.49 cm。真机 wrist 从选中帧到末帧的移动中位数为 17.67 cm，而这次仿真 wrist 固定；这也是末端真机与仿真不可直接视为同一个完整臂手任务的重要限制。

审计验证了 240/240 条都使用本 episode 的 reference 并运行到预定末端，且同一 episode 的三组方法具有完全相同的 27 项初始状态字段和 60 步静置轨迹。新全程轨迹与先前首次 failure 即停止的轨迹在停止点之前逐步完全一致。四条带录像的 episode 0、2、28、51 共 12 组录像也与对应无录像全程轨迹逐步完全一致。

四段不同 episode 的四宫格合成视频位于 `/home/carus/Downloads/Object_state_data_four_distinct_episodes_full.mp4`，顺序为 0、2、28、51；每段左上为原始动作仿真，右上为真机，左下为 guide2/exec2，右下为 guide2/exec1。总长 4418 帧、30 fps、147.27 秒。详细结果在 `full_per_episode_results.csv`、`full_aggregate_results.json`；录像审计在 `selected_full_video_trace_audit.json`。旧的 `single_aggregate_results.json` 属于首次 failure 即停止的实验，不能再用于判断“是否执行到放手末帧”。
