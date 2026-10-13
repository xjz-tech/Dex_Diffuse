# bulb2：50/150条示范抽查

结论：本次50条样本全部呈现手内操作，没有发现完整的“手与物体分离→桌面接近→抓取→抬起”过程。但并不都是单一原地旋转：有换指重定位、倾转翻转、厘米级手内平移，以及停顿后继续调整。

范围：均匀抽样000、003、006……147，共50条（1/3）。读取原始30Hz H5与对应重定向PKL；对所有帧计算运动统计，目视检查每条5个关键帧（250帧），对003、066、084、090再检查6个时刻。不是逐条看完50个视频，也没有将几何回放称为物理仿真。结论不等于已经排除其他100条里的例外。

## 具体案例

| 示范 | 时间 / 秒（原始30Hz） | 观察 |
|---|---|---|
| 003 | 8–12 | 9秒食指距离物体网格约1.0cm；10秒食指约6.4cm而中指约1.3cm；11秒食指约1.2cm、中指约7.0cm；12秒两指都回到约1cm范围。支持食指/中指轮流抬起重定位。 |
| 066 | 0–15 | 有明显倾转和翻转，物体几何中心相对起点最大位移约8.26cm；6秒时中指、无名指、小指伸离，拇指/食指仍近物体；9秒多指又靠近。 |
| 084 | 0–14.43 | 有分段姿态调整和手内位移，中心最大位移约9.17cm。13.63秒食指/中指距网格约10.6/8.5cm；14.43秒回到约1.1/1.4cm。 |
| 090 | 13–15 | 食指明显抬开后回位：13秒约1.1cm，13.53秒约10.7cm，15秒约1.3cm。 |

这里的距离来自原始指尖点与物体网格顶点的最近距离近似，不是碰撞检测或真实接触测量。“换指”指可见的手指位置变化，“换抓成功”仍需接触和动力学验证。

![重点案例](examples.png)

## 50条的数值支持

- 物体几何中心相对初始位置的最大位移：1.87–9.17cm，中位3.63cm；14/50条超过5cm。因此不能简单称为固定位置纯旋转。
- 相对起始姿态的最大旋转角：118.75–180度；这是SO(3)最短角度，最多180度，不能用曲线下降判断旋转反向，也不能用它统计转了几圈。
- 初始时最近指尖距物体网格约0.16–1.17cm；在所有样本所有帧中，“最近指尖距离”的最大值约1.54cm。与“至少某根手指始终靠近物体”的画面一致，没有发现清晰的完全脱手拿取阶段。该指标不能单独证明接触。
- 50条PKL的opt_wrist_pos和opt_wrist_rot全部为0。评测加载器将手腕设为统一参考，当前展示的是手腕相对的手内动作；原H5另存wrist_pose，不能用PKL手腕为0反推采集现场手臂完全没动。
- 本次没有世界坐标桌面高度、接触标注或原始RGB来证实“物体由桌面支撑”。所以措辞是“未发现桌面拿取”，而不是证明原始采集从未拿取。

## ‘切换’有两个层次

1. 示范内部：上面的换指、倾转、平移是真实记录中的变化，可以从单条数据看到。
2. RL/采集中的跨示范subgoal切换：属于任务代码运行时生成的目标选择。enableCrossTrajectoryReset=true，达到目标后以crossTrajectoryGoalProb=0.3触发重新抽示范编号；可能抽回原编号，所以不是每次触发都换成不同示范。该分支更新目标示范映射，没有在此处把实际手和物体重置到新示范姿态。RL需要从当下真实状态接向新目标。

单独回放一条H5/PKL不会出现“003→084”的代码生成切换。是否成功衔接、耗时多久、期间是否掉落，需要查看RL实际rollout的target_data_index_id/target_frame及状态动作记录，或记录新的运行轨迹。本次核对了切换逻辑，没有把它冒充已目视确认的成功衔接实例。

当前控制还会在示范两端改变目标推进方向，构成前后往返的subgoal调度；这也是运行时逻辑，不一定等同于原始示范中的动作反向。

## 卡顿处理

已终止此前启动的8个view_demonstration.py播放器。停止前多个进程分别占用约4–8个CPU核心；停止后短时CPU排队压力降至0，内存可用约27GiB。浏览器工具检查当前任务无剩余网页标签。

后续播放器入口已限制OMP/MKL/OpenBLAS/NumExpr及PyTorch线程数为1，降低多窗口线程争用。未重启这些播放器，以免重新干扰输入。两组GPU评测仍保持运行，采样时4090利用率98%，故剩余延迟可能包含GPU争用或应用渲染因素；未证明唯一根因，也未声称输入问题完全解决。

## 原始检查材料

- [50条关键帧，第1页](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/bulb2_sample50/sheet_1.png)
- [50条关键帧，第2页](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/bulb2_sample50/sheet_2.png)
- [50条关键帧，第3页](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/bulb2_sample50/sheet_3.png)
- [50条关键帧，第4页](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/bulb2_sample50/sheet_4.png)
- [50条关键帧，第5页](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/bulb2_sample50/sheet_5.png)

[逐条指标](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/bulb2_sample50/metrics.json)。

生成时间：2026-09-12T22:40:13+08:00。
