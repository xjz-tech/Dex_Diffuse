# Astra 固定左右转 reference × reference editing：首轮结果

2026-09-28。完成10次真实仿真：左右转各 direct、noise=0 校验、noise ratio 0.1/0.2/0.35。单一物理seed42、单一编辑噪声seed44；不代表跨seed成功率。

**这轮编辑延缓了失稳，但没有解决持续转动。** 左转各编辑组比direct更晚原生failure，最终仍脱手，最大净左转约31–46°。右转三档都运行至10秒且录像末帧仍在指间，但最大右转约44–50°后有明显回退，10秒净右转仅18–26°，没有达到半圈。

## 原生结果与录像复核

|方向|方法|实际步数|原生failure / 观察截止|可见接触截止时净指定转角|此前最大净指定转角|录像结论|
|---|---|---:|---|---:|---:|---|
|左|direct|50|1.67s failure|9.24°（step42）|21.45°|step42仍有接触，step50已分离|
|左|noise0.1|96|3.20s failure|3.13°（step88）|41.85°|step88仍有接触，step96已分离|
|左|noise0.2|142|4.73s failure|14.26°（step134）|30.77°|step134仍有接触，step142已分离|
|左|noise0.35|218|7.27s failure|18.66°（step210）|45.80°|step210仍有接触，step218已分离|
|右|direct|290|9.67s failure|46.43°（step282）|56.14°|step282仍有接触，step290已分离|
|右|noise0.1|300|10.00s观察截止，无failure|18.87°（step300）|50.39°|截止时仍在指间|
|右|noise0.2|300|10.00s观察截止，无failure|17.60°（step300）|44.31°|截止时仍在指间|
|右|noise0.35|300|10.00s观察截止，无failure|25.66°（step300）|46.64°|截止时仍在指间|

指定方向在表中均为正；原始trajectory保留左正右负。脱手组的转角保守截止到表中已复核接触帧，不把终帧下落旋转计入握持内结果。可见接触不证明稳定力闭合，所列帧不是精确脱手时刻。保守分离区间：左direct 1.40–1.67s、0.1为2.93–3.20s、0.2为4.47–4.73s、0.35为7.00–7.27s；右direct 9.40–9.67s。这些录像分析没有改变原生终止逻辑。

在接触截止帧，各组均有手指净接触力记录支持，具体值与审阅帧见 [visual_review.json](visual_review.json)。这些净力未按接触对分解，不单独据此认定手物接触。右转达到10秒只是观察截止，不能推断之后仍保持。

## 视频和曲线

- [左转四组同步实际录像](left_comparison.mp4)
- [右转四组同步实际录像](right_comparison.mp4)
- [角度与倾斜曲线](curves.png)
- [左转末段帧](left_terminal_review.jpg)、[右转过程及终帧](right_progress_review.jpg)

视频排列：左上direct、右上noise0.1、左下noise0.2、右下noise0.35。逐控制步同步、30fps；提前结束的面板标为ENDED/FROZEN并冻结，不能解释为继续保持。曲线画出完整原生轨迹，最后数步可能含滑移/下落旋转；握持内分析以上表的保守截止为准。视频分别218/300帧，全部解码通过。每个原始run保留frames与rollout.mp4，frame0为未同步reset图形，比较视频从第一实际控制步开始。

## 方法与配置

左转复用 `.worktrees/Astra-controller/outputs/astra_left_sequence/v2_s5_pause100`，右转复用 `astra_halfturn/seed42_10b_right_gait_s100_v2`。左转源任务含三次转动和暂停，但本轮统一只观察前300步，不能称完整三段任务评估。两份存档原来依据各自历史运行的反馈编写；本轮固定重播原数值reference，没有重新让Astra按新状态规划，也没有新增换指或插值。

编辑方法沿用最近episode54的ReferenceActionEditor：将3步实际已执行target history和9步未来reference归一化后加噪，用10B EMA prior按本组实时观测去噪；过去3步在各扩散时刻以对应噪声强制约束。DDIM更新4次、eta0、exec2；没有gradient guidance。每窗口复用噪声seed44的同一张量，与近期编辑实验一致。

|请求噪声比|实际噪声比 sqrt((1-alpha)/alpha)|4次时间步|左/右实际执行前缀平均编辑RMSE（rad）|
|---:|---:|---|---|
|0.1|0.105623|5,3,2,0|0.05543 / 0.03947|
|0.2|0.201754|11,7,4,0|0.07129 / 0.06402|
|0.35|0.353170|20,13,7,0|0.10079 / 0.10271|

噪声比不是关节弧度标准差。模型 `/home/carus/data_usb/10B_obs_4-66.ckpt`，EMA、epoch50/globalstep91600。原生随机物理：质量262.262g、物体摩擦1.06、物体scale0.987454，手各连杆摩擦保持原随机值。**不是**episode54的44g/摩擦1.1，也不是后续Astra筛选seed50的物理条件。

原生协议保持demo000–149、位置0.05m、指尖0.1m、旋转180°、立即位置0.15m、FailureToleranceScale10000、fixedToleranceSteps20000、traj_steps_limit12000、resetOnReachGoal=false、跨轨迹目标概率0.3。原生物体相对手腕目标误差、目标更新与容忍逻辑未改；失败终止。唯一观察时限为300实际步，30Hz。详见 [PLAN.md](PLAN.md)。

## 验证与可复现性

- 左右noise=0与本轮direct的完整逐步物理轨迹分别50/290步逐值一致（包括关节、物体、接触记录、实际target、failure）。验证通过后才运行非零编辑。
- 所有10组27项初态与各自源存档逐元素一致；所有预测窗口9步reference与同名源预测逐元素一致；direct和noise=0实际输出精确等于reference前2步。
- 所有窗口reference padding为0；编辑history mask误差为0。源码冻结、reference哈希和模型元数据均保存。
- [run.py](run.py) 可跳过已完成记录并复现批次；[analyze.py](analyze.py) 重建分析与视频。`source/`是独立仿真代码快照，只新增编辑分支和方法录像标签；原工作树及其他实验未修改。
- 合成视频时发现系统PATH无ffmpeg，分析脚本已改为使用现有imageio_ffmpeg二进制；未影响任何物理仿真结果。两段最终视频逐帧完整解码，见 [video_validation.json](video_validation.json)。

本轮没有加入旧gradient-guidance对照或多个噪声seed，因此不能声称优于旧guidance，更不能凭这一噪声seed给三档作普遍排序。能支持的结论是：在这两份既有reference及同一初态上，reference编辑确实改变了执行并延缓失稳；指定方向的持续累积仍未解决。
