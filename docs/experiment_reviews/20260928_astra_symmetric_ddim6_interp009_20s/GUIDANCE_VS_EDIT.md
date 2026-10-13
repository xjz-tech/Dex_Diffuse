# 新对称循环：Direct / Edit / Guidance 同屏对比

|方法|左转脱手区间|右转脱手区间|左转握持时最大净转角|右转握持时最大净转角|
|---|---|---|---:|---:|
|Direct|5.17–5.37秒|2.27–2.47秒|27.15°|14.46°|
|Edit noise 0.15 / DDIM4|14.57–14.77秒|14.00–14.40秒|36.35°|23.94°|
|Guidance scale 25 / DDIM4|20秒截止仍握住|20秒截止仍握住|6.03°|14.73°|

使用同一套新对称循环reference，初态27字段逐值相同，170g、物体摩擦2.2、物理seed42；模型为同一10B EMA checkpoint。Edit为noise0.15（实际0.153397）、DDIM4、exec2。Guidance使用原Astra关节MSE梯度引导、scale25、DDIM4、9步未来reference、exec2；模型从Gaussian prior起步，在4次采样中施加reference梯度。Guidance与Edit都固定noise seed44，每窗口复用噪声。没有附加guidance模型、残差clamp或动态scale。

这是同一reference上的新补跑Guidance，不是拼接旧物理条件或旧左右reference的视频。旧的右转Guidance源实验用scale100和每窗口新噪声；本轮的scale选择和固定噪声配置在此显式记录。Edit具有已执行target的历史mask，Guidance保留既有算法；两者的加噪起点/反向采样时间点不同属于方法差异，并未声称相同扩散时刻。

六组均完整执行600实际步，约20秒；脱手后继续动作、不重置。原生failure阈值与目标更新协议保持：demo000–149，位置0.05m、指尖0.1m、旋转180°、立即位置0.15m、FailureToleranceScale10000、fixedToleranceSteps20000、traj_steps_limit12000、resetOnReachGoal=false、跨轨迹0.3。native failure与录像脱手分开记录；抑制重置并继续至20秒来自用户明确要求。

插值规则仍为reference相邻任一关节差>0.09rad插一个中点；当前共同reference最大差0.0375rad，因此实际新增0点。每个run的300个9步预测窗口与固定reference逐值相同，全部共1800窗口通过核验；Direct预测等于reference，Edit历史mask误差0，Guidance记录的MSE统计有限且正scale分支启用。首实际控制步灯泡位移<1.5mm。

视频上排左转、下排右转，列顺序Direct / Edit / Guidance。600帧逐控制步同步、约30fps、20.00004秒、H.264、1倍速，无终帧冻结或补时。握持角度截止到各自最后确认仍握住的帧，后续下落旋转不计入。此为单一物理seed/噪声seed比较，不能外推普遍方法排序。

[六组同屏视频](direct_edit015_guidance25_ddim4_1x.mp4) · [握持转角曲线](direct_edit015_guidance25_ddim4_curves.png)
