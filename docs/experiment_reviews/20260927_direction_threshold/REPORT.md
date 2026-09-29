# 方向差异多小能改变转向：原生物理闭环实测

固定物理seed3577，10B EMA、DDIM4、guide9、scale25、exec2；各同噪声条件初始化及共同前8步逐值配对。3个政策噪声seed为48/49/50。每个正式运行最多288个控制步（9.6秒），前8步为共同初始化保持，净转角从第8步末起算；原生failure保留。全部采用实际原生仿真录像。

## 干预定义

每8步从该组当前状态，使用原Astra规则生成16步左/右reference，混合为：

\[r_\lambda(s)=\frac{r_L(s)+r_R(s)}{2}+\lambda\frac{r_L(s)-r_R(s)}{2}.\]

lambda=+1为原左reference，−1为原右reference，0为两者中点。scale始终25；中点组仍使用guidance，绝不是scale0组。各组以自身状态更新，只有共同前8步及首个分叉推理输入完全相同；后续动作差异包括反馈效应。

预先网格为−1、−0.5、−0.25、−0.125、0、0.125、0.25、0.5、1；先做noise48端点/中点验证，再补全3×9。noise48的−0.75及后续四次局部区间细化属于看到网格结果后的探索，不能混称预先注册。方向诊断预设净角>5°左、<−5°右，其余小变化；这不替代原生终止或保证持续方向。

## 实际净转角：不是单调增益曲线

| lambda | noise48 | noise49 | noise50 |
|---:|---:|---:|---:|
| -1 | -14.22° | -6.59° | +0.37° |
| -0.75 | -21.92° | — | — |
| -0.625 | -11.91° | — | — |
| -0.5 | +12.92° | -8.59° | -6.42° |
| -0.25 | +22.61° | +7.01° | +3.67° |
| -0.125 | +21.40° | -5.15° | -2.85° |
| 0 | +15.12° | +2.74° | +20.39° |
| 0.125 | +11.54° | +8.64° | +26.67° |
| 0.25 | +31.50° | +17.20° | +27.40° |
| 0.5 | +14.49° | +13.11° | +24.93° |
| 1 | +94.03° | +32.97° | +71.08°* |

正式记录32条，另有一次独立复跑；原生failure 1条。* noise50/lambda+1于第266步（8.87秒）提前结束，其角度不是9.6秒结果，未并入完整时长响应曲线。所有原始结果均保留。净角与末1秒方向可能不同，例如noise48/lambda−1净右转14.22°，末1秒却左摆6.03°。因此不能把净右转称为单调持续右转。

![响应曲线](direction_response.png)

![实际时间轨迹](rotation_traces.png)

## “微小”用什么量衡量

下表均相对于同noise的lambda0组，只取分叉后的**第一个动作和第一个物理控制步**，因此之前状态完全相同。RMS跨22关节；实际指尖差是5个末端刚体位置差的均值。所有组在后续不断施加相应reference差异，不能将首步差异误说成一次脉冲产生全部末端转角。

| noise / lambda | 末端净角 | 首个命令差RMS/rad | 比例项差RMS/mN·m | 实测总关节力矩差RMS/µN·m | 物体净接触力读数差/N | 实测指尖平均差/mm |
|---|---:|---:|---:|---:|---:|---:|
| 48 / -0.75 | -21.92° | 0.00040219 | 1.1085 | 26.253 | 0.09268 | 0.017599 |
| 49 / -0.125 | -5.15° | 0.00007631 | 0.2311 | 3.340 | 0.01497 | 0.003124 |
| 50 / -0.5 | -6.42° | 0.00025599 | 1.0563 | 35.737 | 0.06689 | 0.010523 |

比例项的数学定义是：

\[\tau_{PD}=K_p(a-q)-K_d\dot q,\qquad \Delta\tau_P=K_p\Delta a\quad\text{（相同 }q,\dot q\text{）}.\]

Kp使用每条原生随机化后实际引擎属性，单位N·m/rad。这个量是显式比例控制项的变化，不是声称PhysX隐式驱动器最终恰好施加该力矩；总关节力矩传感器读数包含所有作用的合成，也不能当作电机单独输出。物体净接触力是接触合力读数，不含接触点力臂，不能当作扭转力矩或单指法向力。

**本实验没有测得一个普适的“最小牛顿力”。** 未指定作用点、方向、持续时间的力没有唯一阈值；合力相同的两个接触分布也可产生相反力矩。0.01497N是noise49/−0.125相对中点的首步净接触力读数差，不是外加0.01497N就必然反向的实验。

## 局部数学响应与闭环效应

在同一第8步末状态，令a(lambda)为完整4次DDIM输出，以±0.125计算中央有限差分：

\[J_a d\approx\frac{a(0.125)-a(-0.125)}{0.25},\quad J_{\theta}J_a d\approx\frac{\theta_9(0.125)-\theta_9(-0.125)}{0.25}.\]

| noise | 命令方向导数RMS/rad每单位lambda | 首个物理步转角方向导数/°每单位lambda |
|---:|---:|---:|
| 48 | 0.0005396 | 0.02799 |
| 49 | 0.0005307 | 0.02584 |
| 50 | 0.0006366 | 0.02868 |

这是实际模型加实际物理的局部有限差分，而不是把关节RMS直接当成物体转角。首步响应小，但重复闭环会改变接触与后续观察。完整系统写成：

\[s_{k+1}=F(s_k,a_k),\quad a_k=G(o(s_k),r_\lambda(s_k),\xi_k),\quad S_{k+1}=A_kS_k+B_kG_{r,k}d_k.\]

其中S为状态对lambda的敏感度，A包含策略与reference的状态反馈，d为reference左右半差。跨接触模式时未必可微，有限差分和完整实跑比假定固定Jacobian更可靠。本次全程净角对lambda非单调，且噪声改变方向边界；不能将长期净角除以首步微小输入，包装成通用机械放大倍数。

## 仪表有效性与边界

1. 正式版只读取原生已有DOF总力矩与net-contact张量。noise48/lambda+1的288步关节、物体位姿、速度、控制目标与旧原生轨迹逐值一致（最大误差0）。各同noise组前8步状态逐值一致；全部27项初态字段与旧记录一致。
2. 首次试加物体约束力传感器导致第1个物理步qpos变化0.00321688rad，尽管初态与命令相同；这一pilot被拒绝，单独保留在pilot_force_sensor_source与n48_lambdap1p00000，未纳入正式表。
3. 不用root angular velocity反算物体轴向力矩：在复現左转轨迹中，它与姿态有限差分的相关系数仅约0.443，其他组也存在明显不一致。没有把不可靠的速度差分写成准确的接触力矩测量。
4. 该物理初态来自原生随机化，质量约212.55g、物体摩擦约2.026。只检验一个物理初态、三个政策噪声；不能推广为不同抓姿/质量/摩擦、真机上的最小扰动力。有限分辨率网格也不能证明更小扰动一定无效。
5. native协议：示范000–149，位置0.05m、指尖0.1m、旋转180°、立即位置0.15m，FailureToleranceScale10000、fixedToleranceSteps20000、resetOnReachGoal=false、跨轨迹0.3；目标更新与误差坐标原生。DDIM4、数值reference guide9，不存在第二个guide去噪网络；exec2、30Hz。没有将native failure改名物理掉落。

## 全部视频与原始数据

[汇总CSV](summary.csv) · [逐条完整数值](analysis.json) · [原生仪表化diff](native_instrumentation.diff) · [预设协议](protocol.json) · [全部终帧](final_frames.jpg)
- noise48 / lambda-1: [实际录像](runs/native_n48_lambdam1p00000/rollout.mp4)
- noise48 / lambda-0.75: [实际录像](runs/native_n48_lambdam0p75000/rollout.mp4)
- noise48 / lambda-0.625: [实际录像](runs/native_n48_lambdam0p62500/rollout.mp4)
- noise48 / lambda-0.5625: [实际录像](runs/native_n48_lambdam0p56250/rollout.mp4)
- noise48 / lambda-0.546875: [实际录像](runs/native_n48_lambdam0p54688/rollout.mp4)
- noise48 / lambda-0.53125: [实际录像](runs/native_n48_lambdam0p53125/rollout.mp4)
- noise48 / lambda-0.5: [实际录像](runs/native_n48_lambdam0p50000/rollout.mp4)
- noise48 / lambda-0.25: [实际录像](runs/native_n48_lambdam0p25000/rollout.mp4)
- noise48 / lambda-0.125: [实际录像](runs/native_n48_lambdam0p12500/rollout.mp4)
- noise48 / lambda+0: [实际录像](runs/native_n48_lambdap0p00000/rollout.mp4)
- noise48 / lambda+0.125: [实际录像](runs/native_n48_lambdap0p12500/rollout.mp4)
- noise48 / lambda+0.25: [实际录像](runs/native_n48_lambdap0p25000/rollout.mp4)
- noise48 / lambda+0.5: [实际录像](runs/native_n48_lambdap0p50000/rollout.mp4)
- noise48 / lambda+1: [实际录像](runs/native_n48_lambdap1p00000/rollout.mp4)
- noise49 / lambda-1: [实际录像](runs/native_n49_lambdam1p00000/rollout.mp4)
- noise49 / lambda-0.5: [实际录像](runs/native_n49_lambdam0p50000/rollout.mp4)
- noise49 / lambda-0.25: [实际录像](runs/native_n49_lambdam0p25000/rollout.mp4)
- noise49 / lambda-0.125: [实际录像](runs/native_n49_lambdam0p12500/rollout.mp4)
- noise49 / lambda+0: [实际录像](runs/native_n49_lambdap0p00000/rollout.mp4)
- noise49 / lambda+0.125: [实际录像](runs/native_n49_lambdap0p12500/rollout.mp4)
- noise49 / lambda+0.25: [实际录像](runs/native_n49_lambdap0p25000/rollout.mp4)
- noise49 / lambda+0.5: [实际录像](runs/native_n49_lambdap0p50000/rollout.mp4)
- noise49 / lambda+1: [实际录像](runs/native_n49_lambdap1p00000/rollout.mp4)
- noise50 / lambda-1: [实际录像](runs/native_n50_lambdam1p00000/rollout.mp4)
- noise50 / lambda-0.5: [实际录像](runs/native_n50_lambdam0p50000/rollout.mp4)
- noise50 / lambda-0.25: [实际录像](runs/native_n50_lambdam0p25000/rollout.mp4)
- noise50 / lambda-0.125: [实际录像](runs/native_n50_lambdam0p12500/rollout.mp4)
- noise50 / lambda+0: [实际录像](runs/native_n50_lambdap0p00000/rollout.mp4)
- noise50 / lambda+0.125: [实际录像](runs/native_n50_lambdap0p12500/rollout.mp4)
- noise50 / lambda+0.25: [实际录像](runs/native_n50_lambdap0p25000/rollout.mp4)
- noise50 / lambda+0.5: [实际录像](runs/native_n50_lambdap0p50000/rollout.mp4)
- noise50 / lambda+1: [实际录像](runs/native_n50_lambdap1p00000/rollout.mp4)

## 探索性局部区间细化

noise48的局部有限时净角符号区间为[-0.5625, -0.546875]，端点分别−0.032°和+2.994°，两者都在预设±5°小变化区间，不能称为有效左右转边界。lambda−0.625得到−11.907°，−0.5625得到−0.032°；这也只是一个局部的有效右转/小变化区间。响应非单调，这不是全局最小阈值，也不是跨噪声鲁棒门槛。见[细化记录](local_refinement.json)。

## 两端均超过5°的反向配对

这些配对两端都运行完整288步，且无native failure；首步输入相同，后续持续采用各自reference。RMS只衡量首个动作，不是整个动作序列的最大差。

| noise / 两个lambda | 两个净转角/° | 首命令差RMS/rad | 首命令差RMS/° | 比例项差RMS/mN·m | 首步总关节力矩差RMS/µN·m | 首步净接触力读数差/N | 首步指尖平均差/mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 49 / [0.125, -0.125] | +8.637, -5.150 | 0.00013267 | 0.007601 | 0.475762 | 7.431747 | 0.024556 | 0.006067 |
| 48 / [-0.5, -0.625] | +12.923, -11.907 | 0.00006489 | 0.003718 | 0.154616 | 2.428648 | 0.009224 | 0.001989 |
| 50 / [0, -0.5] | +20.390, -6.423 | 0.00025599 | 0.014667 | 1.056277 | 35.736773 | 0.066889 | 0.010523 |

**不能把首步微小差异推广到全程。** 两条反馈轨迹分开后，280个控制步的下发命令差RMS，noise48为0.119765rad（6.86°），noise49为0.087101rad（4.99°），noise50为0.219657rad（12.59°）。因此实验支持“微小的初始命令差及持续reference干预，能让闭环走向不同转向”，不支持“全程只需微小力矩”或“一次微扰足够”。这些数值包含反馈造成的状态差异，不能归因于同状态下的guidance直接增益。

noise49/lambda−0.125完整独立复跑288步，状态、总关节力矩和净接触力与首次运行逐值一致，最大误差0，末端净角仍为−5.1495°。见[复跑校验](repeat_validation.json)。这证明该仿真配置的可复现性，不证明真机具有微米精度。

[两组反向配对的原生仿真录像](paired_direction_changes.mp4) · [配对原始数值](direction_flip_pairs.json)
