# Astra / 真机 reference：引导长度与实际作用量

2026-09-27。核对已有闭环实验，新增同观察、同噪声的离线采样；本次没有推进物理仿真，没有新闭环成功率或新实跑视频。所有讨论限定为现有 normalized joint MSE guidance，不适用于 FK loss。

**后续数学解释修正：** 用户指出左右reference均为小幅动作，不能由Prior偏左推出左guidance与prior score同向。新增36个同状态左右reference反事实与score核验支持这一质疑：左右幅度相近，存在共同拉回分量；score与左右guidance平均余弦均接近0。[详细核验及修正解释](LEFT_RIGHT_CORRECTION.md)。本文“运动倾向与目标一致”只指闭环行为层面，不能解读为局部梯度同向。

## 结论先限定

不能证明“Astra需要guide9，真机需要guide2”。已有反例：Astra原小幅reference、scale25/guide2/exec2，环境3577、噪声48，首10秒左转3.129rad（约179.28°）且仍持物，但保持阶段继续漂移1.676rad，24.33秒原生失败；不是完整任务成功。Astra scale50配对中，guide2在2.33秒失败，guide9在10.17秒失败；只能说明该配置下后者保持较久。guide2用8步计划、guide9用16步计划，均8步重规划，伴随变化必须记录。

- [Astra scale25/guide2原始reference对照](../../../.worktrees/Astra-controller/docs/astra_controller/20260926_scale25_g2e2_min005_guard/README.md)
- [Astra scale50 guide2与guide9](../../../.worktrees/Astra-controller/docs/astra_controller/20260926_scale50_guide2/README.md)

真机episode53修正初态、插1中点、scale50/exec2时，guide2/4/9的动作结束距竖直误差为24.63/59.13/73.35°，guide2在这组指标最好；对应命令RMSE为0.08787/0.07205/0.09381rad，guide4跟踪反而最好。没有插值、scale50时，guide9/exec2比guide2/exec2更接近竖直；插1、scale25时guide9也优于guide2（48.87° vs 64.47°）。不能把局部最佳扩展到所有真机回放。

- [插值、scale50、guide2/4/9对照](../20260924_object_state_data/corrected_direct_vs_reference/interpolation_1_to_5/insert1_guide4_exec2_scale50/COMPARISON.md)
- [不插值对照](../20260924_object_state_data/corrected_direct_vs_reference/nointerp_scale50/README.md)

## 1. 精确的直接引导系数

令H为guide actions数量，D=22，s为scale，z为归一化的干净动作预测，r为归一化reference：

\[
L_H=\frac{1}{22H}\sum_{k=1}^{H}\sum_{j=1}^{22}(z_{kj}-r_{kj})^2,
\quad s\nabla_z L_H=\frac{2s}{22H}(z-r).
\]

这里的残差是prior动作预测−reference，不是reference相邻两步的差。小幅、平滑reference不必然意味着guidance梯度小。

| 配置 | s/H | 2s/(22H) |
|---|---:|---:|
| scale50 / guide9 | 5.55556 | 0.50505 |
| scale25 / guide2 | 12.5 | 1.13636 |
| scale50 / guide2 | 25 | 2.27273 |

在相同中间状态、噪声级和同一关节残差下，前两步每个分量的直接更新系数比是(50/9)/(25/2)=4/9。第3–9步前者有直接引导，后者为0。系数匹配：guide9/50 ↔ guide2/11.111；guide2/25 ↔ guide9/112.5；这不是最终输出或物理效果相等的保证。

代码冻结epsilon，不经过UNet求guidance梯度：z=(x_t−sqrt(1−α_t)epsilon)/sqrt(α_t)。所以精确地有：

\[
\Delta\hat{x}_0=-\frac{2s}{22H}\frac{1-\alpha_t}{\alpha_t}(z-r),
\]

相对于同一次输入的base x0；base可clip，误差仍用raw z。下一噪声状态相对于base更新为：

\[
\Delta x_{t'}=-K_t(z-r),\quad
K_t=\frac{2s}{22H}\left[\sqrt{\alpha_{t'}}\frac{1-\alpha_t}{\alpha_t}
-\sqrt{1-\alpha_{t'}}\sqrt{\frac{1-\alpha_t}{\alpha_t}}\right].
\]

本checkpoint DDIM4实际时间点75/50/25/0；cosine schedule，α分别0.133495/0.478265/0.835621/0.999369。

| DDIM t | K_t：guide9/50 | K_t：guide2/25 |
|---:|---:|---:|
| 75 | 1.337703 | 3.009833 |
| 50 | 0.289772 | 0.651986 |
| 25 | 0.093692 | 0.210806 |
| 0 | 0.000319 | 0.000718 |

早期更新可能外推超过当前残差，不能解释成把最终动作“拉回某个百分比”的凸组合；后续还要重新过UNet。公式与实际step张量更新最大误差2.98e−8。最后一步日志mse_before已经包含前三次guidance作用，不是无引导Prior误差。

源码：[guided_ddim.py](../../../diffusion_policy/guidance/guided_ddim.py)。工作目录与Astra worktree该文件SHA256一致：7970aacb13ad89ba48716972a7048883ec48e4cdfcf4e6616682f8fa6c77c77f。

## 2. 新复算：最终每个动作实际改变多少

取Astra guide9/50历史轨迹no_gait_e3577_n48前70步的全部35个窗口。每个窗口使用原4×66观察和原噪声，对6个(H,s)组合完整执行4次DDIM。guide9/50历史动作逐值重现，最大误差0。这里scale0仅作离线数学基准，没有新增scale0仿真。

下表排除8步初始化，剩31个左转窗口（62个执行动作）。改变量定义为同观察、同噪声下最终命令相对于全程无guide命令的RMSE，跨窗口及22关节，单位rad。

| 配置 | 第1个动作改变量 | 第2个动作改变量 | 两动作合并改变量 | 输出/reference RMSE |
|---|---:|---:|---:|---:|
| 无guide | 0 | 0 | 0 | 0.04042 |
| guide9/50 | 0.01110 | 0.02481 | 0.01922 | 0.03213 |
| guide2/25 | 0.00555 | 0.00964 | 0.00786 | 0.03521 |
| guide2/50 | 0.01156 | 0.02110 | 0.01701 | 0.03007 |
| guide2/11.111 | 0.00223 | 0.00405 | 0.00327 | 0.03816 |
| guide9/112.5 | 0.01379 | 0.03097 | 0.02397 | 0.02800 |

guide9/50的直接系数虽然仅为guide2/25的4/9，最终前两个输出的改变量分别约为2.00倍、2.57倍，合并约2.44倍。这是具体条件下的测量，不是通用倍数。动作改变大也不等于改善大；reference RMSE和物体表现需另评。

原因：本次梯度本身没有跨时间的UNet反传，但引导第3–9步会改变整个噪声动作序列，下一次UNet预测通过时序卷积影响第1–2步。guide9覆盖全部9个可执行未来动作（模型horizon12，前3个索引对齐观察历史），guide2留下7个未来动作不直接约束。将系数匹配为guide9/50对guide2/11.111后，输出仍不同，全部35窗口差异RMSE为0.01698rad，直接证实了未来引导对前缀输出的影响。

这些观察取自guide9/50轨迹。换成guide2闭环后状态会改变；表格不能预测其存活时间或成功率。未统计末端执行器真实关节位移或接触力。

## 3. 平滑reference为何可能适合长窗口

对同一批Astra左转9步窗口，二阶差分RMS为3.55e−8rad，22维路径直线度均值约1；基本是线性小步推进。第1到第9目标距离的逐关节RMS为0.04196rad。长窗口能表达持续推进趋势，前2步更容易与暂时保持混淆；对线性r_k=r_0+k v，9点首尾位移8v，2点只有v。

这里不是把相同动作复制9遍就自动增强9倍：mean loss会归一化。若所有位置误差相同，每分量强度仍按s/H变化，长窗口收益来自覆盖更多时间点及后续模型的时序响应。

| reference片段 | 9步平均路径直线度 | 二阶差分RMS/rad | 第1至9目标RMS距离/rad |
|---|---:|---:|---:|
| Astra左转31窗口 | 1.0000 | 0.000000036 | 0.04196 |
| ep53原75步，34窗口 | 0.6200 | 0.02875 | 0.13743 |
| ep53插1中点，71窗口 | 0.8575 | 0.00928 | 0.10745 |

路径直线度=首尾22维欧氏距离/分段路径长度；窗口可重叠，不视为独立统计样本。两者任务和初态不同，这些仅描述本批reference形状，不是平滑度的因果试验。

Astra全局也非完全平滑：既有guide9前70步，计划内部相邻目标RMS均值0.00464rad，8步重规划边界0.05036rad（约10.9倍）。真机ep53原始相邻RMS均值0.03069rad，插值后0.01534rad；插值后的全局RMS 0.01762rad与Astra含边界0.01875rad相近，不能只说“Astra所有时刻变化更小”。

真机短窗可能有利的机制：只紧跟马上执行的reference，后面接触阶段仍允许Prior调整；9步为约0.30秒的动作预算，2步约0.067秒（首尾间隔分别8/30与1/30秒）。若未来reference需要特定支撑/松指/回位，而仿真实际接触相位偏离真机，长窗可能把尚未合适的未来动作也约束进去。同scale下guide2同时将近端直接系数增强4.5倍。这两种变化混在现有对照里，尚不能单独确定哪个主导。

## 4. Prior偏左为何让右转更难

已有10B无外力96环境首轮前30秒（保留更早失败）：净左转>30°为85，净右转>30°为8，小净转动3。运动帧左右57.24%/42.76%，累计左/右角度比1.863。短暂右摆并不等于稳定持续右转。此统计关闭外力，不能直接当所有Astra物理条件的精确概率。

另一固定初态32噪声序列中，首1秒21/32净右转>10°，10秒终点只有3/32；强右转guide25/100在8个成对噪声中均得到净右转，但平均只有19.71°/21.53°，未达180°。所以更准确的是“可以改变方向，难以维持大幅右转与稳定接触”，不是右转完全采不到。

数学上joint MSE不包含物体转角、转速或接触模式。对一个有助于换指的动作修改δ：

\[
L(a+\delta)-L(a)=\frac{2(a-r)^T\delta+\|\delta\|^2}{22H}.
\]

当a已接近r，任何偏离该平滑reference的换指修改都增加损失，即使它有利于右转。左转时Prior自身持续转动倾向与目标一致，容许偏离reference可能仍获得目标方向；右转时弱guide可能被偏左动力学轨迹覆盖，强guide又可能压缩完成换指所需的自由度。现有参考缺少可靠的持续换指循环，且部分右转批次为固定动作重放、状态分岔后时机可能失配。这些机制与结果一致，但尚无单独隔离试验能把失败全部归因于训练/先验偏置或换指。

Prior运动不均衡是闭环行为统计，不是训练集标签比例，也不是单状态动作密度比；不能把85/8直接代入某个posterior或宣称需10.6倍scale。

- [多环境Prior方向统计](../20260924_prior10b_free_rotation/report.md)
- [同初态噪声与成对右转guidance](../../../.worktrees/Astra-controller/docs/astra_controller/20260923_noise_direction/README.md)

## 文件与复现

`analyze.py`产生`results.json`、`paired_commands.npz`；`reference_geometry.py`产生`reference_geometry.json`。运行环境`/home/carus/miniforge3/envs/dp/bin/python`，设置`PYTHONDONTWRITEBYTECODE=1`。源观察和reference路径记录在结果中。没有修改控制器、默认参数、参考动作或评估终止口径。

后续若做窗口长度因果试验，应保留同16步规划、同8步重规划、同exec2，至少比较(9,50)/(2,50)/(2,11.111)，或(2,25)/(9,25)/(9,112.5)，分离窗口和直接系数；这些更强scale只用于实验设计，本文没有证明其安全稳定或最优。
