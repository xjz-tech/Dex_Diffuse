# 同速度对照：原始 reference、插值 direct、scale 50 guided

四条均为先通过原始 reference 稳定翻转验收的起点：episode 76/f114、34/f79、54/f110、2/f103。所有组使用各自相同的初态、170 g、摩擦 2.2、固定 wrist、30 Hz、原生 failure 协议，跑完整 reference 尾段并额外保持 60 控制步。本次新增四条「插 1 个线性中点，直接执行 reference」完整回放，未录新视频。已有的原速 direct 与 scale 50 guided 记录未重跑。

下表为首次连续 3 控制步满足几何/接触分离判据时的**原始 reference 动作进度**（从 1 起；`.5` 是插值中点）。越大表示更晚脱手。该分析指标独立于原生环境 failure。

| episode | 原速 direct | 插值 direct | 插值+guide2/exec1 | 插值+guide2/exec2 |
|---:|---:|---:|---:|---:|
| 76 | 159 | 165 | 178.5 | 69.5 |
| 34 | 185 | 180.5 | 183.5 | 184 |
| 54 | 121 | 36 | 55.5 | 127.5 |
| 2 | 195 | 61 | 188.5 | 216 |
| **均值** | **165.0** | **110.6** | **151.5** | **149.3** |

稳定翻转通过数：原速 direct **4/4**；插值 direct **3/4**（episode 2 未通过）；guided exec1 **4/4**；guided exec2 **4/4**。此处「稳定翻转」要求首次分离前物体距竖直不超过 30°、在空中连续至少 30 控制步，并满足两处以上手部接触及物体接触力判据。

**结论：插值造成的执行速度变化是原先原速 direct 与 guided 比较中的重要混杂因素；也有证据表明 prior 的动作修改在同速度条件下能改善这四条中的部分案例。** 插值 direct 相对原速 direct 在 episode 54/2 提前 85/134 个原始动作进度分离；scale 50 guided exec1/exec2 相对插值 direct 分别在 4/4、3/4 条上更晚分离，平均进度分别增加约 40.9/38.6。尤其 episode 2，插值 direct 到 61，guided exec1/exec2 到 188.5/216。但 guided exec2 在 episode 76 从插值 direct 的 165 提前到 69.5，说明收益仍不稳定。与原速 direct 相比，guided exec1/exec2 分别只在 1/4、2/4 条上延长保持。

插值方案是 `a1, (a1+a2)/2, a2, ...`，仍以 30 Hz 执行，所以每个原始 reference 动作段约花两倍物理时间。原速 direct 的动作数为 239/441/462/492；插值 direct 和两种 guided 为 477/881/923/983。对照中所有组均完成各自完整尾段，没有在首次分离或原生 failure 时提前停止。

四个新增回放的初态文件全部字段与各自原速 direct 完全一致；60 步静置记录除了物体接触力浮点归约误差外逐字段相同，该力差小于 `1e-5 N`。插值 direct 的全部实际发出命令与线性插值 reference 数值一致；原生 failure 参数、物理属性和尾段长度均已核验。将分离阈值在 3/5/8/10 mm 间改变，新组脱手进度最多变化 0.5 个原始动作。

这些结果只描述四条经过原速 direct 成功条件筛选的轨迹、固定环境 seed 与 prior 噪声。没有新增录像独立确认四条插值 direct；分离结果由已有几何/接触分析产生。物体网格和手模型外参未独立标定；固定 wrist 的翻转测试不能推广成完整真机任务成功率。因轨迹尾段长度不同，均值只用于这四条成对比较。

可复现分析脚本：`../../analyze_direct_interp1.py`；逐条数据：`RESULTS.json` 和 `qualified_comparison/episode_XX/direct_interp1/`。已有 scale 50 guided 和原速 direct 的首次分离记录分别见各 case 的 `retention.json`。
