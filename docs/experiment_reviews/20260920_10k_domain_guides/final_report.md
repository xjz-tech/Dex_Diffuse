# 固定1B、五种10k数据分布：完成报告

已完成5个新guide的训练，以及7种方法×48个配对案例，共336次闭环评估。1B权重固定。

本轮新guide中，1B + 四域混合的整体截断均值最高，为129.90秒。混合组在4/4个测试物理点的均值高于普通1B；四个单域guide在240g的两个测试点均低于1B。此结果不支持简单地认为训练质量或摩擦范围越窄、越对应测试域，就一定更好。

旧10k在低摩擦两个测试点明显领先混合组；在高摩擦两个点，旧10k与混合组的截断均值差均不足0.5秒，不能据此区分优劣。旧10k缺少逐片段质量/摩擦标签，因此无法确认它的优势来自哪种实际物理比例。

混合组相对1B的均值增益，在三个策略seed分组中分别为：seed8 +68.71秒、seed19 +61.72秒、seed25 +19.12秒。这仍是一个训练seed上的探索性结果。

训练：每组40个250步片段，总10k=9k梯度训练+1k验证；seed42，200epoch/3600次更新。采集设定为轻40–80g、重180–300g，物体摩擦低0.5–1.5、高2.5–4.0；混合组四域各25%。实际保留分布和每段标签均已存档。

评估：4初态（demo079、082、094、真机pose50）×44/240g×物体摩擦1.312/2.572×策略seed8/19/25。xjz原生判据，DDIM4/4、exec2、scale25引导前2个动作，观察上限400秒。物体尺寸scale1。真机pose50使用同一历史手型，但本轮手部物理随机参数重新固定采样，不能把时长直接与过去单独seed50的结果混在一起。

| 方法 | 截断均值/s | 中位数/s | ≥20s | ≥80s | ≥400s |
|---|---:|---:|---:|---:|---:|
| 普通1B | 80.05 | 32.20 | 54.2% | 29.2% | 6.2% |
| 1B + 旧10k | 167.59 | 86.15 | 54.2% | 50.0% | 33.3% |
| 1B + 轻/低 | 95.22 | 1.83 | 33.3% | 25.0% | 16.7% |
| 1B + 轻/高 | 49.10 | 2.17 | 25.0% | 14.6% | 4.2% |
| 1B + 重/低 | 81.77 | 1.67 | 27.1% | 25.0% | 16.7% |
| 1B + 重/高 | 77.86 | 1.68 | 31.2% | 18.8% | 14.6% |
| 1B + 四域混合 | 129.90 | 57.18 | 60.4% | 45.8% | 14.6% |

400秒是观察截断，不是失败时长；原生failure是评估失败代理，并非独立的物理掉落检测。

混合组相比普通1B，截断均值变化+49.85秒，20秒存活率变化+6.2个百分点；相比旧10k，截断均值变化-37.69秒。

| 测试质量/摩擦 | 普通1B | 旧10k | 轻/低 | 轻/高 | 重/低 | 重/高 | 混合 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 44g / 1.312 | 85.71 | 245.65 | 152.74 | 46.97 | 134.29 | 93.54 | 143.89 |
| 44g / 2.572 | 116.11 | 209.52 | 167.69 | 128.72 | 173.34 | 174.90 | 209.95 |
| 240g / 1.312 | 54.17 | 129.58 | 15.06 | 14.44 | 17.27 | 5.24 | 80.35 |
| 240g / 2.572 | 64.21 | 85.63 | 45.41 | 6.26 | 2.18 | 37.77 | 85.42 |

每格12例，均为400秒截断均值。它们是特定诊断初态上的结果，不能直接当作总体成功率或质量/摩擦的通用阈值。没有等比例但不同质量/摩擦的扫描，也没有接触法向力记录，本轮不足以确认一个通用质量/摩擦比定律。

| 方法 | 原本<20s、现≥20s | 原本≥20s、现<20s | 相比1B提升>20s | 下降>20s |
|---|---:|---:|---:|---:|
| 普通1B | 0 | 0 | 0 | 0 |
| 1B + 旧10k | 7 | 7 | 21 | 9 |
| 1B + 轻/低 | 1 | 11 | 11 | 14 |
| 1B + 轻/高 | 0 | 14 | 6 | 19 |
| 1B + 重/低 | 2 | 15 | 11 | 15 |
| 1B + 重/高 | 2 | 13 | 9 | 16 |
| 1B + 四域混合 | 7 | 4 | 22 | 7 |

这些是同一批案例的描述性计数，不是统计显著性检验。

## 对原因的证据与限制

新五组使用共同初始化模板、相同训练预算，能更直接比较数据分配的影响。它们使用四域都成功的共同模板，且只保存连续250步片段，因此是经过成功筛选的保留分布，不能把设定的均匀采样区间当成实际数据分布。

新旧10k还存在片段长度、成功筛选、质量缓存处理和起始观察覆盖差异，不能把新旧性能差全部归因于质量/摩擦。旧训练集有18个目标残差RMS<0.001rad的起始观察；新五组因跳过4步预热而均无此观察，但评估从目标=q、残差0开始。这是候选因素，尚未做消融证明。混合组共享这一限制却改善早期表现，也说明它不能单独解释所有结果。

在同一批留出专家观察上，新guide动作MAE约0.041–0.045rad，旧guide约0.049–0.052rad；这项开环误差不能预测闭环guidance的排名。留出指不参与梯度更新；归一化沿用项目实现，统计整个replay buffer（包含验证片段）。

模型输入只有关节角、上一目标和目标残差，不直接输入质量、摩擦或物体位姿。各guide的初始目标改变量接近，单看动作幅度也不足以解释结果；目前没有接触力证据来确认某个抓握机制。

只有一个训练seed和4个诊断初态，所有guide固定scale25，未单独调优各模型。所有方法初态、实际物理参数、初始RNG及prior噪声配对，原生目标更新仍依赖各自状态；同输入基线重跑也存在后续轨迹变化。

中断前后的两次普通1B运行，初态、物理和第一步动作相同；按共同240秒窗口截断，逐案例时长绝对差平均43.27秒，21/48例相差超过20秒。这提示单条轨迹的改善需要重复验证；它不是置信区间，也不能确定波动的来源。

## 文件与录像

- [训练分布与checkpoint登记](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260920_10k_domain_guides/guide_registry.md)
- [逐案例与分组结果](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260920_10k_domain_guides/evaluation_summary.json)
- [逐初态/seed、早期动作和迁移分析](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260920_10k_domain_guides/transfer_analysis.json)
- [留出专家观察分析](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260920_10k_domain_guides/offline_transfer_report.md)
- [实际训练分布图](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260920_10k_domain_guides/training_distributions.png)
- [结果对比图](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260920_10k_domain_guides/comparison.png)
- demo079 / noise8，44g/μ2.572：[原速并排录像](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260920_10k_domain_guides/videos/case03_all_guides.mp4)、[10倍速预览](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260920_10k_domain_guides/videos/case03_overview_10x.mp4)。结束一侧冻结并标注最后一帧。
- demo082 / noise19，44g/μ2.572：[原速并排录像](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260920_10k_domain_guides/videos/case16_all_guides.mp4)、[10倍速预览](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260920_10k_domain_guides/videos/case16_overview_10x.mp4)。结束一侧冻结并标注最后一帧。
- demo094 / noise25，44g/μ2.572：[原速并排录像](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260920_10k_domain_guides/videos/case29_all_guides.mp4)、[10倍速预览](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260920_10k_domain_guides/videos/case29_overview_10x.mp4)。结束一侧冻结并标注最后一帧。

| 录像方法 | demo079 / seed8 | demo082 / seed19 | demo094 / seed25 |
|---|---:|---:|---:|
| 普通1B | 244.97 | 41.87 | 1.27 |
| 1B + 旧10k | ≥400 | 1.70 | 1.13 |
| 1B + 轻/低 | ≥400 | 0.53 | 1.63 |
| 1B + 轻/高 | ≥400 | 2.03 | 4.43 |
| 1B + 重/低 | ≥400 | 1.53 | 85.57 |
| 1B + 重/高 | ≥400 | 38.87 | ≥400 |
| 1B + 四域混合 | 218.30 | 58.77 | 1.80 |

录像说明了单例与总体排名不同：重/高guide在demo094的seed25案例达到400秒，但同物理条件另外两个seed案例仅1.03秒、2.40秒。这一条录像不能当作对该手型稳定成功的保证。

模型加载、EMA/训练计数、数据哈希、初始化与物理参数配对、DDIM加速实现对原始实现的一致性均已检查。三个录像案例在运行前固定选择；录像不替代336例统计。