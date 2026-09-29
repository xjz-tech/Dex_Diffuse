# 1B guide → 10B prior

2026-09-21。方向为10B主策略、1B guide。按相同48个初始化/物理/噪声配置评估普通10B、10B+1B guide、10B+旧10k guide；1B相关两行复用上一轮已核对相同初态和物理参数的结果。

## 本次结论

在相同48个case、DDIM4/4、exec2、scale25下，**10B+1B guide平均保持54.13秒，普通10B为115.69秒，相差-61.57秒；旧10k guide同样引导10B时为160.18秒。** 1B guide的中位保持16.37秒，普通10B为70.67秒；20秒通过数分别是23/48与31/48。

成对案例中，1B guide救回2组原本不到20秒的案例，却让10组原本超过20秒的案例提前失败。归档真机手型12组均值：普通10B 126.61秒，1B guide 21.10秒，旧10k guide 246.34秒。

首次动作相对普通10B的平均变化，1B guide为0.0040rad，旧10k guide为0.0214rad。首步变化大小不能单独解释闭环保持结果。这里的比较限定在当前scale与四种指定初始化；每个case只运行一轮。

## 结果

平均值截断于400秒；≥400表示到达观察上限。原生failure是评估失败代理，不独立证明物理掉落。

| 方法 | 平均保持/s | 中位/s | ≥20s | ≥80s | ≥400s | 相对普通10B平均变化/s |
|---|---:|---:|---:|---:|---:|---:|
| 普通1B | 80.05 | 32.20 | 26/48 | 14/48 | 3/48 | — |
| 1B + 旧10k guide | 167.59 | 86.15 | 26/48 | 24/48 | 16/48 | — |
| 普通10B | 115.69 | 70.67 | 31/48 | 21/48 | 4/48 | +0.00 |
| 10B + 1B guide | 54.13 | 16.37 | 23/48 | 10/48 | 1/48 | -61.57 |
| 10B + 旧10k guide | 160.18 | 31.77 | 25/48 | 22/48 | 14/48 | +44.49 |

![保持分布](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260921_10b_prior_1b_guide/comparison.png)

10B主策略的成对得失：

| guide | 比普通10B多保持>20s | 比普通10B少保持>20s | 普通10B<20s→guide≥20s | 普通10B≥20s→guide<20s |
|---|---:|---:|---:|---:|
| 1B | 8 | 23 | 2 | 10 |
| 旧10k | 17 | 12 | 4 | 10 |

## 初始化分组

每组12个case，均为400秒截断均值。

| 初始化 | 普通1B | 1B+旧10k | 普通10B | 10B+1B | 10B+旧10k |
|---|---:|---:|---:|---:|---:|
| demo079 | 193.19 | 314.02 | 189.36 | 100.08 | 296.21 |
| demo082 | 17.57 | 0.51 | 32.71 | 37.81 | 0.63 |
| demo094 | 55.28 | 114.28 | 114.10 | 57.52 | 97.53 |
| real_pose50 | 54.17 | 241.56 | 126.61 | 21.10 | 246.34 |

## 质量、摩擦分组

每格12个case，物理参数为引擎实际读回值。

| 质量 / 摩擦 | 普通1B | 1B+旧10k | 普通10B | 10B+1B | 10B+旧10k |
|---|---:|---:|---:|---:|---:|
| 44g_mu1.312 | 85.71 | 245.65 | 181.22 | 34.87 | 244.61 |
| 44g_mu2.572 | 116.11 | 209.52 | 145.74 | 107.26 | 168.99 |
| 240g_mu1.312 | 54.17 | 129.58 | 67.98 | 54.28 | 107.84 |
| 240g_mu2.572 | 64.21 | 85.63 | 67.83 | 20.10 | 119.28 |

## 策略噪声seed分组

每个seed16个case。

| seed | 普通1B | 1B+旧10k | 普通10B | 10B+1B | 10B+旧10k |
|---:|---:|---:|---:|---:|---:|
| 8 | 65.05 | 125.79 | 131.77 | 58.16 | 127.75 |
| 19 | 94.53 | 232.69 | 93.88 | 67.48 | 183.71 |
| 25 | 80.58 | 144.29 | 121.43 | 36.74 | 169.08 |

## 条件与解释边界

固定4初始化（demo079、082、094、归档真机手型）×2质量（44g、240g）×2摩擦（1.312、2.572）×3策略噪声seed（8、19、25）；每方法48组，400秒上限。所有DP仿真统一沿用xjz_test原生任务：0.05m物体相对手腕误差、0.1m指尖误差、180°旋转误差、0.15m立即失效位置误差、FailureToleranceScale10000、fixedToleranceSteps20000、resetOnReachGoal=false、跨轨迹目标概率0.3、demo000–149。目标更新和容忍步数不另写判定。

普通主策略采用真正普通DDIM，不使用guided scale=0。10B/1B/旧10k均使用EMA检查点，66维观测/22维动作；prior和guide各DDIM4、每次执行2步、引导前2个可执行动作、scale25。guide先生成物理角度动作，再按10B主策略的动作normalizer进入引导公式。

本次三个新方法完整重跑；上轮1B两行可作为同48配置对照，初态含CUDA随机状态和实际质量/摩擦/手部参数逐项完全一致。10B先前12000组历史评估不能直接和这些400秒均值混用。

每个方法的每个case只运行一次；这些4个初始手型是有针对性的案例，不代表150个demo总体。闭环轨迹之前观察到重复运行差异，故小幅单例或总体差异不能当成稳定性证明。完整逐case结果、首动作和视频保留以供复查。

## 文件

- verification.json：初态、物理参数、原生协议和模型哈希核对。
- evaluation_summary.json：全部逐case结果及分组汇总。
- evaluation/：原始视频、首次动作、状态、物理读回和完整rollout。

## 四路对比录像

三个录制case预先固定，均为44g、摩擦2.572；10倍播放。左上为上轮1B+旧10k，右上为普通10B，左下为10B+1B guide，右下为10B+旧10k guide。面板时钟为仿真时间，已结束的画面冻结并标注。

- [demo079 / noise8](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260921_10b_prior_1b_guide/videos/case03_comparison_10x.mp4)
- [demo082 / noise19](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260921_10b_prior_1b_guide/videos/case16_comparison_10x.mp4)
- [demo094 / noise25](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260921_10b_prior_1b_guide/videos/case29_comparison_10x.mp4)

完整原速录制保留于本实验的evaluation/ordinary_10b、evaluation/guide_1b、evaluation/guide_old10k目录中，每种方法均有case03_raw.mp4、case16_raw.mp4、case29_raw.mp4。单个录像不能替代48组统计。

[采样数值核对](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260921_10b_prior_1b_guide/guided_sampler_verification.json)；[首动作差异](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260921_10b_prior_1b_guide/first_action_diagnostics.json)；[逐case结果](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260921_10b_prior_1b_guide/evaluation_summary.json)。
