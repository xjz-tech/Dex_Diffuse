# dp_dino 本周实验复盘（2026-09-07—2026-09-12）

核对时间：2026-09-12T21:37:41+08:00。以本地原始 episode JSONL、启动日志和代码版本为依据；结果快照不自动更新。本次没有启动额外训练或修改推理算法。

**核心判断：同域 10k guide 改善强 1B prior 的现象值得继续研究；真实数据 guide 在仿真里造成明显退化。现在最该解决的是噪声协议、任务目标和修正幅度，而不是继续扩大混合权重网格。**

当前分支名是 dp_dino，但本报告的主要闭环结果来自无图像的 Sim-Hand lowdim 策略（4×66 本体状态、22 维关节动作），不能直接称为 DINO 视觉策略能力提升。1B、mixed 是实验中的 checkpoint 标签，本报告没有把 1B 解释成参数量。

## 1. 评测口径

正式比较：150 条轨迹（000–149）、物体/指尖失败阈值 0.05/0.1 m、tolerance scale=10000、fixed tolerance=20000、cross-trajectory probability=0.3；DDIM4、预测未来9步、执行2步、历史 guidance 只约束前2步；每个初始环境只统计首次 episode，12000 控制步=400秒，未失败的记录为 timeout。

- “400秒未失败比例”对应日志 completion_rate；这里完整批次的 success 数均为0，不能将该比例当作旋转/目标任务成功率。
- 限制平均保持时间是 mean(min(T_failure,400s))，含未失败样本；不推断400秒之后的平均寿命。
- 新三 seed 表的中位数由3000条记录合并计算，不是三个中位数的平均。
- 不把中途停止或只有失败记录的文件当完整初始环境队列。旧“累计3000失败episode”与新“3000个初始环境”分开。
- 多个评测 seed 不等于多个训练 seed；同一 guide checkpoint 的泛化到其他训练子集尚未验证。

## 2. 正式主结果：1B / mixed × 无 guide / 10k guide

seeds=8、19、25，每 seed 每方法1000个初始环境。10k guide 固定为 runs/sim_hand_10k_seed42/checkpoints/latest.ckpt；prior 为 /home/carus/data_usb/obs_4-66.ckpt 或 /home/carus/data_usb/8_2_mixed.ckpt。无 guide 行虽通过双模型入口加载了 real1400 checkpoint，但 scale=0 禁用了 guidance 修正。

| 实验 | N | 400 秒未失败 | 限制平均保持 / 秒 | 首次保持中位数 / 秒 |
|---|---:|---:|---:|---:|
| 1B，无 guide | 3000 | 13.33% | 143.13 | 87.53 |
| 1B，10k guide，scale25 | 3000 | 19.57% | 162.30 | 106.58 |
| mixed，无 guide | 3000 | 11.17% | 134.69 | 80.35 |
| mixed，10k guide，scale25 | 3000 | 13.73% | 136.00 | 77.15 |

逐 seed 的400秒未失败比例：

| Seed | 1B 无 guide | 1B + 10k | mixed 无 guide | mixed + 10k |
|---|---:|---:|---:|---:|
| 8 | 13.9% | 18.8% | 10.4% | 13.0% |
| 19 | 12.8% | 20.6% | 10.3% | 13.6% |
| 25 | 13.3% | 19.3% | 12.8% | 14.6% |

- 1B + 10k：未失败比例 +6.23个百分点（相对 +46.75%）；限制平均保持时间 +13.40%；合并中位数 +21.76%。三个 seed 方向一致。未失败比例跨 seed 均值±样本标准差：13.33±0.55% → 19.57±0.93%；这不是置信区间。
- mixed + 10k：未失败比例 +2.57个百分点，但限制平均仅 +0.97%，中位数80.35→77.15秒。主要改善长尾，不能称作全面提升。seed25 的限制平均甚至139.43→139.26秒。
- 1B 无 guide 稳定优于 mixed；加相同10k guide 后差距更大。现有证据支持优先围绕1B做主线，而非默认混合数据一定更优。
- 观察到小幅早期差异，但尚未证明稳定或统计显著的早期副作用。按“失败时间≤5秒”计数，新三seed的1B无guide为325/3000次早期失败，加guide为365/3000；5秒存活比例89.17%→87.83%。长期收益与初期鲁棒性应同时报告。此前使用length≥150的边界口径会将恰在第150步失败的记录算入该时刻存活，本次明确统一为排除失败时间≤t的记录。

来源：[无引导三seed](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/hold_runs/20260911_mixed_vs_1b_3seeds_1000)、[10k引导三seed](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/hold_runs/20260911_10kguide_1B_vs_mixed_3seeds_1000)。

## 3. 9月10日强结果与自引导负对照

| 实验 | N | 400 秒未失败 | 限制平均保持 / 秒 | 首次保持中位数 / 秒 |
|---|---:|---:|---:|---:|
| 旧 1B，scale0 | 3000 | 11.73% | 132.88 | 75.93 |
| 旧 1B + 10k，scale25 | 3000 | 46.43% | 239.61 | 328.87 |
| 1B 自引导，prior42 / guide7 | 3000 | 7.33% | 113.23 | 62.47 |

补充早期核对（按失败时间≤观察时刻计数）：旧seed42实验前5秒失败357→376，增加19/3000=0.63个百分点；10秒时累计失败498→448，guide已更好。400秒累计失败2648→1607，大幅减少1041个。早期少量差异与长期显著收益并不矛盾；目前不足以认定guidance稳定增加早期失败风险，也不能据此否定总体提升。

第一对实验为seed42、每组3000环境：未失败比例11.73%→46.43%（+34.70个百分点），限制平均保持时间132.88→239.61秒（+80.32%）。同 checkpoint 不同噪声的自引导只有7.33%未失败，中位62.47秒，弱于旧无引导基线；因此“只要多采样/多跑一个模型就会提升”没有得到支持。但自引导仅一个种子组合，并且代码处在切换时间点，不能据此排除所有同模型融合方法。

**旧新结果的核心混杂：噪声协议改变。** 9月10日17:16提交 a1bc783 之前的 eval/xjz_eval_guidance.py 给 prior 固定噪声，guide 直接调用 predict_action，没有绑定重置随机生成器，每次使用新噪声。新 eval/guided_pair_policy.py 的 bind_conditional_sample_seed 在 fixed_noise=1 时每次按相同seed重建生成器，guide 也变成固定噪声。旧日志的 weak=/strong= 与新日志的 prior=/guide= 分别对应这两个入口。

因此，46.43% 与19.57%差距混合了环境seed、环境数量、代码和噪声协议；不能归因于某一个因素，也不能合并成同一重复实验。上述代码核对支持这一协议差异，旧运行没有完整源码哈希，仍需严格复跑确定因果。

来源：[旧3000环境对照](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/hold_runs/20260910_n3000_t12000_firstep_strong1b_prior_weak10k_guide2_scale0_25)、[自引导](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/hold_runs/20260910_n3000_t12000_firstep_strong1b_selfguide_prior42_guide7_scale25)、[当前噪声绑定](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/guided_pair_policy.py:42)。

## 4. 9月11—12日：真实数据训练的 guide

real1400：data/real_hand_sft/manifest.json 记录75条episode、48894帧，单位弧度，obs=[qpos,target_before,target_before-qpos]，action=commanded_target_after。下面都是该模型在仿真中评测，不是真机成功率。

| Prior | Seed | scale0：未失败 / 中位秒 | real guide scale25 | real guide scale50 |
|---|---|---|---|---|
| 1B | 42 | 12.30% / 75.93 | 0.83% / 16.82 | 0.00% / 0.57 |
| 1B | 8 | 10.83% / 73.27 | 1.33% / 17.40 | 0.00% / 0.60 |
| Mixed | 42 | 8.70% / 66.15 | 1.27% / 10.28 | 0.07% / 0.47 |
| Mixed | 8 | 8.30% / 67.47 | 2.20% / 8.92 | 0.10% / 0.47 |

real1400单独运行：seed42/8 的400秒未失败比例0.83%/0.93%，中位都约0.23秒。对强prior施加scale25已严重退化，scale50在多数环境几乎立即失败，两个seed趋势一致。

这说明当前真实域动作参考在仿真闭环中不适配；不能直接说“真实数据没用”，也不能单凭此结果断言只是normalizer的问题。需要分开检查状态支持范围、目标关节动作与实际qpos的语义、时序滞后、关节顺序和动力学差异。manifest 声明的语义并不自动证明数据构建无误。

9月10日另一个 real epoch200 的128个仿真验证窗口离线结果：prior的动作RMSE为0.02186，sim10k guidance后0.02866，real guidance后0.36993。sim10k reference自身0.06885，real reference自身0.50387。**离线更贴近示范与在线保持更稳不是同一个指标**；但这里的real200不是上述real1400，不能合并成同checkpoint诊断。

来源：[离线128窗口](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/hold_runs/compare_guided_vs_gt_val128.json)、[真实数据定义](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/data/real_hand_sft/manifest.json)、[真实guide网格](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/run_real1400_grid.sh)。

## 5. 9月8—9日探索实验

| 内容 | 可核对结果 | 可用结论及限制 |
|---|---|---|
| obs22/obs66，DDPM与DDIM2/3/4/8/16 | 早期严格阈值下，已失败episode中位约1.6–1.8秒 | 是早期筛选；不能和后来的宽松hold基线比绝对时长 |
| 宽松hold下 DDIM×exec 网格 | exec2时，DDIM2/4/8/16的已失败episode中位约6.40/25.18/12.27/8.32秒 | DDIM4/exec2是合理候选；按累计失败停止且缺失未失败样本，不能当正式生存比较 |
| mixed exec2 vs exec5 | 已失败episode中位25.33 vs16.37秒 | 支持少执行几步再闭环的候选方向；仍需统一观察窗口确认 |
| TensorRT vs eager，seed7 | 已失败episode中位25.40 vs26.08秒 | 近似接近，不足以证明精度等价或更优；不作为prior创新证据 |
| 弱10k prior + 强1B guide，前2步 | 共同比较前126步=4.2秒：scale0/10/25/50 的未失败比例约1.07/38.48/62.21/59.38% | 强prior相关信息能挽救弱策略；这里只是极短窗口，未证明优于强模型单独运行 |
| guide动作范围×scale | scale100、guide2/exec2的已失败episode中位0.33秒；guide4/exec4约5.20秒；scale150部分组合进一步崩溃 | scale与范围强耦合，不能脱离范围比较“最佳scale” |

10k训练实际是9999 transitions、20条episode、单个训练子集seed42，不是10000条独立轨迹。来源：[子集manifest](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/data/sim_hand_10k_seed42/subset_manifest.json)。

## 6. 已排除、清理和在跑内容

已按用户指令删除 eval/hold_runs/20260911_parallel_1Bbase_fullhorizon_3k 整批失效结果。该批仅8条轨迹、物体/指尖阈值0.012/0.036m、tolerance scale=1、fixed tolerance=200，无法与历史hold基线比较。删除审计保留文件名/哈希及原因，不保留失效结果文件。训练checkpoint保留；供后续重启使用的predecessors.pids已复制到替代目录。

9月12日matched parallel批次仍在跑，不能下结论。其共享normalizer的10k模型已重训200epochs/3600steps，guide范围变为全部9步，同时与旧算法相比改变了checkpoint和guide范围。它能做内部serial/fixed/dynamic对比，但不能直接将与历史差异归因于parallel。当前未取得完成的同批方法对照。

fresh-noise批次计划seeds42/8/19/25、每方法3000环境、两种prior×scale0/25；当前设置 prior和guide都fresh，guide_seed=seed+100000。它也尚无完整对照。注意这仍不等于旧实现的“prior固定、guide fresh”。

其他只排除不删除：seed7短跑、seed25 interrupted目录、未完成mixed guide scale10；real1400 seed42 scale0的113708与132237文件完全重复，只用后者一次。早期其他任务中提到但当前本地无原始文件的5000步/20000步实验，不纳入本报告的重新统计表。

## 7. 推论与优先研究路线

**已经支持的现象**：同域弱guide可以改善强prior的长时稳定性；这种增益依赖prior与guide组合，并非guide单独越强越好；过强修正会毁掉prior；mixed收益集中于长尾；离线动作误差不能替代闭环评测。

**尚未证明的机制**：guide是否提供互补动作信息、是否仅在缩小动作幅度/平滑噪声、是否增加任务完成率、是否推广至新对象/视觉DP/真机。这些要靠消融和任务指标区分。

### 优先级1：先确定噪声效应和真正的提升

固定1B、原始10k checkpoint、guide2/exec2、DDIM4及同一份初始状态清单。在同一批环境上做2×2：prior固定/fresh × guide固定/fresh；每格scale0与25对照（scale0不必因guide模式重复）。使用独立可追踪的仿真seed、prior噪声seed、guide噪声seed，保存每环境初始状态与噪声，而不只依赖batch级seed。先小批筛查，再使用独立评测seed验证。

报告S(5s)、S(100s)、S(400s)、限制平均保持、失败类型和按轨迹/对象分组的区间。之后增加至少三个独立训练子集/训练seed的10k guide，确认不是20条episode的特例。

### 优先级2：证明prior在帮助“完成动作”，而不只是拿稳

在hold之外加入目标角度误差、旋转完成率、累计有效转角/速度、达到目标耗时、掉落率。补充保持当前位置/上一目标的简单基线，以及DP单独、prior单独、简单动作混合、guidance。统一推理预算或报告延迟，否则改善可能只是更多计算。

如果“guided拿得更久但几乎不转”，就不算我们要的DP能力提升。最有说服力的目标是：相同任务成功率下掉落更少，或相同掉落率下动作执行更准确、更快。

### 优先级3：做有边界、可退回prior的修正

主线用强prior承载可行动作分布，任务DP提供目标相关参考；先限制修正幅度，再探索门控。以原始关节单位校验修正，按关节/噪声时间步归一；在reference偏离可行范围或没有改善时减小权重。仅凭两模型分歧大小不能证明谁更可靠，门控需在独立数据校准。

先对比固定小修正、限幅修正、门控修正，配套记录修正量、动作速度、饱和比例、接触/滑移及任务推进。若小幅且有界修正保住1B基线并提升目标进度，就比单纯报告hold涨幅更接近“prior增强DP”的研究目标。

### 优先级4：parallel和全轨迹约束做独立消融

先固定同一checkpoint比较guide2与guide9，再比较serial和parallel，normalizer重训另做一轴。当前MSE对H×22个元素取均值，H由2变9时，相同残差下单坐标梯度约缩小到2/9；保持scale25不代表等效引导强度。可以用匹配修正范数的消融辅助解释，但不能机械把scale放大4.5倍视为等价。

在效果成立后再优化并行速度，报告同硬件P50/P95延迟和控制周期。并行本身先作为实现效率贡献，能力贡献来自它支持的控制频率/修正策略，并需独立验证。

### 优先级5：将真实数据与视觉任务重新接入

先做真实/仿真统一观测、动作定义及延迟检查，建立小规模同状态的动作对照。重新接入DINO/目标条件DP时，让它表达任务意图、让prior约束物理可行性；按对象/轨迹划分未见测试集，比较DP、prior、组合三个层次。当前无图像hold证据仅支持这条路线值得试，还不能替代视觉任务实验。

## 8. 研究定位参考

DexterityGen将低层灵巧运动能力与较粗任务命令结合，这与我们分开“任务意图”和“动作可行性”的路线相关。[DexterityGen原论文](https://arxiv.org/abs/2502.04307)

策略组合和测试时引导已有直接相关工作，因此研究贡献不能仅表述为“两个DP融合”；需要落在可靠性控制、跨域适配、目标进度与稳定性的权衡及受控验证上。[Compose Your Policies!](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f7fc38fdd95fd146a471791b93ff9f12-Abstract-Conference.html)、[DynaGuide作者项目页](https://dynaguide.github.io/)

“弱模型也有帮助”与Autoguidance有表面相似性，但当前代码是将x0动作拉向guide reference的损失引导，不能直接等同于该论文的强弱模型score组合。[Autoguidance官方实现](https://github.com/NVlabs/edm2)

## 9. 插件处理

用户明确范围为GPT/Codex和VS Code，保留Cursor。Codex插件卸载接口对superpower/superpowers均返回未安装，Codex临时插件目录副本已清除。VS Code未发现独立superpowers扩展或技能配置，openai.chatgpt扩展保留。Cursor安装和缓存曾误删，现已按原缓存提交d884ae04edebef577e82ff7c4e143debd0bbec99恢复并核对HEAD、技能文件和插件manifest。仓库docs/superpowers中的设计记录保留。已经加载到现有会话里的文本无法从历史上下文撤回。

[清理审计](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/2026-09-12-cleanup.json)。
