# h8 SDEdit 跨窗口 Gaussian 噪声复用：预注册方案

状态：**只完成方案、代码和离线检查；未加载 checkpoint 推理，未启动仿真。**

## 研究问题与必要的基线区分

在 obs4 / horizon8 / pred5 / exec2 下，只改变重叠未来动作的初始 Gaussian noise 相关性，考察是否减少跨窗口动作计划变化，同时维持翻转与抓握表现。主比较预先指定 N80−B；N50、N100为剂量与极端组，不根据实验结果删 seed 或条件。

现有 `reference_action_editor.py` 每次调用 `set_fixed_noise_from_seeds(seeds)`，重新 manual_seed 再采样；IPC 每轮发送同一个 seed。因此历史基线是**每个局部槽位重复同一张 Gaussian 张量**，不是每轮独立采样，也不是按绝对未来时间移位复用。不能给历史结果贴上 ρ=0 或 ρ=1 标签。

| 组 | 噪声协议 | 角色 | 数量 |
|---|---|---|---:|
| L_archived | 历史固定槽位噪声 | 复用既有结果，衡量相对当前实现的变化 | 55条存档 |
| B | ρ=0，每轮独立重采 | 干净的单变量实验对照 | 55条新运行 |
| N50 | ρ=0.5，重叠未来时间相关 | 弱连续性 | 55条新运行 |
| N80 | ρ=0.8，重叠未来时间相关 | 预先指定主实验组 | 55条新运行 |
| N100 | ρ=1，重叠未来时间完全复用 | 极端机制组 | 55条新运行 |

共计划 **220条新仿真 + 55条存档对照**。L与B的区别还包括历史槽位噪声从固定改为每轮新采，故L→N的差异不能全部归因于未来noise correlation；只有B→N是本实验的干净因果比较。无需额外跑旧action复用、trend loss、adaptive noise或scale=0 guidance组。

## 样本选择与资格

来源：`../20260930_h8_object_state_random10/manifest.json` 和各case的 `audit.json`、`summary.json`、`predictions.json`、`initial_state.npz`、`trace.json`。

严格取原始预选seed顺序的前5个：**486266、426182、662165、766174、234124**。未使用事后“去掉最好最差”的种子表。它们是prior噪声seed，环境seed始终42。

四个episode保留自己的完整源初态、手q和reference尾段：76/f114、34/f79、54/f110、2/f103。固定wrist，仅评价手指控制下的翻转保持，不声称复现真机完整放置。

| 物理条件 | 原始direct最长连续竖直接触步数（ep76/34/54/2） | 纳入 |
|---|---|---|
| 44g / μ=1.1 | 110 / 104 / 40 / 34 | 四个episode |
| 170g / μ=2.0 | 125 / 121 / 93 / **15** | 76、34、54 |
| 130g / μ=2.4 | 98 / 120 / 82 / 39 | 四个episode |

ep2×170g/μ2.0没有通过未经修改原始direct的连续≥30步翻转资格，按AGENTS要求在运行前排除，共11个episode×物理条件。没有改起点、动作、质量或摩擦来让它通过。结论仅适用于该direct合格子集。资格源文件、数值和排除项均写进manifest。

已存档的55条L：稳定翻转 **32/55**，平均观察动作步数 **184.22**。按seed分别为7/11、7/11、7/11、6/11、5/11；这些值是源数据复核，不是新实验结果。

## 固定配置

- checkpoint `/home/carus/data_usb/aggresive_random_ckpt/h8.ckpt`，EMA，obs4×66，horizon8，future5×22，exec2。
- SDEdit requested noise ratio=0.15，历史实际值0.15339703857898712；局部DDIM4时刻 `[8,5,3,0]`，100训练扩散步，eta=0，当前裁剪与normalizer不变。
- 现有SDEdit构造：3个known-history目标 + **5个future reference目标**。前4步guide是另一种guidance算法的配置，不把它误加到当前edit上；guidance scale=0。
- 相邻原始reference的22关节最大跳变严格>0.1rad时插1个中点；每执行2个实际控制步滑动2格。末尾仅padding，不提前跳进度。
- 固定原始初始化顺序：修改物理属性→simulate/fetch/observe→导入手、目标、物体位姿及零速度。qpos静置60步、完整reference尾段、末尾保持60步；30Hz，同步推理等待，不在推理期间推进物理。
- 原生demo000–149；物体位置阈值0.05m、各指尖0.1m、旋转180°、立即失效物体位置0.15m、FailureToleranceScale=10000、fixedToleranceSteps=20000、resetOnReachGoal=false、crossTrajectoryGoalProb=0.3、trajStepsLimit=12000。
- 位置误差/目标更新/容忍步数由原生任务实现；保持既有回放的native failure继续记录而不自动reset。几何分离指标不参与终止或控制。保持时长不能和严格RL默认口径混比。
- 不加入旧action tail、平滑、trend guidance、自适应噪声、额外外力或姿态触发配置。

## 噪声实现与随机性配对

完整模型张量为 `[1,8,22]`，索引0:3是known history，3:8是未来5步。对第k轮完整创新噪声 `xi_k`：

```python
noise = xi_k.clone()
if k > 0:
    noise[:, 3:6] = rho * previous_noise[:, 5:8] + sqrt(1-rho*rho) * xi_k[:, 3:6]
# history 0:3 和新未来 6:8 保持 xi；每轮都采完整 [1,8,22]
```

当前未来b0/b1/b2分别对齐上轮a2/a3/a4。所有新组的历史槽位都重新采样；不将已执行未来噪声移入history，避免同时增加第二个研究变量。ρ=0走exact identity分支，不多做浮点乘加。

每个rollout独立创建CUDA generator，仅初始化时seed一次。四组同seed、同设备、同dtype，按相同顺序消耗完整张量；即使ρ=1也消耗完整xi，以保持common random numbers。第一轮四组及L的初始Gaussian一致；之后rho仅改变融合，不改变随机流推进。

保存的是原始SDEdit Gaussian epsilon，既不是UNet预测epsilon，也不是旧x_tau或去噪后的action。每次调用都根据当前观测的known history和当前reference重新normalize，再构建：

`x_tau = sqrt(alpha_bar_tau) * normalize(C_current) + sqrt(1-alpha_bar_tau) * noise_current`。

后续DDIM及历史mask完全沿用既有editor。新episode/seed/reference必须新建stream。服务限制单环境，reference_index必须0,2,4…；重复、跳步或更换reference直接报错，不静默复用错误缓存。本版本不接到实时异步控制器，也不支持可变exec。

## 记录、分析与解释边界

每次推理保存全部5步动作、当前reference、obs4、完整xi/raw epsilon、有效未来长度、reference_index、clip比例及原有edit误差。动作仍只发送前2步。记录用于以下审计：

1. 每步noise融合满足公式，history和新尾部为fresh；四组xi逐值相等、首轮5步plan逐值相等。
2. 首轮发送给环境前，以1e-6rad绝对容差与L前两条命令比较。不一致立即退出；这不是完整旧协议复现的替代。
3. 新运行与同seed L和原始direct的全部初态字段、60步静置逐值核对，并核对首个实际控制步。不能仅检查导入tensor。
4. 执行动作与plan前2步逐值一致；参考进度每轮+2，末尾有效长度正确。

主结果保持原存档口径：普通翻转、连续≥30步稳定翻转、首次连续3步几何分离前观察动作步数、原始reference进度、native failure单独报告。几何分离：连续3步满足gap>5mm且物体净接触力<0.05N，或gap>2cm；稳定翻转要求距竖直≤30°、离桌>8cm、至少2个近接触link、gap<8mm、物体净接触力>0.1N。

旧报告采用首次分离的一基动作编号，未分离或hold才分离者按action尾段截尾；新报告保留这一对照口径，另存零基的分离前动作数和截尾标记。数值几何代理需配合录像，不能直接把native failure叫物理脱手。

机制指标为 `RMS(plan_k[:3] - plan_(k-1)[2:5])` 和执行边界 `RMS(plan_k[0] - plan_(k-1)[1])`。只使用真实有效未来时刻，排除padding；主配对统计截到两组共同分离前前缀，防止“早掉后不动”被算作更平滑。reference对齐后的residual变化也保存用于审计；同一reference严格对齐时它与plan变化相等，不能当作独立证据。L没有保存完整plan，所以不补造L的overlap指标。

输出整体、逐seed、逐物理条件、逐episode的结果；每个N对B和L列出稳定翻转胜/平/负、保持步数差，N对B另外列共同前缀overlap差。三个rho全部报告，不剔除差seed。5个seed只有11种固定初态/物理组合，不把55条当作55个独立初态，不作强显著性或非劣效结论。

判读预先约定：只有overlap下降且翻转/保持没有一致变差，才能说值得继续；单看平滑不足以证明更好。闭环plan变化也包含观测变化，不能直接命名为sampling-induced mode switching的实测次数。本轮没有受控扰动，所以只能评估自然闭环纠偏是否出现退化迹象，**不能证明扰动恢复能力不受损**。若本轮通过，再单独设计固定时刻/幅度的扰动配对，不能混进这轮主表。

## 录像与运行保护

新运行沿用原生录制函数，包含导入、60步静置、动作起点和完整动作/hold，主图正面与含桌面的宽景。按case保存原始 `guided_front.mp4`。历史L批次未录视频，不能伪称已有配对实拍。

最终对比视频需把同物理条件原始direct放左上、按 `reference_progress.npy` 对齐，并标注各自实际步数；已录direct可以复用，缺视频的存档不能拿其他质量/摩擦录像替代。`compose_video.py`接收一条新运行与同条件direct录像，校验初态、协议、reference后生成上述布局；若direct缺视频则明确报错，不自动启动补录或使用伪造画面。补录不属于220条策略实验，须单独列为可视化工作并逐值核对存档轨迹。

默认入口只dry run；不会隐式启动服务器/仿真。`--prepare`读取存档、计算SHA256并写manifest/命令清单；`--execute`才执行，串行且每条独立进程。输入hash变化拒绝执行；已有不完整输出拒绝覆盖；完成标记绑定manifest hash。当前工作区已有改动被保留并纳入hash。

checkpoint和源码现在的hash已冻结，但旧存档没有checkpoint内容hash，所以不能独立证明归档以来文件从未变化。首轮命令一致性门禁补充行为核对；不同则暂停处理来源差异，不将不匹配结果混表。

## 命令与已验证范围

在仓库根目录执行：

```bash
# 只检查计划（默认，不启动任何模型或仿真）
/home/carus/miniforge3/envs/dp/bin/python docs/experiment_reviews/20260930_h8_noise_reuse/experiment.py

# CPU单元检查；mock denoiser，不读取checkpoint
PYTHONDONTWRITEBYTECODE=1 /home/carus/miniforge3/envs/dp/bin/python docs/experiment_reviews/20260930_h8_noise_reuse/test_noise_reuse.py
PYTHONDONTWRITEBYTECODE=1 /home/carus/miniforge3/envs/decv2/bin/python docs/experiment_reviews/20260930_h8_noise_reuse/test_analysis.py

# 准备/刷新冻结清单，仅在没有runs目录时允许
/home/carus/miniforge3/envs/dp/bin/python docs/experiment_reviews/20260930_h8_noise_reuse/experiment.py --prepare

# 以下仅保存供后续使用，本次未执行
/home/carus/miniforge3/envs/dp/bin/python docs/experiment_reviews/20260930_h8_noise_reuse/experiment.py --execute
/home/carus/miniforge3/envs/decv2/bin/python docs/experiment_reviews/20260930_h8_noise_reuse/analyze.py
```

代码：`noise_stream.py`实现移位与RNG；`server.py`接原有SDEdit；`experiment.py`做准备/启动保护；`analyze.py`复用原几何审计并做配对；`compose_video.py`处理已有实际运行视频。共享editor增加可选`initial_noise`，未传入时保持原有路径；回放器另增纯显示参数`--video-label`，避免h8视频沿用旧的10B标签。两项均保留已有调用的默认行为。

本次CPU检查不验证GPU确定性、checkpoint实际forward、完整IPC/IsaacGym运行或后处理真实结果；这些需未来获准开跑后验证。不会把mock测试通过写成仿真成功。
