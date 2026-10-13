from pathlib import Path
import json,numpy as np
OUT=Path(__file__).resolve().parent
c=json.loads((OUT/'coverage.json').read_text());fc=json.loads((OUT/'fingertip_coverage.json').read_text());r=np.load(OUT/'real_states.npz');n=np.load(OUT/'nearest.npz');f=np.load(OUT/'fingertips.npz');episodes=json.loads((OUT/'episodes.json').read_text());co=r['contact_count'].sum(1)>0
body='''# 真实拧灯泡数据 vs 150条bulb2参考：手型覆盖审计

日期：2026-09-12。

## 结论

已读取真实数据75条、48,894帧（名义30Hz，约27.16分钟），并逐帧与全部150条重定向示范的89,927个原始频率姿态比较。真实数据目录编号为id_0–id_75，缺id_36，所以不是76条。

只能测“手型近似覆盖”，不能从这份数据给出完整手–物体状态的覆盖率，更不能给出RL或prior的成功率：真实Parquet没有灯泡6D位姿。31维state的前9维是机械臂末端位置与旋转表示，后22维是手关节，不能把前9维误当灯泡位姿。

按五个指尖位置差的RMS≤2cm作为说明性标准：全部真实帧47.38%，触觉检测到至少一个接触点的帧58.04%能找到相近参考。该阈值没有通过控制成功实验校准，不能将47%或58%当作已证明的“能力覆盖率”。

## 指尖几何距离

用同一Sharpa URDF，将真实测量关节和参考关节都转成手腕坐标下的五个fingertip link位置。对每个真实帧，搜索同一参考帧上五指联合距离最近的姿态，不能每根手指分别找不同帧拼接。

距离定义为sqrt(sum_f ||p_real,f-p_ref,f||² / 5)。RMS≤2cm并不保证每根手指都≤2cm。

| 相近阈值：五指位置RMS | 全部48,894帧 | 至少一指有触觉接触点：32,698帧 | 至少两指有接触点：15,885帧 |
|---|---:|---:|---:|
'''
for t in ['1','1.5','2','3']:
 vals=[g['threshold_coverage'][t]*100 for g in fc['groups']];body+=f'| ≤{t} cm | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% |\n'
body+='''
全部帧最近指尖距离中位数2.03cm；触觉接触子集中位数1.92cm。

更严格要求“所选最近参考下，每根指尖都≤2cm”时，分别为7.90%、9.93%、11.48%。这说明RMS会掩盖单根手指较大的偏移。

触觉分组只检查tactile_contact_points是否非空，不是完整阶段标注，也不是物体已被稳定握住的证明。图像检查能看到个别空手帧仍报告接触点，或握物帧没有点，可能存在噪声、延迟和漏检。三种分组用于敏感性比较；不能将“有触觉点”直接命名为“纯手内操作阶段”。所有segment_id均为0，数据没有现成的拿取/转正/拧入分段标签。

## 关节角交叉核对

真实state/actions后22维按现有数据转换器约定，已经是policy/Isaac顺序、单位弧度；直接比较，不再做一次SDK顺序重排。现有真实硬件读取器也是先从SDK顺序转成policy顺序再输出hand/joint_angle。参考opt_dof_pos由同一Isaac顺序优化保存。

关节距离为22个关节角差的RMS；每个真实帧搜索同一个完整参考手型的最近邻。

| 22关节角差RMS阈值 | 全部帧 | 至少一指有触觉接触点 |
|---|---:|---:|
'''
for t in ['5','10','15','20']:
 vals=[g['threshold_coverage'][t]*100 for g in c['groups'][:2]];body+=f'| ≤{t}° | {vals[0]:.2f}% | {vals[1]:.2f}% |\n'
body+='''
全部帧最近关节距离中位数14.87°；触觉接触子集中位数15.97°。动作命令另行比较，最近关节RMS中位数分别为15.07°、16.39°；没有用命令值冒充实际手状态。

关节角与指尖位置的最近参考帧可能不同，因为它们优化不同的距离指标。手指中间关节也可能在指尖较相近时明显不同。上述两种距离共同支持真实任务手型与参考存在系统差异，但不能排除真实编码器零位、模型几何、机械柔顺性等因素造成的部分偏差。

## 参考自身的新姿态距离

为了避免只挑一个任意门槛，将150条示范按整条随机分成5折，每轮拿120条作为参考，对另外30条每10个原始帧取一帧查询，总计9,000个查询。不会把同条轨迹相邻帧当成其测试参考。

- 参考留出姿态：95%的最近关节RMS≤7.16°；95%的最近指尖RMS≤0.918cm。
- 真实全部帧：仅0.074%和0.067%分别落入上述两种参考内部的95%范围。
- 真实查询时给了全部150条参考，比留出实验的120条更多，因此不是减少了真实查询可用参考所造成的差距。

这个差异说明：真实任务手型没有紧密落在这批参考的常见变化范围内。该校准是“参考分布的新颖程度”，不是控制成功校准；参考示范之间可能有重复动作与共同采集偏差，也会让参考内部距离较小。不得解释成“99.9%的真实状态都不可控”。

## 当前RL容差为什么改变结论

实际RL采集配置的每指目标位置容差为3.6cm，同时还要求物体位置1.2cm、旋转12°等条件。取指尖RMS最近的参考，再逐指检查3.6cm门槛：全部帧82.68%、触觉接触帧92.32%能通过手指这一项。

这不是完整RL成功率：还没有真实物体位姿，无法检查同一个参考的物体位置/朝向；也没有检查接触动力学、速度和持续保持要求。该查询按RMS寻找最近邻，不是按逐指最大误差寻找最优参考；所以未通过该检查也不证明其他参考都不能通过。

因此“较严格手型相似度低”和“很多手型位于较宽目标容差内”可以同时成立。不能由本结果直接判断controller学得不行。

## 对DP + prior的意义

这次比较针对150条参考，不是训练prior的RL实际rollout。RL在子目标之间产生的手型、修正动作和接触方式可以偏离重定向参考；真实遥操作时RL也可能这样执行。因此参考距离是定位分布缺口的线索，不能替代真实数据对rollout的比较。

更直接的下一步是：对实际用于训练prior的rollout使用相同指标，先比较真实手状态与rollout状态，再比较相近状态下的动作/短期动作序列。若真实状态接近rollout但prior仍表现差，重点查蒸馏、输入观测和guidance；若真实状态也远离rollout，才更支持数据覆盖不足。

真正的完整抓姿覆盖还需真实灯泡相对手腕的位姿。可优先寻找原采集的MoCap日志及时间对齐信息；若原日志没有保存，再做经过验证的图像位姿重建。本审计没有用单目图片臆测精确6D位姿。

图像已抽查6条示范各6个阶段关键帧，并额外检查9个真实帧与最近参考手型的叠加。图像中可见桌面拿取、手内调整、灯座安装和放手；这些阶段的物体支撑条件不同，手型相似不代表完整状态相似。

## 逐条结果

接触子集以tactile_contact_points非空定义。指尖覆盖列均为RMS≤2cm。

| 真实示范 | 帧数 | 触觉接触帧 | 全部指尖覆盖 | 接触子集指尖覆盖 | 指尖距离中位/cm | 关节距离中位/° |
|---|---:|---:|---:|---:|---:|---:|
'''
for e in episodes:
 mask=r['episode']==e['episode'];mc=mask&co
 body+=f"| {e['id']} | {mask.sum()} | {mc.sum()} | {(f['rms_cm'][mask]<=2).mean()*100:.2f}% | {(f['rms_cm'][mc]<=2).mean()*100:.2f}% | {np.median(f['rms_cm'][mask]):.2f} | {np.median(n['rms_deg'][mask]):.2f} |\n"
body+='''
## 复核材料

- [覆盖率随门槛变化曲线](coverage_curves.png)
- [全部75条的距离时间图](all75_distance.png)
- [六条真实示范关键帧](real_overview.jpg)
- [六条示范距离曲线](example_timelines.png)
- [真实图像与最近参考手型叠加1](matches_1.png)
- [真实图像与最近参考手型叠加2](matches_2.png)
- [真实图像与最近参考手型叠加3](matches_3.png)
- [关节指标与逐条结果](coverage.json)
- [指尖指标与所用URDF/关节名](fingertip_coverage.json)
- [逐帧指尖最近参考编号/原始帧号](fingertips.npz)
- [逐帧关节最近参考编号/原始帧号](nearest.npz)
- [25个查询的暴力搜索核验及数值输入校验和](validation.json)

参考PKL插值60Hz，取[::2]还原30Hz的89,927个姿态点。npz的nearest_ref_frame为原始30Hz帧，乘2对应此前viewer帧；所有真实帧都查询，未只选成功或相近片段。真实文件路径/大小/修改时间和schema在sources.json；参考文件SHA256在reference_sources.json。

程序：extract_real.py使用dp环境读取Parquet；compare.py、fingertips.py、make_figures.py、render_matches.py使用decv2环境离线计算。均限制CPU线程，未运行GPU仿真或改动模型和训练数据。

关联代码：[真实数据转换约定](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/scripts/build_real_hand_dataset.py:7)、[当前采集容差](/home/carus/Data/exp_data/hydra_config.yaml:50)。
'''
(OUT/'review.md').write_text(body);print(OUT/'review.md')
