from pathlib import Path
import json
p=Path(__file__).resolve().parent;d=json.loads((p/'analysis.json').read_text())
lines=['# 历史1B基线初始化风险与10k guide改善阶段','','## 口径','','历史20260912批次，seed42/8/19/25，各3000首轮，共12000个配对env。基线为历史guided-DDIM scale0，不是最近单环境实验的普通DDIM；guide为10k scale25，DDIM4/4、执行2步。物理/初态由复现恢复；四seed的guide前5秒1518个失败事件编号与步数全匹配，基线未额外复跑。结论不可直接外推到普通DDIM。','','“快速失败”按各seed基线最短10%（第300例为阈值，含并列，共1204例）。原生failure是失败代理；没有逐例独立标记接触丢失。','','## 初始化特征','','| 特征 | 最短10%均值 | 其余90%均值 | 最低25%中的快速失败率 | 最高25%中的快速失败率 |','|---|---:|---:|---:|---:|']
labels={'mass':('质量/g',1000),'friction':('物体摩擦系数',1),'scale':('尺寸倍数',1),'index_flexion':('食指屈曲关节角之和/rad',1),'ring_flexion':('无名指屈曲关节角之和/rad',1),'thumb_flexion':('拇指屈曲关节角之和/rad',1),'object_local_x':('物体相对腕部局部x/cm',100),'frame_fraction':('起始帧/原始示范长度',1)}
for k,(label,f) in labels.items():
 v=d['feature_statistics'][k];lines.append(f"| {label} | {v['bottom_mean']*f:.3f} | {v['rest_mean']*f:.3f} | {v['low_quartile_early_rate']*100:.2f}% | {v['high_quartile_early_rate']*100:.2f}% |")
lines+=['','屈曲量是URDF对应FE/PIP/DIP/IP关节角之和，作为手型描述，不等同于接触力或抓持包络；没有据此认定已抓稳/未抓稳。物体局部坐标由保存的腕部四元数逆旋转得到，不把x轴擅自解释为靠拇指/掌外侧。分位率为探索性组间关联，未控制示范、初始接触或参数相关性。四个seed中食指/无名指屈曲更小、拇指屈曲更大、局部x更大的方向一致；起始帧阶段没有一致趋势。','','## 快速失败较集中的示范','','| demo | 80例中的快速失败数 | 比例 | 四seed分别的数量 |','|---:|---:|---:|---|']
for v in sorted(d['demos'],key=lambda x:x['baseline_bottom_pct'],reverse=True)[:8]:lines.append(f"| {v['demo']:03d} | {v['baseline_bottom_n']} | {v['baseline_bottom_pct']:.2f}% | {v['early_seed_counts']} |")
lines+=['','以上是事后筛选的高风险示范，不作未经多重比较校正的显著性结论。','','## 从什么时候保持率改善','','![保持率曲线](survival.png)','','| 仿真时间/s | 历史基线保持率 | guide保持率 | 差值/百分点 | 各seed差值/百分点 |','|---:|---:|---:|---:|---|']
for v in d['survival']:lines.append(f"| {v['time']} | {v['baseline_pct']:.2f}% | {v['guide_pct']:.2f}% | {v['difference_pp']:+.2f} | "+', '.join(f'{x:+.2f}' for x in v['seed_differences_pp'])+' |')
lines+=['',f"合并曲线从约{d['sustained_positive_from_seconds']:.1f}秒开始持续领先；约16秒超过5个百分点，约29.5秒超过10个百分点。这是观察到的曲线分离时点，不是guide延后开启：guide从首个推理请求即启用。",'',f"合并中位保持：基线{d['pooled_baseline_median']:.2f}s，guide{d['pooled_guide_median']:.2f}s；400秒截断均值增加{d['capped_mean_gain']:.2f}s。400秒仍未失败是右删失，不是真实失败时间。",'','## 改善集中在哪些案例','','按历史基线实际保持时长分组，再查看同seed同env的guide结果；这是事后分组描述，受随机采样及向均值回归影响，不代表仅凭初态就能预测获益。','','| 基线时长区间/s | 数量 | 基线中位/s | guide中位/s | guide保持更久比例 | guide到400s比例 |','|---|---:|---:|---:|---:|---:|']
for v in d['baseline_time_cohorts']:
 lo,hi=v['baseline_interval'];lines.append(f"| ({lo}, {min(hi,400)}] | {v['n']} | {v['baseline_median']:.2f} | {'≥' if v['guide_median']==400 else ''}{v['guide_median']:.2f} | {v['guide_longer_pct']:.2f}% | {v['guide_reaches400_pct']:.2f}% |")
lines+=['','在1–5秒总体保持率稍差、10秒后四seed均领先，与“guide主要延长中后段保持”相符；无法仅用episode结果证明其具体控制机制（如减小抖动、修正动作偏差或抗滑移），这需要动作和接触时序对照。','','数据：analysis.json，paired_features.npz；代码：analyze.py。','']
(p/'report.md').write_text('\n'.join(lines))
print('bottom_n',d['bottom10_n'])
print(p/'report.md')
