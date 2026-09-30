from pathlib import Path
import json,numpy as np
OUT=Path(__file__).resolve().parents[1];d=json.loads((OUT/'analysis.json').read_text())
lines=['# 历史3000环境初始化恢复与最短10%分析','','## 范围与验证','','恢复 seed8/42/19/25，每批3000环境；使用历史1B+10k guide scale25、DDIM4/4、执行2步、fresh noise、原生xjz失败判据。只将总观察步数改为150（5秒），轨迹上限及失败容忍参数仍按原值。没有修复历史质量缓存缺陷，没有调整物理参数。参数读取不消耗CUDA随机数。','', '| seed | 旧DR汇总完全匹配 | 旧前5秒失败数 | 复跑失败数 | env及失败步数完全一致 |','|---:|---|---:|---:|---:|']
for s in d:lines.append(f"| {s['seed']} | {s['historical_initial_dr_exact_text_match']} | {s['historical_failure_le5s_n']} | {s['replay_failure_le5s_n']} | {s['exact_failure_step_n']} |")
lines+=['','## 1B+10k guide：最短10%与其余90%均值','','每批按第300短时长为阈值，并保留并列；以下所有物理参数均来自初始化后物理引擎API读回。统计标签来自历史完整400秒实验，不使用本次5秒截断值计算最短10%。','', '| seed | 分组 | 质量/g | 物体摩擦 | 尺寸倍数 |','|---:|---|---:|---:|---:|']
for s in d:
 g=next(g for g in s['groups'] if g['prior']=='1B' and g['scale']==25)
 for prefix,label in [('bottom','最短10%'),('rest','其余90%')]:
  lines.append(f"| {s['seed']} | {label} | {g['stats']['mass'][prefix+'_mean']*1000:.2f} | {g['stats']['friction'][prefix+'_mean']:.3f} | {g['stats']['scale'][prefix+'_mean']:.5f} |")
lines+=['','## 各历史配置的标准化均值差','','(最短10%均值 − 其余90%均值) / 全批参数标准差；用于比较差异方向和大小，不是因果效应。scale0保留历史guided-DDIM标签；它和mixed没有额外做轨迹复跑，参数映射依据相同初始化流程及DR汇总。','', '| seed | prior | guide scale | mass | friction | size |','|---:|---|---:|---:|---:|---:|']
for s in d:
 for g in s['groups']:lines.append(f"| {s['seed']} | {g['prior']} | {g['scale']} | "+' | '.join(f"{g['stats'][k]['standardized_mean_difference']:.3f}" for k in ['mass','friction','scale'])+' |')
lines+=['','## 参数分位数与早失败比例','','以下比例表示属于历史最短10%的概率；不是5秒内失败率。低/高四分位在每个seed内定义。','','| seed | 参数 | 最低25%中的早失败比例 | 最高25%中的早失败比例 |','|---:|---|---:|---:|']
for s in d:
 g=next(g for g in s['groups'] if g['prior']=='1B' and g['scale']==25)
 for k in ['mass','friction','scale']:
  v=g['stats'][k];lines.append(f"| {s['seed']} | {k} | {v['bottom10_rate_low_quartile']*100:.2f}% | {v['bottom10_rate_high_quartile']*100:.2f}% |")
lines+=['','## 引擎读回与旧日志的区别','','旧DR日志是在setter调用前打印准备写入的属性，不能保证所有字段最终生效。本次 get_actor_rigid_shape_properties 读回的 rolling_friction/torsion_friction 均为0，不能将旧日志非零值当作已确认生效的摩擦。质量、普通摩擦等使用本次读回值；cached_mass另列，保留旧实现用于外力计算。','','## 初始示范分组（四seed合并）','','以下仅1B+10k guide，每个示范每seed20例，合计80例；分组率用于探索，不作筛选后显著性结论。','']
agg={}
for s in d:
 g=next(g for g in s['groups'] if g['prior']=='1B' and g['scale']==25)
 for v in g['demo_rates']:
  x=agg.setdefault(v['demo'],[0,0]);x[0]+=v['n'];x[1]+=v['bottom_n']
for k,(n,b) in sorted(agg.items(),key=lambda kv:kv[1][1]/kv[1][0],reverse=True)[:10]:lines.append(f'- demo{k:03d}: {b}/{n} ({b/n*100:.1f}%) 落入最短10%。')
lines+=['','## 局限','','5秒轨迹匹配验证覆盖短时失败段，不代表400秒逐帧复现；更重、低摩擦、小尺寸是组间相关性，并未控制初始接触、示范和参数间相关。应通过固定同一初态、单独改变一个物理量验证因果。所有原始初始化数组在各 seed 的 initial_parameters.npz，完整分组数据在 analysis.json。','']
(OUT/'report.md').write_text('\n'.join(lines))
print('\n'.join(lines[:25]))
