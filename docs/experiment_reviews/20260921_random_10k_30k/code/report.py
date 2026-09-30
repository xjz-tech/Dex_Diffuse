from pathlib import Path
import sys,json,hashlib
import numpy as np,zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];ROOT=R.parents[2]
OLD=R.parent/'20260920_10k_domain_guides'
names=['ordinary_1b','old10k','new10k_seed43','new30k_seed44']
labels=['普通1B','1B + 旧10k','1B + 新10k（抽样43）','1B + 新30k（抽样44）']
paths={n:(OLD if i<2 else R)/'evaluation'/n for i,n in enumerate(names)}
all_results={n:json.loads((p/'results.json').read_text()) for n,p in paths.items()}
assert all(v['complete'] and len(v['results'])==48 for v in all_results.values())
base=all_results['ordinary_1b']['results'];b=np.minimum(400,np.asarray([x['seconds'] for x in base]))
initial=np.load(paths['ordinary_1b']/'initial_state.npz')
physics=json.loads((paths['ordinary_1b']/'physical_parameters.json').read_text())
case_keys=['case','initialization','source_seed','source_env','mass_kg','object_friction','policy_seed']
pairing={};summary={};data=json.loads((R/'data_audit.json').read_text())
for n,p in paths.items():
 z=np.load(p/'initial_state.npz');checks={k:bool(np.array_equal(z[k],initial[k])) for k in initial.files}
 checks['physical_parameters']=json.loads((p/'physical_parameters.json').read_text())==physics
 checks['case_identity']=all(all(a[k]==bb[k] for k in case_keys) for a,bb in zip(all_results[n]['results'],base))
 cfg=json.loads((p/'config.json').read_text());refcfg=json.loads((paths['ordinary_1b']/'config.json').read_text())
 checks['native_protocol']=all(cfg[k]==refcfg[k] for k in ['native_overrides','control_dt','environment_seed','dedicated_native_prephysics_rng_seed','cap_steps','prior_ddim','guide_ddim','execution_steps'])
 assert all(checks.values()),(n,checks)
 pairing[n]=checks
 t=np.minimum(400,np.asarray([x['seconds'] for x in all_results[n]['results']]));d=t-b
 summary[n]={'n':48,'mean_capped_seconds':float(t.mean()),'median_seconds':float(np.median(t)),'min_seconds':float(t.min()),'max_seconds':float(t.max()),'ge20_pct':float(100*(t>=20).mean()),'ge80_pct':float(100*(t>=80).mean()),'cap400_pct':float(100*(t>=400).mean()),'mean_paired_gain_seconds':float(d.mean()),'median_paired_gain_seconds':float(np.median(d)),'gain_gt20_count':int((d>20).sum()),'loss_gt20_count':int((d< -20).sum()),'rescued20_count':int(((b<20)&(t>=20)).sum()),'harmed20_count':int(((b>=20)&(t<20)).sum()),'time_bin_counts':np.histogram(t,bins=[0,20,40,80,400,np.inf])[0].tolist(),'by_physics':{},'by_initialization':{},'by_policy_seed':{}}
 for m,mu in [(.044,1.312),(.044,2.572),(.240,1.312),(.240,2.572)]:
  mask=np.array([x['mass_kg']==m and x['object_friction']==mu for x in base]);summary[n]['by_physics'][f'{m*1000:g}g_mu{mu}']={'n':int(mask.sum()),'mean_seconds':float(t[mask].mean()),'mean_gain_seconds':float(d[mask].mean())}
 for key,field in [('by_initialization','initialization'),('by_policy_seed','policy_seed')]:
  for value in dict.fromkeys(x[field] for x in base):
   mask=np.array([x[field]==value for x in base]);summary[n][key][str(value)]={'n':int(mask.sum()),'mean_seconds':float(t[mask].mean()),'mean_gain_seconds':float(d[mask].mean())}
verification={'pairing':pairing,'source_data':{},'checkpoint_hashes':{}}
for n in names[2:]:
 dpath=ROOT/'data'/f'random_20260921_{n}';source=np.load(R/f'{n}_source_arrays.npz');z=zarr.open_group(str(dpath/'replay_buffer.zarr'),mode='r')
 checks={key:bool(np.array_equal(source[key],z['data/'+key][:])) for key in ['obs','action']};assert all(checks.values())
 verification['source_data'][n]=checks
 done=json.loads((ROOT/'runs'/f'random_20260921_{n}_train42/training_complete.json').read_text())
 model=json.loads((paths[n]/'model_manifest.json').read_text());assert done['sha256']==model['guide_sha256']
 verification['checkpoint_hashes'][n]=done
assert len({json.loads((paths[n]/'model_manifest.json').read_text())['prior_sha256'] for n in names})==1
verification['prior_unchanged']=True
(R/'verification.json').write_text(json.dumps(verification,indent=2))
(R/'evaluation_summary.json').write_text(json.dumps({'summary':summary,'results':all_results},indent=2))
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained');colors=['#777777','#2478b4','#d68120','#27916a'];eng=['1B','1B + old 10k','1B + new 10k','1B + new 30k']
for n,l,c in zip(names,eng,colors):
 t=np.asarray([x['seconds'] for x in all_results[n]['results']]);ts=np.sort(np.unique(np.r_[0,t,400]));surv=np.asarray([(t>=x).mean()*100 for x in ts]);ax[0].step(ts,surv,where='pre',label=l,color=c)
ax[0].set(xlim=(0,400),ylim=(0,102),xlabel='Simulation seconds',ylabel='Cases reaching time (%)',title='48 paired cases; native failure proxy');ax[0].legend(fontsize=8)
v=[summary[n]['mean_capped_seconds'] for n in names];bars=ax[1].bar(eng,v,color=colors);ax[1].bar_label(bars,fmt='%.1f s');ax[1].set(ylabel='Mean min(T, 400s)',ylim=(0,max(v)*1.2),title='Same 1B / DDIM4 / exec2 / guide scale25');ax[1].tick_params(axis='x',labelrotation=15);fig.savefig(R/'comparison.png',dpi=170);plt.close(fig)
lines=['# 随机重训10k、30k guide复现实验','','2026-09-21。1B固定；两份新子集从旧10k相同源库独立随机抽取，沿用完整轨迹抽样及200 epoch训练配方。不是新采集数据，不按评估结果选数据。','', '## 48个相同配置的结果','','平均值为截断在400秒的平均保持时间；≥400表示到达观察上限，不知道此后何时失败。环境原生failure仅为失败代理。','', '| 方法 | 平均保持/s | 中位/s | ≥20s | ≥80s | ≥400s | 相对1B平均变化/s |','|---|---:|---:|---:|---:|---:|---:|']
for n,l in zip(names,labels):
 s=summary[n];lines.append(f"| {l} | {s['mean_capped_seconds']:.2f} | {s['median_seconds']:.2f} | {s['ge20_pct']:.1f}% | {s['ge80_pct']:.1f}% | {s['cap400_pct']:.1f}% | {s['mean_paired_gain_seconds']:+.2f} |")
lines+=['','![保持分布](comparison.png)','','| 方法 | 比1B多保持>20s/48 | 比1B少保持>20s/48 | 1B<20s被救至≥20s | 1B≥20s变成<20s |','|---|---:|---:|---:|---:|']
for n,l in zip(names[1:],labels[1:]):
 s=summary[n];lines.append(f"| {l} | {s['gain_gt20_count']} | {s['loss_gt20_count']} | {s['rescued20_count']} | {s['harmed20_count']} |")
lines+=['','## 质量/摩擦分组均值','','每格12次。物理参数是评估引擎实际读回值。','','| 灯泡质量 / 摩擦 | 普通1B | 旧10k | 新10k | 新30k |','|---|---:|---:|---:|---:|']
for k in summary[names[0]]['by_physics']:lines.append('| '+k+' | '+' | '.join(f"{summary[n]['by_physics'][k]['mean_seconds']:.2f}" for n in names)+' |')
lines+=['','## 初始化分组均值','','每个初始化12次，仍是400秒截断均值。','','| 初始化 | 普通1B | 旧10k | 新10k | 新30k |','|---|---:|---:|---:|---:|']
for k in summary[names[0]]['by_initialization']:lines.append('| '+k+' | '+' | '.join(f"{summary[n]['by_initialization'][k]['mean_seconds']:.2f}" for n in names)+' |')
lines+=['','## 数据与训练','','子集名的10k/30k表示总行数，按episode留验证集，因此不是精确的9000/27000梯度训练行。','', '| 项目 | 旧10k | 新10k | 新30k |','|---|---:|---:|---:|','| 抽样seed | 42 | 43 | 44 |','| 总行数 | 9999 | 10000 | 29994 |','| 总轨迹数 | 20 | 25 | 51 |']
for key,label,oldvalue in [('train_rows','梯度训练行数',8910),('validation_rows','验证行数',1089),('train_episodes','训练轨迹数',18),('train_target_demo_count','训练目标demo种类',71),('zero_residual_train_rows','零残差起始帧',18)]:
 lines.append(f'| {label} | {oldvalue} | {data[names[2]][key]} | {data[names[3]][key]} |')
lines.append('| 前3条轨迹的训练窗口占比 | 55.63% | '+ ' | '.join(f"{data[n]['top3_train_episode_pct']:.2f}%" for n in names[2:])+' |')
lines.append('| 优化器更新次数 | 3600 | '+' | '.join(str(verification['checkpoint_hashes'][n]['global_step']) for n in names[2:])+' |')
lines+=['','对应测试demo在梯度训练窗口中的目标帧数（不等价于覆盖同一初始手型/同一物理状态）：','','| 目标demo | 旧10k | 新10k | 新30k |','|---|---:|---:|---:|']
for k,v in [('79',0),('82',62),('94',0),('131',44)]:lines.append(f'| {k} | {v} | '+ ' | '.join(str(data[n]['train_target_demo_counts'].get(k,0)) for n in names[2:])+' |')
lines+=['','三份子集之间无源episode重叠。训练和验证划分seed均为42；两个新模型均从头训练200 epochs。架构、EMA、优化器、batch512等参数沿用旧10k。30k每轮更新更多，因此比较包含数据量与优化预算共同变化，不能声称只测了数据量效应。normalizer也沿用原配方，拟合包括验证行在内的全部子集。','','新子集训练目标demo覆盖更广、对少数长轨迹依赖更低，但这些仅描述数据；不能单独证明它们导致保持性能变化。原101个HDF5分片没有质量/摩擦/尺寸标签，仍然无法给出这三份子集的真实物理参数占比。','', '## 协议与边界','','4个初始化×2质量（44g/240g）×2摩擦（1.312/2.572）×3策略噪声seed（8/19/25）=48。初始化为demo079、082、094及归档真机手型；真机手型继承上轮固定的其他物理参数，不能冒充历史seed50所有参数完全复现。','','所有DP测试沿用xjz_test原生协议：位置阈值0.05m、指尖0.1m、旋转180°、立即失效位置0.15m、FailureToleranceScale10000、fixedToleranceSteps20000、resetOnReachGoal=false、跨轨迹目标概率0.3、demo000–149。位置参考当前示范目标中物体相对手腕位置。prior/guide DDIM=4、exec=2、scale=25、引导可执行动作前2步。普通1B使用普通DDIM；未跑scale0。','','普通1B、旧10k沿用20260920完整评估结果；本次新模型重跑。已逐项确认初始状态、实际物理参数、case配置及原生协议与参照一致。复用了同一评估实现和配对扰动随机流，目标切换仍由原生任务和闭环状态决定。','','每个新子集仅训练一个模型；只有4个有针对性的初始化，不是全部demo总体。之前相同初态/首动作的重复运行在240s截断下曾有21/48个case相差超过20s，故小差异和单个视频不能作为稳定可复现证据。本次检验的是两份新抽样上的性能方向/幅度，未建立多训练seed的统计结论。','', '## 文件','','- data_audit.json：每条轨迹、示范覆盖、权重、数据哈希；物理标签缺失明确记null。','- verification.json：源数据逐元素核对、所有初态/物理参数配对、模型哈希。','- evaluation_summary.json：全量逐case结果与分组统计。','- code/、训练配置和日志保留，原1B/旧10k模型不变。']
(R/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
