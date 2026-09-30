from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];ROOT=R.parents[2];OLD=R.parent/'20260920_10k_domain_guides'
names=['ordinary_1b','prior1b_guide10k','ordinary_10b','guide_1b','guide_old10k']
labels=['普通1B','1B + 旧10k guide','普通10B','10B + 1B guide','10B + 旧10k guide']
dirs={'ordinary_1b':OLD/'evaluation/ordinary_1b','prior1b_guide10k':OLD/'evaluation/old10k',**{n:R/'evaluation'/n for n in names[2:]}}
raw={n:json.loads((dirs[n]/'results.json').read_text()) for n in names}
assert all(v['complete'] and len(v['results'])==48 for v in raw.values())
base=raw['ordinary_10b']['results'];base_t=np.minimum(400,np.asarray([x['seconds'] for x in base]));refinitial=np.load(dirs['ordinary_1b']/'initial_state.npz');refphysical=json.loads((dirs['ordinary_1b']/'physical_parameters.json').read_text());refcfg=json.loads((dirs['ordinary_1b']/'config.json').read_text())
identity=['case','initialization','source_seed','source_env','mass_kg','object_friction','policy_seed']
checks={};summary={}
for n in names:
 p=dirs[n];rs=raw[n]['results'];z=np.load(p/'initial_state.npz');c={k:bool(np.array_equal(z[k],refinitial[k])) for k in refinitial.files};c['physical_parameters']=json.loads((p/'physical_parameters.json').read_text())==refphysical;c['case_identity']=all(all(x[k]==y[k] for k in identity) for x,y in zip(rs,base))
 cfg=json.loads((p/'config.json').read_text());c['native_protocol']=all(cfg[k]==refcfg[k] for k in ['native_overrides','control_dt','environment_seed','dedicated_native_prephysics_rng_seed','cap_steps','prior_ddim','guide_ddim','execution_steps','guidance_steps'])
 assert all(c.values()),(n,c);checks[n]=c
 t=np.minimum(400,np.asarray([x['seconds'] for x in rs]));d=t-base_t
 s={'n':48,'mean_capped_s':float(t.mean()),'median_s':float(np.median(t)),'ge20_count':int((t>=20).sum()),'ge80_count':int((t>=80).sum()),'cap400_count':int((t>=400).sum()),'mean_paired_gain_vs_ordinary10b_s':float(d.mean()),'median_paired_gain_vs_ordinary10b_s':float(np.median(d)),'gain_gt20_count':int((d>20).sum()),'loss_gt20_count':int((d< -20).sum()),'rescued20_count':int(((base_t<20)&(t>=20)).sum()),'harmed20_count':int(((base_t>=20)&(t<20)).sum()),'by_initialization':{},'by_physics':{},'by_policy_seed':{}}
 for init in dict.fromkeys(x['initialization'] for x in base):
  m=np.asarray([x['initialization']==init for x in base]);s['by_initialization'][init]={'n':int(m.sum()),'mean_s':float(t[m].mean()),'mean_gain_s':float(d[m].mean())}
 for mass,mu in [(.044,1.312),(.044,2.572),(.240,1.312),(.240,2.572)]:
  m=np.asarray([x['mass_kg']==mass and x['object_friction']==mu for x in base]);s['by_physics'][f'{mass*1000:g}g_mu{mu}']={'n':int(m.sum()),'mean_s':float(t[m].mean()),'mean_gain_s':float(d[m].mean())}
 for seed in [8,19,25]:
  m=np.asarray([x['policy_seed']==seed for x in base]);s['by_policy_seed'][str(seed)]={'n':int(m.sum()),'mean_s':float(t[m].mean()),'mean_gain_s':float(d[m].mean())}
 summary[n]=s
manifest={n:json.loads((dirs[n]/'model_manifest.json').read_text()) for n in names}
assert len({manifest[n]['prior_sha256'] for n in names[2:]})==1
assert manifest['guide_1b']['guide_sha256']==manifest['ordinary_1b']['prior_sha256']
assert manifest['guide_old10k']['guide_sha256']==manifest['prior1b_guide10k']['guide_sha256']
assert all(manifest[n]['prior_ddim_steps']==4 and manifest[n]['guide_ddim_steps']==4 and manifest[n]['execution_steps']==2 and manifest[n]['guidance_steps']==2 and manifest[n]['guide_scale']==25 for n in names[2:])
verification={'case_pairing':checks,'model_manifest':manifest,'10b_prior_identical_all_new_arms':True,'1b_as_guide_hash_matches_prior_1b':True,'old10k_guide_hash_matches_previous':True,'evaluator_sha256':hashlib.sha256((R/'code/eval.py').read_bytes()).hexdigest(),'previous_evaluator_sha256':hashlib.sha256((OLD/'code/eval.py').read_bytes()).hexdigest()};assert verification['evaluator_sha256']==verification['previous_evaluator_sha256']
(R/'verification.json').write_text(json.dumps(verification,indent=2));(R/'evaluation_summary.json').write_text(json.dumps({'summary':summary,'results':raw},indent=2))
fig,ax=plt.subplots(1,2,figsize=(12,4.1),layout='constrained');colors=['#999999','#4d79bb','#454545','#dc842c','#29936e']
for n,l,c in zip(names,['1B','1B + 10k','10B','10B + 1B','10B + 10k'],colors):
 t=np.minimum(400,np.asarray([x['seconds'] for x in raw[n]['results']]));ts=np.sort(np.unique(np.r_[0,t,400]));surv=np.asarray([(t>=x).mean()*100 for x in ts]);ax[0].step(ts,surv,where='pre',label=l,color=c)
ax[0].set(xlim=(0,400),ylim=(0,102),xlabel='Simulation seconds',ylabel='Cases reaching time (%)',title='48 identical cases; native failure proxy');ax[0].legend(fontsize=8)
bars=ax[1].bar(['1B','1B+10k','10B','10B+1B','10B+10k'],[summary[n]['mean_capped_s'] for n in names],color=colors);ax[1].bar_label(bars,fmt='%.1f');ax[1].set(ylabel='Mean min(T,400s)',title='DDIM4/4, exec2, guide scale25');ax[1].tick_params(axis='x',labelrotation=15);fig.savefig(R/'comparison.png',dpi=170);plt.close(fig)
g=summary['guide_1b'];b=summary['ordinary_10b'];k=summary['guide_old10k'];
lines=['# 1B guide → 10B prior','', '2026-09-21。方向为10B主策略、1B guide。按相同48个初始化/物理/噪声配置评估普通10B、10B+1B guide、10B+旧10k guide；1B相关两行复用上一轮已核对相同初态和物理参数的结果。','', '## 结果','', '平均值截断于400秒；≥400表示到达观察上限。原生failure是评估失败代理，不独立证明物理掉落。','', '| 方法 | 平均保持/s | 中位/s | ≥20s | ≥80s | ≥400s | 相对普通10B平均变化/s |','|---|---:|---:|---:|---:|---:|---:|']
for n,l in zip(names,labels):
 s=summary[n];gain='—' if n.startswith('ordinary_1b') or n=='prior1b_guide10k' else f"{s['mean_paired_gain_vs_ordinary10b_s']:+.2f}"
 lines.append(f"| {l} | {s['mean_capped_s']:.2f} | {s['median_s']:.2f} | {s['ge20_count']}/48 | {s['ge80_count']}/48 | {s['cap400_count']}/48 | {gain} |")
lines+=['',f'![保持分布]({R}/comparison.png)','','10B主策略的成对得失：','','| guide | 比普通10B多保持>20s | 比普通10B少保持>20s | 普通10B<20s→guide≥20s | 普通10B≥20s→guide<20s |','|---|---:|---:|---:|---:|']
for n,l in [('guide_1b','1B'),('guide_old10k','旧10k')]:
 s=summary[n];lines.append(f"| {l} | {s['gain_gt20_count']} | {s['loss_gt20_count']} | {s['rescued20_count']} | {s['harmed20_count']} |")
lines+=['','## 初始化分组','','每组12个case，均为400秒截断均值。','','| 初始化 | 普通1B | 1B+旧10k | 普通10B | 10B+1B | 10B+旧10k |','|---|---:|---:|---:|---:|---:|']
for init in b['by_initialization']:lines.append('| '+init+' | '+' | '.join(f"{summary[n]['by_initialization'][init]['mean_s']:.2f}" for n in names)+' |')
lines+=['','## 质量、摩擦分组','','每格12个case，物理参数为引擎实际读回值。','','| 质量 / 摩擦 | 普通1B | 1B+旧10k | 普通10B | 10B+1B | 10B+旧10k |','|---|---:|---:|---:|---:|---:|']
for cell in b['by_physics']:lines.append('| '+cell+' | '+' | '.join(f"{summary[n]['by_physics'][cell]['mean_s']:.2f}" for n in names)+' |')
lines+=['','## 策略噪声seed分组','','每个seed16个case。','','| seed | 普通1B | 1B+旧10k | 普通10B | 10B+1B | 10B+旧10k |','|---:|---:|---:|---:|---:|---:|']
for seed in ['8','19','25']:lines.append('| '+seed+' | '+' | '.join(f"{summary[n]['by_policy_seed'][seed]['mean_s']:.2f}" for n in names)+' |')
lines+=['','## 条件与解释边界','','固定4初始化（demo079、082、094、归档真机手型）×2质量（44g、240g）×2摩擦（1.312、2.572）×3策略噪声seed（8、19、25）；每方法48组，400秒上限。所有DP仿真统一沿用xjz_test原生任务：0.05m物体相对手腕误差、0.1m指尖误差、180°旋转误差、0.15m立即失效位置误差、FailureToleranceScale10000、fixedToleranceSteps20000、resetOnReachGoal=false、跨轨迹目标概率0.3、demo000–149。目标更新和容忍步数不另写判定。','', '普通主策略采用真正普通DDIM，不使用guided scale=0。10B/1B/旧10k均使用EMA检查点，66维观测/22维动作；prior和guide各DDIM4、每次执行2步、引导前2个可执行动作、scale25。guide先生成物理角度动作，再按10B主策略的动作normalizer进入引导公式。','', '本次三个新方法完整重跑；上轮1B两行可作为同48配置对照，初态含CUDA随机状态和实际质量/摩擦/手部参数逐项完全一致。10B先前12000组历史评估不能直接和这些400秒均值混用。','', '每个方法的每个case只运行一次；这些4个初始手型是有针对性的案例，不代表150个demo总体。闭环轨迹之前观察到重复运行差异，故小幅单例或总体差异不能当成稳定性证明。完整逐case结果、首动作和视频保留以供复查。','', '## 文件','','- verification.json：初态、物理参数、原生协议和模型哈希核对。','- evaluation_summary.json：全部逐case结果及分组汇总。','- evaluation/：原始视频、首次动作、状态、物理读回和完整rollout。']
(R/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({n:{k:v for k,v in summary[n].items() if not k.startswith('by_')} for n in names},indent=2))
