from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];root=r.parents[2];names=['ordinary_1b','old10k','light_low','light_high','heavy_low','heavy_high','balanced'];allrows=[];models=[];pairs=[]
base=r/'evaluation/ordinary_1b'
for name in names:
 p=r/'evaluation'/name
 if not (p/'results.json').exists():continue
 v=json.loads((p/'results.json').read_text());assert v['complete'] and len(v['results'])==48;rows=v['results'];assert all(not x['error_flag'] for x in rows)
 allrows+=rows
 z=np.load(p/'initial_state.npz');b=np.load(base/'initial_state.npz');eq={k:bool(np.array_equal(z[k],b[k])) for k in z.files};assert all(eq.values()),(name,eq)
 assert (p/'physical_parameters.json').read_text()==(base/'physical_parameters.json').read_text()
 cfg=json.loads((p/'config.json').read_text());assert cfg['prior_ddim']==cfg['guide_ddim']==4 and cfg['execution_steps']==2 and cfg['native_overrides']['fixedToleranceSteps']==20000
 pairs.append({'method':name,'initial_state_and_rng_equal':eq,'physical_equal':True})
 t=np.array([x['seconds'] for x in rows]);models.append({'method':name,'n':48,'mean_capped_seconds':float(np.minimum(t,400).mean()),'median_seconds':float(np.median(t)),'survival20_pct':float((t>=20).mean()*100),'survival80_pct':float((t>=80).mean()*100),'survival400_pct':float((t>=400).mean()*100),'failure_le5_pct':float((t<=5).mean()*100)})
if not allrows:raise SystemExit('no completed evaluations')
groups=[]
for name in [x['method'] for x in models]:
 for mass in [.044,.240]:
  for mu in [1.312,2.572]:
   rows=[x for x in allrows if x['method']==name and x['mass_kg']==mass and x['object_friction']==mu];t=np.array([x['seconds'] for x in rows]);brows=[x for x in allrows if x['method']=='ordinary_1b' and x['mass_kg']==mass and x['object_friction']==mu];bt=np.array([x['seconds'] for x in brows]);delta=np.minimum(t,400)-np.minimum(bt,400)
   groups.append({'method':name,'mass_g':mass*1000,'mu':mu,'n':len(rows),'mean_capped_seconds':float(np.minimum(t,400).mean()),'survival20_pct':float((t>=20).mean()*100),'survival400_pct':float((t>=400).mean()*100),'paired_mean_gain_s':float(delta.mean()),'gain_gt20_count':int((delta>20).sum()),'loss_gt20_count':int((delta< -20).sum())})
complete=len(models)==7
(r/'evaluation_summary.json').write_text(json.dumps({'complete':complete,'models':models,'physical_groups':groups,'pairs':pairs,'all_results':allrows},indent=2))
fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');labels=[x['method'].replace('_','\n') for x in models];axs[0].bar(range(len(models)),[x['mean_capped_seconds'] for x in models]);axs[0].set_xticks(range(len(models)),labels);axs[0].set_ylabel('Mean min(T,400s)');axs[0].set_ylim(0,400);axs[0].set_title('48 matched cases per method')
vals=np.array([x['paired_mean_gain_s'] for x in groups]).reshape(len(models),4);limit=max(1,float(abs(vals).max()));im=axs[1].imshow(vals,vmin=-limit,vmax=limit,cmap='RdBu');axs[1].set_yticks(range(len(models)),[x['method'] for x in models]);axs[1].set_xticks(range(4),['44g\nmu1.312','44g\nmu2.572','240g\nmu1.312','240g\nmu2.572']);axs[1].set_title('Paired mean gain vs ordinary 1B (seconds)')
for i in range(len(models)):
 for j in range(4):axs[1].text(j,i,f'{vals[i,j]:+.1f}',ha='center',va='center',fontsize=9,color='white' if abs(vals[i,j])>.55*limit else 'black')
fig.colorbar(im,ax=axs[1]);fig.savefig(r/'comparison.png',dpi=160);plt.close(fig)
lines=['# 固定1B，不同10k训练分布的guidance结果','',f'状态：{len(models)}/7个方法已完成。新仿真对照使用普通DDIM 1B作为基线，未使用guided scale0。','', '| 方法 | n | 400秒截断均值/s | ≥20秒 | ≥80秒 | ≥400秒 | ≤5秒失败 |','|---|---:|---:|---:|---:|---:|---:|']
for x in models:lines.append(f"| {x['method']} | 48 | {x['mean_capped_seconds']:.2f} | {x['survival20_pct']:.1f}% | {x['survival80_pct']:.1f}% | {x['survival400_pct']:.1f}% | {x['failure_le5_pct']:.1f}% |")
lines+=['',f'![结果图]({r / "comparison.png"})','','| 方法 | 质量/g | 摩擦 | n | 截断均值/s | 相比1B差值/s |','|---|---:|---:|---:|---:|---:|']
for x in groups:lines.append(f"| {x['method']} | {x['mass_g']:.0f} | {x['mu']} | {x['n']} | {x['mean_capped_seconds']:.2f} | {x['paired_mean_gain_s']:+.2f} |")
lines+=['','五个新模型的实际分布见guide_registry.json与每个dataset的distribution_manifest.json。新模型均从头训练200epoch、seed42；总10k=9000训练+1000验证。混合组训练/验证均各域25%。这是一轮数据分布对照，不代表已跨训练seed复现。','','测试48种组合：4初态×4物理点×3策略seed；这4个初态是诊断案例，不能当总体随机样本。各方法初态、物理参数、prior噪声及原生prephysics扰动随机流配对；目标切换保留原生状态依赖。400秒为观察截断，不是真实失败时长。新旧10k在片段构成、成功筛选和实际质量缓存处理上不同，因此新旧差异不能全部归因于质量/摩擦分布；五个新模型之间有更严格的匹配。','','原生xjz判据，DDIM4/4、exec2、guide25引导前2步；默认0.05m/0.1m/180deg/0.15m及10000/20000容忍参数保持，示范000–149。没有改动1B或RL训练默认值。','','逐案例结果见evaluation_summary.json；录像在各evaluation方法目录。']
(r/'results_report.md').write_text('\n'.join(lines))
print(json.dumps(models,indent=2))
