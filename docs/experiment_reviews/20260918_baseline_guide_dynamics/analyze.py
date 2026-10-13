from pathlib import Path
import json,pickle
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
OLD=ROOT/'eval/hold_runs/20260912_fresh_noise_3000_seed42_8_19_25';REST=ROOT/'docs/experiment_reviews/20260918_historical_3k_restore'
seeds=[42,8,19,25];arrays=[];zs=[]
for s in seeds:
 pair=[]
 for scale in [0,25]:
  p=OLD/f'seed{s}_1B_scale{scale}';records=[json.loads(x) for x in next(p.glob('*.jsonl')).read_text().splitlines() if x]
  t=np.zeros(3000)
  for x in records:t[x['env']]=x['length']/30
  pair.append(t)
 arrays.append(pair);zs.append(np.load(REST/f'seed{s}_scale25/initial_parameters.npz'))
b=np.concatenate([a[0] for a in arrays]);g=np.concatenate([a[1] for a in arrays]);seed=np.repeat(seeds,3000)
z={k:np.concatenate([a[k] for a in zs]) for k in ['q','wrist','object','demo','frame','mass','friction','scale','restitution','force_prob']}
bottom=np.concatenate([a[0]<=np.sort(a[0])[299] for a in arrays]);demo=z['demo'].astype(int)
lengths=[]
for i in range(150):
 p=Path('/home/carus/Program/dex-controller/data/retargeting/NOKOV-v3/mano2sharpa_rh/bulb2')/f'{i:03d}.pkl'
 d=pickle.load(p.open('rb'));lengths.append(len(d['opt_dof_pos']))
phase=z['frame']/np.maximum(np.array(lengths)[demo]-1,1);assert phase.min()>=0 and phase.max()<=1
rel=Rotation.from_quat(z['wrist'][:,3:7]).inv().apply(z['object'][:,:3]-z['wrist'][:,:3])
features={k:z[k] for k in ['mass','friction','scale','restitution','force_prob']}
features.update(frame_fraction=phase,object_distance=np.linalg.norm(rel,axis=1),object_local_x=rel[:,0],object_local_y=rel[:,1],object_local_z=rel[:,2])
for name,idx in {'index_flexion':[0,2,3],'middle_flexion':[4,6,7],'ring_flexion':[13,15,16],'pinky_flexion':[9,11,12],'thumb_flexion':[17,19,21]}.items():features[name]=z['q'][:,idx].sum(axis=1)
featurestats={}
for k,v in features.items():
 q=np.quantile(v,[.25,.75]);featurestats[k]={'bottom_mean':float(v[bottom].mean()),'rest_mean':float(v[~bottom].mean()),'smd':float((v[bottom].mean()-v[~bottom].mean())/np.std(v)),'low_quartile_threshold':float(q[0]),'high_quartile_threshold':float(q[1]),'low_quartile_early_rate':float(bottom[v<=q[0]].mean()),'high_quartile_early_rate':float(bottom[v>=q[1]].mean())}
milestones=[]
for t in [1,2,3,5,10,15,20,30,60,120,200,300,400]:
 # >=400 represents holding to cap; other times use strictly greater survival.
 sb=b>=t if t==400 else b>t;sg=g>=t if t==400 else g>t
 delta=sg.astype(float)-sb.astype(float);se=delta.std(ddof=1)/len(delta)**.5
 milestones.append(dict(time=t,baseline_pct=sb.mean()*100,guide_pct=sg.mean()*100,difference_pp=delta.mean()*100,approx_paired95ci_pp=[(delta.mean()-1.96*se)*100,(delta.mean()+1.96*se)*100],seed_differences_pp=[(((a[1]>=t).mean()-(a[0]>=t).mean()) if t==400 else ((a[1]>t).mean()-(a[0]>t).mean()))*100 for a in arrays],gained=int((sg&~sb).sum()),lost=int((sb&~sg).sum())))
diff_times=np.arange(1,12001)/30
curves=[]
for a in arrays:curves.append(np.array([[((a[j]>=t).mean() if t==400 else (a[j]>t).mean()) for t in diff_times] for j in [0,1]]))
curves=np.array(curves);mean=curves.mean(axis=0);delta=mean[1]-mean[0]
# Descriptive crossings, not post-hoc significance declarations.
cross={}
for pp in [0,1,5,10,20,30]:
 hit=np.flatnonzero(delta>pp/100);cross[str(pp)]=float(diff_times[hit[0]]) if len(hit) else None
last_nonpositive=np.flatnonzero(delta<=0);sustained_positive=float(diff_times[last_nonpositive[-1]+1]) if last_nonpositive[-1]+1<len(diff_times) else None
bins=[0,1,5,30,120,400.001];cohorts=[]
for lo,hi in zip(bins[:-1],bins[1:]):
 m=(b>lo)&(b<=hi);v=g[m]-b[m]
 cohorts.append(dict(baseline_interval=[lo,hi],n=int(m.sum()),baseline_median=float(np.median(b[m])),guide_median=float(np.median(g[m])),paired_capped_gain_mean=float(v.mean()),paired_capped_gain_median=float(np.median(v)),guide_longer_pct=float((v>0).mean()*100),guide_shorter_pct=float((v<0).mean()*100),guide_reaches400_pct=float((g[m]>=400).mean()*100)))
demos=[]
for i in range(150):
 m=demo==i
 demos.append(dict(demo=i,n=int(m.sum()),baseline_bottom_n=int(bottom[m].sum()),baseline_bottom_pct=float(bottom[m].mean()*100),baseline_median=float(np.median(b[m])),guide_median=float(np.median(g[m])),capped_mean_gain=float((g[m]-b[m]).mean()),baseline_cap_pct=float((b[m]>=400).mean()*100),guide_cap_pct=float((g[m]>=400).mean()*100),early_seed_counts=[int((bottom&(demo==i)&(seed==s)).sum()) for s in seeds]))
np.savez_compressed(OUT/'paired_features.npz',baseline_seconds=b,guide_seconds=g,seed=seed,demo=demo,frame=z['frame'],baseline_bottom10=bottom,**features)
result=dict(baseline='Historical 1B guided-DDIM scale0, not ordinary DDIM',n=12000,bottom10_n=int(bottom.sum()),feature_statistics=featurestats,survival=milestones,first_descriptive_crossings_seconds=cross,sustained_positive_from_seconds=sustained_positive,baseline_time_cohorts=cohorts,demos=demos,pooled_baseline_median=float(np.median(b)),pooled_guide_median=float(np.median(g)),capped_mean_gain=float((g-b).mean()))
(OUT/'analysis.json').write_text(json.dumps(result,indent=2))
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax in axes:
 ax.plot(diff_times,100*mean[0],label='1B historical scale=0');ax.plot(diff_times,100*mean[1],label='1B + 10k guide scale=25');ax.set_xlabel('Simulated time (s)');ax.set_ylabel('Survival (%)');ax.grid(alpha=.2)
axes[0].set_xlim(0,400);axes[0].set_ylim(0,100);axes[0].legend();axes[0].set_title('All 12,000 first episodes')
axes[1].set_xlim(0,30);axes[1].set_ylim(65,100);axes[1].set_title('Early-stage detail')
fig.tight_layout();fig.savefig(OUT/'survival.png',dpi=180)
print('crossings',cross)
print('survival',[(x['time'],round(x['baseline_pct'],2),round(x['guide_pct'],2),round(x['difference_pp'],2)) for x in milestones])
print('features',[(k,round(v['smd'],3),round(v['low_quartile_early_rate']*100,1),round(v['high_quartile_early_rate']*100,1)) for k,v in featurestats.items()])
print('top demos',sorted(demos,key=lambda x:x['baseline_bottom_pct'],reverse=True)[:8])
print('cohorts',cohorts)
