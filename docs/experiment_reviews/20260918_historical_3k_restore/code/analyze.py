from pathlib import Path
import json,re
import numpy as np
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parents[1]
OLD=ROOT/'eval/hold_runs/20260912_fresh_noise_3000_seed42_8_19_25'
summary=[]
for seed in [8,42,19,25]:
 p=OUT/f'seed{seed}_scale25'
 if not (p/'initial_parameters.npz').exists() or not (p/'replay_scale25.jsonl').exists():continue
 rec=[json.loads(x) for x in (p/'replay_scale25.jsonl').read_text().splitlines() if x];
 if len(rec)!=3000:continue
 z=np.load(p/'initial_parameters.npz');new={x['env']:x for x in rec}
 orig=OLD/f'seed{seed}_1B_scale25';old={x['env']:x for x in [json.loads(x) for x in next(orig.glob('*.jsonl')).read_text().splitlines() if x]}
 oldearly={i for i,v in old.items() if v['length']<=150 and v['reason']=='failure'};newearly={i for i,v in new.items() if v['reason']=='failure'}
 exact=[i for i in oldearly if i in newearly and old[i]['length']==new[i]['length']]
 def block(file):
  t=file.read_text();i=t.index('[DR-DEBUG]');return t[i:t.index('\n\n',i)]
 dr_equal=block(p/'console.log')==block(orig/'console.log')
 cols={k:z[k] for k in ['mass','friction','rolling_friction','torsion_friction','restitution','scale','force_prob','cached_mass']}
 for k in ['hand_mass','hand_friction','stiffness','damping']:cols[k+'_mean']=z[k].mean(axis=1)
 groups=[]
 for prior in ['1B','mixed']:
  for scale in [0,25]:
   folder=OLD/f'seed{seed}_{prior}_scale{scale}';records=[json.loads(x) for x in next(folder.glob('*.jsonl')).read_text().splitlines() if x]
   times=np.zeros(3000)
   for x in records:times[x['env']]=x['length']
   cutoff=np.sort(times)[299];mask=times<=cutoff
   stats={}
   for key,v in cols.items():
    lo,hi=np.quantile(v,[.25,.75]);sd=np.std(v)
    stats[key]={'bottom_mean':float(v[mask].mean()),'rest_mean':float(v[~mask].mean()),'bottom_median':float(np.median(v[mask])),'rest_median':float(np.median(v[~mask])),'standardized_mean_difference':float((v[mask].mean()-v[~mask].mean())/sd) if sd else 0,'bottom10_rate_low_quartile':float(mask[v<=lo].mean()),'bottom10_rate_high_quartile':float(mask[v>=hi].mean())}
   demo=z['demo'].astype(int).reshape(-1);demostats=[]
   for demo_id in np.unique(demo):
    m=demo==demo_id;demostats.append({'demo':int(demo_id),'n':int(m.sum()),'bottom_n':int((m&mask).sum()),'rate':float(mask[m].mean())})
   groups.append({'prior':prior,'scale':scale,'bottom_n':int(mask.sum()),'cutoff_s':float(cutoff/30),'stats':stats,'demo_rates':sorted(demostats,key=lambda x:x['rate'],reverse=True)})
 summary.append({'seed':seed,'historical_initial_dr_exact_text_match':dr_equal,'historical_failure_le5s_n':len(oldearly),'replay_failure_le5s_n':len(newearly),'same_failure_env_n':len(oldearly&newearly),'exact_failure_step_n':len(exact),'old_only':sorted(oldearly-newearly),'new_only':sorted(newearly-oldearly),'groups':groups})
(OUT/'analysis.json').write_text(json.dumps(summary,indent=2))
for s in summary:
 print({k:v for k,v in s.items() if k!='groups'})
 g=next(g for g in s['groups'] if g['prior']=='1B' and g['scale']==25)
 for k,v in g['stats'].items():print(k,round(v['bottom_mean'],5),round(v['rest_mean'],5),'SMD',round(v['standardized_mean_difference'],3))
# Export exact physical parameters for all tied shortest historical guide cases.
examples=[]
for s in summary:
 seed=s['seed'];z=np.load(OUT/f'seed{seed}_scale25/initial_parameters.npz')
 p=OLD/f'seed{seed}_1B_scale25'
 rec=[json.loads(x) for x in next(p.glob('*.jsonl')).read_text().splitlines() if x]
 mn=min(x['length'] for x in rec)
 for x in rec:
  if x['length']!=mn:continue
  i=x['env'];examples.append({'seed':seed,'env':i,'seconds':mn/30,**{k:np.asarray(z[k][i]).tolist() for k in ['mass','friction','rolling_friction','torsion_friction','restitution','scale','demo','frame','force_prob','cached_mass']}})
(OUT/'shortest_cases.json').write_text(json.dumps(examples,indent=2))
