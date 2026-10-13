from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
z=np.load(ROOT/'docs/experiment_reviews/20260918_baseline_guide_dynamics/paired_features.npz')
seed=z['seed'];demo=z['demo'];bottom=z['baseline_bottom10'];b=z['baseline_seconds'];g=z['guide_seconds']
restored={int(s):np.load(ROOT/f'docs/experiment_reviews/20260918_historical_3k_restore/seed{s}_scale25/initial_parameters.npz') for s in np.unique(seed)}
q=np.concatenate([restored[s]['q'] for s in [42,8,19,25]])
selected=[]
for dem in [79,82,94]:
 ids=np.flatnonzero((demo==dem)&bottom)
 qs=q[ids];med=np.median(qs,axis=0);dist=np.mean((qs-med)**2,axis=1);i=int(ids[np.argmin(dist)])
 controls=np.flatnonzero((demo==dem)&(b>=400)&(seed==seed[i]))
 if not len(controls):controls=np.flatnonzero((demo==dem)&(b>=400))
 # closest physical conditions among baseline-to-cap examples, not claimed matched control
 v=np.stack([z['mass'],z['friction'],z['scale']],axis=1);std=v.std(axis=0)
 j=int(controls[np.argmin(np.sum(((v[controls]-v[i])/std)**2,axis=1))])
 for label,idx in [('fast',i),('held400',j)]:
  env=idx%3000;s=int(seed[idx]);a=restored[s]
  selected.append(dict(demo=dem,kind=label,seed=s,env=env,frame=int(a['frame'][env]),baseline_s=float(b[idx]),guide_s=float(g[idx]),mass_g=float(a['mass'][env]*1000),friction=float(a['friction'][env]),scale=float(a['scale'][env]),q=a['q'][env].tolist(),wrist=a['wrist'][env].tolist(),object=a['object'][env].tolist()))
(OUT/'selected.json').write_text(json.dumps(selected,indent=2))
mask=z['baseline_bottom10'];mass=z['mass']*1000
stats=[]
for name,m in [('historical_fast10',mask),('historical_rest90',~mask),('mass_180_190g',(mass>=180)&(mass<190)),('mass_214_224g',(mass>=214)&(mass<224))]:
 stats.append(dict(group=name,n=int(m.sum()),mean_mass_g=float(mass[m].mean()),baseline_cap_n=int((b[m]>=400).sum()),baseline_cap_pct=float((b[m]>=400).mean()*100),guide_cap_n=int((g[m]>=400).sum()),guide_cap_pct=float((g[m]>=400).mean()*100)))
(OUT/'group_rates.json').write_text(json.dumps(stats,indent=2))
for x in selected:print({k:v for k,v in x.items() if k not in ['q','wrist','object']})
