from pathlib import Path
import json,numpy as np
from math import sqrt
P=Path(__file__).resolve().parent;Z=np.load(P.parent/'20260918_baseline_guide_dynamics/paired_features.npz')
m=Z['mass']*1000;f=Z['friction'];seed=Z['seed'];ys={'baseline':Z['baseline_seconds']>20,'guide':Z['guide_seconds']>20}
def wilson(n,k):
 z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-r,c+r]
def stat(mask):
 n=int(mask.sum());return {'n':n,**{name:{'gt20':int(y[mask].sum()),'failed_by20':int((~y[mask]).sum()),'pct':float(y[mask].mean()*100),'wilson95_pct':[x*100 for x in wilson(n,int(y[mask].sum()))]} for name,y in ys.items()}}
selected=[]
for ml,fl in [(150,2),(120,2.3),(160,3.3),(160,3.9),(175,3.9)]:
 mask=(m<=ml)&(f>=fl);selected.append({'mass_max_g':ml,'friction_min':fl,**stat(mask),'seeds':{str(s):stat(mask&(seed==s)) for s in np.unique(seed)}})
search=[]
for ml in range(80,351,5):
 for fl in np.round(np.arange(.5,3.91,.1),2):
  mask=(m<=ml)&(f>=fl);n=int(mask.sum())
  if n>=100:search.append({'mass_max_g':ml,'friction_min':float(fl),**stat(mask)})
best={name:{str(n):max((r for r in search if r['n']>=n),key=lambda r:(r[name]['pct'],r['n'])) for n in [100,300,500,1000]} for name in ys}
single={}
for name,y in ys.items():
 for key,vals in [('mass_max_g',np.arange(80,351,5)),('friction_min',np.round(np.arange(.5,3.91,.1),2)),('mass_over_friction_max',np.arange(20,701,5))]:
  candidates=[]
  for v in vals:
   mask=m<=v if key=='mass_max_g' else f>=v if key=='friction_min' else m/f<=v
   if mask.sum()>=500:candidates.append({'threshold':float(v),**stat(mask)})
  single[name+'_'+key]=max(candidates,key=lambda r:(r[name]['pct'],r['n']))
# Independent seed split threshold selection (2 seeds search, 2 seeds held out).
train=np.isin(seed,[42,8]);test=~train;split={}
for name in ys:
 candidates=[]
 for ml in range(80,351,5):
  for fl in np.round(np.arange(.5,3.91,.1),2):
   mask=(m<=ml)&(f>=fl)
   if (mask&train).sum()>=500 and (mask&test).sum()>0:candidates.append((ys[name][mask&train].mean(),int((mask&train).sum()),ml,float(fl)))
 _,_,ml,fl=max(candidates);mask=(m<=ml)&(f>=fl)
 split[name]={'selected_on_seeds':[42,8],'tested_on_seeds':[19,25],'mass_max_g':ml,'friction_min':fl,'train':stat(mask&train),'test':stat(mask&test)}
res={'success':'survives strictly more than 20 simulated seconds','baseline':'historical guided-DDIM scale0, not ordinary DDIM','all':stat(np.ones(len(m),bool)),'selected':selected,'grid_search_best':best,'best_single_condition_min500':single,'heldout_seed_split':split,'note':'Exploratory multiple threshold search; Wilson intervals for selected thresholds do not adjust for selection. Thresholds are associations, not guarantees or isolated causal effects.'}
(P/'results.json').write_text(json.dumps(res,indent=2))
for x in selected:print(x['mass_max_g'],x['friction_min'],x['n'],x['baseline'],x['guide'])
print('single',[(k,v['threshold'],v['n'],v[k.split('_')[0]]['pct']) for k,v in single.items()])
print('heldout',split)
