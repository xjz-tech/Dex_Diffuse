from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent;z=np.load(P.parent/'20260918_baseline_guide_dynamics/paired_features.npz');sc=z['scale'];m=z['mass']*1000;f=z['friction'];seed=z['seed']
edges=[.95,.97,.99,1.01,1.03,1.050001];bi=np.digitize(sc,edges)-1;assert np.all((bi>=0)&(bi<5))
rows=[]
for i in range(5):
 k=bi==i
 row={'size_range':edges[i:i+2],'n':int(k.sum()),'mean_mass_g':float(m[k].mean()),'mean_friction':float(f[k].mean())}
 for name,key in [('baseline','baseline_seconds'),('guide','guide_seconds')]:
  t=z[key][k];row[name]={'gt20_pct':float((t>20).mean()*100),'to400_pct':float((t>=400).mean()*100),'median_s':float(np.median(t))}
 rows.append(row)
# Direct standardization: same seed, 50g mass bins, friction bins of width 1.
stratum=np.stack([seed,np.floor(m/50),np.floor(f)],axis=1)
u,inv=np.unique(stratum,axis=0,return_inverse=True);common=[]
for j in range(len(u)):
 counts=np.bincount(bi[inv==j],minlength=5)
 if np.min(counts)>=5:common.append(j)
common_mask=np.isin(inv,common);weights=np.array([(inv==j).sum() for j in common],float);weights/=weights.sum()
adjusted=[]
for i in range(5):
 row={'size_bin':i,'n':int(((bi==i)&common_mask).sum())}
 for name,key in [('baseline','baseline_seconds'),('guide','guide_seconds')]:
  row[name]={}
  for metric,pred in [('gt20',z[key]>20),('to400',z[key]>=400)]:
   rates=np.array([pred[(inv==j)&(bi==i)].mean() for j in common]);row[name][metric+'_pct']=float(weights@rates*100)
 adjusted.append(row)
res={'baseline':'historical guided-DDIM scale0','raw':rows,'common_strata_count':len(common),'common_support_n':int(common_mask.sum()),'strata':'seed x 50g mass bins x width1 friction bins; all five size groups >=5 cases per cell; common cell weights pooled across size groups','standardized':adjusted,'limitation':'coarse observational adjustment, not random size-only experiment; no adjustment for initial demo/pose'}
(P/'results.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
