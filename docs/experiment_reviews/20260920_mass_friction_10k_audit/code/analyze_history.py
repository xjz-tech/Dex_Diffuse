import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,json
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];root=r.parents[2]
z=np.load(root/'docs/experiment_reviews/20260918_baseline_guide_dynamics/paired_features.npz');mass=z['mass']*1000;mu=z['friction'];b=z['baseline_seconds'];g=z['guide_seconds'];delta=np.minimum(g,400)-np.minimum(b,400);seeds=z['seed'];demo=z['demo']
# These are historical guided-DDIM scale0 vs scale25, not ordinary DDIM.
ratio=mass/mu;lm=np.log(mass);lf=np.log(mu);lr=np.log(ratio)
features={'ratio_only':np.c_[lr,lr**2,lr**3], 'mass_friction_separate':np.c_[lm,lf,lm**2,lf**2,lm*lf]}
initkeys=['scale','restitution','force_prob','frame_fraction','object_local_x','object_local_y','object_local_z','index_flexion','middle_flexion','ring_flexion','pinky_flexion','thumb_flexion']
# Add quadratic terms for geometric/pose covariates; reference demo not included to allow held-out-demo assessment.
init=np.column_stack([z[k] for k in initkeys]);features['mass_friction_initialization']=np.c_[features['mass_friction_separate'],init,init**2]
def auc(y,p):
 n=y.sum();return float((rankdata(p)[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n)))
def logistic(X,y,Z):
 mean=X.mean(0);sd=X.std(0);sd[sd<1e-10]=1
 X=np.c_[np.ones(len(X)),(X-mean)/sd];Z=np.c_[np.ones(len(Z)),(Z-mean)/sd]
 reg=np.ones(X.shape[1])*.001;reg[0]=0
 def fun(w):
  a=X@w;return np.mean(np.logaddexp(0,a)-y*a)+.5*(reg*w*w).sum(),X.T@(expit(a)-y)/len(y)+reg*w
 fit=minimize(fun,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':300});assert fit.success,fit.message
 return expit(Z@fit.x)
targets={'baseline_ge20s':(b>=20).astype(int),'guide_ge20s':(g>=20).astype(int),'guide_ge400s':(g>=400).astype(int),'guide_gain_gt20s':(delta>20).astype(int)}
cv={};predictions={}
for splitname,groups in [('leave_one_seed_out',seeds),('heldout_demo_groups',demo%5)]:
 for name,X in features.items():
  for outcome,y in targets.items():
   p=np.zeros(len(y));null=np.zeros(len(y))
   for group in np.unique(groups):
    test=groups==group;train=~test;p[test]=logistic(X[train],y[train],X[test]);null[test]=y[train].mean()
   key=f'{splitname}/{name}/{outcome}';cv[key]={'auc':auc(y,p),'brier':float(np.mean((y-p)**2)),'null_brier':float(np.mean((y-null)**2)),'base_rate':float(y.mean())};predictions[key.replace('/','_')]=p
  print(splitname,name,'finished',flush=True)
mb=np.array([0,100,150,200,250,400]);fb=np.array([.49,1,2,3,4.01]);grid=[]
for i in range(len(mb)-1):
 for j in range(len(fb)-1):
  s=(mass>=mb[i])&(mass<mb[i+1])&(mu>=fb[j])&(mu<fb[j+1]);n=int(s.sum())
  grid.append({'mass_low_g':float(mb[i]),'mass_high_g':float(mb[i+1]),'friction_low':float(fb[j]),'friction_high':float(fb[j+1]),'n':n,'baseline_ge20_pct':float((b[s]>=20).mean()*100),'guide_ge20_pct':float((g[s]>=20).mean()*100),'baseline_ge400_pct':float((b[s]>=400).mean()*100),'guide_ge400_pct':float((g[s]>=400).mean()*100),'rmst400_delta_s':float(delta[s].mean()),'guide_gain_gt20_pct':float((delta[s]>20).mean()*100),'baseline_gain_gt20_pct':float((delta[s]<-20).mean()*100)})
ratioedges=np.quantile(ratio,np.linspace(0,1,6));ratio_rows=[]
for i in range(5):
 s=(ratio>=ratioedges[i])&((ratio<=ratioedges[i+1]) if i==4 else (ratio<ratioedges[i+1]));ratio_rows.append({'ratio_low_g':float(ratioedges[i]),'ratio_high_g':float(ratioedges[i+1]),'n':int(s.sum()),'baseline_ge20_pct':float((b[s]>=20).mean()*100),'guide_ge20_pct':float((g[s]>=20).mean()*100),'rmst400_delta_s':float(delta[s].mean())})
fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
for ax,key,title,vmin,vmax in zip(axes,['baseline_ge20_pct','guide_ge20_pct','rmst400_delta_s'],['Historical baseline: survival at 20s (%)','10k guide: survival at 20s (%)','Guide minus baseline: mean min(T,400) (s)'],[0,0,0],[100,100,180]):
 values=np.array([x[key] for x in grid]).reshape(5,4);im=ax.imshow(values,origin='lower',vmin=vmin,vmax=vmax,cmap='viridis')
 for i in range(5):
  for j in range(4):ax.text(j,i,f'{values[i,j]:.1f}\nn={grid[i*4+j]["n"]}',ha='center',va='center',fontsize=8,color='white' if values[i,j]<(vmax+vmin)/2 else 'black')
 ax.set_xticks(range(4),['0.5–1','1–2','2–3','3–4']);ax.set_yticks(range(5),['<100','100–150','150–200','200–250','250–350']);ax.set_xlabel('Object friction');ax.set_ylabel('Mass (g)');ax.set_title(title,fontsize=10);fig.colorbar(im,ax=ax,shrink=.7)
fig.suptitle('Exploratory historical data: 12,000 paired environments; baseline = guided-DDIM scale 0',fontsize=11)
fig.savefig(r/'historical_mass_friction.png',dpi=170);plt.close(fig)
result={'scope':'20260912 historical guided-DDIM scale0 baseline vs scale25; not ordinary DDIM; association only; first-episode physics restored; baseline itself not rerun','n':len(b),'mass_range_g':[float(mass.min()),float(mass.max())],'friction_range':[float(mu.min()),float(mu.max())],'spearman_ratio_baseline_time':float(spearmanr(ratio,b).statistic),'spearman_ratio_guide_time':float(spearmanr(ratio,g).statistic),'cross_validation':cv,'ratio_quintiles':ratio_rows,'grid':grid,'feature_sets':{k:v.shape[1] for k,v in features.items()},'initialization_covariates':initkeys}
(r/'historical_analysis.json').write_text(json.dumps(result,indent=2));np.savez_compressed(r/'historical_oof_predictions.npz',**predictions)
print(json.dumps({'ratio_quintiles':ratio_rows,'cv':cv},indent=2),flush=True)
