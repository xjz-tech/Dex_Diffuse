"""Conditional descriptive results with explicit 30-second right censoring."""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from report import METHODS, COLORS, curves
HERE=Path(__file__).resolve().parents[1]
if len(sys.argv)>1:HERE=Path(sys.argv[1])
LABELS=['Astra direct','Prior','Scale 25','Scale 38','Scale 50']
rows=[json.loads(p.read_text()) for p in (HERE/'runs').glob('*/summary.json')]
rows=[r for r in rows if r['phase']=='formal' and r['method'] in METHODS]
assert len(rows)==27
assert all(r['steps']<=900 for r in rows)
assert all(r['terminal_kind'] in ('visually_confirmed_drop','30s_observation_cap') for r in rows)
groups={}
for method in METHODS:
 rr=[r for r in rows if r['method']==method]
 groups[method]=dict(n=len(rr),cap_survivors=sum(r['reached_cap'] for r in rr),
  median_held_net_right_deg=float(np.median([r['held_net_right_deg'] for r in rr])),
  median_held_max_net_right_deg=float(np.median([r['held_max_net_right_deg'] for r in rr])),
  reached_right_180_before_cutoff=sum(r['first_180_seconds'] is not None and r['first_180_seconds']<=r['held_cutoff_step']/30+1e-3 for r in rr))
(HERE/'statistics.json').write_text(json.dumps(dict(groups=groups,interpretation='Conditional on three frozen selected physics seeds. Two paired noise seeds; direct n=3, not n=6. Observation caps are right censored, never drop times. Angular reach does not certify stable task success.'),indent=2)+'\n')
(HERE/'analysis_results.json').write_text(json.dumps(rows,indent=2)+'\n')
selection=json.loads((HERE/'screening.json').read_text())['selected_environment_seeds']
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for j,seed in enumerate(selection):
 for i,noise in enumerate([0,1]):
  ax=axes[i,j]
  for method,label,color in zip(METHODS,LABELS,COLORS):
   r=next(x for x in rows if x['method']==method and x['environment_seed']==seed and x['noise_seed']==(0 if method=='astra_direct' else noise))
   angle=curves(r)[:r['held_cutoff_step']];t=np.arange(1,len(angle)+1)/30
   ax.plot(t,angle,color=color,label=label,lw=1.3)
   ax.scatter(t[-1],angle[-1],color=color,marker='>' if r['reached_cap'] else 'x',s=30)
  ax.axhline(180,color='#334155',ls=':',lw=.8);ax.axhline(0,color='#334155',lw=.5)
  ax.set(title=f'Environment {seed}, noise {noise}',xlabel='Simulation seconds',ylabel='Net right twist (deg)',xlim=(0,30));ax.grid(alpha=.18)
axes[0,0].legend(fontsize=8)
fig.suptitle('Own-state feedback; x = last confirmed held frame, > = observation cap')
fig.savefig(HERE/'angle_curves.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for j,(method,color) in enumerate(zip(METHODS,COLORS)):
 rr=sorted([r for r in rows if r['method']==method],key=lambda r:(r['environment_seed'],r['noise_seed']))
 for k,r in enumerate(rr):
  x=j+(k-(len(rr)-1)/2)*.06
  if r['reached_cap']:axes[0].scatter(x,30,color=color,marker='^',s=40)
  else:
   lo,hi=r['drop_time_bracket_seconds'];mid=(lo+hi)/2
   axes[0].errorbar(x,mid,yerr=[[mid-lo],[hi-mid]],fmt='o',color=color,ms=5,capsize=2)
  axes[1].scatter(x,r['held_max_net_right_deg'],color=color,s=25)
 for ax in axes:ax.set_xticks(range(5),LABELS,rotation=15)
axes[0].set(ylabel='Seconds',title='Drop interval or 30s cap (triangle)',ylim=(0,32))
axes[1].set(ylabel='Maximum net right twist before cutoff (deg)',title='Angular reach is not stable rotation success')
axes[1].axhline(180,color='#334155',ls=':',lw=.8)
for ax in axes:ax.grid(axis='y',alpha=.18)
fig.savefig(HERE/'outcomes.png',dpi=150);plt.close(fig)
print(json.dumps(groups,indent=2))
