"""Descriptive statistics for the frozen paired batch, with visual time bounds."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parents[1]
METHODS=['astra_direct','prior','guided5','guided25','guided100']
LABELS=['Astra direct','Prior','Scale 5','Scale 25','Scale 100']
COLORS=['#c026d3','#64748b','#d97706','#059669','#2563eb']
rows=[json.loads(p.read_text()) for p in (HERE/'runs').glob('*/summary.json') if json.loads(p.read_text())['phase']=='formal']
assert len(rows)==27, 'Final statistics require all 27 planned trials, including retries'
assert all(r['terminal_kind']=='visually_confirmed_drop' for r in rows)
selection=json.loads((HERE/'screening.json').read_text())['selected_environment_seeds']
curves={}
for r in rows:
    run=Path(r['run']);m=json.loads((run/'manifest.json').read_text());axis=np.array(m['axis_world'])
    ts=[];angles=[];max_tilt=0.
    for line in (run/'trajectory.jsonl').open():
        z=json.loads(line)
        if z['step']>r['held_cutoff_step']:
            break
        ts.append(z['step']*m['control_dt']);angles.append(-z['twist_degrees'])
        tilt=np.degrees(np.arccos(np.clip(np.dot(Rotation.from_quat(z['state']['object_xyzw']).apply([0,1,0]),axis),-1,1)))
        max_tilt=max(max_tilt,float(tilt))
    r['held_max_tilt_deg']=max_tilt
    (run/'summary.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
    curves[r['label']]=(ts,angles)

groups={}
for method in METHODS:
    rr=[r for r in rows if r['method']==method]
    bounds=np.array([r['drop_time_bracket_seconds'] for r in rr])
    groups[method]=dict(n=len(rr),mean_drop_interval_seconds=bounds.mean(0).tolist(),
        median_drop_interval_seconds=np.median(bounds,axis=0).tolist(),
        earliest_drop_lower_seconds=float(bounds[:,0].min()),latest_drop_upper_seconds=float(bounds[:,1].max()),
        median_held_net_right_deg=float(np.median([r['held_net_right_deg'] for r in rr])),
        median_held_max_net_right_deg=float(np.median([r['held_max_net_right_deg'] for r in rr])),
        reached_right_180_before_cutoff=sum(r['first_180_confirmed_while_held'] for r in rr),
        median_held_max_tilt_deg=float(np.median([r['held_max_tilt_deg'] for r in rr])))
paired={}
for method in METHODS[2:]:
    differences=[]
    for r in [x for x in rows if x['method']==method]:
        prior=next(x for x in rows if x['method']=='prior' and x['environment_seed']==r['environment_seed'] and x['noise_seed']==r['noise_seed'])
        l,u=r['drop_time_bracket_seconds'];pl,pu=prior['drop_time_bracket_seconds']
        differences.append(dict(environment_seed=r['environment_seed'],noise_seed=r['noise_seed'],difference_interval_seconds=[l-pu,u-pl]))
    paired[method]=dict(pairs=differences,definitely_longer=sum(x['difference_interval_seconds'][0]>0 for x in differences),
        definitely_shorter=sum(x['difference_interval_seconds'][1]<0 for x in differences))
stats=dict(groups=groups,paired_against_prior=paired,interpretation='Descriptive conditional results after screening; three physics configurations, two paired noise sequences. Direct n=3 is reused visually across noise panels, not six independent direct trials. Right180 is an angle crossing, not a certified stable task success.')
(HERE/'statistics.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False)+'\n')
(HERE/'analysis_results.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')

fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for j,seed in enumerate(selection):
    for i,noise in enumerate([0,1]):
        ax=axes[i,j]
        for method,label,color in zip(METHODS,LABELS,COLORS):
            r=next(x for x in rows if x['method']==method and x['environment_seed']==seed and x['noise_seed']==(0 if method=='astra_direct' else noise))
            t,angle=curves[r['label']]
            ax.plot(t,angle,color=color,label=label,lw=1.3)
            ax.scatter(t[-1],angle[-1],color=color,marker='x',s=30)
        ax.axhline(180,color='#334155',ls=':',lw=.8)
        ax.axhline(0,color='#334155',lw=.5)
        ax.set(title=f'Environment {seed}, noise {noise}',xlabel='Simulation seconds',ylabel='Net right twist (deg)')
        ax.grid(alpha=.18)
axes[0,0].legend(fontsize=8)
fig.suptitle('Independent feedback per arm; curves stop at last visually confirmed held frame')
fig.savefig(HERE/'angle_curves.png',dpi=160)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for j,(method,color) in enumerate(zip(METHODS,COLORS)):
    rr=sorted([r for r in rows if r['method']==method],key=lambda r:(r['environment_seed'],r['noise_seed']))
    for k,r in enumerate(rr):
        x=j+(k-(len(rr)-1)/2)*.06
        lo,hi=r['drop_time_bracket_seconds'];mid=(lo+hi)/2
        axes[0].errorbar(x,mid,yerr=[[mid-lo],[hi-mid]],fmt='o',color=color,ms=5,capsize=2)
        axes[1].scatter(x,r['held_max_net_right_deg'],color=color,s=25)
    for ax in axes:ax.set_xticks(range(5),LABELS,rotation=15)
axes[0].set(ylabel='Visual drop time (seconds)',title='Every planned trial; vertical bars = review interval')
axes[1].set(ylabel='Maximum net right twist before cutoff (deg)',title='Angular reach does not certify stable rotation')
axes[1].axhline(180,color='#334155',ls=':',lw=.8)
for ax in axes:ax.grid(axis='y',alpha=.18)
fig.savefig(HERE/'outcomes.png',dpi=160)
plt.close(fig)
print(json.dumps(groups,indent=2))
