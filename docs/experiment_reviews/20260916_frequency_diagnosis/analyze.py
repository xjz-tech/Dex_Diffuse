from pathlib import Path
import json,re
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'eval/hold_runs/20260916_10k_obs466_frequency_3k'
OUT=Path(__file__).resolve().parent
all_rows={};result={}
for hz in (30,45,60):
 rows=sorted([json.loads(x) for x in (RUN/f'hz{hz}'/'episodes.jsonl').read_text().splitlines()],key=lambda r:r['env'])
 assert len(rows)==3000 and [r['env'] for r in rows]==list(range(3000))
 assert all(r['episode']==0 for r in rows)
 length=np.array([r['length'] for r in rows]);failed=np.array([r['reason']=='failure' for r in rows])
 all_rows[hz]=(length,failed)
 timing=json.loads((RUN/f'hz{hz}'/'timing.json').read_text())
 result[str(hz)]={'hold_steps':timing['hold_steps'],'survival_after_steps':{},'survival_after_seconds':{},'capped_mean_steps_at_12000':float(np.minimum(length,12000).mean()),'median_steps_at_12000':float(np.median(np.minimum(length,12000))), 'failures_within_one_second':int(np.sum(failed&(length<=hz)))}
 for step in (100,1000,3000,6000,9000,12000):
  alive=~(failed&(length<=step))
  result[str(hz)]['survival_after_steps'][str(step)]={'survivors':int(alive.sum()),'percent':float(alive.mean()*100),'physical_seconds':step/hz}
 for sec in (1,10,30,60,120,200,400):
  alive=~(failed&(length<=sec*hz))
  result[str(hz)]['survival_after_seconds'][str(sec)]={'survivors':int(alive.sum()),'percent':float(alive.mean()*100)}
 # Count target-arrival messages only until each env's first episode ends.
 # Diagnostic only: these are reference-goal transitions, not measured bulb rotations.
 goals=np.zeros(3000,dtype=np.int64);active=np.ones(3000,bool)
 with (RUN/f'hz{hz}'/'console.log').open() as f:
  for line in f:
   m=re.search(r'\[sim\] reached target \| step=(\d+) env_ids=\[([^]]*)\]',line)
   if m and int(m.group(1))<=12000:
    ids=[int(x) for x in m.group(2).split(',') if x.strip()]
    for i in ids:
     if active[i]:goals[i]+=1
   m=re.search(r'episode end \| env=(\d+) episode=0 length=',line)
   if m:active[int(m.group(1))]=False
 result[str(hz)]['reference_goals_before_first_end_or_12000']={'total':int(goals.sum()),'mean_per_env':float(goals.mean()),'per_1000_at_risk_action_steps':float(goals.sum()/np.minimum(length,12000).sum()*1000)}
initials={hz:np.load(RUN/f'hz{hz}'/'initial_state.npz') for hz in (30,45,60)}
assert all(np.array_equal(initials[30][key],initials[hz][key]) for hz in (45,60) for key in initials[30].files)
paired={}
for hz in (45,60):
 l0,f0=all_rows[30];l1,f1=all_rows[hz]
 s0=~(f0&(l0<=12000));s1=~(f1&(l1<=12000))
 paired[str(hz)]={'30_survives_other_fails':int(np.sum(s0&~s1)), '30_fails_other_survives':int(np.sum(~s0&s1)), 'both_survive':int(np.sum(s0&s1)), 'both_fail':int(np.sum(~s0&~s1))}
 demos=initials[30]['demo'].reshape(-1)
 diffs=[float(s1[demos==d].mean()-s0[demos==d].mean()) for d in np.unique(demos)]
 paired[str(hz)]['initial_demo_groups']=len(diffs)
 paired[str(hz)]['demo_groups_worse']=sum(x<0 for x in diffs)
 paired[str(hz)]['demo_groups_better']=sum(x>0 for x in diffs)
 paired[str(hz)]['demo_groups_tied']=sum(x==0 for x in diffs)
output={'results':result,'paired_12000_steps':paired,'initial_states_equal':True,'endpoint':'survives after K completed actions; failure at K is counted as failed','scope':'Three completed arms only; no per-step action/qpos/contact traces were saved.'}
(OUT/'statistics.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
