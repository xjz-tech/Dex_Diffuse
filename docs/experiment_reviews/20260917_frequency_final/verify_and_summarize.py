from pathlib import Path
import importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'eval/hold_runs'
OUT=Path(__file__).resolve().parent
specs=[('A','20260916_10k_obs466_frequency_3k',[30,45,60,90],180,2),('B','20260916_10k_obs466_40hz_3k',[30,40],360,1),('C','20260916_10k_obs466_20_25hz_3k',[30,25,20],300,1)]
results={};verification={};row_cache={}
reference=np.load(RUN/specs[0][1]/'hz30/initial_state.npz')

def check(folder,hz,physics,substeps):
 rows=[json.loads(x) for x in (folder/'episodes.jsonl').read_text().splitlines()]
 assert len(rows)==3000 and {r['env'] for r in rows}==set(range(3000))
 assert all(r['episode']==0 and 0<r['length']<=hz*400 and r['reason'] in ('failure','timeout') for r in rows)
 assert all(r['reason']!='timeout' or r['length']==hz*400 for r in rows)
 timing=json.loads((folder/'timing.json').read_text());schedule=json.loads((folder/'schedule.json').read_text())
 assert timing['episodes']==3000 and timing['hold_steps']==0 and timing['wait']==1
 assert timing['control_steps']==hz*400 and timing['action_steps']==hz*400
 assert timing['force_physics_ticks']==physics*400
 assert schedule['control_hz']==hz and schedule['physics_hz']==physics and schedule['substeps']==substeps
 assert abs(schedule['physics_dt']*schedule['decimation']-1/hz)<1e-10
 with np.load(folder/'initial_state.npz') as initial:
  assert all(np.array_equal(reference[k],initial[k]) for k in reference.files)
 gate=json.loads((folder/'optimization_validation.json').read_text())
 assert gate['guidance_steps']==2 and gate['execution_steps']==2 and gate['scale']==25 and gate['fresh_noise']
 assert gate['checkpoint']=='/home/carus/data_usb/obs_4-66.ckpt' and gate['microbatch']==256
 assert all(np.isfinite(x) and x<.002 for x in gate['fp32_gate'])
 assert all(np.isfinite(x['max_abs']) and x['max_abs']<.03 for x in gate['fp16_drift'])
 assert np.isfinite(gate['batch_independence_max_abs']) and gate['batch_independence_max_abs']<.03
 key=str(folder.relative_to(RUN))
 verification[key]=dict(records=3000,initial_state_equal=True,control_steps=timing['control_steps'],physics_ticks=timing['force_physics_ticks'],zero_wait_holds=True,optimization_gates_pass=True,max_fp16_abs_error=max(x['max_abs'] for x in gate['fp16_drift']))
 length=np.array([r['length'] for r in rows]);failure=np.array([r['reason']=='failure' for r in rows])
 survived=int((~failure).sum())
 return rows,dict(survivors=survived,survival_percent=100*survived/3000,mean_capped_seconds=float((length/hz).mean()),median_capped_seconds=float(np.median(length/hz)))

for group,name,hzs,physics,substeps in specs:
 assert json.loads((RUN/name/'state.json').read_text())['status']=='complete'
 results[group]={}
 for hz in hzs:
  rows,v=check(RUN/name/f'hz{hz}',hz,physics,substeps)
  results[group][str(hz)]=v;row_cache[(group,str(hz))]=rows
folder=RUN/'20260916_10k_obs466_30hz_clamp003_3k'/'clamp003'
rows,v=check(folder,30,180,2);results['D']={'unclamped':results['A']['30'],'clamp003':v};row_cache[('D','unclamped')]=row_cache[('A','30')];row_cache[('D','clamp003')]=rows
c=json.loads((folder/'clamp_statistics.json').read_text())
assert c['env_action_samples']==sum(r['length'] for r in rows)
assert c['joint_samples']==22*c['env_action_samples']
assert c['step_clipped_joints']==sum(c['per_joint_step_clips'])
assert 0<c['step_clipped_joints']<=c['joint_samples'] and c['max_sent_target_step']<=.030001
assert np.isclose(c['step_clipped_joint_fraction'],c['step_clipped_joints']/c['joint_samples'])
assert np.isclose(c['step_clipped_env_action_fraction'],c['step_clipped_env_actions']/c['env_action_samples'])
verification['clamp_telemetry']={'passed':True,'max_sent_step':c['max_sent_target_step'],'joint_clip_fraction':c['step_clipped_joint_fraction'],'env_action_clip_fraction':c['step_clipped_env_action_fraction']}
step_comparison={}
for group,cap in [('A',12000),('B',12000),('C',8000),('D',12000)]:
 step_comparison[group]={}
 for name in results[group]:
  rr=row_cache[(group,name)]
  survived=sum(not(r['reason']=='failure' and r['length']<=cap) for r in rr)
  step_comparison[group][name]=dict(common_steps=cap,survivors=survived,survival_percent=100*survived/3000)
data=dict(results=results,matched_steps=step_comparison,clamp_statistics=c,verification=verification)
(OUT/'results.json').write_text(json.dumps(data,indent=2)+'\n')
lines=['# Control frequency and 0.03-rad clamp: final verified results','', 'All arms: 3000 identical initial environments, first episodes only, 400 simulated seconds, seed8/100008, 10k guide + obs_4-66, DDIM4/4, guide2/execute2, scale25, original normalizers, TensorRT FP16/fused updates, WAIT=1. Failure is the configured simulator predicate, not a separately verified physical drop. Means and medians are right-censored at400s.','', '| Group | Policy Hz / clamp | Survived400s | Survival | Mean hold(s) | Median hold(s) |','|---|---|---:|---:|---:|---:|']
for group,rr in results.items():
 for name,v in rr.items():lines.append(f"| {group} | {name} | {v['survivors']}/3000 | {v['survival_percent']:.2f}% | {v['mean_capped_seconds']:.2f} | {v['median_capped_seconds']:.2f} |")
lines+=['','A/D:180Hz outer physics,2substeps. B:360Hz,1substep. C:300Hz,1substep. A/B integration step1/360s; C1/300s. Compare each group with its own30Hz bridge; do not merge the30Hz controls or claim identical physics across groups. Original pre-sweep30Hz result46.27% used60Hz outer physics/2substeps and is contextual only.','', '## Equal-action-count survival','', '| Group | Setting | Common action count | Survival |','|---|---|---:|---:|']
for group,rr in step_comparison.items():
 for name,v in rr.items():lines.append(f"| {group} | {name} | {v['common_steps']} | {v['survival_percent']:.2f}% |")
lines+=['','Action-matched endpoints have different physical duration; equal-time and equal-action-count results answer different questions. Sampling of observations, replanning, and terminal checks also changes with action frequency. Single-seed results are not proof of an optimal real-hardware frequency or of the bulb pull-in mechanism.','', '## Step-clamp result','',f"At30Hz,0.03rad clamp changes survival from50.47% to{results['D']['clamp003']['survival_percent']:.2f}% (460 fewer survivors), mean hold269.33s to194.74s. Clamp activated for{100*c['step_clipped_joint_fraction']:.2f}% of joint commands and{100*c['step_clipped_env_action_fraction']:.2f}% of environment-action samples (at least one joint clipped). It is a target-to-target clamp after URDF bounds; the next observation uses the clipped target. No extra inference holds are present. This demonstrates a negative effect in this matched simulation; it does not authorize removing hardware protection or prove causation in the real video.", '', 'Training-data exceedance fractions from the earlier audit are not these observed closed-loop clip fractions. Mean raw-to-sent absolute difference0.002969rad; measured residual0.034815rad in clamped arm only, without matching baseline trace. The0.9rad/s nominal command-increment bound is not a measured joint-velocity bound.','', '## Validation','', 'All10 full arms have3000 unique episode0 records, the expected action/physics tick counts, equal saved observations/demo/frame/root states, zero WAIT holds, matching checkpoint metadata and passing numerical gates. Clamp sample denominators and per-joint counts reconcile exactly; maximum target increment0.030000000000000027rad. See results.json for verification details.','', 'Source reports:']
for _,name,_,_,_ in specs:
 p=RUN/name/'comparison.md';lines.append(f'- [{name}]({p})')
p=folder.parent/'comparison.md';lines.append(f'- [30Hz clamp comparison]({p})')
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'results':results,'matched_steps':step_comparison,'verified_arms':len(verification)-1,'clamp':verification['clamp_telemetry']},indent=2))
