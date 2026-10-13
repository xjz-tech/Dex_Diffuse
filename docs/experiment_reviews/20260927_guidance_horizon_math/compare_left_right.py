"""Same-state left/right reference, residual and noisy-score audit; no physics."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
ASTRA=ROOT/'.worktrees/Astra-controller'
sys.path[:0]=[str(ASTRA),str(ASTRA/'eval')]
from astra_kinematic_step import propose
from inference_dp_controller import GuidedDDIMController
from diffusion_policy.guidance.guided_ddim import guided_ddim_step

RUN=ASTRA/'outputs/astra_gait_pause9/no_gait_e3577_n48'
def rms(a): return float(np.sqrt(np.mean(np.asarray(a,dtype=float)**2)))
def cosine(a,b):
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
groups={k:[] for k in ['turn1','turn2','turn3']}
actual={k:[] for k in groups}
for f in sorted(RUN.glob('request_*.json')):
    req=json.loads(f.read_text());resp=json.loads(f.with_name(f.name.replace('request','response')).read_text())
    phases=resp['action_phases']; phase=phases[0]
    if req['step']>0 and phase in groups and len(set(phases))==1:
        groups[phase].append((req,resp))
        a=np.asarray(resp['actions']);anchor=np.asarray(req['state']['target_before']);q=np.asarray(req['state']['qpos'])
        actual[phase].append(dict(step=req['step'],step_rms=rms(np.diff(a,axis=0)),
            prefix2_from_target_rms=rms(a[:2]-anchor),full9_from_target_rms=rms(a[:9]-anchor),
            prefix2_from_qpos_rms=rms(a[:2]-q),target_qpos_rms=rms(anchor-q)))
selected=[]
for phase, items in groups.items():
    indices=np.linspace(0,len(items)-1,12).round().astype(int)
    selected.extend((phase,*items[i]) for i in indices)
selected.sort(key=lambda x:x[1]['step'])
meta=json.loads((RUN/'model.json').read_text())
torch.manual_seed(meta['seed'])
c=GuidedDDIMController(Path(meta['checkpoint']),torch.device('cuda:0'),inference_steps=4,
    execution_steps=9,guidance_scale=25,eta=0.,fixed_noise=False,seed=meta['noise_seed'],allow_salvage=False)
c.set_guidance_horizon(9)
selected_steps={req['step'] for _,req,_ in selected}
noise_states={}
for step in range(0,max(selected_steps)+1,2):
    if step in selected_steps:noise_states[step]=c.generator.get_state()
    c._noise(1,c.policy.dtype)
records=[];arrays=[];score_rows=[]
max_repro=0.;max_plan_error=0.
for index,(phase,req,resp) in enumerate(selected):
    step=req['step']; data=np.load(RUN/f'prediction_{step:06d}.npz')
    q=np.asarray(req['state']['qpos']);anchor=np.asarray(req['state']['target_before'])
    plans={}
    for key,deg,measured in [('left',15,False),('right',-15,True),('right_same_target_base',-15,False),('left_same_qpos_base',15,True)]:
        plans[key]=propose(req,deg,16,0.,max_delta=.15,measured_base=measured)[0][:9]
    expected=plans['right' if phase=='turn2' else 'left']
    err=float(np.max(np.abs(expected-data['reference'][0])))
    max_plan_error=max(max_plan_error,err)
    np.testing.assert_allclose(expected,data['reference'][0],atol=1e-6,rtol=1e-5)
    outputs={}
    for key,scale,ref in [('pure',0,data['reference'][0]),('historical',25,data['reference'][0]),
                          ('left',25,plans['left']),('right',25,plans['right'])]:
        c.generator.set_state(noise_states[step]);c.guidance_scale=scale
        cmd,_=c.predict(data['observation'],ref[None].astype(np.float32))
        outputs[key]=cmd[0]
    error=float(np.abs(outputs['historical'][:2]-data['prediction'][0]).max())
    max_repro=max(max_repro,error)
    np.testing.assert_allclose(outputs['historical'][:2],data['prediction'][0],atol=1e-6,rtol=1e-5)
    pure=outputs['pure'].astype(float)
    record=dict(step=step,source_phase=phase,reference_pair={},residuals={})
    for h in [2,9]:
        l=plans['left'][:h]; r=plans['right'][:h]; p=pure[:h]
        el=p-l;er=p-r
        record['reference_pair'][str(h)]=dict(left_right_rmse=rms(l-r),
            left_from_target_rms=rms(l-anchor),right_from_target_rms=rms(r-anchor),
            left_right_increment_cosine=cosine(l-anchor,r-anchor),
            residual_cosine=cosine(el,er),residual_same_sign_fraction=float(np.mean(el*er>0)),
            left_right_guided_output_rmse=rms(outputs['left'][:h]-outputs['right'][:h]),
            same_target_base_lr_rmse=rms(l-plans['right_same_target_base'][:h]),
            same_target_base_increment_cosine=cosine(l-anchor,plans['right_same_target_base'][:h]-anchor))
        record['residuals'][str(h)]={}
        for key,ref in plans.items():
            e=p-ref[:h]
            record['residuals'][str(h)][key]=dict(rmse=rms(e),negative_fraction=float(np.mean(e<0)),
                signed_mean=float(e.mean()),guidance_vs_prior_target_increment_cosine=cosine(-e,p-anchor))
    # Evaluate epsilon score and reference guidance at exactly the SAME x_t on an unguided DDIM path.
    obs=torch.as_tensor(data['observation'],device=c.device,dtype=c.policy.dtype)
    cond=c.policy.normalizer['obs'].normalize(obs).reshape(1,-1)
    nr={key:c.policy.normalizer['action'].normalize(torch.as_tensor(ref[None],device=c.device,dtype=c.policy.dtype)) for key,ref in plans.items()}
    c.generator.set_state(noise_states[step]);xt=c._noise(1,c.policy.dtype)
    c.scheduler.set_timesteps(4,device=c.device)
    for t in c.scheduler.timesteps:
        with torch.no_grad():eps=c._predict_epsilon(xt,t,cond)
        alpha=c.scheduler.alphas_cumprod[int(t)].to(c.device);beta=1-alpha
        raw=(xt-beta.sqrt()*eps)/alpha.sqrt()
        score=-eps/beta.sqrt()
        sc=score[:,3:12].detach().cpu().numpy()
        gs={key:(-(raw[:,3:12]-ref)/alpha.sqrt()).detach().cpu().numpy() for key,ref in nr.items()}
        score_rows.append(dict(step=step,source_phase=phase,t=int(t),
            cosine_prior_score_guide_left=cosine(sc,gs['left']),
            cosine_prior_score_guide_right=cosine(sc,gs['right']),
            cosine_guide_left_right=cosine(gs['left'],gs['right'])))
        o=guided_ddim_step(c.scheduler,eps,t,xt,nr['left'],0,slice(3,12))
        xt=o.prev_sample
    terminal=c.policy.normalizer['action'].unnormalize(xt[:,3:12]).detach().cpu().numpy()[0]
    np.testing.assert_array_equal(terminal,outputs['pure'])
    records.append(record)
    arrays.append(dict(step=step,phase=phase,pure=pure,left=plans['left'],right=plans['right'],
        right_same_target_base=plans['right_same_target_base'],anchor=anchor,qpos=q))
    if (index+1)%6==0:print(f'{index+1}/{len(selected)} states verified',flush=True)

summary={}
for phase in ['all','turn1','turn2','turn3']:
    ids=[i for i,r in enumerate(records) if phase=='all' or r['source_phase']==phase]
    summary[phase]={}
    for h in [2,9]:
        p=np.array([arrays[i]['pure'][:h] for i in ids]);l=np.array([arrays[i]['left'][:h] for i in ids]);r=np.array([arrays[i]['right'][:h] for i in ids]);anchor=np.array([arrays[i]['anchor'] for i in ids])[:,None]
        same=np.array([arrays[i]['right_same_target_base'][:h] for i in ids])
        el=p-l;er=p-r
        summary[phase][str(h)]=dict(states=len(ids),reference_lr_rmse=rms(l-r),left_from_target_rms=rms(l-anchor),right_from_target_rms=rms(r-anchor),
            residual_left_rmse=rms(el),residual_right_rmse=rms(er),negative_fraction_left=float(np.mean(el<0)),negative_fraction_right=float(np.mean(er<0)),
            residual_same_sign_fraction=float(np.mean(el*er>0)),residual_cosine_pooled=cosine(el,er),
            residual_cosine_median=float(np.median([records[i]['reference_pair'][str(h)]['residual_cosine'] for i in ids])),
            left_right_increment_cosine=cosine(l-anchor,r-anchor),
            guidance_vs_prior_increment_cosine_left=cosine(-el,p-anchor),guidance_vs_prior_increment_cosine_right=cosine(-er,p-anchor),
            same_target_base_lr_rmse=rms(l-same),same_target_base_increment_cosine=cosine(l-anchor,same-anchor))
scores={}
for t in [75,50,25,0]:
    ss=[r for r in score_rows if r['t']==t]
    scores[str(t)]={key:dict(mean=float(np.mean([r[key] for r in ss])),positive_count=sum(r[key]>0 for r in ss),n=len(ss)) for key in ['cosine_prior_score_guide_left','cosine_prior_score_guide_right','cosine_guide_left_right']}
actual_summary={phase:{'plans':len(rows),**{key:float(np.mean([r[key] for r in rows])) for key in ['step_rms','prefix2_from_target_rms','full9_from_target_rms','prefix2_from_qpos_rms','target_qpos_rms']}} for phase,rows in actual.items()}
result=dict(source=str(RUN),selected_steps=sorted(selected_steps),physics_advanced=False,selection='12 uniformly spaced whole-phase 16-step plans from each turn stage; no gait; excludes initialization and phase transitions',max_reproduction_error_rad=max_repro,max_reference_reconstruction_error_rad=max_plan_error,
    actual_plan_summary=actual_summary,same_state_summary=summary,noisy_score_alignment=scores,records=records,score_records=score_rows)
(OUT/'left_right_results.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez_compressed(OUT/'left_right_arrays.npz',steps=np.array([r['step'] for r in arrays]),**{key:np.array([r[key] for r in arrays]) for key in ['pure','left','right','right_same_target_base','anchor','qpos']})
print(json.dumps({k:result[k] for k in ['max_reproduction_error_rad','max_reference_reconstruction_error_rad','actual_plan_summary','same_state_summary','noisy_score_alignment']},indent=2),flush=True)
