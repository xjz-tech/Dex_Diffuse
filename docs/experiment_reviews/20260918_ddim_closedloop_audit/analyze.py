"""Historical survival plus offline sampler diagnostics. No simulator is started."""
import json
import re
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'eval'),str(ROOT)]
from checkpoint_loader import load_checkpoint,build_policy,configure_policy_sampler
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
result={'scope':'Historical native-protocol logs and offline counterfactual inference; no new simulator runs.'}
def save(): (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
def rms(x): return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))
def stats(x):
    x=np.abs(x)
    return {'mean_abs':float(x.mean()),'rms':rms(x),'p95':float(np.percentile(x,95)),
            'max':float(x.max()),'joint_over_009':float((x>.090001).mean()),
            'step_any_over_009':float((x>.090001).any(axis=-1).mean())}

historical=ROOT/'eval/hold_runs/20260908_153711_xjz_grid'
console=(historical/'console.log').read_text()
result['historical']={}
for steps in [2,4,8,16]:
    name=f'ddim{steps}_exec2'
    rows=[json.loads(l) for l in (historical/(name+'.jsonl')).read_text().splitlines()]
    lengths=np.array([r['length'] for r in rows])
    segment=console.split('[xjz_grid] start '+name+'\n',1)[1].split('[xjz_grid] finished '+name,1)[0]
    progress=[l for l in segment.splitlines() if '[sim] progress' in l]
    last_progress=int(re.search(r'steps=(\d+)',progress[-1]).group(1))
    first={r['env']:r['length'] for r in rows if r['episode']==0}
    byenv={i:[] for i in range(1024)}
    for r in rows: byenv[r['env']].append(r)
    events=[]
    for env_id,rr in byenv.items():
        rr=sorted(rr,key=lambda r:r['episode'])
        assert [r['episode'] for r in rr]==list(range(len(rr)))
        endtime=0
        for r in rr:
            endtime+=r['length'];events.append(dict(r,end_step=endtime))
    common=[r for r in events if r['end_step']<=3000]
    capped=np.array([min(first.get(i,np.inf),3000) for i in range(1024)])
    result['historical'][name]={'completed_episodes':len(rows),'mean_failure_steps':float(lengths.mean()),
        'median_failure_steps':float(np.median(lengths)),'first_failures_recorded':len(first),
        'last_progress_step':last_progress,'last_progress_line':progress[-1],
        'reconstructed_stop_step':max(r['end_step'] for r in events),
        'failures_within_3000steps':None if last_progress<3000 else {
            'all':len(common),'first':sum(r['episode']==0 for r in common),
            'after_reset':sum(r['episode']>0 for r in common),
            'after_reset_length_le30':sum(r['episode']>0 and r['length']<=30 for r in common),
            'after_reset_length_le150':sum(r['episode']>0 and r['length']<=150 for r in common)},
        'common_3000step_first_episode': None if last_progress<3000 else {
            'survival_30s':float(np.mean([first.get(i,np.inf)>900 for i in range(1024)])),
            'survival_60s':float(np.mean([first.get(i,np.inf)>1800 for i in range(1024)])),
            'survival_100s':float(np.mean([first.get(i,np.inf)>3000 for i in range(1024)])),
            'restricted_mean_seconds':float(capped.mean()/30)}}
print('historical',json.dumps(result['historical']),flush=True)

loaded=load_checkpoint('/home/carus/data_usb/obs_4-66.ckpt',allow_salvage=False)
policy,spec=build_policy(loaded);policy.cuda().eval();configure_policy_sampler(policy,'ddim',4)
source=np.load(ROOT/'docs/experiment_reviews/20260918_training_inference_audit/samples.npz')
obs=[source['obs'][:256,:4]]; nextobs=[source['obs'][:256,2:6]]
truth=source['action'][:256,3:5]
rollout_provenance=[]; actual_noise=[]; actual_action=[]
rng=np.random.default_rng(42)
for seed in [50,58,63,65]:
    path=ROOT/f'docs/experiment_reviews/20260918_xjz_realinit_3seeds/seed{seed}_ordinary_1b/rollout.npz'
    z=np.load(path)
    # row k is the post-step observation. At plan t use rows t-4..t-1.
    candidates=np.arange(4,len(z['q'])-4,2)
    times=np.sort(rng.choice(candidates,min(64,len(candidates)),replace=False))
    fullobs=np.concatenate([z['q'],z['action'],z['action']-z['q']],axis=-1)
    obs.append(fullobs[times[:,None]+np.arange(-4,0)[None]])
    nextobs.append(fullobs[times[:,None]+np.arange(-2,2)[None]])
    ng=np.random.default_rng(seed)
    allnoise=ng.standard_normal((len(z['q'])//2+1,1,12,22)).astype(np.float32)
    actual_noise.append(allnoise[times//2,0])
    actual_action.append(z['raw'][times[:,None]+np.arange(2)[None]])
    rollout_provenance.append({'seed':seed,'source':str(path),'plan_times':times.tolist()})
obs=np.concatenate(obs); nextobs=np.concatenate(nextobs)
result['samples']={'training':256,'real_rollout_histories':len(obs)-256,'noise_samples_per_history':8,'rollouts':rollout_provenance}
np.savez_compressed(OUT/'observations.npz',obs=obs,nextobs=nextobs,training_labels=truth)
N=len(obs)
noise=np.random.default_rng(20260918).standard_normal((8,N,12,22)).astype(np.float32)
perturb=np.random.default_rng(19).normal(0,.001,(N,4,22)).astype(np.float32)
jittered=obs.copy(); jittered[:,:,:22]+=perturb;jittered[:,:,44:]-=perturb

def sample(observations,noises,times):
    pieces=[]
    with torch.inference_mode():
        for i in range(0,len(observations),64):
            cond=policy.normalizer['obs'].normalize(torch.tensor(observations[i:i+64],device='cuda')).reshape(-1,264)
            x=torch.tensor(noises[i:i+64],device='cuda')
            alphas=policy.noise_scheduler.alphas_cumprod
            for j,t in enumerate(times):
                eps=policy.model(x,int(t),global_cond=cond)
                at=alphas[t]; ap=alphas[times[j+1]] if j+1<len(times) else policy.noise_scheduler.final_alpha_cumprod
                x0=((x-(1-at)**.5*eps)/at**.5).clamp(-1,1)
                x=ap**.5*x0+(1-ap)**.5*eps
            pieces.append(policy.normalizer['action'].unnormalize(x).cpu().numpy()[:,3:5])
    return np.concatenate(pieces)

# Check arbitrary-time implementation against the production DDIM scheduler.
with torch.inference_mode():
    ob=torch.tensor(obs[:8],device='cuda');no=torch.tensor(noise[0,:8],device='cuda')
    cond=policy.normalizer['obs'].normalize(ob).reshape(8,-1)
    policy.noise_scheduler.set_timesteps(4)
    x=no.clone()
    for t in policy.noise_scheduler.timesteps:
        ep=policy.model(x,t,global_cond=cond);x=policy.noise_scheduler.step(ep,t,x).prev_sample
    ref=policy.normalizer['action'].unnormalize(x).cpu().numpy()[:,3:5]
result['manual_scheduler_vs_production_max_abs']=float(np.max(abs(ref-sample(obs[:8],noise[0,:8],[75,50,25,0]))))
reproduced=sample(obs[256:],np.concatenate(actual_noise),[75,50,25,0])
result['saved_rollout_prediction_replay_max_abs']=float(np.max(abs(reproduced-np.concatenate(actual_action))))
print('replay',result['manual_scheduler_vs_production_max_abs'],result['saved_rollout_prediction_replay_max_abs'],flush=True)

result['samplers']={};predictions={}
for steps in [2,4,8,16]:
    policy.noise_scheduler.set_timesteps(steps)
    times=policy.noise_scheduler.timesteps.tolist()
    alpha=float(policy.noise_scheduler.alphas_cumprod[times[0]])
    pred=np.stack([sample(obs,noise[k],times) for k in range(8)])
    predictions[str(steps)]=pred
    next_same=sample(nextobs,noise[0],times);next_fresh=sample(nextobs,noise[1],times)
    pert=sample(jittered,noise[0],times)
    bysource={}
    for label,sl in [('train',slice(0,256)),('rollout',slice(256,None))]:
        p=pred[:,sl]
        row={'noise_std_first_action':float(np.sqrt(np.mean(np.var(p[:,:,0],axis=0)))),
             'noise_std_first2':float(np.sqrt(np.mean(np.var(p,axis=0)))),
             'same_observation_fresh_noise_first_delta':stats(p[1:,:,0]-p[:-1,:,0]),
             'first_vs_previous_target':stats(p[:,:,0]-obs[None,sl,-1,22:44]),
             'within_chunk_second_vs_first':stats(p[:,:,1]-p[:,:,0]),
             'replan_boundary_same_noise':stats(next_same[sl,0]-p[0,:,1]),
             'replan_boundary_fresh_noise':stats(next_fresh[sl,0]-p[0,:,1]),
             'q_jitter_response':stats(pert[sl]-p[0]),
             'output_over_q_jitter_rms_gain':rms(pert[sl]-p[0])/rms(perturb[sl]),
             'command_residual':stats(p-obs[None,sl,-1,None,:22])}
        if label=='train': row['first2_mse']=float(np.mean((p-truth[None])**2))
        bysource[label]=row
    result['samplers'][str(steps)]={'timesteps':times,'start_alpha_bar':alpha,'start_snr':alpha/(1-alpha),'metrics':bysource}
    print('steps',steps,'alpha',alpha,'train noise',bysource['train']['noise_std_first2'],'rollout noise',bysource['rollout']['noise_std_first2'],'rollout gain',bysource['rollout']['output_over_q_jitter_rms_gain'],flush=True)
    save()

# Separate start timestep from number of reverse evaluations. Offline ablations only.
result['fixed_start_ablations']={}
for start,steps in [(75,8),(75,16),(90,4),(90,8)]:
    times=np.rint(np.linspace(start,0,steps)).astype(int).tolist()
    ps=np.stack([sample(obs,noise[k],times) for k in range(4)])
    result['fixed_start_ablations'][f'start{start}_steps{steps}']={'timesteps':times,
        'train_first2_mse':float(np.mean((ps[:,:256]-truth[None])**2)),
        'train_noise_std_first2':float(np.sqrt(np.mean(np.var(ps[:,:256],axis=0)))),
        'rollout_noise_std_first2':float(np.sqrt(np.mean(np.var(ps[:,256:],axis=0)))),
        'vs_standard4_same_noise_rms':rms(ps-predictions['4'][:4]),
        'vs_standard16_same_noise_rms':rms(ps-predictions['16'][:4])}
    print('ablation',start,steps,result['fixed_start_ablations'][f'start{start}_steps{steps}'],flush=True)
    save()
np.savez_compressed(OUT/'predictions.npz',**predictions)
print('COMPLETE',flush=True)
