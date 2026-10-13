"""Bounded offline audit of data, native/eval parity, and loss interpretation."""
import ast
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import dill
import h5py
import hydra
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/'eval')]
from checkpoint_loader import LoadedCheckpoint, build_policy, configure_policy_sampler
from policy_observation import compose_policy_observation, QPOS_TARGET_RESIDUAL_OBSERVATION
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.replay_buffer import ReplayBuffer

SOURCE = Path('/home/carus/Data/exp_data')
MMAP = SOURCE/'exp_data_mmap_obs4_h12'
CKPT = Path('/home/carus/data_usb/obs_4-66.ckpt')
torch.set_num_threads(4)
result = {'scope':'Offline diagnostics only. No simulator, hardware, training, or protocol changes.', 'checkpoint':str(CKPT)}
def save():
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
def maximum(a,b): return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def array_stats(a):
    a=np.asarray(a,dtype=np.float64)
    return {'mse':float(np.mean(a*a)), 'rmse':float(np.sqrt(np.mean(a*a))),
            'mae':float(np.mean(np.abs(a))), 'abs_p95':float(np.percentile(np.abs(a),95)),
            'abs_p99':float(np.percentile(np.abs(a),99)), 'abs_max':float(np.max(np.abs(a)))}
def delta_stats(d):
    d=np.abs(d)
    return {'joint_over_009':float((d>.090001).mean()),
            'step_any_over_009':float((d>.090001).any(axis=-1).mean()),
            'max_delta_p50':float(np.median(d.max(axis=-1))),
            'max_delta_p95':float(np.percentile(d.max(axis=-1),95)),
            'max':float(d.max())}

payload=torch.load(CKPT,map_location='cpu',pickle_module=dill,weights_only=False)
cfg=payload['cfg']
epoch=dill.loads(payload['pickles']['epoch'])
global_step=dill.loads(payload['pickles']['global_step'])
(OUT/'checkpoint_config.yaml').write_text(OmegaConf.to_yaml(cfg))
result['checkpoint_info']={'epoch':epoch,'global_step':global_step,'wandb_id':cfg.logging.id,
                           'original_dataset_path':cfg.task.dataset_path,'use_ema':cfg.training.use_ema}
loaded=LoadedCheckpoint(cfg,payload['state_dicts']['ema_model'],'EMA model',global_step,epoch,False)
policy,spec=build_policy(loaded)
native=hydra.utils.instantiate(cfg.policy)
native.set_normalizer(policy.normalizer)
native.load_state_dict(loaded.state_dict,strict=True)
policy=policy.cuda().eval()
native=native.cuda().eval()
result['spec']=spec

q=np.load(MMAP/'obs.npy',mmap_mode='r')
a=np.load(MMAP/'action.npy',mmap_mode='r')
ends=np.load(MMAP/'episode_ends.npy')
starts=np.r_[0,ends[:-1]]
episode_ids=np.load(MMAP/'episode_ids.npy')
env_ids=np.load(MMAP/'env_ids.npy')
val=np.load(MMAP/'val_mask.npy')
metadata=json.loads((MMAP/'metadata.json').read_text())
manifest=json.loads((SOURCE/'manifest.json').read_text())
result['dataset']={'q_shape':list(q.shape),'action_shape':list(a.shape),
                   'episodes':len(ends),'split_matches_seed42':bool(np.array_equal(val,get_val_mask(len(ends),.1,42))),
                   'manifest_hash_matches_cache':hashlib.sha256((SOURCE/'manifest.json').read_bytes()).hexdigest()==metadata['source']['manifest_sha256'],
                   'policy_dt':manifest['metadata']['policy_dt']}

cache_norm=np.load(MMAP/'normalizer.npz')
norm_check={}
for field in ['obs','action']:
    for stat in ['min','max','mean','std']:
        actual=loaded.state_dict[f'normalizer.params_dict.{field}.input_stats.{stat}'].numpy()
        norm_check[field+'_'+stat]=maximum(actual[:22],cache_norm[field+'_'+stat])
result['normalizer_vs_full_source_cache_max_abs']=norm_check

# Validate mmap materialization and target_before semantics against raw HDF5.
catalog={(int(ep),int(en)):i for i,(ep,en) in enumerate(zip(episode_ids,env_ids))}
h5checks=[]
for shard_i in [0,50,99]:
    shard=manifest['shards'][shard_i]
    with h5py.File(SOURCE/shard['path'],'r') as f:
        offset=min(100000,len(f['index/step'])//2)
        sl=slice(offset,offset+4096)
        ep=f['index/episode_id'][sl]; en=f['index/env_id'][sl]; st=f['index/step'][sl]
        keep=np.array([(int(e),int(n)) in catalog for e,n in zip(ep,en)])
        cat=np.array([catalog[(int(e),int(n))] for e,n in zip(ep[keep],en[keep])])
        ix=starts[cat]+st[keep]
        qr=f['robot/qpos'][sl][keep]; ar=f['robot/target_after'][sl][keep]
        before=f['robot/target_before'][sl][keep]
        interior=st[keep]>0
        h5checks.append({'shard':shard['path'],'rows':len(ix),
            'q_max_abs_error':maximum(q[ix],qr),'action_max_abs_error':maximum(a[ix],ar),
            'target_before_vs_prev_action_error':maximum(before[interior],a[ix[interior]-1]),
            'episode_indices_in_bounds':bool(np.all(ix<ends[cat]))})
result['raw_hdf5_checks']=h5checks
print('data checks',json.dumps(result['dataset']),json.dumps(h5checks),flush=True)

# Sample real training/validation windows, matching the unpadded interior sampler.
rng=np.random.default_rng(20260918)
selected={False:[],True:[]}
while len(selected[False])<1024 or len(selected[True])<256:
    ix=int(rng.integers(4,len(a)-12)); ep=int(np.searchsorted(ends,ix,side='right'))
    split=bool(val[ep]); cap=256 if split else 1024
    if len(selected[split])<cap and ix>=starts[ep]+4 and ix+8<ends[ep]:
        selected[split].append(ix)
indices=np.array(selected[False]+selected[True])
ix=indices[:,None]+np.arange(-3,9)[None]
qs=np.asarray(q[ix]); ts=np.asarray(a[ix-1]); acts=np.asarray(a[ix])
obs=np.concatenate([qs,ts,ts-qs],axis=-1)
np.savez_compressed(OUT/'samples.npz',indices=indices,obs=obs,action=acts)
previous=ts[:,3]
target=acts[:,3:12]
result['sample_sizes']={'train':1024,'validation':256,'padding':'none; interior windows'}
result['observation_builder_max_abs_error']=maximum(compose_policy_observation(qs,ts,QPOS_TARGET_RESIDUAL_OBSERVATION),obs)

# Run actual SequenceSampler on independently extracted episode interiors.
small_obs=obs[:8].reshape(-1,66); small_actions=acts[:8].reshape(-1,22)
rb=ReplayBuffer(root={'data':{'obs':small_obs,'action':small_actions},'meta':{'episode_ends':np.arange(1,9)*12}})
sampler=SequenceSampler(rb,12,pad_before=3,pad_after=8)
matches=[]
for k in range(8):
    j=int(np.flatnonzero(np.all(sampler.indices==[k*12,(k+1)*12,0,12],axis=1))[0])
    sample=sampler.sample_sequence(j)
    matches.append(max(maximum(sample['obs'],obs[k]),maximum(sample['action'],acts[k])))
result['sequence_sampler_interior_max_abs_error']=max(matches)

normed=policy.normalizer['obs'].normalize(torch.as_tensor(obs[:64],device='cuda'))
roundtrip=policy.normalizer['obs'].unnormalize(normed).cpu().detach().numpy()
result['obs_normalizer_roundtrip_error']=maximum(roundtrip,obs[:64])
action_scale=policy.normalizer['action'].params_dict['scale'].detach().cpu().numpy()
result['action_normalizer_scale_range']=[float(action_scale.min()),float(action_scale.max())]

# Native training class and inference adapter must agree with identical RNG.
b={'obs':torch.as_tensor(obs[:64],device='cuda'),'action':torch.as_tensor(acts[:64],device='cuda')}
parity={}
with torch.inference_mode():
    torch.manual_seed(123)
    ln=native.compute_loss(b)
    torch.manual_seed(123)
    le=policy.compute_loss(b)
    parity['noise_loss_abs_error']=float(abs(ln-le))
    parity['noise_loss']=float(ln)
    for sampler_name,steps in [('ddpm',100),('ddim',4)]:
        native.noise_scheduler=DDPMScheduler(**{k:v for k,v in OmegaConf.to_container(cfg.policy.noise_scheduler).items() if k!='_target_'})
        policy.noise_scheduler=DDPMScheduler.from_config(native.noise_scheduler.config)
        configure_policy_sampler(native,sampler_name,steps); configure_policy_sampler(policy,sampler_name,steps)
        torch.manual_seed(321); pn=native.predict_action({'obs':b['obs']})['action']
        torch.manual_seed(321); pe=policy.predict_action({'obs':b['obs']})['action']
        parity[sampler_name+'_action_max_abs_error']=float((pn-pe).abs().max())
        torch.manual_seed(321); ph=policy.predict_action({'obs':b['obs'][:,:4]})['action']
        parity[sampler_name+'_full_vs_history_only_max_abs_error']=float((ph-pe).abs().max())
result['native_eval_parity']=parity
print('policy parity',json.dumps(parity),flush=True)
del native
torch.cuda.empty_cache()
save()

def predict(model,sampler_name,steps,seed,count):
    # Reconstruct DDPM first: setting sampler='ddpm' does not convert an already-DDIM scheduler.
    model.noise_scheduler=DDPMScheduler(**{k:v for k,v in OmegaConf.to_container(cfg.policy.noise_scheduler).items() if k!='_target_'})
    configure_policy_sampler(model,sampler_name,steps)
    model.n_action_steps=9
    chunks=[]
    with torch.inference_mode():
        for start in range(0,count,64):
            torch.manual_seed(seed+start)
            chunks.append(model.predict_action({'obs':torch.as_tensor(obs[start:start+64,:4],device='cuda')})['action'].cpu().numpy())
    return np.concatenate(chunks)

def prediction_metrics(pred,truth,prev):
    err=pred-truth
    metrics={'first':array_stats(err[:,0]),'first2':array_stats(err[:,:2]),
             'first5_training_metric':array_stats(err[:,:5]),'all9':array_stats(err),
             'mse_by_future_step':np.mean(err.astype(np.float64)**2,axis=(0,2)).tolist(),
             'first_delta':delta_stats(pred[:,0]-prev),
             'within_chunk_delta':delta_stats(np.diff(pred,axis=1))}
    return metrics

result['predictions']={}
saved_predictions={}
for name,steps in [('ddpm',100),('ddim',4),('ddim',8),('ddim',100)]:
    pred=predict(policy,name,steps,4200,len(obs))
    key=f'ema_{name}{steps}'
    saved_predictions[key]=pred
    metrics={}
    for split,sl in [('train',slice(0,1024)),('validation',slice(1024,None))]:
        metrics[split]=prediction_metrics(pred[sl],target[sl],previous[sl])
    result['predictions'][key]=metrics
    print(key,'train first5',metrics['train']['first5_training_metric'],'first',metrics['train']['first'],flush=True)
    save()

# Stochasticity and base vs EMA: matched first 256 training windows.
for label,state in [('ema_second_noise',loaded.state_dict),('base',payload['state_dicts']['model'])]:
    policy.load_state_dict(state,strict=True)
    policy.cuda()
    pred=predict(policy,'ddpm',100,4300 if label=='ema_second_noise' else 4200,256)
    saved_predictions[label]=pred
    result['predictions'][label]=prediction_metrics(pred,target[:256],previous[:256])
    if label=='ema_second_noise':
        result['two_samples_same_history']=array_stats(pred-saved_predictions['ema_ddpm100'][:256])
    print(label,result['predictions'][label]['first5_training_metric'],flush=True)
    save()

# Teacher-forced noise loss across every training noise timestep, not action-space MSE.
result['denoising']={}
for label,state in [('base',payload['state_dicts']['model']),('ema',loaded.state_dict)]:
    policy.load_state_dict(state,strict=True)
    policy.cuda()
    sched=DDPMScheduler(**{k:v for k,v in OmegaConf.to_container(cfg.policy.noise_scheduler).items() if k!='_target_'})
    nb=policy.normalizer.normalize({'obs':b['obs'],'action':b['action']})
    cond=nb['obs'][:,:4].reshape(64,-1)
    eps_mse=[]; x0_mse=[]
    with torch.inference_mode():
        for t in range(100):
            torch.manual_seed(6000+t)
            noise=torch.randn_like(nb['action'])
            time=torch.full((64,),t,device='cuda',dtype=torch.long)
            xt=sched.add_noise(nb['action'],noise,time)
            eps=policy.model(xt,time,global_cond=cond)
            eps_mse.append(float(((eps-noise)**2).mean()))
            alpha=sched.alphas_cumprod[t].to('cuda')
            x0=(xt-torch.sqrt(1-alpha)*eps)/torch.sqrt(alpha)
            x0_mse.append(float(((x0-nb['action'])**2).mean()))
    result['denoising'][label]={'mean_epsilon_mse':float(np.mean(eps_mse)),
                              'epsilon_mse_by_t':eps_mse,'unclipped_normalized_x0_mse_by_t':x0_mse}
    print('epsilon',label,np.mean(eps_mse),flush=True)
    save()

# First-step boundaries: measure support leakage separately from reconstruction error.
pred=saved_predictions['ema_ddim4'][:1024,0]
gt=target[:1024,0]; prev=previous[:1024]
delta_gt=gt-prev; delta_pred=pred-prev
sat=np.abs(np.abs(delta_gt)-.09)<1e-6
over=np.abs(delta_pred)>.090001
result['boundary_analysis']={'gt_saturation_joint_fraction':float(sat.mean()),
 'gt_over_009':delta_stats(delta_gt),
 'pred_over_009_fraction':float(over.mean()),
 'fraction_of_overlimit_predictions_whose_gt_is_saturated':float((sat&over).sum()/max(1,over.sum())),
 'error_at_saturated_labels':array_stats((pred-gt)[sat]),
 'excess_over_limit_among_exceedances':array_stats((np.abs(delta_pred)-.09)[over])}

# Test the exact target-to-environment adapter without importing or starting Isaac Gym.
tree=ast.parse((ROOT/'eval/sim_eval.py').read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_absolute_targets_to_env_action')
namespace={'torch':torch}
exec(compile(ast.Module(body=[function],type_ignores=[]),'sim_eval_adapter','exec'),namespace)
lower=torch.tensor(manifest['metadata']['joint_lower'])
upper=torch.tensor(manifest['metadata']['joint_upper'])
raw=torch.from_numpy(saved_predictions['ema_ddim4'])
u=namespace['_absolute_targets_to_env_action'](raw,lower,upper,'cpu')
back=lower+(u+1)*.5*(upper-lower)
expected=torch.maximum(torch.minimum(raw,upper),lower)
result['environment_action_adapter']={'roundtrip_vs_clipped_target_max_abs':float((back-expected).abs().max()),
 'physical_limit_clip_max_abs':float((expected-raw).abs().max()),
 'physical_limit_clip_fraction':float((abs(expected-raw)>1e-6).float().mean())}
np.savez_compressed(OUT/'predictions.npz',**saved_predictions)
save()
print('COMPLETE',flush=True)
