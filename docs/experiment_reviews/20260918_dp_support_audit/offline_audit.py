"""Offline conditional-sampling audit; no simulator or hardware is started."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]
from checkpoint_loader import load_checkpoint, build_policy, configure_policy_sampler

OUT = Path(__file__).resolve().parent
SOURCE = Path('/home/carus/Data/exp_data/exp_data_mmap_obs4_h12')

def stats(delta):
    delta = np.abs(delta)
    maximum = delta.max(axis=-1)
    return {'joint_mean': float(delta.mean()),
            'joint_over_009_fraction': float((delta > .090001).mean()),
            'step_any_over_009_fraction': float((maximum > .090001).mean()),
            'step_max_quantiles': dict(zip(['p50','p95','p99','max'],
                map(float, np.percentile(maximum, [50,95,99,100]))))}

torch.set_num_threads(4)
loaded = load_checkpoint(Path('/home/carus/data_usb/obs_4-66.ckpt'), allow_salvage=False)
policy, spec = build_policy(loaded)
policy = policy.cuda().eval()
policy.n_action_steps = 2
q = np.load(SOURCE/'obs.npy', mmap_mode='r')
a = np.load(SOURCE/'action.npy', mmap_mode='r')
ends = np.load(SOURCE/'episode_ends.npy')
val = np.load(SOURCE/'val_mask.npy')
starts = np.r_[0, ends[:-1]]
rng = np.random.default_rng(20260918)
indices = []
while len(indices) < 256:
    t = int(rng.integers(4, len(a)-10))
    ep = int(np.searchsorted(ends, t, side='right'))
    if not val[ep] and t >= starts[ep]+4 and t+9 < ends[ep]:
        indices.append(t)
indices = np.asarray(indices)
hist_idx = indices[:, None] + np.arange(-3, 1)[None]
qh = np.asarray(q[hist_idx])
th = np.asarray(a[hist_idx-1])
obs = np.concatenate([qh, th, th-qh], axis=-1)
previous = np.asarray(a[indices-1])
gt = np.asarray(a[indices[:,None]+np.arange(2)[None]])
result = {'scope':'Offline only; 256 source training-split interior windows; no state rollout, no guidance. Paired diffusion noise across conditions and DDIM step counts. Rates use >0.09+1e-6 rad.',
          'checkpoint':'/home/carus/data_usb/obs_4-66.ckpt', 'weight_source':loaded.weight_source,
          'spec':spec, 'source':str(SOURCE), 'indices':indices.tolist(),
          'checkpoint_task':str(loaded.cfg.get('task',{})),
          'gt_first':stats(gt[:,0]-previous), 'gt_second':stats(gt[:,1]-gt[:,0]), 'conditions':{}}
conditions = {'real_history':obs, 'repeat_current':np.repeat(obs[:,-1:],4,axis=1)}
reset_obs = obs[:,-1].copy()
reset_obs[:,22:44] = reset_obs[:,:22]
reset_obs[:,44:] = 0
conditions['repeat_current_zero_residual'] = np.repeat(reset_obs[:,None],4,axis=1)
for steps in [4,8,16,100]:
    configure_policy_sampler(policy, 'ddim', steps)
    for name, observations in conditions.items():
        predictions=[]
        with torch.inference_mode():
            for start in range(0,len(obs),64):
                torch.manual_seed(42+start)
                prediction=policy.predict_action({'obs':torch.as_tensor(observations[start:start+64], device='cuda')})
                predictions.append(prediction['action'].cpu().numpy())
        pred=np.concatenate(predictions)
        anchor = previous if name != 'repeat_current_zero_residual' else reset_obs[:,:22]
        row={'first_action_vs_previous_target':stats(pred[:,0]-anchor),
             'second_action_vs_first':stats(pred[:,1]-pred[:,0]),
             'mae_to_recorded_actions':float(np.abs(pred-gt).mean())}
        result['conditions'][f'{name}_ddim{steps}']=row
        print(name,steps,json.dumps(row),flush=True)
    (OUT/'statistics.json').write_text(json.dumps(result,indent=2)+'\n')
