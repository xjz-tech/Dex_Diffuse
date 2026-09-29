"""Offline synthetic reset-observation probe; not a simulator experiment."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path[:0] = [str(ROOT / 'eval'), str(ROOT)]
from checkpoint_loader import load_checkpoint, build_policy, configure_policy_sampler
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
source = np.load(OUT / 'observations.npz')['obs'][:256]
q = source[:, -1, :22]
single = np.concatenate([q, q, np.zeros_like(q)], axis=-1)
obs = np.repeat(single[:, None], 4, axis=1)
noise = np.random.default_rng(20260918).standard_normal((8, 256, 12, 22)).astype(np.float32)
policy, spec = build_policy(load_checkpoint('/home/carus/data_usb/obs_4-66.ckpt', allow_salvage=False))
policy.cuda().eval()
configure_policy_sampler(policy, 'ddim', 4)
result = {'scope': 'Synthetic repeat([q,q,0],4) using 256 training q poses; no physical reset, no expert labels, no closed-loop causal test.'}
with torch.inference_mode():
    for steps in [4, 8, 16]:
        policy.noise_scheduler.set_timesteps(steps)
        preds = []
        for k in range(8):
            chunks = []
            for b in range(0, 256, 64):
                cond = policy.normalizer['obs'].normalize(torch.tensor(obs[b:b+64], device='cuda')).reshape(-1, 264)
                x = torch.tensor(noise[k,b:b+64], device='cuda')
                for t in policy.noise_scheduler.timesteps:
                    eps = policy.model(x, t, global_cond=cond)
                    x = policy.noise_scheduler.step(eps, t, x).prev_sample
                chunks.append(policy.normalizer['action'].unnormalize(x).cpu().numpy()[:,3:5])
            preds.append(np.concatenate(chunks))
        p = np.stack(preds)
        delta = p[:,:,0] - q[None]
        result[str(steps)] = {
            'noise_std_first2': float(np.sqrt(np.var(p,axis=0).mean())),
            'first_delta_rms': float(np.sqrt(np.mean(delta**2))),
            'first_delta_joint_over_009': float((abs(delta)>.090001).mean()),
            'first_delta_any_joint_over_009': float((abs(delta)>.090001).any(axis=-1).mean()),
        }
        print(steps, result[str(steps)], flush=True)
(OUT / 'reset_probe.json').write_text(json.dumps(result, indent=2)+'\n')
