"""Check the evaluator's fused update against production autograd guidance."""
from pathlib import Path
from types import MethodType
import sys, json
import torch, numpy as np
from omegaconf import OmegaConf

r = Path(__file__).resolve().parents[1]
root = r.parents[2]
sys.path[:0] = [str(root / '.worktrees/sim-real-holding-comparison/eval/real'),
               str(root / '.worktrees/sim-real-holding-comparison/eval'), str(root)]
from policy_loader import load_policy
from real_sim_policy_server import sample
from trt_unet import _cached_fused_coeffs, fused_guided_ddim_update
from diffusion_policy.guidance.guided_ddim import sample_guided_trajectory

OmegaConf.register_new_resolver('eval', eval, replace=True)
torch.set_num_threads(4)
_, prior, _ = load_policy('/home/carus/data_usb/obs_4-66.ckpt', torch.device('cpu'), 'ddim', 4)
ids = [0, 12, 24, 36]
q = np.load(r / 'evaluation/ordinary_1b/initial_state.npz')['q'][ids]
hist = torch.tensor(np.repeat(np.concatenate([q, q, np.zeros_like(q)], axis=-1)[:, None], 4, axis=1))
noise = torch.tensor(np.tile(np.random.default_rng(8).standard_normal((1, 12, 22)).astype(np.float32), (4, 1, 1)))
gnoise = torch.tensor(np.tile(np.random.default_rng(100008).standard_normal((1, 12, 22)).astype(np.float32), (4, 1, 1)))
cond = prior.normalizer['obs'].normalize(hist).reshape(4, -1)
rows = []
for entry in json.loads((r / 'guide_registry.json').read_text()):
    loaded, guide, _ = load_policy(entry['checkpoint_expected'], torch.device('cpu'), 'ddim', 4)
    del loaded
    guide.conditional_sample = MethodType(sample, guide)
    guide.recorded_noise = gnoise
    with torch.no_grad():
        ref = prior.normalizer['action'].normalize(guide.predict_action({'obs': hist})['action_pred'][:, 3:5])
        coeffs = _cached_fused_coeffs(prior, noise)
        fused = noise.clone()
        for i, t in enumerate(coeffs.timesteps):
            eps = prior.model(fused, t, local_cond=None, global_cond=cond)
            fused = fused_guided_ddim_update(fused, eps, i, coeffs, ref, 25., slice(3, 5))
    production = sample_guided_trajectory(prior.model, prior.noise_scheduler, noise.clone(), cond, ref, 4, 25., slice(3, 5)).trajectory
    max_error = float((fused - production).abs().max())
    rad_error = float((prior.normalizer['action'].unnormalize(fused[:, 3:5]) - prior.normalizer['action'].unnormalize(production[:, 3:5])).abs().max())
    assert max_error < 2e-5 and rad_error < 2e-5, (entry['name'], max_error, rad_error)
    rows.append({'guide': entry['name'], 'prior_ddim_timesteps': [int(t) for t in coeffs.timesteps],
                 'max_normalized_trajectory_error': max_error, 'max_executed_target_error_rad': rad_error,
                 'guide_scale': 25, 'guided_action_slice': [3, 5], 'passed': True})
    print(rows[-1], flush=True)
    del guide
(r / 'guided_sampler_verification.json').write_text(json.dumps(rows, indent=2))
