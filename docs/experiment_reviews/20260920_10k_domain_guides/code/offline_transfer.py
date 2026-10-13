"""CPU-only held-out expert action comparison; not a closed-loop success metric."""
from pathlib import Path
from types import MethodType
import json, sys
import numpy as np
import torch, zarr
from omegaconf import OmegaConf

r = Path(__file__).resolve().parents[1]
root = r.parents[2]
sys.path[:0] = [str(root / '.worktrees/sim-real-holding-comparison/eval/real'),
               str(root / '.worktrees/sim-real-holding-comparison/eval'), str(root)]
from policy_loader import load_policy
from real_sim_policy_server import sample

OmegaConf.register_new_resolver('eval', eval, replace=True)
torch.set_num_threads(4)
domains = ['light_low', 'light_high', 'heavy_low', 'heavy_high']
histories, targets, metadata = [], [], []
for domain in domains:
    path = root / 'data' / f'domain10k_20260920_{domain}'
    manifest = json.loads((path / 'distribution_manifest.json').read_text())
    z = zarr.open_group(str(path / 'replay_buffer.zarr'), mode='r')
    obs, act = z['data/obs'][:], z['data/action'][:]
    for ep in manifest['episodes_detail']:
        if ep['split'] != 'validation':
            continue
        start = ep['episode_index'] * 250
        for t in range(0, 250, 25):
            histories.append(obs[start + np.maximum(np.arange(t - 3, t + 1), 0)])
            targets.append(act[start + t:start + t + 2])
            metadata.append({'domain': domain, 'episode_index': ep['episode_index'], 'local_step': t})
histories, targets = np.stack(histories), np.stack(targets)
# Reuse the same 40 noises across domains and all models.
noise = np.tile(np.random.default_rng(20260920).standard_normal((40, 12, 22)).astype(np.float32), (4, 1, 1))
models = {'old10k': root / 'runs/sim_hand_10k_seed42/checkpoints/latest.ckpt'}
models.update({x['name']: Path(x['checkpoint_expected']) for x in json.loads((r / 'guide_registry.json').read_text())})
output = {}
predictions = {}
for name, path in models.items():
    loaded, policy, spec = load_policy(path, torch.device('cpu'), 'ddim', 4)
    del loaded
    policy.conditional_sample = MethodType(sample, policy)
    batches = []
    with torch.no_grad():
        for start in range(0, len(histories), 16):
            policy.recorded_noise = torch.tensor(noise[start:start + 16])
            p = policy.predict_action({'obs': torch.tensor(histories[start:start + 16])})['action_pred'][:, 3:5]
            batches.append(p.numpy())
    pred = np.concatenate(batches)
    assert pred.shape == targets.shape and np.isfinite(pred).all()
    predictions[name] = pred
    output[name] = {}
    for domain in domains:
        mask = np.array([x['domain'] == domain for x in metadata])
        err = pred[mask] - targets[mask]
        output[name][domain] = {'windows': int(mask.sum()), 'mean_abs_action_error_rad': float(np.abs(err).mean()),
                                'action_rmse_rad': float(np.sqrt(np.square(err).mean()))}
    print(name, output[name], flush=True)
    del policy
np.savez_compressed(r / 'offline_transfer_predictions.npz', targets=targets, histories=histories, noise=noise, **predictions)
(r / 'offline_transfer.json').write_text(json.dumps({
    'protocol': '4 shared held-out templates x 10 windows per physical domain, first two executable target actions, DDIM4, fixed shared noise, physical radians, CPU FP32.',
    'limitations': 'Teacher-conditioned open-loop imitation only; not guidance success or proof of a failure mechanism. One noise per window. All five new guides exclude these four template identities from training.',
    'normalization_note': 'Held out means excluded from gradient updates. The existing project normalizer uses the full replay buffer, including validation segments.',
    'window_metadata': metadata, 'models': output}, indent=2))
