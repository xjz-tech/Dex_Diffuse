"""Replay the largest first-20s guided target jump at the exact saved observation.

Load the experiment server's model/predict definitions without its validation or
socket loop. This does not step physics or alter the original rollout.
"""
import sys, json
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]).resolve()
source = (out/'code/server.py').read_text()
namespace = {'__name__':'spike_model'}
exec(compile(source.split('\nold=PROJECT')[0], str(out/'code/server.py'), 'exec'), namespace)
torch = namespace['torch']
predict = namespace['predict']
prior = namespace['prior']
with np.load(out/'guide10k_scale25/rollouts.npz') as z:
    actions = z['action']; raw = z['raw']; q = z['qpos']; active = z['active']
diff = abs(np.diff(actions[:600,32:48], axis=0))
diff[~(active[1:600,32:48] & active[:599,32:48])] = 0
s, k, joint = np.unravel_index(diff.argmax(), diff.shape)
target_step = int(s+1)
chunk_start = target_step//2*2
assert target_step == chunk_start+1, 'This audit expects an intra-chunk spike'
ids = np.flatnonzero(active[chunk_start])
obs = np.concatenate([q[chunk_start-4:chunk_start,ids], actions[chunk_start-4:chunk_start,ids], actions[chunk_start-4:chunk_start,ids]-q[chunk_start-4:chunk_start,ids]], axis=2).transpose(1,0,2)
chunks = chunk_start//2+1
noise = np.stack([np.random.default_rng(50+i-32).standard_normal((chunks,12,22))[-1].astype(np.float32) for i in ids])
gnoise = np.stack([np.random.default_rng(100050+i-32).standard_normal((chunks,12,22))[-1].astype(np.float32) for i in ids])
results = {}
row = int(np.flatnonzero(ids == 32+k)[0])
for arm, name in enumerate(['ordinary_1b', 'guided_scale0', 'guide10k_scale25']):
    pred, refs, _, _ = predict(obs, noise, gnoise, [arm]*len(ids))
    results[name] = {'joint_targets_rad':pred[row,:,joint].tolist(), 'joint_delta_rad':float(abs(pred[row,1,joint]-pred[row,0,joint])), 'max_joint_delta_rad':float(abs(pred[row,1]-pred[row,0]).max())}
    if arm == 2:
        error = float(abs(pred-raw[chunk_start:chunk_start+2,ids].transpose(1,0,2)).max())
        assert error < 1e-5, error
        ref_raw = prior.normalizer['action'].unnormalize(refs).detach().cpu().numpy()
        results['guide_reference'] = {'joint_targets_rad':ref_raw[row,:,joint].tolist(), 'joint_delta_rad':float(abs(ref_raw[row,1,joint]-ref_raw[row,0,joint]))}
        results['replay_max_error_rad'] = error
results.update(seed=int(50+k), joint_index=int(joint), target_time_s=(target_step+1)/30, chunk_start_step=chunk_start, batch_size=len(ids), interpretation='Same saved guided-policy observation and same prior noise; ordinary and scale0 predictions are counterfactual at this observation, not their independent rollout at this time.')
(out/'action_spike_audit.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
