from pathlib import Path
import json
import numpy as np
import zarr

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent

def stats(x):
    return {'transitions':len(x),'joint_values':int(x.size),'mean_abs_rad':float(x.mean()),
            'percentiles_rad':dict(zip(['p50','p90','p95','p99','max'],map(float,np.percentile(x,[50,90,95,99,100])))),
            'thresholds':{str(v):{'joint_fraction':float((x>v+1e-6).mean()),'step_any_fraction':float((x>v+1e-6).any(axis=1).mean())} for v in (.03,.045,.06,.09)},
            'mean_per_joint_rad':x.mean(axis=0).tolist()}

r=zarr.open(str(ROOT/'data/sim_hand_10k_seed42/replay_buffer.zarr'),mode='r')
o=r['data/obs'][:];a=r['data/action'][:];ends=r['meta/episode_ends'][:]
valid=np.ones(len(a),bool);valid[np.r_[0,ends[:-1]]]=False
assert np.allclose(o[1:,22:44][valid[1:]],a[:-1][valid[1:]],rtol=0,atol=0)
result={'scope':'RL simulation training trajectories, not the guided policy evaluation rollout; episode-boundary jumps excluded.',
        '10k_target_delta':stats(np.abs(a-o[:,22:44])[valid]),
        '10k_measured_joint_delta':stats(np.abs(np.diff(o[:,:22],axis=0))[valid[1:]])}
source=Path('/home/carus/Data/exp_data/exp_data_mmap_obs4_h12')
a=np.load(source/'action.npy',mmap_mode='r');o=np.load(source/'obs.npy',mmap_mode='r');ends=np.load(source/'episode_ends.npy')
assert a.shape==o.shape and a.shape[1]==22
rng=np.random.default_rng(42);starts=np.sort(rng.choice(len(a)-513,size=256,replace=False))
deltas=[];qdeltas=[]
for start in starts:
    idx=np.arange(start+1,start+513)
    valid=~np.isin(idx,ends[:-1])
    deltas.append(np.abs(np.diff(a[start:start+513],axis=0))[valid])
    qdeltas.append(np.abs(np.diff(o[start:start+513],axis=0))[valid])
result['global_sample']={'source':str(source),'seed':42,'sampling':'256 random contiguous windows of 512 transitions, excluding episode boundaries','window_starts':starts.tolist(),
                         'target_delta':stats(np.concatenate(deltas)),
                         'measured_joint_delta':stats(np.concatenate(qdeltas))}
(OUT/'statistics.json').write_text(json.dumps(result,indent=2)+'\n')
for name,v in [('10k target',result['10k_target_delta']),('10k qpos',result['10k_measured_joint_delta']),('global target',result['global_sample']['target_delta'])]:
    print(name,json.dumps({k:v[k] for k in ('transitions','mean_abs_rad','percentiles_rad','thresholds')}))
