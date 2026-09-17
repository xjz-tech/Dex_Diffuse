"""Select exact, reproducible episode-start rows; no hardware I/O."""
import json
from pathlib import Path
import sys
import hashlib
import h5py
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'eval/real'))
from hardware import POLICY_LOWER_LIMITS,POLICY_UPPER_LIMITS,POLICY_SHARPA_DOF_NAMES,REAL_SHARPA_DOF_NAMES,policy_to_real
DATA=Path('/media/frankagvl/U393/20260907173246')
for line in (ROOT/'reports/hold_comparison_20260917/pair1_seed50_hold0.jsonl').open():
    row=json.loads(line)
    if row['stage']=='inference' and row['event']=='model':
        stats=row['components']['prior']['normalizer_state'];break
mean=np.asarray(stats['params_dict.obs.input_stats.mean'][:22])
std=np.asarray(stats['params_dict.obs.input_stats.std'][:22])
rng=np.random.default_rng(20260918)
selected=[]
for shard in (75,317,589):
    path=DATA/f'shards/shard_{shard:06d}.h5'
    with h5py.File(path,'r') as f:
        candidates=np.flatnonzero((f['index/step'][:]==0)&~f['index/done'][:]&~f['index/failure'][:])
        q=f['robot/qpos'][candidates].astype(np.float64)
        targets=f['robot/target_before'][candidates].astype(np.float64)
        mask=(np.isfinite(q).all(1)&(q>=POLICY_LOWER_LIMITS-1e-7).all(1)&(q<=POLICY_UPPER_LIMITS+1e-7).all(1)&
              (np.max(abs(q-targets),axis=1)<=1e-5)&(np.max(abs((q-mean)/np.maximum(std,1e-8)),axis=1)<=3))
        # The first attempt at +0.3491 rad did not settle on this physical hand.
        # Choose another exact row; do not clip the dataset pose or relax tolerance.
        mask &= np.abs(q[:, POLICY_SHARPA_DOF_NAMES.index('right_index_MCP_AA')]) <= .18
        for prior in selected:
            mask &= np.sqrt(np.mean((q-np.asarray(prior['qpos_policy_rad']))**2,axis=1))>=.15
        qualified=candidates[mask]
        assert len(qualified)>0
        index=int(rng.choice(qualified))
        values=f['robot/qpos'][index].astype(np.float64)
        previous=f['robot/target_before'][index].astype(np.float64)
        action=f['robot/target_after'][index].astype(np.float64)
        entry=dict(name=f'dataset_pose_{len(selected)+1}',shard=str(path),row=index,episode_id=int(f['index/episode_id'][index]),
                   env_id=int(f['index/env_id'][index]),step=int(f['index/step'][index]),qualifying_rows_in_shard=len(qualified),
                   qpos_policy_rad=values.tolist(),qpos_real_rad=policy_to_real(values).tolist(),
                   target_before_policy_rad=previous.tolist(),target_after_policy_rad=action.tolist(),
                   source_target_step_max_rad=float(np.max(abs(action-previous))),
                   max_abs_marginal_z=float(np.max(abs((values-mean)/std))))
        assert np.array_equal(f['robot/qpos'][index],entry['qpos_policy_rad'])
        selected.append(entry)
        print(entry['name'],'shard',shard,'row',index,'episode',entry['episode_id'],'max_z',entry['max_abs_marginal_z'])
        (OUT/(entry['name']+'.json')).write_text(json.dumps(entry,ensure_ascii=False,indent=2)+'\n')
manifest=dict(dataset_root=str(DATA),dataset_manifest_sha256=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
 selection_seed=20260918,inference_seed=50,criteria='step=0, not done/failure, finite, URDF bounds (1e-7 float tolerance), target_before=qpos within 1e-5, max marginal z<=3, abs(index MCP_AA)<=0.18rad after failed physical initialization, pairwise joint RMS>=0.15rad; one reproducible random row from each of shards 75/317/589; no grasp/orientation filter',
 dataset_identity_note='本地数据集；与checkpoint原训练目录的完全等价性未核实。',
 policy_joint_names=POLICY_SHARPA_DOF_NAMES,real_joint_names=REAL_SHARPA_DOF_NAMES,poses=selected)
(OUT/'selected_poses.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
