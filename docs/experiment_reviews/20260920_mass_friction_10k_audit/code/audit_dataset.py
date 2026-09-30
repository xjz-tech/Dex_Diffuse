from pathlib import Path
import json,sys,hashlib
import numpy as np,h5py,zarr
ROOT=Path(__file__).resolve().parents[4];out=Path(__file__).resolve().parents[1]
source=Path('/home/carus/Data/exp_data');subset=ROOT/'data/sim_hand_10k_seed42'
m=json.loads((subset/'subset_manifest.json').read_text());manifest=json.loads((source/'manifest.json').read_text());episodes=m['episodes'];ids=np.array([e['episode_id'] for e in episodes]);n=len(ids)
z=zarr.open_group(str(subset/'replay_buffer.zarr'),mode='r');ends=np.asarray(z['meta/episode_ends']);starts=np.r_[0,ends[:-1]]
val=np.zeros(n,bool);val[np.random.default_rng(42).choice(n,2,replace=False)]=True
pieces=[{} for e in episodes];schemas=set();fields=['index/step','index/data_index_id','target/data_index_id','target/demo_frame_index','index/sim_frame','index/skip_steps','index/failure','index/success','index/done','robot/qpos','robot/target_before','robot/target_after','fingertip/contact_force','object/linear_velocity_world']
for si,shard in enumerate(manifest['shards']):
 with h5py.File(source/shard['path'],'r') as f:
  names=[];f.visititems(lambda k,v:names.append(k) if isinstance(v,h5py.Dataset) else None);schemas.update(names)
  ep=f['index/episode_id'][:];matching=np.flatnonzero(np.isin(ep,ids))
  if len(matching):
   for j,e in enumerate(episodes):
    indices=matching[ep[matching]==e['episode_id']]
    if not len(indices):continue
    assert np.all(f['index/env_id'][indices]==e['env_id'])
    for key in fields:pieces[j].setdefault(key,[]).append(f[key][indices])
 if si%10==0:print('scanned',si+1,'/',len(manifest['shards']),flush=True)
rows=[];demo_train={};demo_all={};extras={}
for j,e in enumerate(episodes):
 a={k:np.concatenate(v) for k,v in pieces[j].items()};order=np.argsort(a['index/step']);a={k:v[order] for k,v in a.items()};assert np.array_equal(a['index/step'],np.arange(e['length']))
 obs=np.concatenate([a['robot/qpos'],a['robot/target_before'],a['robot/target_before']-a['robot/qpos']],axis=-1)
 assert np.array_equal(obs,z['data/obs'][starts[j]:ends[j]]) and np.array_equal(a['robot/target_after'],z['data/action'][starts[j]:ends[j]])
 a['target/data_index_id']=np.array([int(x.decode().split('@')[-1]) for x in a['target/data_index_id']])
 ds,counts=np.unique(a['target/data_index_id'],return_counts=True)
 for d,c in zip(ds,counts):
  demo_all[int(d)]=demo_all.get(int(d),0)+int(c)
  if not val[j]:demo_train[int(d)]=demo_train.get(int(d),0)+int(c)
 force=np.linalg.norm(a['fingertip/contact_force'],axis=-1)
 row={**e,'split':'validation' if val[j] else 'train','all_transition_pct':100*e['length']/9999,'train_transition_pct':0 if val[j] else 100*e['length']/sum(x['length'] for i,x in enumerate(episodes) if not val[i]),'demo_counts':{int(d):int(c) for d,c in zip(ds,counts)},'first_demo':int(a['target/data_index_id'][0]),'first_demo_frame':int(a['target/demo_frame_index'][0]),'sim_frame_range':[int(a['index/sim_frame'].min()),int(a['index/sim_frame'].max())],'recorded_failure_flags':int(a['index/failure'].sum()),'recorded_success_flags':int(a['index/success'].sum()),'mean_fingertip_force_norm_sum_N':float(force.sum(axis=-1).mean()),'mass_kg':None,'object_friction':None}
 rows.append(row)
 for k,v in a.items():extras[f'ep{j:02d}_{k.replace("/","_")}']=v
train=np.array([e['length'] for j,e in enumerate(episodes) if not val[j]]);w=train/train.sum()
result={'actual_transitions':9999,'episodes':20,'training_episodes':int((~val).sum()),'training_transitions':int(train.sum()),'validation_episodes':int(val.sum()),'validation_transitions':int(sum(e['length'] for j,e in enumerate(episodes) if val[j])),'train_windows_with_padding':int(train.sum()),'train_unpadded_windows':int((train-11).sum()),'largest_train_episode_pct':float(w.max()*100),'top3_train_episode_pct':float(np.sort(w)[-3:].sum()*100),'top5_train_episode_pct':float(np.sort(w)[-5:].sum()*100),'weight_concentration_equivalent_episode_count':float(1/(w@w)),'train_demo_counts':demo_train,'all_demo_counts':demo_all,'training_unique_target_demos':len(demo_train),'all_unique_target_demos':len(demo_all),'hdf5_shards_checked':len(manifest['shards']),'hdf5_field_names':sorted(schemas),'physical_label_fields':[k for k in schemas if any(t in k.lower() for t in ['mass','friction','inertia','scale'])],'source_metadata':manifest['metadata'],'episodes_detail':rows,'subset_matched_source_exactly':True}
(out/'dataset_audit.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'selected_source_fields.npz',**extras)
print(json.dumps({k:v for k,v in result.items() if k not in ['source_metadata','episodes_detail','hdf5_field_names']},indent=2),flush=True)
