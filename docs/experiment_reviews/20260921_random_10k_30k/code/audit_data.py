from pathlib import Path
import sys,json,hashlib
import numpy as np,h5py,zarr
R=Path(__file__).resolve().parents[1]; ROOT=R.parents[2]
sys.path[:0]=[str(ROOT),str(R/'code')]
from build_sim_hand_subset import _load_catalog,select_episode_keys
from diffusion_policy.common.sampler import get_val_mask
source=Path('/home/carus/Data/exp_data')
manifest=json.loads((source/'manifest.json').read_text())
catalog=_load_catalog(source)
specs=[('new10k_seed43',10000,43),('new30k_seed44',30000,44)]
selections={name:select_episode_keys(*catalog,max_transitions=cap,seed=seed) for name,cap,seed in specs}
pieces={(e.episode_id,e.env_id):{} for es in selections.values() for e in es}
ids=np.asarray([k[0] for k in pieces])
fields=['index/step','target/data_index_id','target/demo_frame_index','index/sim_frame','robot/qpos','robot/target_before','robot/target_after']
schemas=set()
for si,shard in enumerate(manifest['shards']):
 with h5py.File(source/shard['path'],'r') as f:
  f.visititems(lambda k,v:schemas.add(k) if isinstance(v,h5py.Dataset) else None)
  ep=f['index/episode_id'][:]; match=np.flatnonzero(np.isin(ep,ids))
  if len(match):
   env=f['index/env_id'][match]
   for eid in np.unique(ep[match]):
    sub=np.flatnonzero(ep[match]==eid)
    for en in np.unique(env[sub]):
     key=(int(eid),int(en))
     if key not in pieces:continue
     ix=match[sub[env[sub]==en]]
     for field in fields:pieces[key].setdefault(field,[]).append(f[field][ix])
 if si%10==0:print('audit shards',si+1,flush=True)
results={};extras={}
for name,cap,seed in specs:
 es=selections[name];val=get_val_mask(len(es),.1,42)
 train_count=sum(e.length for j,e in enumerate(es) if not val[j]);total=sum(e.length for e in es)
 rows=[];train_demos={};all_demos={};obs_parts=[];act_parts=[]
 for j,e in enumerate(es):
  a={k:np.concatenate(v) for k,v in pieces[(e.episode_id,e.env_id)].items()};order=np.argsort(a['index/step']);a={k:v[order] for k,v in a.items()}
  assert np.array_equal(a['index/step'],np.arange(e.length))
  q=a['robot/qpos'].astype(np.float32);target=a['robot/target_before'].astype(np.float32);action=a['robot/target_after'].astype(np.float32)
  obs=np.concatenate([q,target,target-q],axis=-1);obs_parts.append(obs);act_parts.append(action)
  demo=np.asarray([int(x.decode().split('@')[-1]) for x in a['target/data_index_id']]);ds,counts=np.unique(demo,return_counts=True)
  for d,c in zip(ds,counts):
   all_demos[int(d)]=all_demos.get(int(d),0)+int(c)
   if not val[j]:train_demos[int(d)]=train_demos.get(int(d),0)+int(c)
  rms=np.sqrt(np.mean(obs[:,44:]**2,axis=-1))
  row={'episode_id':e.episode_id,'env_id':e.env_id,'length':e.length,'split':'validation' if val[j] else 'train','train_window_pct':0 if val[j] else 100*e.length/train_count,'first_target_demo':int(demo[0]),'first_target_frame':int(a['target/demo_frame_index'][0]),'target_demo_counts':{int(d):int(c) for d,c in zip(ds,counts)},'zero_residual_rows':int((rms<1e-8).sum()),'first_residual_rms_rad':float(rms[0]),'mass_kg':None,'object_friction':None}
  rows.append(row)
  extras[name+'_ep%02d_demo'%j]=demo
  extras[name+'_ep%02d_frame'%j]=a['target/demo_frame_index']
  extras[name+'_ep%02d_simframe'%j]=a['index/sim_frame']
 obs=np.concatenate(obs_parts);actions=np.concatenate(act_parts)
 # Store source-derived arrays for subsequent exact comparison to built Zarr.
 np.savez_compressed(R/f'{name}_source_arrays.npz',obs=obs,action=actions)
 lens=np.asarray([e.length for j,e in enumerate(es) if not val[j]]);w=lens/lens.sum()
 results[name]={'sampling_seed':seed,'requested_rows':cap,'actual_rows':total,'episode_count':len(es),'train_rows':train_count,'validation_rows':total-train_count,'train_episodes':int((~val).sum()),'validation_episodes':int(val.sum()),'train_target_demo_count':len(train_demos),'train_target_demo_counts':train_demos,'all_target_demo_count':len(all_demos),'top3_train_episode_pct':float(np.sort(w)[-3:].sum()*100),'equivalent_equal_weight_episodes':float(1/np.sum(w*w)),'zero_residual_train_rows':sum(e['zero_residual_rows'] for e in rows if e['split']=='train'),'obs_sha256':hashlib.sha256(obs.tobytes()).hexdigest(),'action_sha256':hashlib.sha256(actions.tobytes()).hexdigest(),'episodes':rows}
old=json.loads((ROOT/'data/sim_hand_10k_seed42/subset_manifest.json').read_text())
keysets={'old10k':{(e['episode_id'],e['env_id']) for e in old['episodes']},**{k:{(e.episode_id,e.env_id) for e in v} for k,v in selections.items()}}
overlap={a+'__'+b:len(keysets[a]&keysets[b]) for a in keysets for b in keysets if a<b}
results['provenance']={'source':str(source),'shards_checked':len(manifest['shards']),'physical_fields':[k for k in schemas if any(x in k.lower() for x in ['mass','friction','inertia','scale'])],'episode_overlap_counts':overlap,'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),'normalizer':'fits all subset rows including validation; unchanged from original recipe','training_seed':42,'split_seed':42}
(R/'data_audit.json').write_text(json.dumps(results,indent=2));np.savez_compressed(R/'selected_target_fields.npz',**extras)
print(json.dumps({n:{k:v for k,v in results[n].items() if k not in ['episodes','train_target_demo_counts']} for n,_,_ in specs},indent=2),flush=True)
