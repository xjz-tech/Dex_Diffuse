from pathlib import Path
import json,hashlib
import numpy as np,zarr
r=Path(__file__).resolve().parents[1];root=r.parents[2];pool=np.load(r/'collection_pool.npz');selected=json.loads((r/'collection_selected.json').read_text());phys=json.loads((r/'physical_slots.json').read_text());domains=['light_low','light_high','heavy_low','heavy_high'];L=250
assert len(selected)==40
val=np.zeros(40,bool);val[np.random.default_rng(42).choice(40,4,replace=False)]=True
assignment=np.empty(40,int);rg=np.random.default_rng(20260921);assignment[np.flatnonzero(val)]=rg.permutation(4);assignment[np.flatnonzero(~val)]=rg.permutation(np.repeat(np.arange(4),9))
summaries=[]
for name in domains+['balanced']:
 dest=root/'data'/f'domain10k_20260920_{name}';dest.mkdir(exist_ok=False,parents=True);obs=[];act=[];rows=[]
 for i,entry in enumerate(selected):
  base=entry['base_env'];j=domains.index(name) if name!='balanced' else int(assignment[i]);domain=domains[j];key=f'b{base}_{domain}_';p=phys[j*150+base]
  o=pool[key+'obs'];a=pool[key+'action'];assert o.shape==(L,66) and a.shape==(L,22)
  assert np.isfinite(o).all() and np.isfinite(a).all();assert np.array_equal(o[:,44:],o[:,22:44]-o[:,:22]);assert np.allclose(o[1:,22:44],a[:-1],atol=1e-6)
  obs.append(o);act.append(a);rows.append({'episode_index':i,'length':L,'split':'validation' if val[i] else 'train','source_domain':domain,'base_env':base,'source_round':entry['round'],'initial_demo':entry['initial_demo'],'initial_frame':entry['initial_frame'],'mass_kg':p['mass_kg'],'object_friction':p['object_friction'],'hand_friction':p['hand_friction'],'physical_slot':j*150+base,'train_window_weight':0 if val[i] else 1/36,'all_window_weight':1/40,'target_demo_counts':{str(int(d)):int(n) for d,n in zip(*np.unique(pool[key+'demo'],return_counts=True))}})
 z=zarr.open_group(str(dest/'replay_buffer.zarr'),mode='w');data=z.create_group('data');meta=z.create_group('meta');data.array('obs',np.concatenate(obs).astype(np.float32),chunks=(1000,66));data.array('action',np.concatenate(act).astype(np.float32),chunks=(1000,22));meta.array('episode_ends',np.arange(1,41,dtype=np.int64)*L);meta.array('episode_ids',np.arange(40,dtype=np.int64));meta.array('env_ids',np.array([x['physical_slot'] for x in rows],dtype=np.int32))
 weights={d:{'all_steps':sum(x['length'] for x in rows if x['source_domain']==d),'train_steps':sum(x['length'] for x in rows if x['source_domain']==d and x['split']=='train'),'validation_steps':sum(x['length'] for x in rows if x['source_domain']==d and x['split']=='validation')} for d in domains}
 summary={'name':name,'dataset':str(dest),'total_steps':10000,'episodes':40,'train_steps':9000,'validation_steps':1000,'episode_length':250,'selection':'matched initial templates successful for all 4 domains; 250-step segments, not full terminal episodes','split_seed':42,'physical_labels':'per-episode actual engine readbacks; fixed within each segment','weights_by_domain':weights,'mass_range_g':[min(x['mass_kg'] for x in rows)*1000,max(x['mass_kg'] for x in rows)*1000],'friction_range':[min(x['object_friction'][0] for x in rows),max(x['object_friction'][0] for x in rows)],'episodes_detail':rows}
 (dest/'distribution_manifest.json').write_text(json.dumps(summary,indent=2));summaries.append(summary)
(r/'datasets.json').write_text(json.dumps(summaries,indent=2));print('BUILT',[(x['name'],x['mass_range_g'],x['friction_range']) for x in summaries])
