import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,numpy as np,pyarrow.parquet as pq
ROOT=Path('/home/carus/Data/realworld_bulb_sft_260909');OUT=Path(__file__).resolve().parent
paths=sorted(ROOT.glob('rank_*/id_*/data/chunk-*/episode_*.parquet'),key=lambda p:(int(p.parts[-5].split('_')[-1]),int(p.parts[-4].split('_')[-1]),str(p)))
arrays={k:[] for k in ['qpos','action','arm','segment','episode','frame','force','contact_count']};rows=[];sources=[]
for e,p in enumerate(paths):
 f=pq.ParquetFile(p);cols=f.schema_arrow.names
 t=pq.read_table(p,columns=['state','actions','segment_id','tactile_f6','tactile_contact_points'],use_threads=False)
 q=np.array(t['state'].to_pylist(),dtype=np.float32);act=np.array(t['actions'].to_pylist(),dtype=np.float32);seg=np.array(t['segment_id'].to_pylist());force=np.array(t['tactile_f6'].to_pylist(),dtype=np.float32)
 cp=[json.loads(x) for x in t['tactile_contact_points'].to_pylist()];cnt=np.array([[len(y) for y in x] for x in cp],dtype=np.int16)
 assert q.shape[1]==31 and q.shape==act.shape and np.isfinite(q).all() and np.isfinite(act).all()
 arrays['qpos'].append(q[:,9:]);arrays['action'].append(act[:,9:]);arrays['arm'].append(q[:,:9]);arrays['segment'].append(seg);arrays['episode'].append(np.full(len(q),e));arrays['frame'].append(np.arange(len(q)));arrays['force'].append(force);arrays['contact_count'].append(cnt)
 u,c=np.unique(seg,return_counts=True);rows.append(dict(episode=e,id=p.parts[-4],rank=p.parts[-5],frames=len(q),segments=dict(zip(map(str,u),map(int,c))),contact_any_frames=int((cnt.sum(1)>0).sum()),state_hand_min=float(q[:,9:].min()),state_hand_max=float(q[:,9:].max()),path=str(p)))
 sources.append(dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,schema=cols,rows=f.metadata.num_rows))
np.savez_compressed(OUT/'real_states.npz',**{k:np.concatenate(v) for k,v in arrays.items()})
(OUT/'episodes.json').write_text(json.dumps(rows,indent=2));(OUT/'sources.json').write_text(json.dumps(sources,indent=2))
print('episodes',len(rows),'frames',sum(r['frames'] for r in rows));print('segments',np.unique(np.concatenate(arrays['segment']),return_counts=True));print('contact any',sum(r['contact_any_frames'] for r in rows));print('first episodes',rows[:3])
