import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,numpy as np,h5py
from pathlib import Path
from scipy.ndimage import median_filter
p=Path(__file__).resolve().parent
sys.path.insert(0,str(p))
from segment import events
meta=json.loads((p/'initial_scan.json').read_text());axis=np.array(meta['mesh_long_axis']);ref=np.array(meta['reference_axis']);axes={};stamps={}
for i in range(150):
 with h5py.File(f'/home/carus/Program/dex-controller/data/NOKOV-v3/data/bulb2/{i:03d}.h5') as f:axes[i]=f['object/6dpose'][:,:3,:3]@axis;stamps[i]=f['timestamp'][:]
a=np.concatenate(list(axes.values()));select=a[np.degrees(np.arccos(np.clip(a@ref,-1,1)))<30];alt=np.median(select,axis=0);alt/=np.linalg.norm(alt)
out=[]
for name,r in [('primary',ref),('all150_aligned_median',alt)]:
 for width,sidehold,readyhold in [(9,9,15),(5,6,9),(15,15,30)]:
  evs=[];nd=0;frames=0
  for i,arr in axes.items():
   ang=median_filter(np.degrees(np.arccos(np.clip(arr@r,-1,1))),size=width,mode='nearest');ev,_=events(ang,60,30,sidehold,readyhold);nd+=bool(ev);frames+=sum(e['end']-e['start'] for e in ev);evs.extend(ev)
  out.append(dict(reference=name,smoothing_frames=width,lateral_hold=sidehold,aligned_hold=readyhold,demos=nd,events=len(evs),frames=frames,percent=frames/89927*100))
s=json.loads((p/'segments.json').read_text());real_total=sum(ts[-1]-ts[0] for ts in stamps.values());real_event=0
for row in s['rows']:
 ts=stamps[row['index']]
 for e in row['events']:
  assert 0<=e['start']<e['end']<len(ts)
  real_event+=ts[e['end']]-ts[e['start']]
(p/'validation.json').write_text(json.dumps(dict(alt_reference=alt.tolist(),angle_between_references=float(np.degrees(np.arccos(ref@alt))),sensitivity=out,actual_timestamp_total_s=float(real_total),actual_timestamp_event_s=float(real_event),actual_timestamp_fraction=real_event/real_total),indent=2))
print(json.dumps(json.loads((p/'validation.json').read_text()),indent=2))
