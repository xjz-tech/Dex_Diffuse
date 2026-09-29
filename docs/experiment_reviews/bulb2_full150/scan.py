"""Single-threaded full-frame geometric audit; thresholds are heuristic labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib
import numpy as np,h5py,trimesh
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/carus/Program/dex-controller/data/NOKOV-v3')
OUT=Path(__file__).resolve().parent
mesh=trimesh.load(ROOT/'object/mesh/bulb2.obj',process=False)
_,axes=np.linalg.eigh(np.cov(np.array(mesh.vertices).T));long=axes[:,-1]
if long[1]<0:long=-long
clips={};sources=[]
for i in range(150):
 p=ROOT/f'data/bulb2/{i:03d}.h5'
 with h5py.File(p) as f:o=f['object/6dpose'][:];j=f['joint_position'][:];ts=f['timestamp'][:]
 assert np.isfinite(o).all() and np.isfinite(j).all()
 axis=o[:,:3,:3]@long
 clips[i]=dict(o=o,j=j,axis=axis)
 sources.append(dict(index=i,frames=len(o),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),timestamp_first=float(ts[0]),timestamp_last=float(ts[-1])))
ref=np.median(np.concatenate([clips[i]['axis'] for i in range(60)]),axis=0);ref/=np.linalg.norm(ref)
rows=[]
for i,c in clips.items():
 angle=np.degrees(np.arccos(np.clip(c['axis']@ref,-1,1)))
 c['angle']=median_filter(angle,size=9,mode='nearest')
 a=c['angle'];t=np.argmax(a)
 rows.append(dict(index=i,frames=len(a),max_angle=float(a.max()),max_time=float(t/30),start_angle=float(a[0]),end_angle=float(a[-1]),above45=int((a>=45).sum()),above60=int((a>=60).sum())))
np.savez_compressed(OUT/'angles.npz',**{f'{i:03d}':c['angle'] for i,c in clips.items()})
(OUT/'sources.json').write_text(json.dumps(sources,indent=2))
(OUT/'initial_scan.json').write_text(json.dumps(dict(reference_axis=ref.tolist(),mesh_long_axis=long.tolist(),rows=rows),indent=2))
fig,axs=plt.subplots(3,1,figsize=(15,17),layout='constrained')
for k,ax in enumerate(axs):
 mat=np.full((50,max(len(c['angle']) for c in clips.values())),np.nan)
 for row,i in enumerate(range(k*50,(k+1)*50)):mat[row,:len(clips[i]['angle'])]=clips[i]['angle']
 im=ax.imshow(mat,aspect='auto',origin='upper',vmin=0,vmax=100,cmap='magma',extent=[0,mat.shape[1]/30,k*50+49.5,k*50-.5]);ax.set_yticks(np.arange(k*50,(k+1)*50,2));ax.set_xlabel('Original recording time (nominal seconds)');ax.set_ylabel('Demo index');ax.set_title('Long-axis deviation from typical spin-ready direction (degrees)')
fig.colorbar(im,ax=axs,label='Axis deviation (degrees)',shrink=.6);fig.savefig(OUT/'all150_angles.png',dpi=130);plt.close(fig)
print('ref',ref,'frames',sum(r['frames'] for r in rows))
for t in [35,40,45,50,55,60,65,70,80]:print('max>=',t,':',sum(r['max_angle']>=t for r in rows),'frames',sum((c['angle']>=t).sum() for c in clips.values()))
print('candidates',[(r['index'],round(r['max_angle'],1),round(r['max_time'],2),r['above60']) for r in rows if r['max_angle']>=40])
print('084/090',[rows[i] for i in [84,90]])
