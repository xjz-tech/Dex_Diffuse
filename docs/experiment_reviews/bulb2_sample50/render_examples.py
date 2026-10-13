import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import h5py,numpy as np,trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
root=Path('/home/carus/Program/dex-controller');out=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/bulb2_sample50')
m=trimesh.load(root/'data/NOKOV-v3/object/mesh/bulb2.obj',process=False);v=np.asarray(m.vertices);v=v[np.linspace(0,len(v)-1,1000).astype(int)]
cases=[(3,[8,9,10,11,12,13]),(66,[0,3,6,9,12,15]),(84,[0,4,8,12,13.63,14.43]),(90,[12,13,13.53,14,15,16])]
colors=['#d64b43','#319751','#4273ca','#d19b26','#9749ae']
fig=plt.figure(figsize=(16,10),facecolor='white')
for row,(idx,seconds) in enumerate(cases):
 with h5py.File(root/f'data/NOKOV-v3/data/bulb2/{idx:03d}.h5') as f:o=f['object/6dpose'][:];j=f['joint_position'][:]
 allp=j.reshape(-1,3);mid=(allp.min(0)+allp.max(0))/2;span=np.ptp(allp,axis=0).max()*.53
 for col,sec in enumerate(seconds):
  t=round(sec*30);ax=fig.add_subplot(4,6,row*6+col+1,projection='3d');posed=v@o[t,:3,:3].T+o[t,:3,3]
  ax.scatter(*posed.T,c='#7b838c',s=1,alpha=.32,depthshade=False)
  for fi,c in enumerate(colors):
   pts=j[t,np.r_[0,np.arange(4+fi*4,8+fi*4)]];ax.plot(*pts.T,c=c,lw=2);ax.scatter(*pts[-1],c=c,s=14)
  ax.set(xlim=(mid[0]-span,mid[0]+span),ylim=(mid[1]-span,mid[1]+span),zlim=(mid[2]-span,mid[2]+span));ax.set_box_aspect((1,1,1));ax.view_init(elev=28,azim=-60);ax.set_axis_off();ax.set_title(f'{idx:03d} | {t/30:.2f}s',fontsize=11,pad=-8)
fig.suptitle('Finger repositioning and in-hand reorientation examples\nRed: thumb | Green: index | Blue: middle | Orange: ring | Purple: little\nRecorded landmarks and object mesh; geometric proximity does not prove physical contact.',fontsize=14,y=.995)
fig.subplots_adjust(left=0,right=1,bottom=0,top=.89,wspace=-.05,hspace=.06)
fig.savefig(out/'examples.png',dpi=130)
print(out/'examples.png')
