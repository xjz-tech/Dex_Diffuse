import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,numpy as np,h5py,trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;ROOT=Path('/home/carus/Program/dex-controller/data/NOKOV-v3')
s=json.loads((OUT/'segments.json').read_text());angles=np.load(OUT/'angles.npz');meta=json.loads((OUT/'initial_scan.json').read_text());ref=np.array(meta['reference_axis']);long=np.array(meta['mesh_long_axis'])
m=trimesh.load(ROOT/'object/mesh/bulb2.obj',process=False);v=np.array(m.vertices);center=(v.min(0)+v.max(0))/2;v=v[np.linspace(0,len(v)-1,450).astype(int)]
colors=['#d64b43','#319751','#4273ca','#d19b26','#9749ae'];cache={}
def get(i):
 if i not in cache:
  with h5py.File(ROOT/f'data/bulb2/{i:03d}.h5') as f:cache[i]=(f['object/6dpose'][:],f['joint_position'][:])
 return cache[i]
def draw_pose(ax,i,t):
 o,j=get(i);posed=v@o[t,:3,:3].T+o[t,:3,3];c=o[t,:3,:3]@center+o[t,:3,3];axis=o[t,:3,:3]@long
 ax.scatter(*posed.T,c='#8a9096',s=.8,alpha=.25,depthshade=False)
 ax.plot(*np.array([c-axis*.065,c+axis*.065]).T,c='black',lw=1.4)
 for fi,col in enumerate(colors):
  p=j[t,np.r_[0,np.arange(4+fi*4,8+fi*4)]];ax.plot(*p.T,c=col,lw=1.3)
 ax.set(xlim=(-.13,.13),ylim=(-.20,.06),zlim=(-.05,.21));ax.set_box_aspect((1,1,1));ax.view_init(elev=28,azim=-65);ax.set_axis_off();ax.set_title(f'{i:03d} | {t/30:.2f}s | {angles[f"{i:03d}"][t]:.0f} deg',fontsize=8,pad=-5)
cases=[]
for row in s['rows']:
 for e in row['events']:cases.append((row['index'],e))
for page in range((len(cases)+7)//8):
 selected=cases[page*8:(page+1)*8];fig=plt.figure(figsize=(14,18),facecolor='white')
 for r,(i,e) in enumerate(selected):
  times=[e['start'],e['peak'],(e['last_side']+e['end'])//2,e['end']]
  for col,t in enumerate(times):draw_pose(fig.add_subplot(8,5,r*5+col+1,projection='3d'),i,t)
  ax=fig.add_subplot(8,5,r*5+5);a=angles[f'{i:03d}'];ax.plot(np.arange(len(a))/30,a,lw=1);ax.axhline(60,color='red',ls='--',lw=.7);ax.axhline(30,color='green',ls='--',lw=.7);ax.axvspan(e['start']/30,e['end']/30,color='orange',alpha=.3);ax.set_ylim(0,125);ax.set_xlim(0,20);ax.tick_params(labelsize=7);ax.set_title(f'{i:03d}: {e["seconds"]:.2f}s episode',fontsize=9);ax.grid(alpha=.2)
 fig.suptitle(f'Full150 candidate review | events {page*8+1}-{page*8+len(selected)}/58\nFirst sideways frame / peak tilt / final correction / aligned; black line = bulb long axis\nOrange = sideways-to-aligned interval, including sideways dwell. Geometry, no physics.',fontsize=12,y=.99)
 fig.subplots_adjust(top=.935,bottom=.02,left=.005,right=.98,wspace=.03,hspace=.23);fig.savefig(OUT/f'events_{page+1}.png',dpi=115);plt.close(fig);print('events',page+1,flush=True)
# Review all candidates rejected by primary rule, plus negative control 090.
positive={r['index'] for r in s['rows'] if r['events']}
neg=[r['index'] for r in meta['rows'] if r['max_angle']>=40 and r['index'] not in positive]+[90]
for page in range((len(neg)+7)//8):
 fig=plt.figure(figsize=(14,18),facecolor='white')
 for r,i in enumerate(neg[page*8:(page+1)*8]):
  a=angles[f'{i:03d}'];peak=int(np.argmax(a));times=[max(0,peak-30),peak,min(len(a)-1,peak+15),min(len(a)-1,peak+60)]
  for col,t in enumerate(times):draw_pose(fig.add_subplot(8,5,r*5+col+1,projection='3d'),i,t)
  ax=fig.add_subplot(8,5,r*5+5);ax.plot(np.arange(len(a))/30,a,lw=1);ax.axhline(60,color='red',ls='--',lw=.7);ax.axhline(30,color='green',ls='--',lw=.7);ax.set_ylim(0,125);ax.set_xlim(0,20);ax.tick_params(labelsize=7);ax.set_title(f'{i:03d}: no primary complete event',fontsize=8)
 fig.suptitle('Borderline / truncated candidate review + control 090\nBefore peak / peak / +0.5s / +2s; black line = bulb long axis',fontsize=12,y=.99)
 fig.subplots_adjust(top=.935,bottom=.02,left=.005,right=.98,wspace=.03,hspace=.23);fig.savefig(OUT/f'borderline_{page+1}.png',dpi=115);plt.close(fig);print('borderline',page+1,flush=True)
