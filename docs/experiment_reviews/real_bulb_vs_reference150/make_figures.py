import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;r=np.load(OUT/'real_states.npz');n=np.load(OUT/'nearest.npz');f=np.load(OUT/'fingertips.npz');co=r['contact_count'].sum(1)>0;rows=json.loads((OUT/'episodes.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
for ax,values,cal,unit in [(axs[0],n['rms_deg'],n['calibration_rms_deg'],'22-joint RMS difference (degrees)'),(axs[1],f['rms_cm'],f['calibration_cm'],'Five-fingertip RMS 3D difference (cm)')]:
 for label,x,color in [('Reference held-out demos',cal,'#379063'),('Real: all frames',values,'#73767e'),('Real: tactile points detected',values[co],'#c26730')]:
  y=np.sort(x);ax.plot(y,np.arange(1,len(y)+1)/len(y)*100,label=label,color=color,lw=2)
 ax.set_xlabel(unit);ax.set_ylabel('Fraction with nearest reference within distance (%)');ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
axs[0].set_xlim(0,30);axs[1].set_xlim(0,4)
fig.suptitle('Real bulb demonstrations vs 150 retargeted references\nHand-configuration similarity only: no measured object pose / no physical coverage claim.',fontsize=13)
fig.savefig(OUT/'coverage_curves.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(3,1,figsize=(13,13),layout='constrained')
for page,ax in enumerate(axs):
 mat=np.zeros((25,100))
 for row,e in enumerate(range(page*25,(page+1)*25)):
  x=f['rms_cm'][r['episode']==e];mat[row]=np.interp(np.linspace(0,len(x)-1,100),np.arange(len(x)),x)
 im=ax.imshow(mat,aspect='auto',vmin=0,vmax=3.5,cmap='magma',extent=[0,100,24.5,-.5]);ax.set_yticks(range(25));ax.set_yticklabels([rows[e]['id'] for e in range(page*25,(page+1)*25)],fontsize=7);ax.set_xlabel('Normalized episode progress (%)');ax.set_title('Nearest reference: five-fingertip RMS distance (cm)')
fig.colorbar(im,ax=axs,label='Distance (cm)',shrink=.6);fig.savefig(OUT/'all75_distance.png',dpi=120);plt.close(fig)
fig,axs=plt.subplots(3,2,figsize=(12,9),layout='constrained')
for ax,e in zip(axs.ravel(),[0,14,29,44,59,74]):
 mask=r['episode']==e;time=r['frame'][mask]/30;x=f['rms_cm'][mask];ax.plot(time,x,color='#336ca3');ax.fill_between(time,0,3.5,where=co[mask],color='#a6cbb1',alpha=.3,label='tactile points');ax.axhline(2,color='#c26730',ls='--',lw=1);ax.set_ylim(0,3.5);ax.set_title(rows[e]['id']);ax.set_xlabel('seconds');ax.set_ylabel('Fingertip RMS difference (cm)');ax.grid(alpha=.2)
fig.suptitle('Example episodes: green shading = tactile points detected (not a stage label)');fig.savefig(OUT/'example_timelines.png',dpi=130);plt.close(fig)
print('figures written')
