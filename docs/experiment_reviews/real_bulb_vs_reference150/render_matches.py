import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import ast,json
import numpy as np,torch,pytorch_kinematics as pk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;r=np.load(OUT/'real_states.npz');n=np.load(OUT/'nearest.npz');f=np.load(OUT/'fingertips.npz');meta=json.loads((OUT/'episodes.json').read_text());names=json.loads((OUT/'fingertip_coverage.json').read_text())['policy_joint_names'];torch.set_num_threads(1);torch.set_num_interop_threads(1)
chain=pk.build_chain_from_urdf(Path('/home/carus/Program/dex-controller/maniptrans_envs/assets/sharpa_hand/v3right_sharpa_wave-forhammer5.urdf').read_bytes());order=[names.index(x) for x in chain.get_joint_parameter_names()]
fingers=[['right_thumb_CMC_VL','right_thumb_MC','right_thumb_MCP_VL','right_thumb_PP','right_thumb_DP','right_thumb_fingertip']]+[[f'right_{x}_{j}' for j in (['MC','MCP_VL','PP','MP','DP','fingertip'] if x=='pinky' else ['MCP_VL','PP','MP','DP','fingertip'])] for x in ['index','middle','ring','pinky']]
sel=json.loads((OUT/'example_indices.json').read_text())
for page,chosen in enumerate([sel[:3],sel[3:6],sel[6:]]):
 fig=plt.figure(figsize=(13,11),facecolor='white')
 for row,i in enumerate(chosen):
  e=int(r['episode'][i]);t=int(r['frame'][i]);qs=np.stack([r['qpos'][i],f['nearest_ref_qpos'][i]])
  with torch.no_grad():fk=chain.forward_kinematics(torch.tensor(qs[:,order],dtype=torch.float32))
  ax=fig.add_subplot(3,3,row*3+1);ax.imshow(plt.imread(OUT/f'example_{e}_{t}.jpg'));ax.axis('off');ax.set_title(f"Real {meta[e]['id']} frame {t} ({t/30:.2f}s)",fontsize=10)
  for col,az in [(1,-65),(2,30)]:
   ax=fig.add_subplot(3,3,row*3+col+1,projection='3d')
   for k,(color,label) in enumerate([('#cc6f27','Real qpos FK'),('#3269ad','Nearest reference FK')]):
    for finger in fingers:
     pts=np.stack([fk[x].get_matrix()[k,:3,3].numpy() for x in ['right_hand_C_MC']+finger]);ax.plot(*pts.T,c=color,lw=1.7,alpha=.8);ax.scatter(*pts[-1],s=18,c=color)
   ax.set(xlim=(-.1,.1),ylim=(-.14,.06),zlim=(-.02,.18));ax.set_box_aspect((1,1,1));ax.view_init(25,az);ax.set_axis_off()
   ax.set_title(f"Reference {f['nearest_ref_index'][i]:03d}/{f['nearest_ref_frame'][i]} | fingertip RMS {f['rms_cm'][i]:.2f}cm",fontsize=9)
 fig.suptitle('Orange = real hand from measured joints; blue = nearest reference in fingertip space\nSame robot kinematics; object pose is unavailable and is NOT matched here.',fontsize=13)
 fig.subplots_adjust(top=.9,bottom=.01,left=.01,right=.99,wspace=.04,hspace=.22);fig.savefig(OUT/f'matches_{page+1}.png',dpi=140);plt.close(fig)
print('matches written')
