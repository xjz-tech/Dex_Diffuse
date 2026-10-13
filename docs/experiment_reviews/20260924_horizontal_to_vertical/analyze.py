from pathlib import Path
import json,numpy as np,cv2
from scipy.spatial.transform import Rotation as R
import imageio.v2 as imageio
P=Path(__file__).resolve().parent
rows=[]
for mode in ['guided','direct']:
 p=P/mode
 if not (p/'trajectory.npz').exists():continue
 t=np.load(p/'trajectory.npz');m=json.loads((p/'run_metadata.json').read_text());ref=np.load(P/'reference/reference.npz');idx=np.flatnonzero(t['phase']=='reference');b=idx[0]-1
 mapping=[]
 for k in range(len(t['phase'])):
  mapping.append(k)
  if k in [0,b,idx[-1],len(t['phase'])-1]:mapping.append(k)
 axes=R.from_quat(t['object_pose'][...,3:].reshape(-1,4)).apply([0,1,0]).reshape(len(t['phase']),3,3)
 angles=np.degrees(np.arccos(np.clip(axes[...,2],-1,1)))
 caps=[cv2.VideoCapture(str(p/f'env{i:02d}.mp4')) for i in range(3)]
 results=[]
 for i in range(3):
  bad=np.flatnonzero(t['failure'][:,i]);first=int(bad[0]) if len(bad) else None
  success=(not len(bad)) and bool((angles[-15:,i]<=20).all())
  results.append(dict(mode=mode,env=i,noise_seed=42+i if mode=='guided' else None,first_native_failure_seconds_from_reference=(first-b)/30 if first is not None else None,initial_vertical_error_deg=float(angles[b,i]),reference_end_vertical_error_deg=float(angles[idx[-1],i]),hold_end_vertical_error_deg=float(angles[-1,i]),best_prefailure_reference_vertical_error_deg=float(angles[idx[idx<(first if first is not None else len(angles))],i].min()) if first is None or first>idx[0] else None,task_numeric_success=success,joint_reference_rmse_rad=float(np.sqrt(np.mean((t['q'][idx,i]-ref['hand_qpos_rad'][1:])**2))),command_reference_rmse_rad=float(np.sqrt(np.mean((t['command'][idx,i]-ref['hand_target_rad'])**2))),native_failure=bool(len(bad))))
 with imageio.get_writer(str(p/'comparison.mp4'),fps=30,codec='libx264',quality=8) as w:
  for k in mapping:
   phase=str(t['phase'][k]);j=int(t['index'][k]);f=160 if phase=='settle' else min(250,161+j) if phase=='reference' else 250
   real=cv2.imread(f'/home/carus/Data/bulb_tac_260909/episode_37/wrist/{f:06d}.png');real=cv2.resize(real,(640,480));ims=[real];labels=[f'REAL id37 frame{f} | {phase}'];details=['Lifted horizontal bulb -> screw-down vertical']
   for i,c in enumerate(caps):
    ok,im=c.read();assert ok;ims.append(im);labels.append(f'10B + recorded reference | env{i}' if mode=='guided' else f'Direct reference | env{i}');bad=bool(t['failure'][:k+1,i].any());details.append(f'{phase} | vertical error {angles[k,i]:.1f} deg | native failure {bad}')
   canvas=np.zeros((1088,1280,3),np.uint8)
   for n,im in enumerate(ims):
    x=(n%2)*640;y=(n//2)*544;canvas[y+64:y+544,x:x+640]=im
    cv2.putText(canvas,labels[n],(x+10,y+24),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA);cv2.putText(canvas,details[n],(x+10,y+48),cv2.FONT_HERSHEY_SIMPLEX,.47,(190,220,255),1,cv2.LINE_AA)
   w.append_data(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
   if k in [b,idx[29],idx[59],idx[-1],len(t['phase'])-1]:cv2.imwrite(str(p/f'comparison_trace{k:03d}.jpg'),canvas)
 for c in caps:c.release()
 (p/'summary.json').write_text(json.dumps(results,indent=2));rows+=results
(P/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
