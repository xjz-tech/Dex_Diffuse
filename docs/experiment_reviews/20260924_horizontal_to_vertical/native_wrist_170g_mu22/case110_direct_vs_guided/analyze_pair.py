from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
ref=np.load(P.parent.parent/'reference/reference.npz')['hand_target_rad'];i=110
runs={name:np.load(P/name/'trajectory.npz') for name in ['direct','guided']}
zs={name:np.load(P/name/'initial_state.npz') for name in runs}
fields={k:bool(np.array_equal(zs['direct'][k],zs['guided'][k])) for k in zs['direct'].files}
assert all(fields.values()),fields
original=np.load(P.parent/'sweep_final/initial_state.npz')
original_fields={k:bool(np.array_equal(zs['guided'][k],original[k])) for k in original.files if k in zs['guided']}
np.testing.assert_array_equal(runs['direct']['command'][60:150,i],ref)
result={}
for name,t in runs.items():
 z=zs[name];np.testing.assert_allclose(z['object_mass'],.17,atol=1e-7);np.testing.assert_allclose(z['object_friction'],2.2,atol=1e-6);np.testing.assert_allclose(z['hand_friction'],2.2,atol=1e-6)
 failure=np.flatnonzero(t['failure'][:,i]);first=int(failure[0]) if len(failure) else None
 v=t['vertical_error_deg'][:,i];axis=R.from_quat(t['relative_quaternion'][:,i]).apply([0,1,0]);change=np.degrees(np.arccos(np.clip(axis@axis[59],-1,1)))
 wr=t['wrist_pose'][:,i];w0=z['wrist'][i,:7];assert np.max(abs(wr[:,:3]-w0[:3]))<1e-6;assert np.max(np.minimum(np.linalg.norm(wr[:,3:]-w0[3:],axis=1),np.linalg.norm(wr[:,3:]+w0[3:],axis=1)))<1e-6
 result[name]=dict(initial_vertical_deg=float(v[59]),reference_end_vertical_deg=float(v[149]),hold_end_vertical_deg=float(v[179]),reference_axis_change_deg=float(change[149]),first_failure_reference_s=None if first is None else (first-59)/30,first_failure_frame=first,native_failure=first is not None,command_vs_recorded_rmse_rad=float(np.sqrt(np.mean((t['command'][60:150,i]-ref)**2))),q_vs_command_rmse_rad=float(np.sqrt(np.mean((t['q'][60:150,i]-t['command'][60:150,i])**2))),reference_end_contact_count=int((t['contact_force_norm'][149,i]>.05).sum()),hold_end_contact_count=int((t['contact_force_norm'][179,i]>.05).sum()),horizontal_to_vertical_success=bool(60<=v[59]<=120 and first is None and (v[-15:]<=20).all()))
d,g=runs['direct'],runs['guided'];prefix={k:bool(np.array_equal(d[k][:60],g[k][:60])) for k in d.files}
settle_pos_mm=float(np.max(np.linalg.norm(d['object_pose'][:60,i,:3]-g['object_pose'][:60,i,:3],axis=-1))*1000)
settle_rot_deg=float(np.max((R.from_quat(d['object_pose'][:60,i,3:]).inv()*R.from_quat(g['object_pose'][:60,i,3:])).magnitude())*180/np.pi)
forces=bool(np.array_equal(d['applied_forces'][:,i],g['applied_forces'][:,i]));force_prefix=bool(np.array_equal(d['applied_forces'][:60,i],g['applied_forces'][:60,i]))
case_prefix={k:bool(np.array_equal(d[k][:60,i],g[k][:60,i])) for k in d.files if d[k].ndim>=2 and d[k].shape[1]==189}
verification=dict(static_prefix_exact_case110=case_prefix,initial_fields_exact_between_methods=fields,initial_fields_exact_vs_original=original_fields,static_prefix_exact_all_environments=prefix,static_case110_max_position_difference_mm=settle_pos_mm,static_case110_max_rotation_difference_deg=settle_rot_deg,case110_random_forces_equal=forces,case110_static_random_forces_equal=force_prefix,direct_commands_equal_recorded_actions=True,original_wrist_retained=True)
(P/'metrics.json').write_text(json.dumps(dict(results=result,verification=verification),indent=2))
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
focal=640/(2*np.tan(np.deg2rad(21)))
caps={n:cv2.VideoCapture(str(P/n/'case110_two_views.mp4')) for n in runs};labels={'direct':'原始动作直接回放（不使用 prior）','guided':'10B prior + 原始轨迹引导（scale 25）'}
writers=[imageio.get_writer(str(P/'comparison.mp4'),fps=30,codec='libx264',quality=8),imageio.get_writer(str(P/'comparison_slow.mp4'),fps=15,codec='libx264',quality=8)]
for k in range(180):
 panels=[]
 for name,t in runs.items():
  ok,frame=caps[name].read();assert ok
  target=zs[name]['object'][i,:3].astype(float);pos=t['object_pose'][k,i,:3];axis=R.from_quat(t['object_pose'][k,i,3:]).apply([0,1,0])
  for view,offset in enumerate([[.12,.42,.14],[-.32,.24,-.08]]):
   eye=target+offset;eye[2]=max(eye[2],-.32);forward=target-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
   def project(point):
    delta=point-eye;depth=delta@forward
    if depth<=.02:return None
    xy=np.array([320+focal*(delta@right)/depth,240-focal*(delta@up)/depth]);return tuple(np.round(xy+[view*640,64]).astype(int))
   p0,p1=project(pos),project(pos+.05*axis)
   if p0 and p1 and all(-200<v<1600 for v in p0+p1):cv2.arrowedLine(frame,p0,p1,(70,235,90),2,cv2.LINE_AA,tipLength=.2)
  im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(im);draw.rectangle((0,0,1279,63),fill=(14,19,25));failed=bool(t['failure'][:k+1,i].any());phase={'settle':'静态抓持','reference':'动作执行','hold':'末态保持'}[str(t['phase'][k])]
  draw.text((12,1),labels[name]+'  |  case110 / 170 g / 摩擦 2.2 / 原生 wrist',font=font,fill='white')
  draw.text((12,33),f'{phase} {(int(t["index"][k])+1)/30:.2f} s  |  距竖直 {t["vertical_error_deg"][k,i]:.1f}°  |  原生 failure: {failed}  |  两个视角 / 绿箭头指向灯泡顶部',font=small,fill=(255,170,150) if failed else (190,230,255))
  panels.append(np.asarray(im))
 composite=np.concatenate(panels,axis=0)
 writers[0].append_data(composite)
 slow=Image.fromarray(composite);draw=ImageDraw.Draw(slow);draw.rectangle((1130,100,1279,130),fill=(14,19,25));draw.text((1138,100),'0.5× 半速',font=small,fill='white');writers[1].append_data(np.asarray(slow))
 if k in [59,89,119,149,179]:Image.fromarray(composite).save(P/f'comparison{k:03d}.jpg')
for w in writers:w.close()
for c in caps.values():c.release()
checks={}
for file in ['comparison.mp4','comparison_slow.mp4']:
 cap=cv2.VideoCapture(str(P/file));count=0;fps=cap.get(cv2.CAP_PROP_FPS)
 while True:
  ok,frame=cap.read()
  if not ok:break
  count+=1
 cap.release();assert count==180;checks[file]=dict(frames=count,fps=fps,duration=count/fps)
verification['videos']=checks
(P/'verification.json').write_text(json.dumps(verification,indent=2))
fig,axs=plt.subplots(2,1,figsize=(10,6),sharex=True)
for name,t in runs.items():
 time=(np.arange(180)-59)/30
 axs[0].plot(time,t['vertical_error_deg'][:,i],label=labels[name].split('（')[0] if False else name)
 axs[1].plot(time,np.linalg.norm(t['relative_position'][:,i],axis=1),label=name)
 f=result[name]['first_failure_reference_s']
 if f is not None:
  axs[0].axvline(f,linestyle=':',alpha=.5);axs[0].text(f,160,f'{name} native failure',rotation=90)
for ax in axs:
 ax.axvspan(-2,0,color='gray',alpha=.1);ax.axvspan(3,4,color='gray',alpha=.1);ax.legend();ax.grid(alpha=.2)
axs[0].axhline(20,color='green',ls='--',alpha=.4);axs[0].set_ylabel('Angle from screw-down vertical (deg)');axs[0].set_ylim(0,180)
axs[1].set_ylabel('Bulb-wrist distance (m)');axs[1].set_xlabel('Time relative to reference start (s)');fig.tight_layout();fig.savefig(P/'comparison_metrics.png',dpi=150)
print(json.dumps(dict(results=result,verification={k:v for k,v in verification.items() if not isinstance(v,dict)}),indent=2))
