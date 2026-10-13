from pathlib import Path
import sys,json,cv2,numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parent;run=P/(sys.argv[1] if len(sys.argv)>1 else 'expanded');m=json.loads((run/'metadata.json').read_text());archive=np.load(run/'trajectory.npz');t={k:archive[k] for k in ['phase','index','object_pose','relative_quaternion','vertical_error_deg','failure']};z=np.load(run/'initial_state.npz');ref=np.load(P/'reference/reference.npz');summary=json.loads((run/'summary.json').read_text());rows={x['case_id']:x for x in summary['results']};recorded=m['recorded_cases'];centers=np.load(run/'camera_centers.npy') if (run/'camera_centers.npy').exists() else z['object'][:,:3];offset=int(m.get('video_first_trajectory_index',0));num=len(t['phase'])-offset
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22);small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20);focal=640/(2*np.tan(np.deg2rad(21)))
selected=[]
for ri in range(2):
 pool=[rows[i] for i in recorded if rows[i]['reference_id']==ri];pool.sort(key=lambda r:(not r['similar_reference_turn'],r['native_failure'],not r['initial_horizontal'],r['hold_end_goal_axis_error_deg']));selected.append(pool[0]['case_id'])
(run/'video_selection.json').write_text(json.dumps(dict(cases=selected,basis='posthoc perreference best among recorded: similar reference turn, nofailure, horizontal start, then reference-end-axis-error'),indent=2))
def axis_error(i,k):
 desired=R.from_matrix(ref['object_pose_wrist'][rows[i]['reference_id'],-1,:3,:3]).apply([0,1,0]);axis=R.from_quat(t['relative_quaternion'][k,i]).apply([0,1,0]);return np.degrees(np.arccos(np.clip(axis@desired,-1,1)))
def annotated(i,k,frame):
 target=centers[i].astype(float);pos=t['object_pose'][k,i,:3];axis=R.from_quat(t['object_pose'][k,i,3:]).apply([0,1,0])
 for view,off in enumerate([[.12,.42,.14],[-.32,.24,-.08]]):
  eye=target+off;eye[2]=max(eye[2],-.32);forward=target-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
  def project(point):
   delta=point-eye;depth=delta@forward
   if depth<=.02:return None
   xy=np.array([320+focal*(delta@right)/depth,240-focal*(delta@up)/depth]);return tuple(np.round(xy+[view*640,64]).astype(int))
  p0,p1=project(pos),project(pos+.05*axis)
  if p0 and p1 and all(-200<v<1600 for v in p0+p1):cv2.arrowedLine(frame,p0,p1,(70,235,90),2,cv2.LINE_AA,tipLength=.2)
 return Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
caps={i:cv2.VideoCapture(str(run/f'case{i:03d}_two_views.mp4')) for i in selected};check={}
with imageio.get_writer(str(run/'source_vs_guided_slow.mp4'),fps=15,codec='libx264',quality=8) as writer:
 for vindex in range(num):
  k=offset+vindex;out=Image.new('RGB',(1280,1152),(13,18,25));draw=ImageDraw.Draw(out);draw.text((12,0),'Object_state_data  |  左：真机参考  /  右：10B prior + 引导实际仿真',font=font,fill='white');draw.text((12,33),'原生 wrist · 170 g · 摩擦 2.2 · 0.5× 半速  |  绿箭头为灯泡顶部方向',font=small,fill=(190,225,245))
  for row,i in enumerate(selected):
   r=rows[i];ri=r['reference_id'];ep=r['episode'];phase=str(t['phase'][k]);index=int(t['index'][k]);src=r['start'] if phase=='settle' else min(r['end'],r['start']+index+1) if phase=='reference' else r['end'];y=64+row*544
   original=Image.open(Path('/home/carus/Data/Object_state_data')/f'episode_{ep}/wrist/{src:06d}.png').convert('RGB').resize((640,480));out.paste(original,(0,y+64));ok,frame=caps[i].read();assert ok;sim=annotated(i,k,frame);out.paste(sim.crop((640,64,1280,544)),(640,y+64));draw.rectangle((0,y,1279,y+63),fill=(21,30,40));draw.text((12,y+1),f'episode {ep} / 原始帧 {src}',font=font,fill='white');phase_label={'settle':'起始帧停留','reference':'原始片段','hold':'源片段结束，停留末帧'}[phase];draw.text((12,y+34),phase_label,font=small,fill=(190,225,245));fail=bool(t['failure'][:k+1,i].any());draw.text((652,y+1),f'case {i} / 距竖直 {t["vertical_error_deg"][k,i]:.1f}° / failure {fail}',font=font,fill='white');draw.text((652,y+34),f'参考末态轴误差 {axis_error(i,k):.1f}° / '+('末态保持' if phase=='hold' else '动作执行')+f' {(index+1)/30:.2f}s',font=small,fill=(190,225,245))
  draw.line((640,64,640,1151),fill=(110,130,145),width=2);writer.append_data(np.asarray(out))
  if k in [60,97,134,164]:out.save(run/f'comparison{k:03d}.jpg')
for cap in caps.values():cap.release()
# Preserve selected case full two-view clips with clear labels for closer inspection.
for i in selected:
 cap=cv2.VideoCapture(str(run/f'case{i:03d}_two_views.mp4'))
 with imageio.get_writer(str(run/f'case{i:03d}_clear.mp4'),fps=30,codec='libx264',quality=8) as writer:
  for k in range(offset,len(t['phase'])):
   ok,frame=cap.read();assert ok;im=annotated(i,k,frame);draw=ImageDraw.Draw(im);draw.rectangle((0,0,1279,63),fill=(14,19,25));draw.text((12,1),f'episode {rows[i]["episode"]} / case {i}  |  10B prior + 记录轨迹引导  |  170 g / 摩擦 2.2 / 原生 wrist',font=font,fill='white');draw.text((12,33),f'距竖直 {t["vertical_error_deg"][k,i]:.1f}° / 参考末态轴误差 {axis_error(i,k):.1f}° / native failure {bool(t["failure"][:k+1,i].any())}',font=small,fill=(190,225,245));writer.append_data(np.asarray(im))
 cap.release()
for file in ['source_vs_guided_slow.mp4']+[f'case{i:03d}_clear.mp4' for i in selected]:
 cap=cv2.VideoCapture(str(run/file));count=0;fps=cap.get(cv2.CAP_PROP_FPS)
 while True:
  ok,_=cap.read()
  if not ok:break
  count+=1
 cap.release();assert count==num;check[file]=dict(frames=count,fps=fps,duration_s=count/fps)
(run/'video_verification.json').write_text(json.dumps(check,indent=2));print('Selected',selected);print(json.dumps(check,indent=2))
