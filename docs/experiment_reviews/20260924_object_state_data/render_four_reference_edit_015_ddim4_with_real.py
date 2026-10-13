"""Four-episode raw / traditional prior / SDEdit / real-front comparison."""
import json,time
from pathlib import Path
import cv2,imageio.v2 as imageio,numpy as np
from PIL import Image,ImageDraw
from reference_resampling import interpolate_large_jumps
import render_episode54_adaptive010_scale50_g4e2_base as base
from render_full_episode53_axes import draw_axes,object_pose
from run_four_reference_edit_015_ddim4_video import O,R,folder
DATA=Path('/home/carus/Data/Object_state_data')
OLD=R/'four_reference_edit_noise_ddim_sweep_20260928'
VIDEO=O/'four_episodes_raw_traditional_prior_edit015_real_front.mp4'
LABELS=('原速 reference · 仿真正面','传统 prior · DDIM4 scale25 guide2 exec1','reference 编辑 · 噪声0.15 DDIM4 exec2','原数据集 front · 真机')

def load_episode(ep):
 case=R/f'qualified_comparison/episode_{ep:02d}'
 editpath=folder(ep,.15,4)
 for _ in range(900):
  if (editpath/'summary.json').exists() and (editpath/'guided_front.mp4').exists() and (editpath/'predictions.json').exists() and (editpath/'trace.json').exists():break
  time.sleep(2)
 else:raise TimeoutError(editpath)
 runs=[base.load_run(case/'direct_m044_mu11','direct'),base.load_run(case/'guide1_adaptive010_ddim4_scale25_m044_mu11_video','guided'),base.load_run(editpath,'guided')]
 with np.load(case/'reference_full.npz') as f:
  ref=f['hand_target_rad'];source_ids=f['source_state_frame_indices'][0].astype(int)
 expanded,progress=interpolate_large_jumps(ref,.1)
 n=ref.shape[1];ng=expanded.shape[1];assert len(source_ids)==n+1
 assert runs[0]['summary']['steps']['action']==n
 assert all(r['summary']['steps']['action']==ng for r in runs[1:])
 assert runs[0]['summary']['source_start_frame']==source_ids[0]
 for r in runs[1:]:
  f=Path(r['folder']);s=r['summary']
  assert s['reference_interpolation_threshold']==.1 and np.array_equal(np.load(f/'reference_progress.npy'),progress)
  assert s['native_protocol']==runs[0]['summary']['native_protocol'] and s['source_start_frame']==source_ids[0]
  with np.load(Path(runs[0]['folder'])/'initial_state.npz') as a,np.load(f/'initial_state.npz') as b:
   assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
  keys=[k for k in runs[0]['trace'][0] if k!='object_contact_force']
  assert all(all(a[k]==b[k] for k in keys) for a,b in zip(runs[0]['trace'][:60],r['trace'][:60]))
 prior=runs[1]['summary'];assert prior['prior']['ddim']==4 and prior['guidance_scale']==25 and prior['execution_steps']==1 and prior['prior']['guidance_steps']==2
 editor=runs[2]['summary'];assert editor['prior']['ddim']==4 and editor['execution_steps']==2 and editor['guidance_scale']==0 and editor['prior']['editor']['requested_noise_ratio']==.15
 # Recording must reproduce the exact rollout used in the numeric sweep.
 old=OLD/f'episode_{ep:02d}/edit_015_ddim4_seed44'
 with np.load(old/'initial_state.npz') as a,np.load(editpath/'initial_state.npz') as b:
  assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
 baseline=json.loads((old/'trace.json').read_text());recorded=runs[2]['trace'];assert len(baseline)==len(recorded)
 keys=[k for k in baseline[0] if k!='object_contact_force']
 assert all(all(a[k]==b[k] for k in keys) for a,b in zip(baseline,recorded))
 force=float(np.max(np.abs(np.asarray([a['object_contact_force'] for a in baseline])-np.asarray([b['object_contact_force'] for b in recorded]))));assert force<1e-5
 oldpred=json.loads((old/'predictions.json').read_text());newpred=json.loads((editpath/'predictions.json').read_text());assert len(oldpred)==len(newpred) and all({k:v for k,v in a.items() if k!='inference_seconds'}=={k:v for k,v in b.items() if k!='inference_seconds'} for a,b in zip(oldpred,newpred))
 zero=Image.open(R/f'm044_mu11_adaptive010_ddim4_scale25_four_video_20260927/episode{ep}_zero_physics.png').convert('RGB');assert zero.size==(640,480)
 losses=[json.loads((case/'direct_m044_mu11/m044_mu11_result.json').read_text())['first_separation'],json.loads((case/'guide1_adaptive010_ddim4_scale25_m044_mu11/m044_mu11_result.json').read_text())['first_separation'],json.loads((old/'analysis.json').read_text())['first_separation']]
 source_dir=DATA/f'episode_{ep}'/'front';assert source_dir.is_dir() and all((source_dir/f'{i:06d}.png').is_file() for i in source_ids)
 return dict(ep=ep,n=n,ng=ng,source_ids=source_ids,source_dir=source_dir,progress=progress,runs=runs,zero=zero,losses=losses,force_max_delta_N=force)

def render_episode(data,writer):
 ep,n,ng=data['ep'],data['n'],data['ng'];runs=data['runs'];progress=data['progress']
 timeline=[('title',0)]*30+[('import',0)]*30+[('settle',i) for i in range(60)]+[('ready',59)]*30+[('action',i) for i in range(ng)]+[('hold',i) for i in range(60)]
 source_id=None;source_image=None
 try:
  for fi,(phase,j) in enumerate(timeline):
   canvas=Image.new('RGB',(1280,1152),(18,25,35));draw=ImageDraw.Draw(canvas)
   prog=float(progress[j]) if phase=='action' else (float(n) if phase=='hold' else 0.)
   title=f'Episode {ep:02d} · 44g / 摩擦1.1 · 原始 reference 进度 {prog:g}/{n} · seed44'
   caption={'title':'原速reference / 传统prior / 加噪编辑 / 真机front','import':'物理运行前 · 同一源初态的FK重建','settle':'60步静置检查；真机停留在源起点','ready':'共同动作起点（定格1秒）','action':'按原始reference进度对齐；插值组物理控制步更多','hold':'完整动作尾段后保持2秒；真机停留在末帧'}[phase]
   draw.text((12,0),title,font=base.FONT,fill='white');draw.text((12,32),caption,font=base.SMALL,fill=(190,225,250))
   for panel,run in enumerate(runs):
    x=(panel%2)*640;y=64+(panel//2)*544;row=None
    if phase in ('title','import'):
     canvas.paste(data['zero'],(x,y+64))
    else:
     if phase in ('settle','ready'):idx=j
     elif phase=='action':idx=60+(max(0,int(np.floor(prog))-1) if panel==0 else j)
     else:idx=60+(n if panel==0 else ng)+j
     row=run['trace'][idx];frame=base.read_to(run,idx+1)
     canvas.paste(Image.fromarray(cv2.cvtColor(frame[:,:640],cv2.COLOR_BGR2RGB)),(x,y))
     draw_axes(canvas,object_pose(row),run['projectors'][0],(x,y))
    draw=ImageDraw.Draw(canvas);draw.rectangle((x,y,x+639,y+63),fill=(25,36,49))
    draw.text((x+8,y),LABELS[panel],font=base.FONT,fill='white')
    detail='精确导入位姿 · 尚未运行物理' if row is None else f'{phase} · 控制步 {row["index"]+1} · 竖直偏差 {row["vertical_error_deg"]:.1f}°'
    loss=data['losses'][panel]
    if row is not None and loss is not None and idx>=loss['trace_index']:detail+=' · 已几何分离'
    draw.text((x+8,y+35),detail,font=base.SMALL,fill=(255,150,130) if '已几何分离' in detail else (188,231,208))
   si=min(int(np.floor(prog)),n);sid=int(data['source_ids'][si])
   if sid!=source_id:
    source_image=Image.open(data['source_dir']/f'{sid:06d}.png').convert('RGB');assert source_image.size==(640,480);source_id=sid
   canvas.paste(source_image,(640,672));draw=ImageDraw.Draw(canvas)
   draw.rectangle((640,608,1279,671),fill=(25,36,49));draw.text((648,608),LABELS[3],font=base.FONT,fill='white')
   draw.text((648,643),f'源帧 {sid} · 原始动作进度 {prog:g}',font=base.SMALL,fill=(188,231,208))
   if fi in (30,150,220,300):canvas.save(O/f'episode{ep}_preview_{fi:04d}.jpg',quality=90)
   writer.append_data(np.asarray(canvas))
 finally:
  for r in runs:r['cap'].release()
 return dict(episode=ep,frames=len(timeline),source_actions=n,expanded_actions=ng,inserted=ng-n,recorded_editor_matches_numeric_rollout=True,recording_force_max_delta_N=data['force_max_delta_N'])

def main():
 writer=imageio.get_writer(str(VIDEO),fps=30,codec='libx264',quality=7,macro_block_size=16,ffmpeg_params=['-preset','fast','-threads','2','-movflags','+faststart'])
 chapters=[];total=0
 try:
  for ep in (76,34,54,2):
   d=load_episode(ep);c=render_episode(d,writer);c['start_s']=total/30;total+=c['frames'];chapters.append(c);print('COMPOSED',c,flush=True)
 finally:writer.close()
 cap=cv2.VideoCapture(str(VIDEO));fps=cap.get(cv2.CAP_PROP_FPS);count=0
 while True:
  ok,frame=cap.read()
  if not ok:break
  assert frame.shape==(1152,1280,3);count+=1
 cap.release();assert fps==30 and count==total
 info=dict(video=str(VIDEO),fps=fps,frames=count,duration_s=count/30,fully_decoded=True,chapters=chapters,panels=LABELS,alignment='Original reference action progress; raw simulation and real source frames repeat at inserted midpoints. Physical times differ.',checkpoint='/home/carus/data_usb/10B_obs_4-66.ckpt')
 (O/'video_verification.json').write_text(json.dumps(info,indent=2)+'\n')
 print(VIDEO,count,count/30,flush=True)
if __name__=='__main__':main()
