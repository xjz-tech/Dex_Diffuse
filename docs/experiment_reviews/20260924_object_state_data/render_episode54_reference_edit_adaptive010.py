"""Six front-camera panels aligned by original reference progress."""
import json
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
from reference_resampling import interpolate_large_jumps
import render_episode54_adaptive010_scale50_g4e2_base as base
from render_full_episode53_axes import draw_axes,object_pose
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926';O=R/'episode54_reference_edit_adaptive010_video_20260928';C=R/'qualified_comparison/episode_54'
VIDEO=O/'episode54_reference_edit_gt010_raw_direct_real.mp4'
LABELS=['原速 reference · 仿真','>0.1 插值 direct · 无编辑','真机 · 原数据集 front','>0.1 插值 + prior 编辑 · 噪声 0.10','>0.1 插值 + prior 编辑 · 噪声 0.20','>0.1 插值 + prior 编辑 · 噪声 0.35']

def main():
 assert json.loads((O/'zero_verification.json').read_text())['full_physics_trace_exact']
 runs={0:base.load_run(C/'direct_m044_mu11','direct')}
 for panel,method in [(1,'zero'),(3,'edit010'),(4,'edit020'),(5,'edit035')]:runs[panel]=base.load_run(O/f'{method}_seed44','guided')
 with np.load(C/'reference_full.npz') as f:
  ref=f['hand_target_rad'];source_ids=f['source_state_frame_indices'][0].astype(int)
 expanded,progress=interpolate_large_jumps(ref,.1);n=ref.shape[1];ng=expanded.shape[1]
 assert (n,ng)==(462,612)
 for panel,run in runs.items():
  if panel==0:continue
  folder=Path(run['folder']);s=run['summary']
  assert s['reference_interpolation_threshold']==.1 and s['steps']['action']==ng
  assert np.array_equal(progress,np.load(folder/'reference_progress.npy'))
  with np.load(C/'direct_m044_mu11/initial_state.npz') as a,np.load(folder/'initial_state.npz') as b:
   assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
  keys=[k for k in runs[0]['trace'][0] if k!='object_contact_force']
  assert all(all(a[k]==b[k] for k in keys) for a,b in zip(runs[0]['trace'][:60],run['trace'][:60]))
  if panel>1:assert json.loads((folder/'analysis.json').read_text())['validation']['native_protocol_exact']
 losses={}
 for panel,run in runs.items():
  f=(C/('direct_m044_mu11' if panel==0 else 'direct_adaptive010_m044_mu11')/'m044_mu11_result.json') if panel<2 else Path(run['folder'])/'analysis.json'
  losses[panel]=json.loads(f.read_text())['first_separation']
 zero=Image.open(R/'m044_mu11_adaptive010_ddim4_scale25_video_20260927/episode54_zero_physics.png').convert('RGB')
 assert zero.size==(640,480)
 timeline=[('title',0)]*30+[('import',0)]*30+[('settle',i) for i in range(60)]+[('ready',59)]*30+[('action',i) for i in range(ng)]+[('hold',i) for i in range(60)]
 writer=imageio.get_writer(str(VIDEO),fps=30,codec='libx264',quality=7,macro_block_size=16,ffmpeg_params=['-preset','fast','-threads','2','-movflags','+faststart'])
 source_cache={}
 try:
  for fi,(phase,j) in enumerate(timeline):
   canvas=Image.new('RGB',(1920,1152),(18,25,35));draw=ImageDraw.Draw(canvas)
   prog=float(progress[j]) if phase=='action' else (float(n) if phase=='hold' else 0.)
   draw.text((12,0),f'Episode 54 · 44g / 摩擦1.1 · 原始 reference 进度 {prog:g}/{n} · seed44',font=base.FONT,fill='white')
   subtitle={'title':'参考动作加噪后由 10B prior 去噪编辑 · 4次去噪 / 执行2步 · 无额外 MSE guidance', 'import':'物理运行前：同一导入状态的 FK 重建', 'settle':'60步静置检查；真机停留在源起点', 'ready':'共同动作起点；噪声数值是归一化扩散噪声比，不是 rad', 'action':'按原始动作进度对齐；>0.1 rad 插一个中点 · 462 → 612 步 · 插值组物理时间更长', 'hold':'完整 reference 尾段结束后保持2秒；真机停留在末帧'}[phase]
   draw.text((12,32),subtitle,font=base.SMALL,fill=(190,225,250))
   for panel,run in runs.items():
    x=(panel%3)*640;y=64+(panel//3)*544;row=None
    if phase in ('title','import'):
     canvas.paste(zero,(x,y+64))
    else:
     if phase in ('settle','ready'):idx=j
     elif phase=='action':idx=60+(max(0,int(np.floor(prog))-1) if panel==0 else j)
     else:idx=60+(n if panel==0 else ng)+j
     row=run['trace'][idx];frame=base.read_to(run,idx+1)
     canvas.paste(Image.fromarray(cv2.cvtColor(frame[:,:640],cv2.COLOR_BGR2RGB)),(x,y))
     draw_axes(canvas,object_pose(row),run['projectors'][0],(x,y))
    draw=ImageDraw.Draw(canvas);draw.rectangle((x,y,x+639,y+63),fill=(25,36,49));draw.text((x+8,y),LABELS[panel],font=base.FONT,fill='white')
    detail='精确导入位姿 · 尚未运行物理' if row is None else f'{phase} · 控制步 {row["index"]+1} · 竖直偏差 {row["vertical_error_deg"]:.1f}°'
    if row is not None and losses[panel] is not None and idx>=losses[panel]['trace_index']:detail+=' · 已几何分离'
    draw.text((x+8,y+35),detail,font=base.SMALL,fill=(255,150,130) if '已几何分离' in detail else (188,231,208))
   source_idx=min(int(np.floor(prog)),n);source_id=int(source_ids[source_idx])
   if source_id not in source_cache:source_cache[source_id]=Image.open(Path('/home/carus/Data/Object_state_data/episode_54/front')/f'{source_id:06d}.png').convert('RGB')
   canvas.paste(source_cache[source_id],(1280,128));draw=ImageDraw.Draw(canvas);draw.rectangle((1280,64,1919,127),fill=(25,36,49));draw.text((1288,64),LABELS[2],font=base.FONT,fill='white');draw.text((1288,99),f'源帧 {source_id} · 原始动作进度 {prog:g}',font=base.SMALL,fill=(188,231,208))
   if fi in (30,149,190,220,280,350):canvas.save(O/f'preview_{fi:04d}.jpg',quality=90)
   writer.append_data(np.asarray(canvas))
 finally:
  writer.close()
  for run in runs.values():run['cap'].release()
 cap=cv2.VideoCapture(str(VIDEO));fps=cap.get(cv2.CAP_PROP_FPS);count=0
 while True:
  ok,frame=cap.read()
  if not ok:break
  assert frame.shape==(1152,1920,3);count+=1
 cap.release();assert count==len(timeline) and fps==30
 (O/'video_verification.json').write_text(json.dumps(dict(video=str(VIDEO),frames=count,duration_s=count/30,fps=fps,fully_decoded=True,source_actions=n,expanded_actions=ng,inserted=ng-n,threshold_rad=.1,panels=LABELS,initial_states_equal=True,settle_states_equal=True,seed=44,alignment='Original reference progress; inserted midpoints repeat preceding source frame for raw simulation and real front. Physical time differs.',zero_edit_matches_interpolated_direct=True),indent=2)+'\n')
 print(VIDEO,count,count/30,flush=True)
if __name__=='__main__':main()
