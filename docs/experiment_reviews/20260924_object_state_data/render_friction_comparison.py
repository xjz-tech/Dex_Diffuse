"""Friction comparison with raw/guide front views and aligned source front."""
import json
from pathlib import Path
import cv2
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from render_friction_comparison_base import load_run,read_to,FONT,SMALL,ROOT,CASES
from render_full_episode53_axes import draw_axes,object_pose
from reference_resampling import interpolate_large_jumps

OUT=ROOT/'m044_mu11_vs_mu14_adaptive010_video_20260927'
OUT.mkdir(exist_ok=True)
VIDEO=OUT/'four_episodes_mu11_vs_mu14_raw_guide_real.mp4'
EPS=(76,34,54,2)
KEYS=('command','q','object_pose','vertical_error_deg','native_failure','hand_body_pose')

def runs_for(ep):
 case=CASES/f'episode_{ep:02d}';runs=[]
 for mode,mu in [('direct',11),('direct',14),('guided',11),('guided',14)]:
  if ep==54 and mode=='guided' and mu==14:
   assert not json.loads((case/'direct_m044_mu14/m044_mu14_result.json').read_text())['completed_turn']
   runs.append(None);continue
  stem=f'direct_m044_mu{mu}' if mode=='direct' else f'guide1_adaptive010_ddim4_scale25_m044_mu{mu}'
  baseline=case/stem;folder=baseline if mode=='direct' else case/(stem+'_video')
  run=load_run(folder,mode);run.update(mode=mode,mu=mu)
  assert run['summary']['friction']==mu/10
  run['loss']=json.loads((baseline/f'm044_mu{mu}_result.json').read_text())['first_separation']
  with np.load(baseline/'initial_state.npz') as a,np.load(folder/'initial_state.npz') as b:
   assert all(np.array_equal(a[k],b[k]) for k in a.files)
  if mode=='guided':
   tr=json.loads((baseline/'trace.json').read_text());assert len(tr)==len(run['trace'])
   assert all(x[k]==y[k] for x,y in zip(tr,run['trace']) for k in KEYS)
   s=run['summary'];assert s['reference_interpolation_threshold']==.1 and s['guidance_scale']==25 and s['execution_steps']==1 and s['prior']['ddim']==4 and s['prior']['guidance_steps']==2
  runs.append(run)
 with np.load(case/'reference_full.npz') as ref:
  source=ref['source_state_frame_indices'][0].astype(int);_,progress=interpolate_large_jumps(ref['hand_target_rad'],.1)
 for run in runs:
  if run and run['mode']=='guided':assert np.array_equal(progress,np.load(Path(run['folder'])/'reference_progress.npy'))
 with np.load(case/'direct_m044_mu11/initial_state.npz') as a,np.load(case/'direct_m044_mu14/initial_state.npz') as b:
  assert set(k for k in a.files if not np.array_equal(a[k],b[k]))=={'hand_friction','object_friction'}
 return runs,source,progress

def main():
 verify=dict(video=str(VIDEO),fps=30,episodes=[],layout='top: raw mu1.1 / raw mu1.4 / real front; bottom: guide mu1.1 / guide mu1.4 / legend',alignment='Original reference progress. Raw and real frames repeat at inserted midpoints. Physics times differ; camera exposure not independently synchronized.',guide=dict(ddim=4,scale=25,guide=2,exec=1,threshold=.1),mass_kg=.044,original_size=True)
 writer=imageio.get_writer(str(VIDEO),fps=30,codec='libx264',quality=7,macro_block_size=16,ffmpeg_params=['-preset','fast','-threads','2','-movflags','+faststart'])
 total=0
 try:
  for ep in EPS:
   runs,source,progress=runs_for(ep);n=len(source)-1;ng=len(progress);start=total
   zero=Image.open(ROOT/f'm044_mu11_adaptive010_ddim4_scale25_four_video_20260927/episode{ep}_zero_physics.png').convert('RGB')
   timeline=[('title',0)]*30+[('import',0)]*30+[('settle',j) for j in range(60)]+[('ready',59)]*30+[('action',j) for j in range(ng)]+[('hold',j) for j in range(60)]
   cached_id=None;real=None
   for phase,j in timeline:
    p=float(progress[j]) if phase=='action' else (float(n) if phase=='hold' else 0.)
    si=min(int(np.floor(p)),n);source_id=int(source[si]);im=Image.new('RGB',(1920,1152),(18,25,35));d=ImageDraw.Draw(im)
    d.text((12,2),f'episode {ep:02d} | 摩擦1.1 与1.4 | 原始reference进度 {p:g}/{n}',font=FONT,fill='white')
    subtitle={'title':'四条依次播放；上排原速reference，下排guide；左右对应摩擦1.1/1.4','import':'物理运行前：精确FK重建；导入位姿相同','settle':'60步静置；真机停留在该episode起点','ready':'各组静置后的动作起点','action':'按原始reference进度对齐；插值组实际物理步更多','hold':'完整动作尾段结束，末目标保持2秒'}[phase]
    d.text((12,32),subtitle,font=SMALL,fill=(185,225,250))
    if phase=='title':writer.append_data(np.asarray(im));total+=1;continue
    for panel,run in enumerate(runs):
     x=(panel%2)*640;y=64+(panel//2)*544
     label=('原速reference' if panel<2 else '>0.1插值 guide')+f' · 摩擦{1.1 if panel%2==0 else 1.4}'
     if run is None:
      d.rectangle((x,y,x+639,y+543),fill=(32,35,41));d.text((x+10,y+5),label,font=FONT,fill='white')
      for line,text in enumerate(['此组未运行','Episode54在摩擦1.4下','原速reference未通过翻转基线','没有补跑或用其他轨迹替代']):d.text((x+45,y+160+42*line),text,font=FONT,fill=(245,191,145))
      continue
     if phase=='import':row=None;frame=zero
     else:
      if phase in ('settle','ready'):idx=j
      elif phase=='action':idx=60+(max(0,int(np.floor(p))-1) if run['mode']=='direct' else j)
      else:idx=60+(n if run['mode']=='direct' else ng)+j
      row=run['trace'][idx];bgr=read_to(run,idx+1);frame=Image.fromarray(cv2.cvtColor(bgr[64:544,:640],cv2.COLOR_BGR2RGB))
     im.paste(frame,(x,y+64))
     if row is not None:draw_axes(im,object_pose(row),run['projectors'][0],(x,y))
     d=ImageDraw.Draw(im);d.rectangle((x,y,x+639,y+63),fill=(25,36,49));d.text((x+8,y+2),label,font=FONT,fill='white')
     if row is None:detail='原始q与物体位姿；未运行物理'
     else:
      loss=run['loss'];sep=bool(loss and (phase=='hold' or (phase=='action' and row['index']>=loss['zero_based_control_index'])))
      detail=f"控制步 {row['index']+1} | 竖直角 {row['vertical_error_deg']:.1f}° | 分离标记 {'已到' if sep else '未到'}"
     d.text((x+8,y+34),detail,font=SMALL,fill=(255,160,140) if '已到' in detail else (180,230,205))
    if cached_id!=source_id:
     real=Image.open(Path('/home/carus/Data/Object_state_data')/f'episode_{ep}/front/{source_id:06d}.png').convert('RGB');cached_id=source_id
    im.paste(real,(1280,128));d=ImageDraw.Draw(im)
    d.text((1290,66),'原数据集 front · 真机',font=FONT,fill='white');d.text((1290,100),f'源帧 {source_id} | 原始进度 {p:g}',font=SMALL,fill=(180,230,205))
    for line,text in enumerate(['固定配置','灯泡44g，原尺寸','Guide：10B / DDIM4 / scale25','引导2步，执行1步','相邻最大关节跳变 >0.1 rad 插1中点','左列：摩擦1.1；右列：摩擦1.4','仿真正面；灯泡XYZ轴：红/绿/蓝','分离标记来自几何/接触判据','按原始动作进度对齐，不是相同物理时间']):d.text((1295,640+40*line),text,font=FONT if line<4 else SMALL,fill='white' if line<4 else (180,210,225))
    if phase=='action' and j in (0,ng//2,ng-1):im.save(OUT/f'ep{ep}_action{j}.jpg',quality=90)
    writer.append_data(np.asarray(im));total+=1
   for run in runs:
    if run:run['cap'].release()
   verify['episodes'].append(dict(episode=ep,start_frame=start,end_frame=total-1,source_actions=n,guide_actions=ng,mu14_guide_recorded=ep!=54,guide_recordings_exact=True))
   print('COMPOSED',ep,len(timeline),flush=True)
 finally:writer.close()
 cap=cv2.VideoCapture(str(VIDEO));count=0
 while True:
  ok,im=cap.read()
  if not ok:break
  assert im.shape==(1152,1920,3);count+=1
 assert count==total and abs(cap.get(cv2.CAP_PROP_FPS)-30)<1e-6;cap.release()
 verify.update(frames=count,duration_s=count/30,fully_decoded=True)
 (OUT/'verification.json').write_text(json.dumps(verify,indent=2)+'\n');print('VERIFIED',VIDEO,count,count/30,flush=True)

if __name__=='__main__':main()
