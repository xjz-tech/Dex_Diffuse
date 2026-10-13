"""Raw / guided / autonomous prior, synchronized by physical control time."""
import json
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import render_episode2_adaptive010_scale50_g4e2_base as base
from render_full_episode53_axes import draw_axes,object_pose

P=Path(__file__).resolve().parent
R=P/'reference_turn_baseline_20260926'
C=R/'qualified_comparison/episode_02'
OUT=R/'m044_mu11_ddim4_prior_exec2_episode2_20260928'
VIDEO=OUT/'episode2_raw_guided_autonomous_prior_physical_time.mp4'

def main():
 data=base.episode_data(2)
 prior=base.load_run(C/'guide4exec2_adaptive010_ddim4_scale0_m044_mu11_video','guided')
 runs=[data['direct'],data['guide'],prior]
 folders=[C/'direct_m044_mu11',C/'guide4exec2_adaptive010_ddim4_scale50_m044_mu11',Path(prior['folder'])]
 results=[json.loads((f/'m044_mu11_result.json').read_text()) for f in folders]
 labels=['原速 reference','DDIM4 / scale50 / guide4 exec2','纯 10B prior / DDIM4 / exec2']
 n=prior['summary']['steps']['action'];assert n==668
 zero=Image.open(base.OUT/'episode2_zero_physics.png').convert('RGB')
 timeline=[('title',0)]*30+[('import',0)]*30+[('settle',i) for i in range(60)]+[('ready',59)]*30+[('action',i) for i in range(n)]+[('hold',i) for i in range(60)]
 writer=imageio.get_writer(str(VIDEO),fps=30,codec='libx264',quality=7,macro_block_size=16,ffmpeg_params=['-preset','fast','-threads','2','-movflags','+faststart'])
 try:
  for frame_index,(phase,j) in enumerate(timeline):
   canvas=Image.new('RGB',(1920,608),(18,25,35));d=ImageDraw.Draw(canvas)
   elapsed=(j+1)/30 if phase=='action' else (n+j+1)/30 if phase=='hold' else 0
   d.text((12,0),f'Episode 02 · 44g / 摩擦1.1 · 相同初态 · 控制开始后 {elapsed:.2f} 秒',font=base.FONT,fill='white')
   subtitle={'title':'原始 reference / reference guidance / 无 reference 的自主 prior', 'import':'物理运行前 · 导入状态 FK 重建', 'settle':'相同初态的60步静置检查','ready':'共同动作起点','action':'按相同物理时间对齐；纯 prior 未使用 reference，也不插值生成动作','hold':'自主 prior 完成668控制步，末目标保持2秒；较短回放结束后定格'}[phase]
   d.text((12,32),subtitle,font=base.SMALL,fill=(190,225,250))
   if phase=='title':writer.append_data(np.asarray(canvas));continue
   for k,run in enumerate(runs):
    frozen=False
    if phase=='import':row=None;frame=None
    else:
     idx=j if phase in ('settle','ready') else 60+j if phase=='action' else 60+n+j
     frozen=idx>=len(run['trace']);idx=min(idx,len(run['trace'])-1);row=run['trace'][idx]
     frame=base.read_to(run,idx+1)
    if row is None:rgb=np.vstack([np.zeros((64,640,3),dtype=np.uint8),np.asarray(zero)])
    else:rgb=cv2.cvtColor(frame[:,:640],cv2.COLOR_BGR2RGB)
    x=k*640;canvas.paste(Image.fromarray(rgb),(x,64))
    if row is not None:draw_axes(canvas,object_pose(row),run['projectors'][0],(x,64))
    d=ImageDraw.Draw(canvas);d.rectangle((x,64,x+639,127),fill=(25,36,49));d.text((x+8,64),labels[k],font=base.FONT,fill='white')
    if row is None:detail='导入位姿；尚未运行物理'
    else:
     loss=results[k]['first_separation'];sep=loss is not None and idx>=loss['trace_index']
     detail=f'{row["phase"]} {row["index"]+1} | 竖直角 {row["vertical_error_deg"]:.1f}° | '+('已分离' if sep else '未分离')+(' | 已结束定格' if frozen else '')
    d.text((x+8,99),detail,font=base.SMALL,fill=(255,150,130) if '已分离' in detail else (188,231,208))
   if frame_index in [180,210,270,330]:canvas.save(OUT/f'comparison_frame_{frame_index}.jpg')
   writer.append_data(np.asarray(canvas))
 finally:
  writer.close()
  for run in runs:run['cap'].release()
 cap=cv2.VideoCapture(str(VIDEO));count=0
 while True:
  ok,f=cap.read()
  if not ok:break
  assert f.shape==(608,1920,3);count+=1
 assert count==len(timeline) and cap.get(cv2.CAP_PROP_FPS)==30
 cap.release()
 (OUT/'video_verification.json').write_text(json.dumps(dict(video=str(VIDEO),frames=count,fps=30,duration_s=count/30,fully_decoded=True,alignment='equal physical control time, not reference progress',panels=labels),indent=2)+'\n')
 print(VIDEO,count,flush=True)

if __name__=='__main__':main()
