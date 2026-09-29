from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
P=Path(__file__).resolve().parent/'reference_turn_baseline_20260926/qualified_comparison';OUT=Path('/home/carus/Downloads/Object_state_data_reference_turn_verified.mp4');EPS=[76,34,54,2]
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22);small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17)
runs=[]
for ep in EPS:
 f=P/f'episode_{ep:02d}/direct';t=json.load(open(f/'trace.json'));s=json.load(open(f/'summary.json'));cap=cv2.VideoCapture(str(f/'direct_front.mp4'));runs.append(dict(ep=ep,t=t,s=s,cap=cap,last=-1,frame=None))
# Imported state, settle, shared start, first100 raw controls (half-speed), endpoint freeze.
timeline=[0]*30+list(range(1,61))+[60]*30+[61+j//2 for j in range(200)]+[160]*60
with imageio.get_writer(str(OUT),fps=30,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-preset','fast','-threads','2','-movflags','+faststart']) as writer:
 for k,idx in enumerate(timeline):
  canvas=Image.new('RGB',(1280,1152),(17,24,33));d=ImageDraw.Draw(canvas);d.text((12,0),'4条原始 reference 翻转验收：无prior / 无插值 / 170g / 摩擦2.2',font=font,fill='white')
  phase='导入与静置' if k<120 else ('原始动作前100步 · 视频放慢2倍（实际仍30Hz）' if k<320 else '验收片段末帧定格2秒 · 后续完整回放另存')
  d.text((12,34),phase,font=small,fill=(190,226,244))
  for run,(x,y) in zip(runs,[(0,64),(640,64),(0,608),(640,608)]):
   while run['last']<idx:
    ok,fr=run['cap'].read();assert ok;run['frame']=fr;run['last']+=1
   rgb=cv2.cvtColor(run['frame'],cv2.COLOR_BGR2RGB);canvas.paste(Image.fromarray(rgb[:,:640]),(x,y));canvas.paste(Image.fromarray(rgb[64:,640:]).resize((192,144)),(x+448,y+400));d=ImageDraw.Draw(canvas);d.rectangle((x,y,x+639,y+63),fill=(25,36,49))
   row=run['t'][max(0,idx-1)];step=idx-60 if idx>=61 else 0
   d.text((x+8,y),f'episode {run["ep"]} · 自身源帧 {run["s"]["source_start_frame"]} 起',font=font,fill='white');d.text((x+8,y+34),f'原始动作 {step} | 灯泡距竖直 {row["vertical_error_deg"]:.1f}° | 宽景含桌面',font=small,fill=(177,245,205))
  writer.append_data(np.asarray(canvas))
  if k==319:canvas.save(P/'all_four_reference_turned.jpg')
for run in runs:run['cap'].release()
cap=cv2.VideoCapture(str(OUT));n=0
while cap.read()[0]:n+=1
assert n==len(timeline);cap.release();(P/'proof_video_verification.json').write_text(json.dumps(dict(file=str(OUT),frames=n,duration_s=n/30,raw_reference_steps=100,actual_control_hz=30,action_video_speed=.5,last_frame_freeze_seconds=2),indent=2));print(OUT,flush=True)
