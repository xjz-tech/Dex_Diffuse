from pathlib import Path
import json,hashlib
import cv2,numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;run=P/'review_repeat';out=P/'four_cases_comparison';out.mkdir(exist_ok=True)
ids=[66,110,137,182];t=np.load(run/'trajectory.npz');font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',23);small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
caps={i:cv2.VideoCapture(str(run/f'case{i:03d}_clear.mp4')) for i in ids}
with imageio.get_writer(str(out/'four_cases_slow.mp4'),fps=15,codec='libx264',quality=8) as writer:
 for k in range(180):
  im=Image.new('RGB',(1280,1152),(13,18,25));draw=ImageDraw.Draw(im)
  phase={'settle':'初始静态抓持','reference':'参考动作执行','hold':'末态保持'}[str(t['phase'][k])]
  draw.text((14,0),'4 个 case 同初态补录  |  10B prior + 引导  |  170 g / 摩擦 2.2 / 原生 wrist',font=font,fill='white')
  draw.text((14,33),f'{phase} {(int(t["index"][k])+1)/30:.2f} s  |  0.5× 半速  |  角度越小越竖直  |  本轮结果与首次筛选可能不同',font=small,fill=(190,220,240))
  for j,i in enumerate(ids):
   ok,frame=caps[i].read();assert ok;assert frame.shape==(544,1280,3)
   x=(j%2)*640;y=64+(j//2)*544
   # Use the full-resolution unobstructed side camera; existing measured-axis overlay preserved.
   view=Image.fromarray(cv2.cvtColor(frame[64:,640:,:],cv2.COLOR_BGR2RGB));im.paste(view,(x,y+64));draw.rectangle((x,y,x+639,y+63),fill=(20,29,39));angle=float(t['vertical_error_deg'][k,i]);start=float(t['vertical_error_deg'][59,i]);fail=bool(t['failure'][:k+1,i].any())
   draw.text((x+12,y+1),f'Case {i}  |  当前距竖直 {angle:.1f}°',font=font,fill=(115,245,155) if i==110 else 'white')
   draw.text((x+12,y+34),f'动作起点 {start:.1f}°  |  原生 failure: {fail}',font=small,fill=(190,220,240))
   draw.line((x,y+543,x+639,y+543),fill=(120,130,140),width=2)
  draw.line((640,64,640,1151),fill=(120,130,140),width=2);writer.append_data(np.asarray(im))
  if k in [59,149,179]:im.save(out/f'frame{k:03d}.jpg')
for cap in caps.values():
 ok,_=cap.read();assert not ok;cap.release()
cap=cv2.VideoCapture(str(out/'four_cases_slow.mp4'));n=0;fps=cap.get(cv2.CAP_PROP_FPS)
while True:
 ok,frame=cap.read()
 if not ok:break
 assert frame.shape==(1152,1280,3);n+=1
cap.release();assert n==180 and fps==15
meta=dict(cases=ids,source_run='review_repeat',layout='top-left66,top-right110,bottom-left137,bottom-right182',view='right side camera from existing two-view recording; no new simulation',frames=n,fps=fps,duration_s=n/fps,simulated_duration_s=6,repeat_not_original_sweep=True,angles={i:dict(start=float(t['vertical_error_deg'][59,i]),end=float(t['vertical_error_deg'][179,i])) for i in ids},sources={str(run/f'case{i:03d}_clear.mp4'):hashlib.sha256((run/f'case{i:03d}_clear.mp4').read_bytes()).hexdigest() for i in ids})
(out/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
