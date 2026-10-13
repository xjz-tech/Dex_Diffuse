import json
from pathlib import Path
import cv2,imageio.v2 as imageio,numpy as np
from PIL import Image,ImageDraw,ImageFont
from render_full_episode53_axes import camera_projector,draw_axes,object_pose
P=Path(__file__).resolve().parent;O=P/'reference_turn_baseline_20260926/episode2_prior_palm_direction_20260928';VIDEO=O/'paired_prior_direction_seeds44_45_46.mp4'
FONT=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20);SMALL=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
def main():
 writer=imageio.get_writer(str(VIDEO),fps=30,codec='libx264',quality=7,macro_block_size=16,ffmpeg_params=['-preset','fast','-threads','2','-movflags','+faststart']);total=0
 try:
  for seed in [44,45,46]:
   runs=[]
   for name in ['original','toward_palm_20']:
    f=O/name/f'prior_seed{seed}';s=json.loads((f/'summary.json').read_text());t=json.loads((f/'trace.json').read_text());result=json.loads((f/'direction_result.json').read_text());cap=cv2.VideoCapture(str(f/'guided_front.mp4'));assert cap.isOpened();cap.read();m=s['camera_metadata'][0];projector=camera_projector(np.array(m['eye']),np.array(m['target']),m['horizontal_fov']);runs.append(dict(cap=cap,trace=t,result=result,projector=projector))
   canvas=Image.new('RGB',(1280,608),(18,25,35));d=ImageDraw.Draw(canvas)
   d.text((12,0),f'10B prior · seed{seed} · 物理运行前导入位姿（FK重建）',font=FONT,fill='white')
   for k,name in enumerate(['original','toward_palm_20']):
    canvas.paste(Image.open(O/(name+'_zero_physics.png')).convert('RGB'),(k*640,128))
    d=ImageDraw.Draw(canvas);d.text((k*640+8,64),'原位置' if k==0 else '向掌心平移2.24cm，尚未静置调整',font=FONT,fill='white')
   for _ in range(30):writer.append_data(np.asarray(canvas));total+=1
   for i in range(300):
    canvas=Image.new('RGB',(1280,608),(18,25,35));d=ImageDraw.Draw(canvas);phase='静置' if i<60 else '自主prior' if i<240 else '末目标保持';step=i+1 if i<60 else i-59 if i<240 else i-239
    d.text((12,0),f'10B prior · DDIM4 exec2 · 无guidance · seed{seed} · {phase} 第{step}步',font=FONT,fill='white');d.text((12,32),'44g / 摩擦1.1 · 同初始手型与灯泡朝向；仅平移灯泡 · 按相同物理时间对齐',font=SMALL,fill=(190,225,250))
    for k,run in enumerate(runs):
     ok,frame=run['cap'].read();assert ok;row=run['trace'][i];x=k*640;rgb=cv2.cvtColor(frame[:,:640],cv2.COLOR_BGR2RGB);canvas.paste(Image.fromarray(rgb),(x,64));draw_axes(canvas,object_pose(row),run['projector'],(x,64));d=ImageDraw.Draw(canvas);d.rectangle((x,64,x+639,127),fill=(25,36,49));label='原位置 · episode2' if k==0 else '向掌心平移2.24cm · 非完整掌心包握';d.text((x+8,64),label,font=FONT,fill='white');loss=run['result']['first_separation'];sep=loss is not None and i>=loss['trace_index'];d.text((x+8,99),f'竖直角 {row["vertical_error_deg"]:.1f}° | '+('已分离' if sep else '未分离'),font=SMALL,fill=(255,150,130) if sep else (188,231,208))
    if seed==44 and i in [59,74,89,113]:canvas.save(O/f'paired_seed44_trace{i}.jpg')
    writer.append_data(np.asarray(canvas));total+=1
   for run in runs:run['cap'].release()
 finally:writer.close()
 cap=cv2.VideoCapture(str(VIDEO));count=0
 while True:
  ok,f=cap.read()
  if not ok:break
  assert f.shape==(608,1280,3);count+=1
 assert count==total==990;cap.release();(O/'video_verification.json').write_text(json.dumps(dict(video=str(VIDEO),frames=count,fps=30,duration_s=33,fully_decoded=True,alignment='physical control time',import_note='One second FK pre-physics import, two seconds settling, six seconds autonomous actions, two seconds terminal hold per seed.'),indent=2)+'\n');print(VIDEO,flush=True)
if __name__=='__main__':main()
