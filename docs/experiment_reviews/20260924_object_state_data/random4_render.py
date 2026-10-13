"""Full front-view comparison, source-progress aligned, with initialization and wide insets."""
from pathlib import Path
import json,sys
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from render_full_episode53_axes import camera_projector,draw_axes,object_pose
P=Path(__file__).resolve().parent;ROOT=P/'random4_ep53_standard_20260926'
FONT=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
SMALL=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
METHODS=[('direct','原始 reference actions · 仿真', (0,64)),('guide2','10B guide2 / exec2 · scale50 · 插1', (0,608)),('guide1','10B guide2 / exec1 · scale50 · 插1',(640,608))]
def render(ep):
 folder=ROOT/f'episode_{ep:02d}';runs=[]
 for mode,title,corner in METHODS:
  sub=folder/mode;s=json.load(open(sub/'summary.json'));t=json.load(open(sub/'trace.json'));r=json.load(open(sub/'retention.json'))
  cap=cv2.VideoCapture(str(sub/('direct_front.mp4' if mode=='direct' else 'guided_front.mp4')))
  assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==1+len(t)
  c=s['camera_metadata'][0]
  runs.append(dict(mode=mode,title=title,corner=corner,s=s,t=t,r=r,cap=cap,last=-1,frame=None,project=camera_projector(np.array(c['eye']),np.array(c['target']),c['horizontal_fov'])))
 n=runs[0]['s']['steps']['action'];ng=2*n-1;start=runs[0]['s']['source_start_frame'];end=start+n
 timeline=[('import',0)]*30+[('settle',j) for j in range(60)]+[('ready',59)]*30+[('action',j) for j in range(ng)]+[('hold',j) for j in range(60)]
 path=folder/f'episode_{ep:02d}_comparison.mp4'
 writer=imageio.get_writer(str(path),fps=30,codec='libx264',quality=7,macro_block_size=16,ffmpeg_params=['-preset','fast','-threads','2','-movflags','+faststart'])
 cachedf=-1;real=None
 markers={120}
 for run in runs:
  loss=run['r']['first_separation']
  if loss and loss['phase']=='action':markers.update(120+int((loss['reference_action_number']-1)*2)+d for d in [-6,0,6,18])
 for frameidx,(phase,j) in enumerate(timeline):
  canvas=Image.new('RGB',(1280,1152),(17,24,33));draw=ImageDraw.Draw(canvas)
  prog=1+j/2 if phase=='action' else (n if phase=='hold' else 0)
  header=f'episode {ep:02d} · 自身源帧 {start} 起 · 170g / 摩擦2.2 · 同初态'
  draw.text((12,1),header,font=FONT,fill='white')
  names={'import':'导入瞬间（暂停1秒）','settle':'开始前静置检查','ready':'抓稳后的共同动作起点（暂停1秒）','action':f'原始 reference 进度：第 {prog:g} 个动作 / {n}','hold':'全部动作结束，再保持2秒'}
  draw.text((12,33),names[phase]+'   |   按动作进度对齐，direct画面放慢2倍',font=SMALL,fill=(190,222,250))
  realf=start if phase in ['import','settle','ready'] else (min(end,start+j//2+1) if phase=='action' else end)
  if realf!=cachedf:real=Image.open(f'/home/carus/Data/Object_state_data/episode_{ep}/front/{realf:06d}.png').convert('RGB');cachedf=realf
  canvas.paste(real,(640,128));draw.rectangle((640,64,1279,127),fill=(25,36,49));draw.text((648,65),f'真机原视频 · episode {ep:02d}',font=FONT,fill='white');draw.text((648,96),f'状态帧 {realf} | 仿真固定wrist，仅回放手指',font=SMALL,fill=(190,230,213))
  for run in runs:
   count=run['s']['steps']['action'];mode=run['mode'];x,y=run['corner'];row=None
   if phase=='import':idx=0
   elif phase in ['settle','ready']:idx=1+j;row=run['t'][j]
   elif phase=='action':act=j//2 if mode=='direct' else j;idx=61+act;row=run['t'][60+act]
   else:idx=61+count+j;row=run['t'][60+count+j]
   while run['last']<idx:
    ok,fr=run['cap'].read();assert ok,(ep,mode,idx);run['frame']=fr;run['last']+=1
   rgb=cv2.cvtColor(run['frame'],cv2.COLOR_BGR2RGB);canvas.paste(Image.fromarray(rgb[:,:640]),(x,y))
   if row:draw_axes(canvas,object_pose(row),run['project'],(x,y))
   inset=Image.fromarray(rgb[64:,640:]).resize((192,144));canvas.paste(inset,(x+448,y+400));draw=ImageDraw.Draw(canvas);draw.text((x+452,y+400),'宽景：含桌面',font=SMALL,fill='white',stroke_width=1,stroke_fill='black')
   draw.rectangle((x,y,x+639,y+63),fill=(25,36,49));draw.text((x+8,y),run['title'],font=FONT,fill='white')
   loss=run['r']['first_separation'];lost=bool(loss and row and (idx-1)>=loss['trace_index'])
   status=f'脱手标记：ref {loss["reference_action_number"]:g}' if lost else ('抓稳起点' if phase=='ready' else '尚未达到脱手标记')
   cnt=f'执行 {row["index"]+1}' if row and phase=='action' else phase
   native=f' | native失败 {int(row["native_failure"])}' if row else ''
   draw.text((x+8,y+34),f'{cnt} | {status}{native}',font=SMALL,fill=(255,132,115) if lost else (178,241,204))
   if phase=='ready':
    m=json.load(open(folder/'static/grasp_metrics.json'))['static']
    draw.text((x+10,y+515),f'静置末1秒位移变化 {m["last30_translation_range_m"]*1000:.2f}mm',font=SMALL,fill='white',stroke_width=1,stroke_fill='black')
  writer.append_data(np.asarray(canvas))
  if frameidx in markers or frameidx in [0,119,len(timeline)-1]:canvas.save(folder/f'comparison_{frameidx:04d}.jpg')
 writer.close()
 for run in runs:run['cap'].release()
 (folder/'video_manifest.json').write_text(json.dumps(dict(episode=ep,file=str(path),frames=len(timeline),fps=30,source_start=start,source_end=end,initialization_frames=120,action_frames=ng,hold_frames=60,alignment='Original reference action progress; raw direct held twice; guide actual every control step. Real post-action state frame uses start+floor(j/2)+1. Independent sim times differ.',axes='sim object XYZ: red green blue; exact simulation camera projection; real not overlaid without calibration'),indent=2))
 print('RENDERED',ep,path,flush=True)
if __name__=='__main__':
 for ep in map(int,sys.argv[1:]):render(ep)
