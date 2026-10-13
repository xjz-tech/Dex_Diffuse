from pathlib import Path
import sys,json,cv2,numpy as np
from scipy.spatial.transform import Rotation as R
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
P=Path(__file__).resolve().parent;run=P/(sys.argv[1] if len(sys.argv)>1 else 'sweep_final');t=np.load(run/'trajectory.npz');z=np.load(run/'initial_state.npz');m=json.loads((run/'metadata.json').read_text());selected=m['recorded_cases'];focal=640/(2*np.tan(np.deg2rad(42/2)))
for i in selected:
 cap=cv2.VideoCapture(str(run/f'case{i:03d}_two_views.mp4'));target=z['object'][i,:3].astype(float);frames=[]
 with imageio.get_writer(str(run/f'case{i:03d}_clear.mp4'),fps=30,codec='libx264',quality=8) as writer:
  for k in range(180):
   ok,frame=cap.read();assert ok;pos=t['object_pose'][k,i,:3];axis=R.from_quat(t['object_pose'][k,i,3:]).apply([0,1,0])
   for view,offset in enumerate([[.12,.42,.14],[-.32,.24,-.08]]):
    eye=target+offset;eye[2]=max(eye[2],-.32);forward=target-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
    def project(point):
     d=point-eye;depth=d@forward
     if depth<=.02:return None
     xy=np.array([320+focal*(d@right)/depth,240-focal*(d@up)/depth]);return tuple(np.round(xy+[view*640,64]).astype(int))
    p0=project(pos);p1=project(pos+.05*axis)
    if p0 and p1 and all(-200<v<1600 for v in p0+p1):cv2.arrowedLine(frame,p0,p1,(70,235,90),2,cv2.LINE_AA,tipLength=.2)
   rgb=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(rgb);draw.rectangle((0,0,1279,63),fill=(14,19,25))
   phase=str(t['phase'][k]);label={'settle':'静态抓持','reference':'轨迹引导','hold':'末态保持'}[phase]
   failed=bool(t['failure'][:k+1,i].any());angle=float(t['vertical_error_deg'][k,i])
   draw.text((12,2),f'案例 {i}  |  10B + 轨迹引导  |  170 g / 摩擦 2.2  |  原生 wrist',font=font,fill='white')
   draw.text((12,33),f'{label} {int(t["index"][k])/30:.2f} 秒  |  距竖直 {angle:.1f}°  |  原生 failure: {failed}  |  绿箭头：灯泡顶部方向',font=font,fill=(190,230,255))
   frame=cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2BGR);writer.append_data(np.asarray(rgb))
   if k in [59,89,119,149,179]:cv2.imwrite(str(run/f'case{i:03d}_clear{k:03d}.jpg'),frame)
 cap.release()
for page in range((len(selected)+2)//3):
 ids=selected[page*3:(page+1)*3];sheet=Image.new('RGB',(1280,3*300),(20,20,20));draw=ImageDraw.Draw(sheet)
 for row,i in enumerate(ids):
  for col,k in enumerate([59,149]):
   im=Image.open(run/f'case{i:03d}_clear{k:03d}.jpg').resize((640,272));sheet.paste(im,(col*640,row*300));draw.text((col*640+8,row*300+279),f'case {i}: '+('reference start' if col==0 else 'reference end'),fill='white')
 sheet.save(run/f'review_page{page}.jpg')
print(str(len(selected))+' full-size two-view H264 videos saved; arrows = measured bulb dome axis.')
