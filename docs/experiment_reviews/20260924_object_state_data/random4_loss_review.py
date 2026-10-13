import json,cv2
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent/'random4_ep53_standard_20260926'
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
for page,episodes in enumerate([[50,76],[70,40]]):
 canvas=Image.new('RGB',(960,6*266),(18,24,32));draw=ImageDraw.Draw(canvas);i=0
 for ep in episodes:
  for mode in ['direct','guide2','guide1']:
   f=P/f'episode_{ep:02d}'/mode
   if not (f/'retention.json').exists():continue
   loss=json.load(open(f/'retention.json'))['first_separation'];cap=cv2.VideoCapture(str(f/('direct_front.mp4' if mode=='direct' else 'guided_front.mp4')))
   for j,delta in enumerate([-4,0,8]):
    idx=1+loss['trace_index']+delta;cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,img=cap.read();assert ok
    canvas.paste(Image.fromarray(cv2.cvtColor(img[64:,:640],cv2.COLOR_BGR2RGB)).resize((320,240)),(j*320,i*266+26))
    draw.text((j*320+4,i*266+2),f'ep{ep} {mode} 脱手标记{delta:+}控制步',font=font,fill='white')
   cap.release();i+=1
 canvas.save(P/f'loss_review_{page+1}.jpg')
