from pathlib import Path
import json,numpy as np,cv2,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;D=Path('/home/carus/Data/Object_state_data');specs=json.loads((P/'reference/initial_state.json').read_text())['specs'];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
for spec in specs:
 ep=spec['episode'];path=P/'reference'/f'episode{ep}_source.mp4'
 with imageio.get_writer(str(path),fps=30,codec='libx264',quality=8) as w:
  for k in range(spec['start'],spec['end']+1):
   ims=[Image.open(D/f'episode_{ep}/{view}/{k:06d}.png').convert('RGB').resize((640,480)) for view in ['front','wrist']];out=Image.new('RGB',(1280,544),(15,20,26));out.paste(ims[0],(0,64));out.paste(ims[1],(640,64));draw=ImageDraw.Draw(out);draw.text((12,1),f'Object_state_data / episode{ep} / 原始帧{k}  |  真机双视角',font=font,fill='white');draw.text((12,33),'已拿起后的空中调整抓握；仅手指轨迹用于后续仿真引导',font=font,fill=(190,230,255));w.append_data(np.asarray(out))
   if k in [spec['start'],spec['end']]:out.save(P/'reference'/f'episode{ep}_frame{k}.jpg')
 print(path)
