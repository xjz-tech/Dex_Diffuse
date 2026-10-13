import json,sys
from pathlib import Path
import cv2
import numpy as np

p=Path(__file__).resolve().parent/'runs'/sys.argv[1]
rows=[json.loads(x) for x in (p/'trajectory.jsonl').read_text().splitlines()]
native=next((r['step'] for r in rows if r['failure']),None)
steps=([int(x) for x in sys.argv[2:]] if len(sys.argv)>2 else sorted(set([1,24,80,160,240,320,400,480,len(rows)])))
if len(sys.argv)<=2 and native:
    steps=sorted(set([1,24, max(1,native-24),max(1,native-12),max(1,native-6),native,min(len(rows),native+12),480,len(rows)]))
images=[]
for s in steps:
    im=cv2.imread(str(p/'frames'/f'{s:06d}.png'));assert im is not None
    im=cv2.resize(im,(480,360));cv2.putText(im,f'{sys.argv[1]}  step {s}  t={s/30:.2f}s',(8,345),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,255),1,cv2.LINE_AA);images.append(im)
while len(images)%3: images.append(np.zeros_like(images[0]))
out=cv2.vconcat([cv2.hconcat(images[i:i+3]) for i in range(0,len(images),3)])
path=p/('review_'+('_'.join(str(s) for s in steps))+'.jpg');cv2.imwrite(str(path),out)
print(json.dumps(dict(run=p.name,native_failure=native,frames=steps,image=str(path),final_twist=rows[-1]['twist_degrees'])))
