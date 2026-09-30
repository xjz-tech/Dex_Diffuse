"""Contact sheets of existing camera frames for explicit physical-loss review."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--run')
p.add_argument('--root',type=Path,default=HERE)
p.add_argument('--steps',type=int,nargs='+')
a=p.parse_args()
HERE=a.root
pending=list((HERE/'runs').glob('*/physical_review_pending.json'))
if a.run:
    pending=[HERE/'runs'/a.run/'physical_review_pending.json']
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
for path in pending:
    data=json.loads(path.read_text());run=path.parent;step=data['step']
    steps=a.steps or sorted(set(max(1,step-d) for d in (45,24,12,6,3,0)))
    states={}
    for line in (run/'trajectory.jsonl').open():
        r=json.loads(line)
        if r['step'] in steps:
            states[r['step']]=r
    nrows=(len(steps)+2)//3
    canvas=Image.new('RGB',(1440,55+420*nrows),'#0f172a');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),run.name+' | first native terminal review; actual recorded camera frames',font=font,fill='white')
    for i,s in enumerate(steps):
        x,y=(i%3)*480,55+(i//3)*420
        im=Image.open(run/'frames'/f'{s:06d}.png').convert('RGB').resize((480,360))
        canvas.paste(im,(x,y))
        r=states[s];state=r['state'];obj=np.array(state['object_position']);wrist=np.array(state['wrist_state'][:3])
        draw.text((x+8,y+362),f"step {s} / {s/30:.3f}s | right={-r['twist_degrees']:.1f} deg",font=font,fill='white')
        draw.text((x+8,y+385),f"object-wrist distance {np.linalg.norm(obj-wrist):.3f}m | native={r['failure']}",font=font,fill='#cbd5e1')
    out=run/f'review_contact_sheet_{step:06d}.jpg'
    canvas.save(out,quality=94)
    print(json.dumps(dict(run=run.name,pending=data,steps=steps,sheet=str(out))),flush=True)
