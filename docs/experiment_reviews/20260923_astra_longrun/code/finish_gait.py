import json,subprocess,time,sys
from pathlib import Path
h=Path(__file__).resolve().parents[1];version=sys.argv[1] if len(sys.argv)>1 else 'v1';v=h/'gait_trials'/version;py='/home/carus/miniforge3/envs/dp/bin/python'
while not (v/'batch_finished.json').exists():time.sleep(1)
assert json.loads((v/'batch_finished.json').read_text())['complete'], 'Missing runs require retained-log review and retry'
for script,args in [('gait_audit.py',[str(v)]),('clips.py',[str(v)]),('analyze.py',[str(v)]),('gait_results.py',['--version',version,'--videos'])]:
 subprocess.run([py,str(h/'code'/script),*args],check=True)
import cv2
validation=[]
clip_validation=[]
for q in sorted((v/'runs').glob('*/summary.json')):
 r=json.loads(q.read_text())
 if r['phase']!='formal':continue
 p=q.parent/'comparison_clip.mp4';c=cv2.VideoCapture(str(p));n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));fps=c.get(cv2.CAP_PROP_FPS);assert n==r['steps'] and fps==30
 c.set(cv2.CAP_PROP_POS_FRAMES,n-1);ok,_=c.read();assert ok;c.release();clip_validation.append(dict(label=r['label'],frames=n,fps=fps,seconds=n/fps,terminal_frame_decoded=True))
assert len(clip_validation)==27
(v/'clip_validation.json').write_text(json.dumps(clip_validation,indent=2))
for p in sorted(v.glob('comparison_*.mp4')):
 c=cv2.VideoCapture(str(p));n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));fps=c.get(cv2.CAP_PROP_FPS);w=c.get(cv2.CAP_PROP_FRAME_WIDTH);hh=c.get(cv2.CAP_PROP_FRAME_HEIGHT);assert n>0 and fps==10 and (w,hh)==(1440,940)
 c.set(cv2.CAP_PROP_POS_FRAMES,n-1);ok,f=c.read();assert ok;c.release();cv2.imwrite(str(p.with_name(p.stem+'_final.jpg')),f)
 validation.append(dict(path=str(p),frames=n,fps=fps,seconds=n/fps,terminal_frame_decoded=True))
assert len(validation)==6
(v/'video_validation.json').write_text(json.dumps(validation,indent=2))
print('All 27 trials, six real comparison videos and audits complete. Cap camera reviews and visual layout inspection still require agent review.',flush=True)
