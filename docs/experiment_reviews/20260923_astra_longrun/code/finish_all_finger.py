import json,time,subprocess
from pathlib import Path
import cv2
h=Path(__file__).resolve().parents[1];v=h/'gait_trials/v3_all_fingers';py='/home/carus/miniforge3/envs/dp/bin/python'
while not (v/'batch_finished.json').exists():time.sleep(1)
assert json.loads((v/'batch_finished.json').read_text())['complete']
for name,args in [('all_finger_diagnostics.py',[]),('clips.py',[str(v)]),('all_finger_report.py',[])]:subprocess.run([py,str(h/'code'/name),*args],check=True)
results=[]
for p in sorted((v/'runs').glob('*/summary.json')):
 r=json.loads(p.read_text())
 if r['phase']!='formal':continue
 movie=p.parent/'comparison_clip.mp4';cap=cv2.VideoCapture(str(movie));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.set(cv2.CAP_PROP_POS_FRAMES,count-1);ok,im=cap.read();cap.release();assert count==r['steps'] and abs(fps-30)<1e-6 and ok
 results.append(dict(path=str(movie),frames=count,fps=fps,last_frame_decoded=ok))
p=v/'comparison_env50_noise0.mp4';cap=cv2.VideoCapture(str(p));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.set(cv2.CAP_PROP_POS_FRAMES,count-1);ok,im=cap.read();cap.release();assert ok and fps==10;cv2.imwrite(str(v/'comparison_final.jpg'),im);results.append(dict(path=str(p),frames=count,fps=fps,last_frame_decoded=ok))
(v/'video_validation.json').write_text(json.dumps(results,indent=2));print('All7 clips and comparison video validated',flush=True)
