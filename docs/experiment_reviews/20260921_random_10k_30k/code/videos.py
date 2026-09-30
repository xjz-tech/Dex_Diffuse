from pathlib import Path
import json,subprocess,cv2,numpy as np,imageio_ffmpeg
R=Path(__file__).resolve().parents[1];OLD=R.parent/'20260920_10k_domain_guides';ff=imageio_ffmpeg.get_ffmpeg_exe()
names=['ordinary_1b','old10k','new10k_seed43','new30k_seed44']
paths=[(OLD if j<2 else R)/'evaluation'/n for j,n in enumerate(names)]
results=[json.loads((p/'results.json').read_text()) for p in paths]
assert all(r['complete'] for r in results)
dest=R/'videos';dest.mkdir(exist_ok=True);records=[]
for i in results[0]['recorded_cases']:
 caps=[cv2.VideoCapture(str(p/f'case{i:02d}_raw.mp4')) for p in paths]
 counts=[int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
 assert counts==[r['results'][i]['steps'] for r in results]
 frames=[None]*4;tmp=dest/f'case{i:02d}_raw.mp4';target=dest/f'case{i:02d}_comparison_10x.mp4'
 writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,1160));assert writer.isOpened()
 case=results[0]['results'][i]
 sample_indices=list(range(0,max(counts),10));sample_indices[-1]=max(counts)-1;sample_indices=set(sample_indices)
 for t in range(max(counts)):
  for j,c in enumerate(caps):
   if t<counts[j]:ok,frames[j]=c.read();assert ok
  if t not in sample_indices and t!=(max(counts)//20)*10:continue
  canvas=np.zeros((1160,1280,3),np.uint8)
  cv2.putText(canvas,f"{case['initialization']} | noise {case['policy_seed']} | 44g mu2.572 | 10x playback",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(240,240,240),2)
  cv2.putText(canvas,'Top: 1B / old10k (previous run). Bottom: new10k / new30k. Same paired configuration.',(12,57),cv2.FONT_HERSHEY_SIMPLEX,.56,(220,220,220),1)
  for j,frame in enumerate(frames):
   y=80+(j//2)*540;x=(j%2)*640;canvas[y:y+540,x:x+640]=frame
   if t>=counts[j]:cv2.putText(canvas,'ENDED - LAST FRAME',(x+10,y+525),cv2.FONT_HERSHEY_SIMPLEX,.65,(80,120,255),2)
  if t in sample_indices:writer.write(canvas)
  if t in [0,(max(counts)//20)*10,max(counts)-1]:cv2.imwrite(str(dest/f'case{i:02d}_frame{t:05d}.png'),canvas)
 writer.release()
 for c in caps:c.release()
 subprocess.run([ff,'-hide_banner','-loglevel','error','-y','-i',str(tmp),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
 tmp.unlink();c=cv2.VideoCapture(str(target));count=int(c.get(cv2.CAP_PROP_FRAME_COUNT));assert count==(max(counts)+9)//10
 for idx in [0,count//2,count-1]:c.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,img=c.read();assert ok
 c.release()
 records.append({'case':i,'path':str(target),'source_methods':names,'source_frames':counts,'output_frames':count,'speed':10,'verified':True,'terminal_behavior':'labelled last frame held'})
 print('VIDEO COMPLETE',i,flush=True)
(R/'video_verification.json').write_text(json.dumps(records,indent=2))
