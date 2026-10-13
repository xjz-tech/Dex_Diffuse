from pathlib import Path
import json,subprocess,cv2,numpy as np,imageio_ffmpeg
r=Path(__file__).resolve().parents[1];names=['ordinary_1b','old10k','light_low','light_high','heavy_low','heavy_high','balanced'];ff=imageio_ffmpeg.get_ffmpeg_exe()
complete=[name for name in names if (r/'evaluation'/name/'results.json').exists()]
for name in complete:
 p=r/'evaluation'/name;res=json.loads((p/'results.json').read_text())
 for i in res['recorded_cases']:
  src=p/f'case{i:02d}_raw.mp4';dest=p/f'case{i:02d}.mp4'
  if not dest.exists():subprocess.run([ff,'-hide_banner','-loglevel','error','-y','-i',str(src),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
  cap=cv2.VideoCapture(str(dest));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert n==res['results'][i]['steps']
  for idx in [0,n//2,n-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,img=cap.read();assert ok
  cap.release()
if len(complete)<7:print('single videos verified',len(complete));raise SystemExit
base=json.loads((r/'evaluation/ordinary_1b/results.json').read_text());dest=r/'videos';dest.mkdir(exist_ok=True);records=[]
for i in base['recorded_cases']:
 target=dest/f'case{i:02d}_all_guides.mp4'
 if not target.exists():
  cap=[cv2.VideoCapture(str(r/'evaluation'/name/f'case{i:02d}_raw.mp4')) for name in names];counts=[int(x.get(cv2.CAP_PROP_FRAME_COUNT)) for x in cap];frames=[None]*7;tmp=dest/f'case{i:02d}_mosaic_raw.mp4';w=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),30,(1920,890));assert w.isOpened();case=base['results'][i]
  quick_tmp=dest/f'case{i:02d}_overview_raw.mp4';quick=dest/f'case{i:02d}_overview_10x.mp4';qw=cv2.VideoWriter(str(quick_tmp),cv2.VideoWriter_fourcc(*'mp4v'),30,(1920,890));assert qw.isOpened()
  for t in range(max(counts)):
   panel=np.zeros((890,1920,3),np.uint8);title=f"{case['initialization']} | prior noise seed {case['policy_seed']} | 44g, friction 2.572 | DDIM 4/4 exec 2 | 1x"
   cv2.putText(panel,title,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.75,(245,245,245),2)
   for j,c in enumerate(cap):
    if t<counts[j]:ok,frames[j]=c.read();assert ok
    img=cv2.resize(frames[j],(480,405));row=j//4;col=j%4;panel[50+row*420:50+row*420+405,col*480:col*480+480]=img
    if t>=counts[j]:cv2.putText(panel,'ENDED - LAST FRAME',(col*480+10,50+row*420+400),cv2.FONT_HERSHEY_SIMPLEX,.5,(80,120,255),1)
   legend=['New guide data: actual train means','light: 61.3g | heavy: 243.8g','low mu: 0.983 | high mu: 3.225','balanced: 4 domains, 25% each','New guides: 9k train + 1k validation','1B fixed | guide scale 25','Native xjz failure | 400s observation cap','ENDED: source ended; last frame held']
   for k,line in enumerate(legend):cv2.putText(panel,line,(1450,505+k*40),cv2.FONT_HERSHEY_SIMPLEX,.47,(230,230,230),1)
   w.write(panel)
   if t%10==0:
    preview=panel.copy();preview[:50]=0;cv2.putText(preview,title[:-2]+'10x playback | panel clocks show simulation time',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.64,(245,245,245),2);qw.write(preview)
   if t in [0,max(counts)//2,max(counts)-1]:cv2.imwrite(str(dest/f'case{i:02d}_frame{t:05d}.png'),panel)
  w.release();qw.release()
  for c in cap:c.release()
  subprocess.run([ff,'-hide_banner','-loglevel','error','-y','-i',str(tmp),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
  subprocess.run([ff,'-hide_banner','-loglevel','error','-y','-i',str(quick_tmp),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(quick)],check=True);quick_tmp.unlink()
  (dest/f'case{i:02d}_metadata.json').write_text(json.dumps({'source_frames':counts,'output_frames':max(counts),'fps':30,'methods':names,'shorter_side':'holds labelled terminal frame'},indent=2));tmp.unlink()
 meta=json.loads((dest/f'case{i:02d}_metadata.json').read_text());c=cv2.VideoCapture(str(target));assert int(c.get(cv2.CAP_PROP_FRAME_COUNT))==meta['output_frames']
 for idx in [0,meta['output_frames']//2,meta['output_frames']-1]:c.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,img=c.read();assert ok
 c.release();quick=dest/f'case{i:02d}_overview_10x.mp4';c=cv2.VideoCapture(str(quick));expected=(meta['output_frames']+9)//10;assert int(c.get(cv2.CAP_PROP_FRAME_COUNT))==expected
 for idx in [0,expected//2,expected-1]:c.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,img=c.read();assert ok
 c.release();records.append({'path':str(target),'verified':True,'overview_10x':str(quick),'overview_verified':True});print('VIDEO',target,flush=True)
(r/'video_verification.json').write_text(json.dumps({'single_videos':21,'mosaics':records},indent=2))
