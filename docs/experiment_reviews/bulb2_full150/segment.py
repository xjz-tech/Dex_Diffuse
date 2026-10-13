from pathlib import Path
import numpy as np,json
from scipy.ndimage import label
OUT=Path(__file__).resolve().parent
stored=np.load(OUT/'angles.npz')
clips={int(k):dict(angle=stored[k]) for k in stored.files}
ref=np.array(json.loads((OUT/'initial_scan.json').read_text())['reference_axis'])

def runs(mask,minlen):
 labs,n=label(mask);out=[]
 for k in range(1,n+1):
  ids=np.flatnonzero(labs==k)
  if len(ids)>=minlen:out.append((int(ids[0]),int(ids[-1])+1))
 return out

def events(angle,side=60,ready=30,sidehold=9,readyhold=15):
 lateral=runs(angle>=side,sidehold);aligned=runs(angle<=ready,readyhold)
 completed=[];incomplete=[];end=-1
 for start,stop in lateral:
  if start<end:continue
  dest=next(((a,b) for a,b in aligned if a>=stop),None)
  if dest is None:
   incomplete.append(dict(start=start,end=len(angle)));break
  finish=dest[0];end=finish
  peak=start+int(np.argmax(angle[start:finish+1]));last_side=start+int(np.flatnonzero(angle[start:finish]>=side)[-1])
  completed.append(dict(start=start,end=finish,peak=peak,last_side=last_side,seconds=(finish-start)/30,correction_tail_seconds=(finish-last_side)/30,max_angle=float(angle[peak])))
 return completed,incomplete
if __name__ == '__main__':
 summaries=[];total=sum(len(c['angle']) for c in clips.values());main=[]
 for side in [50,55,60,65,70]:
  for ready in [25,30,35]:
   result=[]
   for i,c in clips.items():
    ev,un=events(c['angle'],side,ready)
    result.append(dict(index=i,events=ev,incomplete=un))
   n=sum(bool(r['events']) for r in result);frames=sum(e['end']-e['start'] for r in result for e in r['events']);tail=sum(e['end']-e['last_side'] for r in result for e in r['events'])
   summaries.append(dict(lateral_deg=side,aligned_deg=ready,demos=n,demo_percent=n/150*100,events=sum(len(r['events']) for r in result),frames=frames,frame_percent=frames/total*100,tail_frames=tail,tail_percent=tail/total*100,incomplete_demos=[r['index'] for r in result if r['incomplete']]))
   if side==60 and ready==30:main=result
 (OUT/'segments.json').write_text(json.dumps(dict(fps=30,total_frames=total,method='9-frame median; lateral >=60deg for >=9 frames, then aligned <=30deg for >=15 frames; interval first sustained lateral frame to first sustained aligned frame; excludes outbound tilt, includes lateral dwell; incomplete excluded',reference_axis=ref.tolist(),rows=main,sensitivity=summaries),indent=2))
 print('SENSITIVITY',json.dumps(summaries,indent=2))
 print('PRIMARY',[(r['index'],[(round(e['start']/30,2),round(e['end']/30,2)) for e in r['events']]) for r in main if r['events']])
