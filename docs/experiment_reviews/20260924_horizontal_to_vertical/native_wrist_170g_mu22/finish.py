from pathlib import Path
import json,numpy as np,hashlib,cv2
P=Path(__file__).resolve().parent
s=P/'sweep_final';r=P/'review_repeat';a=np.load(s/'initial_state.npz');b=np.load(r/'initial_state.npz');assert set(a.files)==set(b.files)
for k in a.files:np.testing.assert_array_equal(a[k],b[k],err_msg=k)
x=np.load(s/'trajectory.npz');y=np.load(r/'trajectory.npz');diffs={}
for k in x.files:
 if x[k].dtype.kind in 'fiu':diffs[k]=float(np.max(np.abs(x[k].astype(float)-y[k].astype(float))))
 else:diffs[k]=bool(np.array_equal(x[k],y[k]))
# This repeat changes camera inventory only; report any physics disagreement.
validation=dict(initial_fields_equal=len(a.files),trajectory_max_abs_differences=diffs,all_trajectory_arrays_exact=all(np.array_equal(x[k],y[k]) for k in x.files),repeat_counts_not_independent=True)
summary=json.loads((s/'summary.json').read_text());repeat=json.loads((r/'summary.json').read_text());validation['changed_outcome_case_ids']=[i for i,(v,w) in enumerate(zip(summary['results'],repeat['results'])) if (v['static_pass'],v['native_failure'])!=(w['static_pass'],w['native_failure'])]
validation['repeat_summary']={k:v for k,v in repeat.items() if k!='results'}
validation['case110_first_run']=summary['results'][110]
validation['case110_recorded_repeat']=repeat['results'][110]
video=[]
for f in r.glob('*_clear.mp4'):
 c=cv2.VideoCapture(str(f));n=0
 while True:
  ok,frame=c.read()
  if not ok:break
  n+=1
 c.release();assert n==180;video.append(f.name)
validation['review_h264_videos_verified']=video
(P/'reproduction_verification.json').write_text(json.dumps(validation,indent=2));best=summary['results'][110];(P/'case110.json').write_text(json.dumps(best,indent=2))
np.savez_compressed(P/'case110_initial_slice.npz',**{k:(a[k][110:111] if a[k].ndim and a[k].shape[0]==189 else a[k]) for k in a.files})
(P/'artifact_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.glob('*.py')},indent=2))
print(json.dumps(validation,indent=2))
