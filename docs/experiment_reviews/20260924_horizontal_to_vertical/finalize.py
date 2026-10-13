from pathlib import Path
import json,hashlib,cv2,numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
A=np.load(P/'guided/initial_state.npz');B=np.load(P/'direct/initial_state.npz')
assert A.files==B.files
for k in A.files:np.testing.assert_array_equal(A[k],B[k],err_msg=k)
a=np.load(P/'guided/trajectory.npz');b=np.load(P/'direct/trajectory.npz')
for k in a.files:np.testing.assert_array_equal(a[k][:60],b[k][:60],err_msg=k)
checks=dict(initial_fields_exact=A.files,static_60_steps_exact=True,reference_source_exact=json.loads((P/'reference/verification.json').read_text()),videos={})
for mode in ['guided','direct']:
 for v in [*(P/mode).glob('env*.mp4'),P/mode/'comparison.mp4']:
  c=cv2.VideoCapture(str(v));fps=c.get(cv2.CAP_PROP_FPS);n=0
  while True:
   ok,f=c.read()
   if not ok:break
   n+=1
  c.release();assert n==184,(v,n);checks['videos'][str(v.relative_to(P))]=dict(decoded_frames=n,fps=fps)
fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,mode in zip(axs,['guided','direct']):
 t=np.load(P/mode/'trajectory.npz');secs=(np.arange(59,180)-59)/30
 for i in range(3):
  angle=np.degrees(np.arccos(np.clip(R.from_quat(t['object_pose'][59:,i,3:]).apply([0,1,0])[:,2],-1,1)));f=np.flatnonzero(t['failure'][59:,i]);stop=f[0] if len(f) else len(angle)
  line,=ax.plot(secs[:stop],angle[:stop],label=f'env {i}')
  if len(f):ax.plot(secs[stop:],angle[stop:],':',color=line.get_color(),alpha=.6);ax.axvline(secs[stop],ls='--',color=line.get_color(),alpha=.3)
 ax.axhspan(0,20,color='green',alpha=.10);ax.axvline(3,color='gray',ls='--');ax.set_title('10B + reference guidance' if mode=='guided' else 'Direct recorded targets');ax.set_xlabel('Seconds from reference start');ax.set_ylim(0,180);ax.grid(alpha=.2);ax.legend()
axs[0].set_ylabel('Angle from screw-down vertical (deg)');fig.suptitle('Horizontal-to-vertical trial | dotted curves: after native failure');fig.tight_layout();fig.savefig(P/'vertical_error.png',dpi=150);plt.close(fig)
checks['visual_review']={'guided_env0':'empty hand at reference end; bulb separated','guided_env1':'bulb remains between fingers, not vertical','guided_env2':'empty hand at reference end; bulb separated','basis':'inspected actual-camera comparison frames at reference end; no physical drop onset time inferred'}
(P/'verification.json').write_text(json.dumps(checks,indent=2))
(P/'artifact_hashes.json').write_text(json.dumps({str(x.relative_to(P)):hashlib.sha256(x.read_bytes()).hexdigest() for x in P.glob('*.py')},indent=2))
print('Verified paired initial states, static prefix, eight videos and reference arrays.')
