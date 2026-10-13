from pathlib import Path
import json,sys,cv2,numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;run=P/(sys.argv[1] if len(sys.argv)>1 else 'sweep')
t=np.load(run/'trajectory.npz');z=np.load(run/'initial_state.npz');cases=json.loads((run/'cases.json').read_text());m=json.loads((run/'metadata.json').read_text());native=np.load(run/'native_wrist_before_pairing.npy')
np.testing.assert_allclose(z['object_mass'],.17,atol=1e-7);np.testing.assert_allclose(z['object_friction'],2.2,atol=1e-6);np.testing.assert_allclose(z['hand_friction'],2.2,atol=1e-6)
np.testing.assert_array_equal(z['wrist'],native)
results=[]
for i,c in enumerate(cases):
 failure=np.flatnonzero(t['failure'][:,i]);first=int(failure[0]) if len(failure) else None;settle_failed=bool(t['failure'][:60,i].any());vertical=t['vertical_error_deg'][:,i];q=t['relative_quaternion'][:,i];axis=R.from_quat(q).apply([0,1,0]);change=np.degrees(np.arccos(np.clip(axis@axis[59],-1,1)));initial=float(vertical[59]);is_horizontal=60<=initial<=120
 held_no_failure=first is None;upright=held_no_failure and bool((vertical[-15:]<=20).all());task_success=upright and is_horizontal
 results.append(dict(**c,static_pass=not settle_failed,initial_vertical_deg=initial,initial_horizontal_30deg=is_horizontal,reference_end_vertical_deg=float(vertical[149]),hold_end_vertical_deg=float(vertical[-1]),relative_axis_change_deg=float(change[149]),native_failure=first is not None,first_failure_reference_s=None if first is None else (first-59)/30,held_no_failure=held_no_failure,end_upright_20deg=upright,horizontal_to_vertical_success=task_success,reference_end_tip_contacts=int((t['contact_force_norm'][149,i]>.05).sum())))
summary=dict(cases=len(cases),poses=63,static_pass=sum(r['static_pass'] for r in results),no_failure_through_hold=sum(r['held_no_failure'] for r in results),static_pass_horizontal=sum(r['static_pass'] and r['initial_horizontal_30deg'] for r in results),end_upright=sum(r['end_upright_20deg'] for r in results),horizontal_to_vertical_success=sum(r['horizontal_to_vertical_success'] for r in results),results=results)
(run/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))
# Select representative follow-up views transparently: best completed angle per
# anchor among static passes, plus one failed case. This is review, not rate estimation.
selected=[]
for a in range(3):
 pool=[r for r in results if r['anchor']==a and r['static_pass']]
 if pool:selected.append(min(pool,key=lambda r:(r['native_failure'],r['hold_end_vertical_deg']))['case_id'])
(run/'review_selection.json').write_text(json.dumps(dict(cases=selected,basis='posthoc best retained vertical angle per anchor; all189 results remain reported'),indent=2))
video_checks={}
for f in run.glob('case*_two_views.mp4'):
 cap=cv2.VideoCapture(str(f));n=0
 while True:
  ok,im=cap.read()
  if not ok:break
  n+=1
 cap.release();assert n==180,(f,n);video_checks[f.name]=n
(run/'verification.json').write_text(json.dumps(dict(mass_kg=.17,hand_object_friction=2.2,native_wrist_each_environment_exact=True,remaining_nuisance_native=True,video_decoded_frames=video_checks),indent=2))
fig,axs=plt.subplots(1,3,figsize=(14,4),sharey=True)
for a,ax in enumerate(axs):
 rows=[r for r in results if r['anchor']==a];x=[r['pose_id']%21 for r in rows];y=[r['hold_end_vertical_deg'] if not r['native_failure'] else np.nan for r in rows]
 ax.scatter(x,y,c=[r['noise_seed'] for r in rows],cmap='viridis',s=30);ax.axhspan(0,20,color='green',alpha=.1);ax.set_title(f'Anchor {a}: completed cases');ax.set_xlabel('Nearby pose index');ax.set_ylim(0,180);ax.grid(alpha=.2)
axs[0].set_ylabel('Final angle from screw-down vertical (deg)');fig.suptitle('Native wrist | bulb170g | hand/object friction2.2 | failed cases omitted from angle scatter');fig.tight_layout();fig.savefig(run/'outcomes.png',dpi=150)
