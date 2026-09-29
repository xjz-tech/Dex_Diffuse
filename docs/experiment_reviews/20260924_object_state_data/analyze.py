from pathlib import Path
import json,sys,numpy as np
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parent;run=P/(sys.argv[1] if len(sys.argv)>1 else 'sweep');archive=np.load(run/'trajectory.npz');t={k:archive[k] for k in archive.files if k!='applied_forces'};z=np.load(run/'initial_state.npz');cases=json.loads((run/'cases.json').read_text());ref=np.load(P/'reference/reference.npz');results=[]
for name,val in [('object_mass',.17),('object_friction',2.2),('hand_friction',2.2)]:np.testing.assert_allclose(z[name],val,atol=1e-6)
np.testing.assert_array_equal(z['wrist'],np.load(run/'native_wrist_before_pairing.npy'));assert len(t['phase'])==165
for i,c in enumerate(cases):
 ri=c['reference_id'];axis=R.from_quat(t['relative_quaternion'][:,i]).apply([0,1,0]);desired=R.from_matrix(ref['object_pose_wrist'][ri,-1,:3,:3]).apply([0,1,0]);axis_error=np.degrees(np.arccos(np.clip(axis@desired,-1,1)));change=np.degrees(np.arccos(np.clip(axis@axis[59],-1,1)));v=t['vertical_error_deg'][:,i];f=np.flatnonzero(t['failure'][:,i]);first=int(f[0]) if len(f) else None;held=first is None;horizontal=bool(60<=v[59]<=120);no_static_failure=not bool(t['failure'][:60,i].any())
 originalpos=z['object'][i,:3];staticdrift=np.linalg.norm(t['object_pose'][59,i,:3]-originalpos);tracking=np.sqrt(np.mean((t['command'][60:135,i]-ref['hand_target_rad'][ri])**2))
 wrist=t['wrist_pose'][:,i];base=z['wrist'][i,:7];assert np.max(abs(wrist[:,:3]-base[:3]))<1e-5;assert np.max(np.minimum(np.linalg.norm(wrist[:,3:]-base[3:],axis=1),np.linalg.norm(wrist[:,3:]+base[3:],axis=1)))<1e-5
 results.append(dict(**c,static_pass=no_static_failure,static_position_drift_m=float(staticdrift),initial_vertical_deg=float(v[59]),initial_horizontal=horizontal,reference_end_vertical_deg=float(v[134]),hold_end_vertical_deg=float(v[-1]),initial_goal_axis_error_deg=float(axis_error[59]),reference_end_goal_axis_error_deg=float(axis_error[134]),hold_end_goal_axis_error_deg=float(axis_error[-1]),reference_axis_change_deg=float(change[134]),native_failure=not held,first_failure_reference_s=None if first is None else (first-59)/30,first_failure_frame=first,command_reference_rmse_rad=float(tracking),no_failure_through_hold=held,similar_reference_turn=bool(held and (axis_error[-15:]<=30).all() and change[134]>=30),horizontal_to_upright30=bool(held and horizontal and (v[-15:]<=30).all()),horizontal_to_upright20=bool(held and horizontal and (v[-15:]<=20).all())))
summary={}
for ri in range(2):
 rows=[r for r in results if r['reference_id']==ri];summary[str(rows[0]['episode'])]=dict(cases=len(rows),static_pass=sum(r['static_pass'] for r in rows),no_failure_through_hold=sum(r['no_failure_through_hold'] for r in rows),static_horizontal=sum(r['static_pass'] and r['initial_horizontal'] for r in rows),similar_reference_turn=sum(r['similar_reference_turn'] for r in rows),horizontal_to_upright30=sum(r['horizontal_to_upright30'] for r in rows),horizontal_to_upright20=sum(r['horizontal_to_upright20'] for r in rows))
(run/'summary.json').write_text(json.dumps(dict(summary=summary,results=results),indent=2));print(json.dumps(summary,indent=2))
selected=[]
for ri in range(2):
 rows=[r for r in results if r['reference_id']==ri];rows.sort(key=lambda r:(r['native_failure'],not r['initial_horizontal'],r['hold_end_goal_axis_error_deg']))
 selected.extend([r['case_id'] for r in rows[:2]])
(run/'review_selection.json').write_text(json.dumps(dict(cases=selected,basis='posthoc representative perreference: nofailure first, horizontal start next, lower measured reference-axis error; no change to original counts'),indent=2));print('Selected',selected)
print('Retained cases:',json.dumps([r for r in results if r['no_failure_through_hold']],indent=2))
