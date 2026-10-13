"""Verify exact paired interventions, native protocol, references, and video parity."""
import hashlib
import json
from pathlib import Path
import numpy as np
from run_mass_scale_diagnostic import OUT,CASE,P,PROTOCOL

def main():
    checks=[]
    pairs=[('reproduce_m044_mu11_s25',CASE/'guide1_adaptive012_ddim4_scale25_m044_mu11',set())]
    for label,mass,mu in [('m170_mu11',.17,1.1),('m044_mu22',.044,2.2)]:
        baseline=OUT/('direct_'+label)
        for scale in (25,50):pairs.append(('ddim4_'+label+f'_s{scale}',baseline,set()))
        pairs.append(('direct_'+label+'_video',baseline,set()))
        # Relative to the old 170g/2.2 snapshot, only the assigned properties may differ.
        allowed={'object_friction','hand_friction'} if mass==.17 else {'object_mass','cached_object_mass','object_inertia'}
        pairs.append(('direct_'+label,CASE/'direct',allowed))
    for name,other,allowed in pairs:
        folder=OUT/name
        with np.load(folder/'initial_state.npz') as a,np.load(other/'initial_state.npz') as b:
            assert set(a.files)==set(b.files)
            different=[k for k in a.files if not np.array_equal(a[k],b[k])]
            assert set(different)==allowed,(name,different,allowed)
        s=json.loads((folder/'summary.json').read_text())
        assert s['native_protocol']==PROTOCOL and s['steps']==s['intended_steps']
        assert s['source_start_frame']==110 and s['control_hz']==30
        t=json.loads((folder/'trace.json').read_text());u=json.loads((other/'trace.json').read_text())
        same_physics=not allowed
        if same_physics:
            for k in ('q','command','executed_target','object_pose','native_failure','hand_body_pose'):
                assert all(a[k]==b[k] for a,b in zip(t[:60],u[:60])),(name,k)
        if name.endswith('_video'):
            for k in ('q','command','executed_target','object_pose','native_failure','hand_body_pose'):
                assert all(a[k]==b[k] for a,b in zip(t,u)),(name,k)
            import cv2
            cap=cv2.VideoCapture(str(folder/'direct_front.mp4'));count=0
            while cap.read()[0]:count+=1
            cap.release();assert count==len(t)+1,(name,count,len(t))
        checks.append(dict(run=name,compared_to=str(other),initial_differences=different,paired_settle_exact=same_physics,full_video_parity=name.endswith('_video')))
    sources=[P/n for n in ('server.py','compare_corrected_rollouts.py','run_mass_scale_diagnostic.py','run_archived_command_diagnostic.py','replay_archived_command_server.py','analyze_mass_scale_diagnostic.py','analyze_command_replay_diagnostic.py','verify_mass_scale_diagnostic.py')]
    repo=P.parents[2]
    sources += [repo/n for n in ('eval/inference_dp_controller.py','eval/sim_eval.py','eval/policy_observation.py','diffusion_policy/guidance/guided_ddim.py')]
    digest={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    (OUT/'validation.json').write_text(json.dumps(dict(checks=checks,source_sha256=digest,reference_sha256=hashlib.sha256((CASE/'reference_full.npz').read_bytes()).hexdigest()),indent=2)+'\n')
    print('All paired physics, native protocol, and video parity checks passed.')

if __name__=='__main__':main()
