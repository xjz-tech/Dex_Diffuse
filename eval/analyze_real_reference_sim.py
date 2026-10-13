#!/usr/bin/env python3
"""Report object axial rotation relative to the wrist, separate from failures."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

def rotation_metrics(quaternions):
    r=R.from_quat(quaternions)
    # Mesh geometry: +Y points toward the bulb dome, -Y toward the thread.
    # A clockwise turn viewed from the dome is negative body-Y rotation.
    increments=(r[:-1].inv()*r[1:]).as_rotvec()
    right=np.r_[0,np.cumsum(-np.degrees(increments[:,1]))]
    axes=r.apply(np.tile([0,1,0],(len(r),1)))
    tilt=np.degrees(np.arccos(np.clip(axes@axes[0],-1,1)))
    delta=(r[0].inv()*r).as_quat()
    twist=-np.degrees(np.unwrap(2*np.arctan2(delta[:,1],delta[:,3])))
    return right,tilt,twist

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    # Verify sign convention and the exclusion of a perpendicular-axis tilt.
    right,_,_=rotation_metrics(R.from_euler('y',[0,-10,-20],degrees=True).as_quat())
    assert np.allclose(right,[0,10,20])
    right,_,_=rotation_metrics(R.from_euler('x',[0,10,20],degrees=True).as_quat())
    assert np.allclose(right,0)
    run=args.run;t=np.load(run/'trajectory.npz',allow_pickle=False)
    meta=json.loads((run/'run_metadata.json').read_text())
    ref=np.load(meta['reference'],allow_pickle=False)
    idx=np.flatnonzero(t['phase']=='reference');baseline=idx[0]-1
    select=np.arange(baseline,idx[-1]+1)
    results=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for env in range(t['q'].shape[1]):
        right,tilt,twist=rotation_metrics(t['relative_quaternion'][select,env])
        dist=np.linalg.norm(t['relative_position'][select,env]-t['relative_position'][baseline,env],axis=1)
        contacts=(t['contact_force_norm'][idx,env]>.05).sum(1)
        failures=t['failure'][idx,env]
        bad=np.flatnonzero(failures)
        valid_last=(int(bad[0]) if len(bad) else len(idx))
        qrmse=np.sqrt(((t['q'][idx,env]-ref['hand_qpos_rad'][1:])**2).mean(1))
        armse=np.sqrt(((t['command'][idx,env]-ref['hand_target_rad'])**2).mean(1))
        results.append(dict(environment=env,noise_seed=42+env if meta['mode']=='guided' else None,
            native_failure_during_settle=meta['native_failure_during_settle'][env],
            native_failure_during_reference=bool(failures.any()),
            first_native_failure_reference_step=None if not len(bad) else int(bad[0]),
            right_axial_rotation_deg=float(right[-1]),right_axial_rotation_before_first_native_failure_deg=float(right[valid_last]),
            signed_twist_about_initial_axis_deg=float(twist[-1]),
            final_axis_tilt_deg=float(tilt[-1]),max_axis_tilt_deg=float(tilt.max()),
            final_relative_position_drift_m=float(dist[-1]),max_relative_position_drift_m=float(dist.max()),
            contacting_tip_count_median=float(np.median(contacts)),
            fraction_frames_two_or_more_contacting_tips=float((contacts>=2).mean()),
            joint_reference_rmse_rad=float(np.sqrt(((t['q'][idx,env]-ref['hand_qpos_rad'][1:])**2).mean())),
            command_reference_rmse_rad=float(np.sqrt(((t['command'][idx,env]-ref['hand_target_rad'])**2).mean())),
            positive_axial_increment_fraction=float((np.diff(right)>0).mean()),
            note='Positive is right/clockwise. Angles after native failure are not evidence of successful manipulation. Contact counts are diagnostics, not independent drop detection.'))
        seconds=np.arange(len(select))*meta['control_dt_s']
        label='env %d / noise %d'%(env,42+env) if meta['mode']=='guided' else 'env %d / direct'%env
        line,=axes[0].plot(seconds[:valid_last+1],right[:valid_last+1],label=label)
        if len(bad):
            axes[0].plot(seconds[valid_last:],right[valid_last:],linestyle=':',color=line.get_color(),alpha=.6)
            axes[0].axvline(seconds[min(valid_last+1,len(seconds)-1)],color=line.get_color(),ls='--',alpha=.35)
        axes[1].plot(seconds,dist*1000)
        axes[2].plot(seconds[1:],qrmse)
    axes[0].set_ylabel('Right axial rotation (deg)');axes[0].axhline(0,color='black',lw=.7);axes[0].legend()
    axes[1].set_ylabel('Object displacement\nin wrist frame (mm)')
    axes[2].set_ylabel('Joint reference RMSE (rad)');axes[2].set_xlabel('Reference time (s)')
    for axis in axes:axis.grid(alpha=.2)
    fig.suptitle('Real reference: free bulb; dotted rotation curves are after native failure')
    fig.tight_layout();fig.savefig(run/'metrics.png',dpi=150);plt.close(fig)
    summary=dict(protocol=meta,results=results,
        analysis='Body-local Y rotation of object relative to actual wrist; excludes rigid wrist rotation. Native failure flags retained; displacement diagnostics do not redefine environment failure.',
        metric_sanity_checks_passed=True)
    (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
