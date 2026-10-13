"""Author paired cyclic targets: common grasp schedule +/- one rotation term."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE/'source/eval'))
sys.path.insert(0, str(ROOT/'docs/experiment_reviews/20260924_object_state_data'))
from astra_kinematic_step import point_jacobian
from reference_resampling import interpolate_large_jumps

def main():
    old = HERE.parent/'20260928_astra_reference_edit_170g_mu22/runs/right_direct'
    archive = ROOT/'.worktrees/Astra-controller/outputs/astra_halfturn/seed42_10b_right_gait_s100_v2'
    request = json.loads((archive/'request_0001.json').read_text())
    rows = [json.loads(x) for x in (old/'trajectory.jsonl').read_text().splitlines()]
    state = rows[7]['state']  # Actual synchronized 170 g / mu2.2 frame, same initial hold.
    names = request['joint_names']
    q0 = np.asarray(request['state']['target_before'])
    lo, hi = np.asarray(request['joint_lower']), np.asarray(request['joint_upper'])
    axis = np.asarray(json.loads((old/'manifest.json').read_text())['axis_world'])
    center = np.asarray(state['object_position'])
    fingers = ['middle', 'index', 'ring', 'thumb', 'pinky']
    release = {}
    for finger in fingers:
        target = q0.copy()
        joints = [('MCP_FE', .06), ('PIP', .30), ('DIP', .225)] if finger != 'thumb' else [('MCP_FE', .06), ('IP', .225)]
        for joint, amount in joints:
            i = names.index('right_'+finger+'_'+joint)
            target[i] = max(lo[i], q0[i]-amount)
        release[finger] = target-q0
    common_extremes = np.array([q0]+[q0+x for x in release.values()])
    margin = np.minimum(np.min(common_extremes-lo, axis=0), np.min(hi-common_extremes, axis=0))
    margin = np.maximum(0., np.minimum(margin, .15))
    rotation = np.zeros(22)
    diagnostics = []
    for finger in fingers:
        body = 'right_'+finger+'_elastomer'
        tip = body.replace('_elastomer', '_fingertip')
        pad = (np.array(state['body_pose_world'][body][:3])+np.array(state['body_pose_world'][tip][:3]))/2
        point, jac = point_jacobian(state, names, body, pad)
        desired = np.deg2rad(15.)*np.cross(axis, point-center)
        active = np.flatnonzero((np.linalg.norm(jac, axis=0)>1e-10) & (margin>1e-7))
        solution = lsq_linear(np.vstack([jac[:,active], .008*np.eye(len(active))]),
            np.r_[desired, np.zeros(len(active))], bounds=(-margin[active], margin[active]))
        rotation[active] = solution.x
        diagnostics.append(dict(finger=finger,desired_tangent_m=desired.tolist(),estimated_tangent_m=(jac@rotation).tolist()))
    # Eight actual initial hold steps, then a common-length signed push.
    common, signed, labels = [], [], []
    def append(c, s, label):
        common.append(c.copy()); signed.append(s.copy()); labels.append(label)
    for _ in range(8): append(q0, np.zeros(22), 'initial_sync')
    for k in range(1,17): append(q0, rotation*k/16, 'initial_push')
    cycle_start = len(common)
    current_c, current_s = q0.copy(), rotation.copy()
    def ramp(c, s, count, label):
        nonlocal current_c, current_s
        for k in range(1,count+1): append(current_c+(c-current_c)*k/count, current_s+(s-current_s)*k/count, label)
        current_c, current_s = c.copy(), s.copy()
    for finger in fingers:
        ids = [i for i,n in enumerate(names) if n.startswith('right_'+finger+'_')]
        reset = rotation.copy(); reset[ids] *= -1
        ramp(q0+release[finger], rotation, 8, finger+':unload')
        ramp(q0+release[finger], reset, 8, finger+':reset')
        ramp(q0, reset, 8, finger+':reclose')
        ramp(q0, rotation, 16, finger+':push')
    cycle_end = len(common)
    cycle_common = np.array(common[cycle_start:]); cycle_signed = np.array(signed[cycle_start:])
    cycle_labels = labels[cycle_start:].copy()
    assert np.array_equal(current_c,q0) and np.array_equal(current_s,rotation)
    for cycle in range(1,5):
        common.extend(cycle_common.copy()); signed.extend(cycle_signed.copy()); labels.extend(cycle_labels)
    common, signed = np.array(common), np.array(signed)
    expanded = {}
    for direction, sign in [('left',1),('right',-1)]:
        raw = (common+sign*signed).astype(np.float32)
        assert (raw>=lo-1e-6).all() and (raw<=hi+1e-6).all()
        full, progress = interpolate_large_jumps(raw[None], .09)
        actions = full[0]
        assert len(actions)>=610
        expanded[direction] = (actions,progress)
        np.savez_compressed(HERE/(direction+'_reference.npz'), actions=actions,
            raw_actions=raw, source_progress=progress, common=common, signed_rotation=signed,
            raw_phases=np.array(labels), joint_names=np.array(names))
    left, lp = expanded['left']; right, rp = expanded['right']
    np.testing.assert_array_equal(lp,rp)
    # Independent interpolation of the center must retain the exact paired phase.
    indices = lp-1; a=np.floor(indices).astype(int); b=np.minimum(a+1,len(common)-1); f=(indices-a)[:,None]
    center_expanded = common[a]*(1-f)+common[b]*f
    symmetry_error=float(np.max(np.abs((left.astype(float)+right.astype(float))/2-center_expanded)))
    assert symmetry_error<1e-6
    config=dict(definition='left=common_grasp+rotation; right=common_grasp-rotation; identical unload/reset/reclose/push phases',
        construction='New common cyclic template using the archived Astra Jacobian and finger-gait conventions, not the two asymmetric archived action streams.',
        anchor=str(old), synchronized_anchor_step=8, fingers=fingers, initial_sync_steps=8, initial_push_steps=16,
        raw_cycle_steps=cycle_end-cycle_start, phases=dict(unload=8,reset=8,reclose=8,push=16),
        requested_tangent_degrees=15, max_symmetric_rotation_joint_offset_rad=.15,
        interpolation_threshold_rad=.09, interpolation='One midpoint where any joint difference exceeds threshold; no recursive subdivision.',
        inserted_steps=len(lp)-len(common), inserted_in_first600=int(np.sum(lp[:600]%1!=0)),
        paired_progress_exact=True, max_center_symmetry_error_rad=symmetry_error,
        raw_max_jump_rad=float(max(np.abs(np.diff((common+signed).astype(np.float32).astype(float),axis=0)).max(),np.abs(np.diff((common-signed).astype(np.float32).astype(float),axis=0)).max())),
        control_steps=600, ddim_steps=6, execution_steps=2, noise_ratios=[.1,.2,.35],
        mass_kg=.17,object_friction=2.2,diagnostics=diagnostics,
        outcome_note='Opposite symmetric reference intent does not guarantee symmetric physical motion or continued grasp.')
    (HERE/'reference_config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({k:v for k,v in config.items() if k!='diagnostics'},indent=2))

if __name__=='__main__': main()
