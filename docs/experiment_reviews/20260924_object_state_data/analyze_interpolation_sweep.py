"""Summarize completed episode53 action-interpolation rollouts."""
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

P = Path(__file__).resolve().parent


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=5)
    args = parser.parse_args()
    assert 1 <= args.start <= args.end
    out = P/f'corrected_direct_vs_reference/interpolation_{args.start}_to_{args.end}'
    out.mkdir(exist_ok=True)
    ref = np.load(P/'reference/reference.npz')
    original = ref['hand_target_rad'][1]
    real_axis = Rotation.from_matrix(ref['object_pose_wrist'][1, -1, :3, :3]).apply([0, 1, 0])
    records = []
    prefix = None
    baseline = P/'corrected_direct_vs_reference'
    for inserted in [0, *range(args.start,args.end+1)]:
        for execute in [2, 4]:
            folder = (baseline/'guided9' if execute == 2 else baseline/'guided9_exec4') if inserted == 0 else out/f'insert{inserted}_exec{execute}'
            if not (folder/'summary.json').exists():
                continue
            s = json.loads((folder/'summary.json').read_text())
            t = json.loads((folder/'trace.json').read_text())
            initial = np.load(folder/'initial_state.npz')
            if prefix is None:
                prefix = {k: initial[k] for k in initial.files}
            assert all(np.array_equal(prefix[k], initial[k]) for k in prefix), folder
            settle = [x for x in t if x['phase'] == 'settle']
            if inserted > 0:
                initial_trace = json.loads(((baseline/'guided9')/'trace.json').read_text())
                assert settle == [x for x in initial_trace if x['phase'] == 'settle'], folder
            action = [x for x in t if x['phase'] == 'action']
            from reference_resampling import interpolate_actions
            target = interpolate_actions(ref['hand_target_rad'], inserted)[1]
            command = np.array([x['command'] for x in action])
            q = np.array([x['q'] for x in action])
            assert len(action) == len(target)
            ref_idx = np.arange(0, len(target), inserted+1)
            assert len(ref_idx) == len(original)
            np.testing.assert_array_equal(target[ref_idx], original)
            failure = next((x['index'] for x in action if x['native_failure']), None)
            wrist = Rotation.from_quat(s['initial_wrist'][3:])
            object_end = s['action_end']['object_pose']
            object_axis_wrist = (wrist.inv()*Rotation.from_quat(object_end[3:])).apply([0, 1, 0])
            axis_error = float(np.degrees(np.arccos(np.clip(object_axis_wrist@real_axis, -1, 1))))
            records.append(dict(inserted=inserted, execute=execute,
                action_steps=len(target), action_seconds=len(target)/30,
                model_calls=(len(target)+execute-1)//execute,
                reference_max_adjacent_joint_rad=float(abs(np.diff(target,axis=0)).max()),
                command_max_adjacent_joint_rad=float(abs(np.diff(command,axis=0)).max()),
                command_reference_rmse_rad=rms(command-target),
                q_reference_rmse_rad=rms(q-target),
                q_command_rmse_rad=rms(q-command),
                original_anchor_command_rmse_rad=rms(command[ref_idx]-original),
                original_anchor_q_rmse_rad=rms(q[ref_idx]-original),
                action_end_vertical_deg=s['action_end']['vertical_error_deg'],
                hold_end_vertical_deg=s['hold_end']['vertical_error_deg'],
                action_end_displacement_cm=100*s['action_end']['displacement_from_import_m'],
                hold_end_displacement_cm=100*s['hold_end']['displacement_from_import_m'],
                action_end_axis_error_to_real_deg=axis_error,
                first_native_failure_action_step=failure,
                any_native_failure=s['any_native_failure'],
                initial_state_equal_baseline=True, static_prefix_equal_baseline=True,
                folder=str(folder)))
    (out/'comparison.json').write_text(json.dumps(records, indent=2)+'\n')
    if not records:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    measures = [('command_reference_rmse_rad', 'Command vs interpolated reference RMSE (rad)'),
        ('original_anchor_command_rmse_rad', 'Command vs original actions at anchor steps (rad)'),
        ('action_end_vertical_deg', 'Bulb angle from vertical at action end (deg)'),
        ('action_end_axis_error_to_real_deg', 'Bulb axis error vs real final (deg)')]
    for ax, (key, title) in zip(axes.flat, measures):
        for execute in [2, 4]:
            rr = sorted((r for r in records if r['execute']==execute), key=lambda x:x['inserted'])
            ax.plot([r['inserted'] for r in rr], [r[key] for r in rr], marker='o', label=f'guide9 / exec{execute}')
            for r in rr:
                if r['any_native_failure']:
                    ax.annotate('failure', (r['inserted'],r[key]), xytext=(0,6), textcoords='offset points', ha='center', fontsize=8)
        ax.set_xticks([0,*range(args.start,args.end+1)]);ax.set_xlabel('Interpolated targets inserted per pair');ax.set_title(title)
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160)
    for r in records:
        print(r['inserted'],r['execute'],r['action_steps'],round(r['command_reference_rmse_rad'],4),
              round(r['action_end_vertical_deg'],2),r['first_native_failure_action_step'])


if __name__ == '__main__':
    main()
