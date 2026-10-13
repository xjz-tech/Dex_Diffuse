"""Report paired full-tail native-failure and tracking statistics for all episodes."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


P = Path(__file__).resolve().parent / 'corrected_direct_vs_reference/all_full_episodes'
RUNS = [('direct', 'Direct recorded actions'),
        ('guide2_exec2', '10B guide2/exec2 scale50'),
        ('guide2_exec1', '10B guide2/exec1 scale50')]


def analyze():
    selection = json.loads((P / 'selection.json').read_text())['episodes']
    assert len(selection) == 80
    results = {}
    initial = []
    trajectory = []
    for name, _ in RUNS:
        folder = P / 'all80' / name
        summary = json.loads((folder / 'summary.json').read_text())
        assert summary['source_episode_ids'] == list(range(80))
        data = np.load(folder / 'trajectory.npz')
        assert data['native_failure'].shape[1] == 80
        initial.append(np.load(folder / 'initial_state.npz'))
        trajectory.append(data)
        results[name] = dict(summary=summary, data=data, rows=[])
    paired_fields = [k for k in initial[0].files if all(k in run.files for run in initial)]
    exact_fields = [k for k in paired_fields if all(
        initial[0][k].dtype.kind in 'ifub' and initial[0][k].shape == run[k].shape
        for run in initial)]
    assert len(exact_fields) == 27
    assert all(np.array_equal(initial[0][k], run[k])
               for run in initial[1:] for k in exact_fields)
    static_keys = ['q', 'command', 'object_pose', 'relative_position',
                   'vertical_deg', 'native_failure']
    assert all(np.array_equal(trajectory[0][k][:60], run[k][:60])
               for run in trajectory[1:] for k in static_keys)

    for name, _ in RUNS:
        run = results[name]
        summary, data = run['summary'], run['data']
        interpolation = summary['reference_interpolation']
        for e, selected in enumerate(selection):
            length = int(summary['action_lengths'][e])
            original_length = int(summary['original_action_lengths'][e])
            assert length == (original_length - 1) * (interpolation + 1) + 1
            assert original_length == selected['actions']
            action_slice = slice(60, 60 + length)
            hold_slice = slice(60 + length, 120 + length)
            static_failure = data['native_failure'][:60, e]
            action_failure = data['native_failure'][action_slice, e]
            hold_failure = data['native_failure'][hold_slice, e]
            first_static = int(np.flatnonzero(static_failure)[0]) if static_failure.any() else None
            first_action = int(np.flatnonzero(action_failure)[0]) if action_failure.any() else None
            first_hold = int(np.flatnonzero(hold_failure)[0]) if hold_failure.any() else None
            if first_static is not None:
                first_phase, first_step = 'settle', first_static
            elif first_action is not None:
                first_phase, first_step = 'action', first_action
            elif first_hold is not None:
                first_phase, first_step = 'hold', first_hold
            else:
                first_phase, first_step = None, None
            targets = np.load(P / f'episode_{e:02d}/reference_full.npz')['hand_target_rad'][0]
            if interpolation:
                expanded = np.empty((2*len(targets)-1, 22), dtype=targets.dtype)
                expanded[::2] = targets
                expanded[1::2] = (targets[:-1] + targets[1:]) / 2
                targets = expanded
            assert targets.shape == (length, 22)
            commands = data['command'][action_slice, e]
            joints = data['q'][action_slice, e]
            all_command_rmse = float(np.sqrt(np.mean((commands - targets)**2)))
            all_joint_rmse = float(np.sqrt(np.mean((joints - targets)**2)))
            valid_length = 0 if first_static is not None else (first_action if first_action is not None else length)
            valid_commands = commands[:valid_length]
            valid_joints = joints[:valid_length]
            valid_targets = targets[:valid_length]
            pre_command_rmse = (float(np.sqrt(np.mean((valid_commands - valid_targets)**2)))
                                if valid_length else None)
            pre_joint_rmse = (float(np.sqrt(np.mean((valid_joints - valid_targets)**2)))
                              if valid_length else None)
            angles = data['vertical_deg'][action_slice, e]
            valid_angles = angles[:valid_length]
            best_angle = float(valid_angles.min()) if valid_length else None
            sustained_30 = bool(valid_length >= 15 and
                np.convolve((valid_angles <= 30).astype(int), np.ones(15, dtype=int), 'valid').max() == 15)
            end_idx = 59 + length
            hold_end_idx = 119 + length
            wrist = np.asarray(summary['initial_wrist'][e])
            local = np.asarray(summary['initial_object_pose_wrist'][e])
            source_rotation = Rotation.from_quat(wrist[3:]) * Rotation.from_matrix(local[:3, :3])
            first_rotation = Rotation.from_quat(data['object_pose'][0, e, 3:7])
            settle_rotation = Rotation.from_quat(data['object_pose'][59, e, 3:7])
            first_rotation_drift = float(np.degrees((source_rotation.inv() * first_rotation).magnitude()))
            settle_rotation_drift = float(np.degrees((source_rotation.inv() * settle_rotation).magnitude()))
            first_position_drift = float(np.linalg.norm(data['relative_position'][0, e] - local[:3, 3]))
            settle_position_drift = float(np.linalg.norm(data['relative_position'][59, e] - local[:3, 3]))
            calibration_gate = (first_position_drift < .005 and settle_position_drift < .01 and
                                settle_rotation_drift < 10 and first_static is None)
            source_failure_frame = (selected['source_start_frame'] +
                first_action / (interpolation + 1) if first_action is not None else None)
            progress_fraction = (0.0 if first_static is not None else
                float(first_action / length) if first_action is not None else 1.0)
            row = dict(episode=e, method=name, lifted_start=selected['lifted_start'],
                source_start_frame=selected['source_start_frame'],
                source_last_state_frame=selected['source_end_state_frame'],
                original_actions=original_length, executed_actions=length,
                start_clearance_cm=selected['estimated_mesh_clearance_m']*100,
                start_source_angle_deg=selected['start_world_vertical_angle_deg'],
                first_failure_phase=first_phase, first_failure_step=first_step,
                first_action_failure_source_frame=source_failure_frame,
                first_action_failure_progress_fraction=progress_fraction,
                static_no_failure=first_static is None,
                action_no_failure=first_static is None and first_action is None,
                full_no_failure=first_phase is None,
                first_physics_step_displacement_cm=first_position_drift*100,
                first_physics_step_rotation_deg=first_rotation_drift,
                settle_displacement_cm=settle_position_drift*100,
                settle_rotation_deg=settle_rotation_drift,
                calibration_gate=calibration_gate,
                best_prefailure_vertical_angle_deg=best_angle,
                sustained_30deg_for_15_steps_prefailure=sustained_30,
                action_end_vertical_angle_deg=float(data['vertical_deg'][end_idx, e]),
                hold_end_vertical_angle_deg=float(data['vertical_deg'][hold_end_idx, e]),
                action_end_relative_displacement_cm=float(np.linalg.norm(
                    data['relative_position'][end_idx, e] - local[:3, 3])*100),
                command_vs_reference_rmse_rad=all_command_rmse,
                joint_vs_reference_rmse_rad=all_joint_rmse,
                prefailure_command_vs_reference_rmse_rad=pre_command_rmse,
                prefailure_joint_vs_reference_rmse_rad=pre_joint_rmse,
                mean_adjacent_command_jump_rad=float(np.mean(np.abs(np.diff(commands, axis=0))))
                if length > 1 else None)
            run['rows'].append(row)

    rows = [row for name, _ in RUNS for row in results[name]['rows']]
    with (P / 'per_episode_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (P / 'per_episode_results.json').write_text(json.dumps(rows, indent=2) + '\n')
    def group_summary(name, lifted):
        group = [r for r in results[name]['rows'] if lifted is None or r['lifted_start'] == lifted]
        failures = [r['first_action_failure_progress_fraction'] for r in group]
        def median_nullable(field):
            values = [r[field] for r in group if r[field] is not None]
            return float(np.median(values)) if values else None
        return dict(count=len(group), static_no_failure=sum(r['static_no_failure'] for r in group),
            calibration_gate=sum(r['calibration_gate'] for r in group),
            action_no_failure=sum(r['action_no_failure'] for r in group),
            full_no_failure=sum(r['full_no_failure'] for r in group),
            retained_through_25pct=sum(v >= .25 for v in failures),
            retained_through_50pct=sum(v >= .5 for v in failures),
            retained_through_75pct=sum(v >= .75 for v in failures),
            sustained_30deg_prefailure=sum(r['sustained_30deg_for_15_steps_prefailure'] for r in group),
            median_failure_progress_fraction=float(np.median([v for v in failures if v < 1]))
            if any(v < 1 for v in failures) else None,
            median_prefailure_command_rmse_rad=median_nullable('prefailure_command_vs_reference_rmse_rad'),
            median_prefailure_joint_rmse_rad=median_nullable('prefailure_joint_vs_reference_rmse_rad'),
            median_first_physics_step_displacement_cm=median_nullable('first_physics_step_displacement_cm'),
            median_settle_displacement_cm=median_nullable('settle_displacement_cm'))
    groups = {label: {name: group_summary(name, condition) for name, _ in RUNS}
              for label, condition in [('all80', None), ('lifted43', True), ('near_table37', False)]}
    pairwise = {}
    for label, condition in [('all80', None), ('lifted43', True), ('near_table37', False)]:
        subset = [e for e in range(80) if condition is None or selection[e]['lifted_start'] == condition]
        e2 = results['guide2_exec2']['rows']
        e1 = results['guide2_exec1']['rows']
        direct = results['direct']['rows']
        pairwise[label] = dict(
            exec1_later_than_exec2=sum(e1[e]['first_action_failure_progress_fraction'] >
                                        e2[e]['first_action_failure_progress_fraction'] for e in subset),
            exec2_later_than_exec1=sum(e2[e]['first_action_failure_progress_fraction'] >
                                        e1[e]['first_action_failure_progress_fraction'] for e in subset),
            exec1_exec2_tie=sum(e2[e]['first_action_failure_progress_fraction'] ==
                                e1[e]['first_action_failure_progress_fraction'] for e in subset),
            exec1_later_than_direct=sum(e1[e]['first_action_failure_progress_fraction'] >
                                        direct[e]['first_action_failure_progress_fraction'] for e in subset),
            exec2_later_than_direct=sum(e2[e]['first_action_failure_progress_fraction'] >
                                        direct[e]['first_action_failure_progress_fraction'] for e in subset))
    report = dict(episodes=80, lifted_start=sum(r['lifted_start'] for r in selection),
                  near_table_start=sum(not r['lifted_start'] for r in selection),
                  paired_initial_fields=len(exact_fields), paired_initial_exact=True,
                  paired_settle_exact=True, groups=groups, pairwise=pairwise)
    (P / 'aggregate_results.json').write_text(json.dumps(report, indent=2) + '\n')

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = {'direct': '#5b6770', 'guide2_exec2': '#e0842c', 'guide2_exec1': '#2371b4'}
    grid = np.linspace(0, 1, 101)
    for ax, (label, condition) in zip(axes,
                                     [('all80', None), ('lifted43', True), ('near_table37', False)]):
        subset = [e for e in range(80) if condition is None or selection[e]['lifted_start'] == condition]
        for name, title in RUNS:
            fractions = np.array([results[name]['rows'][e]['first_action_failure_progress_fraction']
                                  for e in subset])
            surv = [(fractions >= x).mean() for x in grid]
            ax.plot(grid, surv, label=title, linewidth=2, color=colors[name])
        ax.set_title(f'{label} (n={len(subset)})')
        ax.set_xlabel('Fraction of source actions completed')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.25)
    axes[0].set_ylabel('No native failure so far')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(P / 'failure_free_progress.png', dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    analyze()
