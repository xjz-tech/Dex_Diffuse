"""Aggregate paired, early-terminal 0–79 episode rollouts."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from audit_paired_early import audit_folder


P = Path(__file__).resolve().parent / 'corrected_direct_vs_reference/all_full_episodes'
METHODS = [('direct', 'Direct recorded actions'),
           ('guide2_exec2', '10B guide2/exec2 scale50'),
           ('guide2_exec1', '10B guide2/exec1 scale50')]


def expanded_reference(reference, inserted):
    if not inserted:
        return reference
    output = np.empty((2*len(reference)-1, reference.shape[1]), dtype=reference.dtype)
    output[::2] = reference
    output[1::2] = (reference[:-1] + reference[1:]) / 2
    return output


def result_for_episode(episode, selected, method, summary, data, position):
    original_actions = int(summary['original_action_lengths'][position])
    length = int(summary['action_lengths'][position])
    inserted = int(summary['reference_interpolation'])
    assert original_actions == selected['actions']
    assert length == (original_actions - 1)*(inserted+1)+1
    targets = np.load(P / f'episode_{episode:02d}/reference_full.npz')['hand_target_rad'][0]
    targets = expanded_reference(targets, inserted)
    assert targets.shape == (length, 22)
    static_steps = int(summary['steps']['settle'])
    total = data['native_failure'].shape[0]
    assert total == sum(summary['steps'].values())
    horizon_end = static_steps + length + 60
    relevant_end = min(total, horizon_end)
    relevant_failure = data['native_failure'][:relevant_end, position]
    first = int(np.flatnonzero(relevant_failure)[0]) if relevant_failure.any() else None
    if first is None:
        phase, index = None, None
    elif first < static_steps:
        phase, index = 'settle', first
    elif first < static_steps + length:
        phase, index = 'action', first-static_steps
    else:
        phase, index = 'hold', first-static_steps-length
    static_no_failure = phase != 'settle'
    action_no_failure = phase not in ('settle', 'action')
    full_no_failure = first is None and total >= horizon_end
    if first is None and total < horizon_end:
        raise AssertionError((episode, method, 'trajectory ended without failure', total, horizon_end))
    action_available = min(length, max(0, total-static_steps))
    commands = data['command'][static_steps:static_steps+action_available, position]
    joints = data['q'][static_steps:static_steps+action_available, position]
    pre_count = 0 if phase == 'settle' else (index if phase == 'action' else action_available)
    pre_count = min(pre_count, action_available)
    if pre_count:
        command_error = float(np.sqrt(np.mean((commands[:pre_count]-targets[:pre_count])**2)))
        joint_error = float(np.sqrt(np.mean((joints[:pre_count]-targets[:pre_count])**2)))
        angle = data['vertical_deg'][static_steps:static_steps+pre_count, position]
        axis_tilt = np.minimum(angle, 180-angle)
        best_vertical = float(np.min(axis_tilt))
        good = (axis_tilt <= 30).astype(int)
        sustained = bool(len(good) >= 15 and np.convolve(good, np.ones(15, int), 'valid').max() == 15)
        good45 = (axis_tilt <= 45).astype(int)
        sustained45 = bool(len(good45) >= 15 and
                           np.convolve(good45, np.ones(15, int), 'valid').max() == 15)
        adjacent_jump = (float(np.mean(np.abs(np.diff(commands[:pre_count], axis=0))))
                         if pre_count > 1 else None)
    else:
        command_error = joint_error = best_vertical = adjacent_jump = None
        sustained = sustained45 = False
    if phase == 'settle':
        progress = 0.
    elif phase == 'action':
        progress = float(index/length)
    else:
        progress = 1.
    failure_source_frame = (selected['source_start_frame'] + index/(inserted+1)
                            if phase == 'action' else None)
    failure_control_time_s = (index/30 if phase == 'action' else
                              (length+index)/30 if phase == 'hold' else None)
    local = np.asarray(summary['initial_object_pose_wrist'][position])
    wrist = np.asarray(summary['initial_wrist'][position])
    imported_rotation = Rotation.from_quat(wrist[3:]) * Rotation.from_matrix(local[:3, :3])
    first_pose = data['object_pose'][0, position]
    first_position_drift = float(np.linalg.norm(data['relative_position'][0, position] - local[:3, 3]))
    first_rotation_drift = float(np.degrees((imported_rotation.inv() *
        Rotation.from_quat(first_pose[3:7])).magnitude()))
    last_static = min(static_steps, 60)-1
    settled_pose = data['object_pose'][last_static, position]
    settled_angle = float(data['vertical_deg'][last_static, position])
    settle_position_drift = float(np.linalg.norm(data['relative_position'][last_static, position] - local[:3, 3]))
    settle_rotation_drift = float(np.degrees((imported_rotation.inv() *
        Rotation.from_quat(settled_pose[3:7])).magnitude()))
    calibration_gate = bool(static_steps == 60 and static_no_failure and
                            first_position_drift < .005 and settle_position_drift < .01 and
                            settle_rotation_drift < 10)
    action_end_index = static_steps+length-1
    hold_end_index = static_steps+length+59
    return dict(episode=episode, method=method,
        lifted_start=bool(selected['lifted_start']),
        source_start_frame=selected['source_start_frame'],
        source_end_state_frame=selected['source_end_state_frame'],
        original_actions=original_actions, intended_executed_actions=length,
        actual_action_steps_in_batch=int(summary['steps']['global_action']),
        start_clearance_cm=float(selected['estimated_mesh_clearance_m']*100),
        start_real_vertical_angle_deg=float(selected['start_world_vertical_angle_deg']),
        real_wrist_displacement_cm=float(selected['real_wrist_displacement_m']*100),
        first_failure_phase=phase, first_failure_local_step=index,
        first_action_failure_source_frame=failure_source_frame,
        first_native_failure_control_time_s=failure_control_time_s,
        source_progress_before_failure=progress,
        static_no_failure=static_no_failure,
        action_no_failure=action_no_failure,
        full_no_failure=full_no_failure,
        first_physics_step_displacement_cm=first_position_drift*100,
        first_physics_step_rotation_deg=first_rotation_drift,
        settle_displacement_cm=settle_position_drift*100,
        settle_rotation_deg=settle_rotation_drift,
        settle_end_vertical_angle_deg=settled_angle,
        settle_still_horizontal=bool(60 <= settled_angle <= 120),
        calibration_gate=calibration_gate,
        best_prefailure_nearest_vertical_angle_deg=best_vertical,
        sustained_30deg_for_15_steps_prefailure=sustained,
        sustained_45deg_for_15_steps_prefailure=sustained45,
        horizontal_to_30deg_prefailure=bool(static_no_failure and
                                           60 <= settled_angle <= 120 and sustained),
        horizontal_to_45deg_prefailure=bool(static_no_failure and
                                           60 <= settled_angle <= 120 and sustained45),
        action_end_vertical_angle_deg=(float(data['vertical_deg'][action_end_index, position])
                                       if action_end_index < total else None),
        hold_end_vertical_angle_deg=(float(data['vertical_deg'][hold_end_index, position])
                                     if hold_end_index < total else None),
        command_vs_reference_rmse_prefailure_rad=command_error,
        joint_vs_reference_rmse_prefailure_rad=joint_error,
        mean_adjacent_command_jump_prefailure_rad=adjacent_jump)


def main():
    selection = json.loads((P / 'selection.json').read_text())['episodes']
    assert len(selection) == 80
    audit = json.loads((P / 'pairing_audit.json').read_text())
    assert audit['completed'] == 80, 'Finish all 40 pairs before aggregating'
    affected = set(audit['affected'])
    rows_by_method = {method: {} for method, _ in METHODS}
    chosen_folders = {}
    for episode in range(80):
        folder = (P / f'singleton_{episode:02d}' if episode in affected else
                  P / f'pair_{episode//2*2:02d}_{episode//2*2+1:02d}')
        if episode in affected:
            check = audit_folder(folder)
            assert len(check) == 1 and check[0]['initial_exact'] and check[0]['static_exact'], episode
        chosen_folders[episode] = str(folder)
        for method, _ in METHODS:
            summary = json.loads((folder / method / 'summary.json').read_text())
            position = summary['source_episode_ids'].index(episode)
            data = np.load(folder / method / 'trajectory.npz')
            rows_by_method[method][episode] = result_for_episode(
                episode, selection[episode], method, summary, data, position)
    rows = [rows_by_method[method][episode] for method, _ in METHODS for episode in range(80)]
    with (P / 'paired_per_episode_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (P / 'paired_per_episode_results.json').write_text(json.dumps(rows, indent=2) + '\n')

    def summarize(method, episodes):
        group = [rows_by_method[method][episode] for episode in episodes]
        def median(field):
            valid = [r[field] for r in group if r[field] is not None]
            return float(np.median(valid)) if valid else None
        return dict(n=len(group),
            static_no_failure=sum(r['static_no_failure'] for r in group),
            calibration_gate=sum(r['calibration_gate'] for r in group),
            action_no_failure=sum(r['action_no_failure'] for r in group),
            full_no_failure=sum(r['full_no_failure'] for r in group),
            reached_25pct_without_failure=sum(r['source_progress_before_failure'] >= .25 for r in group),
            reached_50pct_without_failure=sum(r['source_progress_before_failure'] >= .50 for r in group),
            reached_75pct_without_failure=sum(r['source_progress_before_failure'] >= .75 for r in group),
            sustained_30deg_prefailure=sum(r['sustained_30deg_for_15_steps_prefailure'] for r in group),
            median_failure_progress=float(np.median([
                r['source_progress_before_failure'] for r in group if r['first_failure_phase'] is not None]))
            if any(r['first_failure_phase'] is not None for r in group) else None,
            median_prefailure_command_rmse_rad=median('command_vs_reference_rmse_prefailure_rad'),
            median_prefailure_joint_rmse_rad=median('joint_vs_reference_rmse_prefailure_rad'),
            median_first_step_displacement_cm=median('first_physics_step_displacement_cm'),
            median_settle_displacement_cm=median('settle_displacement_cm'))

    groups = {'all80': list(range(80)),
              'lifted43': [e for e in range(80) if selection[e]['lifted_start']],
              'fallback37': [e for e in range(80) if not selection[e]['lifted_start']]}
    assert len(groups['lifted43']) == 43 and len(groups['fallback37']) == 37
    aggregate = {label: {method: summarize(method, episodes) for method, _ in METHODS}
                 for label, episodes in groups.items()}
    pairwise = {}
    for label, episodes in groups.items():
        direct = rows_by_method['direct']
        e2 = rows_by_method['guide2_exec2']
        e1 = rows_by_method['guide2_exec1']
        pairwise[label] = dict(
            exec1_later_than_exec2=sum(e1[e]['source_progress_before_failure'] >
                                        e2[e]['source_progress_before_failure'] for e in episodes),
            exec2_later_than_exec1=sum(e2[e]['source_progress_before_failure'] >
                                        e1[e]['source_progress_before_failure'] for e in episodes),
            exec1_exec2_tie=sum(e1[e]['source_progress_before_failure'] ==
                                 e2[e]['source_progress_before_failure'] for e in episodes),
            exec1_later_than_direct=sum(e1[e]['source_progress_before_failure'] >
                                        direct[e]['source_progress_before_failure'] for e in episodes),
            exec2_later_than_direct=sum(e2[e]['source_progress_before_failure'] >
                                        direct[e]['source_progress_before_failure'] for e in episodes))
    report = dict(dataset_episodes=80, strict_lifted_starts=43, fallback_starts=37,
        paired_batch_exact=80-len(affected), singleton_reruns=len(affected),
        all_final_comparisons_initial_and_settle_paired=True,
        protocol='native eval/xjz_test.sh thresholds; 170 g; friction 2.2; DDIM4; seed44 fixed noise; guide2 scale50, insert1; exec2/exec1; fixed native wrist; stop once every environment in a pair has native failure',
        groups=aggregate, pairwise=pairwise, selected_folders=chosen_folders)
    (P / 'paired_aggregate_results.json').write_text(json.dumps(report, indent=2) + '\n')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = {'direct': '#5b6770', 'guide2_exec2': '#e0842c', 'guide2_exec1': '#2371b4'}
    grid = np.linspace(0, 1, 101)
    for ax, (label, episodes) in zip(axes, groups.items()):
        for method, title in METHODS:
            progress = np.asarray([rows_by_method[method][e]['source_progress_before_failure']
                                   for e in episodes])
            ax.step(grid, [(progress >= p).mean() for p in grid], where='post',
                    label=title, color=colors[method], linewidth=2)
        ax.set_title(f'{label} (n={len(episodes)})')
        ax.set_xlabel('Fraction of source actions reached')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.25)
    axes[0].set_ylabel('No native failure so far')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(P / 'paired_failure_free_progress.png', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(groups=aggregate, pairwise=pairwise,
                          singleton_reruns=len(affected)), indent=2))


if __name__ == '__main__':
    main()
