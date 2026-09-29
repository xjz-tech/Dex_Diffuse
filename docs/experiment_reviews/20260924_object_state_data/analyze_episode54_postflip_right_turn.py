"""Audit post-flip right-turn intent and physical rotation for edit015, ep54, 170g/mu2.0."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from reference_resampling import interpolate_large_jumps


P = Path(__file__).resolve().parent
R = P / 'reference_turn_baseline_20260926'
CASE = R / 'qualified_comparison/episode_54'
FOLDER = R / 'm170_mu20_guidance4_vs_edit015_20260928/episode_54/edit015'
OUT = R / 'm170_mu20_episode54_postflip_right_turn_20260928'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trace = json.loads((FOLDER/'trace.json').read_text())
    summary = json.loads((FOLDER/'summary.json').read_text())
    analysis = json.loads((FOLDER/'analysis.json').read_text())
    geometry = json.loads((FOLDER/'grasp_metrics.json').read_text())['frames']
    progress = np.load(FOLDER/'reference_progress.npy')
    with np.load(CASE/'reference_full.npz') as z:
        reference_actions, _ = interpolate_large_jumps(z['hand_target_rad'], .1)
        reference_actions = reference_actions[0]
        reference_pose = z['object_pose_wrist'][0]

    assert summary['mass_kg'] == .17 and summary['friction'] == 2.0
    assert summary['prior']['editor']['requested_noise_ratio'] == .15
    assert summary['prior']['editor']['timesteps'] == [8, 5, 3, 0]
    assert summary['execution_steps'] == 2
    assert summary['reference_interpolation_threshold'] == .1
    assert analysis['first_separation_actual_action_step'] == 190

    separation_step = analysis['first_separation_actual_action_step']
    separation_trace = 60 + separation_step - 1
    valid = []
    for i, (row, geo) in enumerate(zip(trace, geometry)):
        force = float(np.linalg.norm(row['object_contact_force']))
        valid.append(i < separation_trace and row['phase'] == 'action' and
                     row['vertical_error_deg'] <= 30 and
                     geo['mesh_table_clearance_m'] > .08 and
                     geo['near_contact_link_count'] >= 2 and
                     geo['mesh_vertex_gap_m'] < .008 and force > .1)
    runs = []
    start = None
    for i, flag in enumerate(valid + [False]):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    stable_start_trace, stable_end_trace = max(runs, key=lambda x: x[1]-x[0]+1)
    stable_start = trace[stable_start_trace]['index']
    stable_end = trace[stable_end_trace]['index']
    assert stable_end - stable_start + 1 == 31

    last_contact = max(row['index'] for i, (row, geo) in enumerate(zip(trace, geometry))
        if i < separation_trace and row['phase'] == 'action' and
        geo['near_contact_link_count'] >= 2 and geo['mesh_vertex_gap_m'] < .008 and
        np.linalg.norm(row['object_contact_force']) > .1)

    start_ref = int(np.floor(progress[stable_start]))
    end_ref = int(np.floor(progress[last_contact]))
    reference_delta = (Rotation.from_matrix(reference_pose[end_ref, :3, :3]) *
                       Rotation.from_matrix(reference_pose[start_ref, :3, :3]).inv()).as_rotvec()
    reference_rotation_deg = float(np.degrees(np.linalg.norm(reference_delta)))
    axis_wrist = reference_delta / np.linalg.norm(reference_delta)
    axis_world = Rotation.from_quat(summary['initial_wrist'][3:]).apply(axis_wrist)

    physical_rows = trace[60+stable_start:60+last_contact+1]
    physical_rot = Rotation.from_quat(np.asarray([x['object_pose'][3:] for x in physical_rows]))
    physical_increment_deg = np.degrees(
        (physical_rot[1:] * physical_rot[:-1].inv()).as_rotvec() @ axis_world)
    physical_cumulative_deg = np.r_[0., np.cumsum(physical_increment_deg)]
    min_index = int(np.argmin(physical_cumulative_deg))
    right_onset = next(i for i in range(len(physical_cumulative_deg))
        if physical_cumulative_deg[i] > 0 and np.all(physical_cumulative_deg[i:] > 0))

    commands = np.asarray([x['command'] for x in trace if x['phase'] == 'action'])
    block_cosines = []
    for j in range(stable_start, last_contact-1, 2):
        k = min(j+2, last_contact)
        issued = commands[k] - commands[j]
        wanted = reference_actions[k] - reference_actions[j]
        denom = np.linalg.norm(issued) * np.linalg.norm(wanted)
        if denom > 1e-12:
            block_cosines.append(float(issued @ wanted / denom))
    assert block_cosines and all(x > 0 for x in block_cosines)

    reference_curve = []
    reference_start_rot = Rotation.from_matrix(reference_pose[start_ref, :3, :3])
    for j in range(stable_start, last_contact+1):
        rid = int(np.floor(progress[j]))
        delta = (Rotation.from_matrix(reference_pose[rid, :3, :3]) *
                 reference_start_rot.inv()).as_rotvec()
        reference_curve.append(float(np.degrees(delta @ axis_wrist)))
    reference_curve = np.asarray(reference_curve)

    result = dict(
        episode=54, mass_kg=.17, friction=2.0,
        method='edit015_ddim4_exec2',
        conditioning='previous 3 issued actions plus future 9 interpolated reference actions',
        stable_vertical_start_action_step=stable_start+1,
        stable_vertical_start_reference_progress=float(progress[stable_start]),
        stable_vertical_block_steps=stable_end-stable_start+1,
        last_effective_contact_action_step=last_contact+1,
        last_effective_contact_reference_progress=float(progress[last_contact]),
        first_geometric_separation_action_step=separation_step,
        first_geometric_separation_reference_progress=analysis['first_separation_reference_progress'],
        postflip_effective_contact_intervals=last_contact-stable_start,
        postflip_effective_contact_seconds=(last_contact-stable_start)/30,
        issued_two_step_blocks=len(block_cosines),
        reference_aligned_two_step_blocks=sum(x > 0 for x in block_cosines),
        command_direction_cosine_mean=float(np.mean(block_cosines)),
        command_direction_cosine_median=float(np.median(block_cosines)),
        initial_physical_drift_min_deg=float(physical_cumulative_deg[min_index]),
        initial_physical_drift_min_step=min_index,
        persistent_net_right_onset_step_after_flip=right_onset,
        persistent_net_right_onset_action_step=stable_start+1+right_onset,
        persistent_net_right_duration_steps=last_contact-(stable_start+right_onset),
        persistent_net_right_duration_seconds=(last_contact-(stable_start+right_onset))/30,
        physical_signed_rotation_before_contact_loss_deg=float(physical_cumulative_deg[-1]),
        reference_signed_rotation_same_progress_deg=float(reference_curve[-1]),
        reference_full_rotation_magnitude_same_progress_deg=reference_rotation_deg,
        rotation_axis_wrist=axis_wrist.tolist(), rotation_axis_world=axis_world.tolist(),
        direction_definition=('positive axis is the recorded reference object rotation from '
                              'post-flip progress 80 to last-contact progress 143'),
        validation=dict(native_protocol_unchanged=True, existing_verified_rollout=True,
                        actions_are_replanned_every_two_steps=True,
                        history_actions_are_previous_three_issued_targets=True,
                        future_reference_window_steps=9))
    (OUT/'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')

    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    x = np.arange(len(physical_cumulative_deg)) / 30
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(x, physical_cumulative_deg, label='仿真灯泡：累计右转角', linewidth=2)
    ax.plot(x, reference_curve, label='数据集 reference：同进度右转角', linewidth=2)
    ax.axvline(right_onset/30, color='0.45', linestyle='--', linewidth=1,
               label=f'净右转持续起点 {right_onset/30:.2f}s')
    ax.axvline((last_contact-stable_start)/30, color='0.2', linestyle=':', linewidth=1,
               label='最后有效接触')
    ax.axhline(0, color='0.75', linewidth=1)
    ax.set_xlabel('完成翻转后的时间（秒）')
    ax.set_ylabel('沿 reference 右转方向的累计角度（度）')
    ax.set_title('Episode 54 · 170g / 摩擦2.0 · edit0.15')
    ax.grid(alpha=.2)
    ax.legend(loc='upper left')
    fig.savefig(OUT/'postflip_right_turn.png', dpi=180)
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
