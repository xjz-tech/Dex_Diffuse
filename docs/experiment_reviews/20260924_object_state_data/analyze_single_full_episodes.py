"""Analyze all 80 complete, non-truncated single-environment rollouts."""

import csv
import json
from pathlib import Path

import numpy as np

from analyze_paired_early_episodes import result_for_episode
from analyze_single_early_episodes import METHODS, P, to_trajectory
from compare_corrected_rollouts import PROTOCOL


def nearest_vertical(angle):
    return min(float(angle), 180.0 - float(angle))


def sustained_within(values, threshold=45.0, count=15):
    good = (np.minimum(values, 180.0 - values) <= threshold).astype(int)
    return bool(len(good) >= count and np.convolve(good, np.ones(count, int), 'valid').max() == count)


def main():
    selections = json.loads((P / 'selection.json').read_text())['episodes']
    assert len(selections) == 80
    methods = [name for name, _ in METHODS]
    rows = []
    for episode in range(80):
        runs = []
        reference = np.load(P / f'episode_{episode:02d}/reference_full.npz')
        real_end_relative = reference['object_pose_wrist'][0, -1, :3, 3]
        real_end_object = reference['object_pose_base'][0, -1]
        real_end_angle = np.degrees(np.arccos(np.clip(real_end_object[2, 1], -1, 1)))
        real_end_hand = reference['hand_qpos_rad'][0, -1]
        for method in methods:
            folder = P / f'episode_{episode:02d}/full/{method}'
            summary = json.loads((folder / 'summary.json').read_text())
            trace = json.loads((folder / 'trace.json').read_text())
            initial = np.load(folder / 'initial_state.npz')
            early_folder = P / f'episode_{episode:02d}/early/{method}'
            early_trace = json.loads((early_folder / 'trace.json').read_text())
            early_initial = np.load(early_folder / 'initial_state.npz')
            assert trace[:len(early_trace)] == early_trace, (episode, method, 'prefix')
            assert all(np.array_equal(initial[key], early_initial[key])
                       for key in initial.files), (episode, method, 'initial')
            assert summary['source_episode'] == episode
            assert summary['source_start_frame'] == selections[episode]['source_start_frame']
            assert np.allclose(summary['initial_object_pose_wrist'],
                               reference['object_pose_wrist'][0, 0], atol=1e-9)
            assert Path(summary['reference']).resolve() == (
                P / f'episode_{episode:02d}/reference_full.npz').resolve()
            assert summary['native_protocol'] == PROTOCOL
            assert summary['mass_kg'] == .17 and summary['friction'] == 2.2
            assert not summary['stop_on_native_failure'] and not summary['video_recorded']
            assert summary['steps'] == summary['intended_steps']
            assert summary['steps']['settle'] == summary['steps']['hold'] == 60
            if method == 'direct':
                assert summary['prior'] is None and summary['reference_interpolation'] == 0
            else:
                prior = summary['prior']
                assert prior['ddim'] == 4 and prior['guidance_steps'] == 2
                assert prior['guidance_scale'] == 50 and prior['reference_interpolation'] == 1
                assert prior['execution_steps'] == int(method[-1])
            runs.append(dict(method=method, summary=summary, trace=trace, initial=initial))
        keys = runs[0]['initial'].files
        assert len(keys) == 27
        assert all(np.array_equal(runs[0]['initial'][key], run['initial'][key])
                   for run in runs[1:] for key in keys), episode
        settled = [[row for row in run['trace'] if row['phase'] == 'settle'] for run in runs]
        assert all(settled[0] == group for group in settled[1:]), episode
        for run in runs:
            method, summary, trace = run['method'], run['summary'], run['trace']
            length = summary['intended_steps']['action']
            synthetic = dict(original_action_lengths=[selections[episode]['actions']],
                             action_lengths=[length],
                             reference_interpolation=summary['reference_interpolation'],
                             steps=dict(settle=60, global_action=length, hold=60),
                             initial_wrist=[summary['initial_wrist']],
                             initial_object_pose_wrist=[summary['initial_object_pose_wrist']])
            result = result_for_episode(episode, selections[episode], method,
                                        synthetic, to_trajectory(trace), 0)
            action = trace[60:60+length]
            hold = trace[60+length:]
            assert len(action) == length and len(hold) == 60
            assert all(row['phase'] == 'action' for row in action)
            assert all(row['phase'] == 'hold' for row in hold)
            start = action[0]
            end = action[-1]
            final = hold[-1]
            angles = np.asarray([row['vertical_error_deg'] for row in action])
            end_relative = np.asarray(end['relative_position'])
            final_relative = np.asarray(final['relative_position'])
            end_hand = np.asarray(end['q'])
            result.update(
                complete_reference_executed=True,
                real_end_nearest_vertical_deg=nearest_vertical(real_end_angle),
                sim_action_end_nearest_vertical_deg=nearest_vertical(end['vertical_error_deg']),
                sim_hold_end_nearest_vertical_deg=nearest_vertical(final['vertical_error_deg']),
                sim_action_end_within_45deg=nearest_vertical(end['vertical_error_deg']) <= 45,
                sim_hold_end_within_45deg=nearest_vertical(final['vertical_error_deg']) <= 45,
                reached_45deg_for_15_action_steps_anytime=sustained_within(angles),
                action_end_bulb_in_wrist_position_error_to_real_cm=float(
                    np.linalg.norm(end_relative - real_end_relative) * 100),
                hold_end_bulb_in_wrist_position_error_to_real_cm=float(
                    np.linalg.norm(final_relative - real_end_relative) * 100),
                action_end_hand_q_rmse_to_real_rad=float(
                    np.sqrt(np.mean((end_hand - real_end_hand)**2))),
                object_motion_during_hold_cm=float(
                    np.linalg.norm(final_relative - end_relative) * 100),
                native_failure_at_action_start=bool(start['native_failure']),
                native_failure_at_action_end=bool(end['native_failure']),
                native_failure_at_hold_end=bool(final['native_failure']),
                native_failure_steps_during_action=sum(row['native_failure'] for row in action),
            )
            rows.append(result)
    with (P / 'full_per_episode_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (P / 'full_per_episode_results.json').write_text(json.dumps(rows, indent=2) + '\n')

    by_method = {method: {int(row['episode']): row for row in rows if row['method'] == method}
                 for method in methods}
    groups = dict(all80=list(range(80)),
                  lifted43=[e for e in range(80) if selections[e]['lifted_start']],
                  fallback37=[e for e in range(80) if not selections[e]['lifted_start']])
    groups['horizontal_after_settle'] = [e for e in range(80)
        if by_method['direct'][e]['static_no_failure'] and
           by_method['direct'][e]['settle_still_horizontal']]
    aggregate = {}
    for label, episodes in groups.items():
        aggregate[label] = {}
        for method in methods:
            group = [by_method[method][e] for e in episodes]
            aggregate[label][method] = dict(
                n=len(group),
                full_reference_executed=sum(row['complete_reference_executed'] for row in group),
                native_failure_before_end=sum(row['first_failure_phase'] is not None for row in group),
                native_failure_during_settle=sum(row['first_failure_phase'] == 'settle' for row in group),
                real_end_within_45deg=sum(row['real_end_nearest_vertical_deg'] <= 45
                                          for row in group),
                reached_45deg_prefailure=sum(row['horizontal_to_45deg_prefailure'] for row in group),
                reached_45deg_anytime=sum(row['reached_45deg_for_15_action_steps_anytime'] for row in group),
                action_end_within_45deg=sum(row['sim_action_end_within_45deg'] for row in group),
                hold_end_within_45deg=sum(row['sim_hold_end_within_45deg'] for row in group),
                action_end_within_10cm_of_real_relative_pose=sum(
                    row['action_end_bulb_in_wrist_position_error_to_real_cm'] <= 10
                    for row in group),
                action_end_vertical_and_within_10cm=sum(
                    row['sim_action_end_within_45deg'] and
                    row['action_end_bulb_in_wrist_position_error_to_real_cm'] <= 10
                    for row in group),
                median_action_end_bulb_in_wrist_error_cm=float(np.median([
                    row['action_end_bulb_in_wrist_position_error_to_real_cm'] for row in group])),
                median_action_end_hand_q_rmse_rad=float(np.median([
                    row['action_end_hand_q_rmse_to_real_rad'] for row in group])),
                median_object_motion_during_hold_cm=float(np.median([
                    row['object_motion_during_hold_cm'] for row in group])),
            )
    report = dict(valid_for_real_task_success_rate=False,
                  interpretation='Descriptive only: start support/contact not validated; '
                                 'fixed wrist and missing socket invalidate full-tail task success claims. '
                                 'See visibility_audit_20260926/PROCESS_AUDIT.md.',
                  protocol='native eval/xjz_test.sh thresholds; n1 runs, 170 g, friction 2.2, '
                           'DDIM4, seed44 fixed noise, guide2 scale50 insert1 exec2/exec1; '
                           'complete reference and 60-step hold regardless of native failure',
                  episode_count=80, methods=methods, initial_fields_exact=27,
                  settle_trajectories_exact=True,
                  complete_runs_match_early_runs_through_first_failure=True,
                  groups=aggregate)
    (P / 'full_aggregate_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
