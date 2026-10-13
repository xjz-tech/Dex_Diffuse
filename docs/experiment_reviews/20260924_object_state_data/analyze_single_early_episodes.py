"""Summarize all single-environment, native-terminal full-tail experiments."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_paired_early_episodes import result_for_episode
from compare_corrected_rollouts import PROTOCOL


P = Path(__file__).resolve().parent / 'corrected_direct_vs_reference/all_full_episodes'
METHODS = [('direct', 'Direct recorded actions'),
           ('guide2_exec2', '10B guide2/exec2 scale50'),
           ('guide2_exec1', '10B guide2/exec1 scale50')]


def to_trajectory(trace):
    columns = dict(q='q', command='command', object_pose='object_pose',
                   relative_position='relative_position', vertical_deg='vertical_error_deg',
                   native_failure='native_failure')
    return {target: np.asarray([record[source] for record in trace])[:, None]
            for target, source in columns.items()}


def main():
    selection = json.loads((P / 'selection.json').read_text())['episodes']
    axis_audit = json.loads((P / 'selection_axis_audit.json').read_text())['rows']
    assert len(selection) == 80
    results = {method: {} for method, _ in METHODS}
    paired_initial_fields = 27
    for episode in range(80):
        runs = []
        for method, _ in METHODS:
            folder = P / f'episode_{episode:02d}/early/{method}'
            summary = json.loads((folder / 'summary.json').read_text())
            assert summary['source_episode'] == episode
            assert summary['native_protocol'] == PROTOCOL
            assert summary['mass_kg'] == .17 and summary['friction'] == 2.2
            assert summary['stop_on_native_failure'] and not summary['video_recorded']
            if method == 'direct':
                assert summary['prior'] is None and summary['reference_interpolation'] == 0
            else:
                prior = summary['prior']
                assert prior['ddim'] == 4 and prior['guidance_steps'] == 2
                assert prior['guidance_scale'] == 50 and prior['reference_interpolation'] == 1
                assert prior['execution_steps'] == summary['execution_steps']
                assert summary['execution_steps'] == int(method[-1])
            trace = json.loads((folder / 'trace.json').read_text())
            initial = np.load(folder / 'initial_state.npz')
            runs.append(dict(summary=summary, trace=trace, initial=initial))
        fields = [key for key in runs[0]['initial'].files if all(
            key in run['initial'].files and run['initial'][key].shape == runs[0]['initial'][key].shape
            and run['initial'][key].dtype.kind in 'ifub' for run in runs)]
        assert len(fields) == paired_initial_fields
        assert all(np.array_equal(runs[0]['initial'][key], run['initial'][key])
                   for run in runs[1:] for key in fields), episode
        static = [[record for record in run['trace'] if record['phase'] == 'settle']
                  for run in runs]
        assert len({len(group) for group in static}) == 1, episode
        for group in static[1:]:
            for left, right in zip(static[0], group):
                for field in ('q', 'command', 'object_pose', 'relative_position',
                              'vertical_error_deg', 'native_failure'):
                    assert left[field] == right[field], (episode, field)
        for (method, _), run in zip(METHODS, runs):
            summary = run['summary']
            original_actions = selection[episode]['actions']
            intended_actions = summary['intended_steps']['action']
            synthetic = dict(original_action_lengths=[original_actions],
                             action_lengths=[intended_actions],
                             reference_interpolation=summary['reference_interpolation'],
                             steps=dict(settle=summary['steps']['settle'],
                                        global_action=summary['steps']['action'],
                                        hold=summary['steps']['hold']),
                             initial_wrist=[summary['initial_wrist']],
                             initial_object_pose_wrist=[summary['initial_object_pose_wrist']])
            results[method][episode] = result_for_episode(
                episode, selection[episode], method, synthetic,
                to_trajectory(run['trace']), 0)
    rows = [results[method][episode] for method, _ in METHODS for episode in range(80)]
    with (P / 'single_per_episode_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (P / 'single_per_episode_results.json').write_text(json.dumps(rows, indent=2) + '\n')

    def group_summary(method, episodes):
        group = [results[method][e] for e in episodes]
        def median(field):
            valid = [r[field] for r in group if r[field] is not None]
            return float(np.median(valid)) if valid else None
        return dict(n=len(group), static_no_failure=sum(r['static_no_failure'] for r in group),
            calibration_gate=sum(r['calibration_gate'] for r in group),
            action_no_failure=sum(r['action_no_failure'] for r in group),
            full_no_failure=sum(r['full_no_failure'] for r in group),
            reached_25pct_without_failure=sum(r['source_progress_before_failure'] >= .25 for r in group),
            reached_50pct_without_failure=sum(r['source_progress_before_failure'] >= .50 for r in group),
            reached_75pct_without_failure=sum(r['source_progress_before_failure'] >= .75 for r in group),
            sustained_30deg_prefailure=sum(r['sustained_30deg_for_15_steps_prefailure'] for r in group),
            sustained_45deg_prefailure=sum(r['sustained_45deg_for_15_steps_prefailure'] for r in group),
            horizontal_to_30deg_prefailure=sum(r['horizontal_to_30deg_prefailure'] for r in group),
            horizontal_to_45deg_prefailure=sum(r['horizontal_to_45deg_prefailure'] for r in group),
            settle_still_horizontal=sum(r['settle_still_horizontal'] for r in group),
            median_failure_progress=float(np.median([
                r['source_progress_before_failure'] for r in group if r['first_failure_phase'] is not None]))
            if any(r['first_failure_phase'] is not None for r in group) else None,
            median_prefailure_command_rmse_rad=median('command_vs_reference_rmse_prefailure_rad'),
            median_prefailure_joint_rmse_rad=median('joint_vs_reference_rmse_prefailure_rad'),
            median_first_step_displacement_cm=median('first_physics_step_displacement_cm'),
            median_settle_displacement_cm=median('settle_displacement_cm'))
    groups = {'all80': list(range(80)),
              'lifted43': [e for e in range(80) if selection[e]['lifted_start']],
              'lifted_axis41': [e for e in range(80) if selection[e]['lifted_start'] and
                                axis_audit[e]['object_axis_wrist_error_to_nearest_template_deg'] <= 30],
              'fallback37': [e for e in range(80) if not selection[e]['lifted_start']]}
    groups['static_survivors'] = [e for e in range(80)
                                  if results['direct'][e]['static_no_failure']]
    groups['horizontal_after_settle'] = [e for e in groups['static_survivors']
                                         if results['direct'][e]['settle_still_horizontal']]
    groups['lifted_static_survivors'] = [e for e in groups['static_survivors']
                                         if selection[e]['lifted_start']]
    assert len(groups['lifted43']) == 43 and len(groups['fallback37']) == 37
    assert len(groups['lifted_axis41']) == 41
    aggregate = {label: {method: group_summary(method, episodes) for method, _ in METHODS}
                 for label, episodes in groups.items()}
    pairwise = {}
    for label, episodes in groups.items():
        d, e2, e1 = (results['direct'], results['guide2_exec2'], results['guide2_exec1'])
        pairwise[label] = dict(
            exec1_later_than_exec2=sum(e1[e]['source_progress_before_failure'] >
                                        e2[e]['source_progress_before_failure'] for e in episodes),
            exec2_later_than_exec1=sum(e2[e]['source_progress_before_failure'] >
                                        e1[e]['source_progress_before_failure'] for e in episodes),
            exec1_exec2_tie=sum(e1[e]['source_progress_before_failure'] ==
                                 e2[e]['source_progress_before_failure'] for e in episodes),
            exec1_later_than_direct=sum(e1[e]['source_progress_before_failure'] >
                                        d[e]['source_progress_before_failure'] for e in episodes),
            exec2_later_than_direct=sum(e2[e]['source_progress_before_failure'] >
                                        d[e]['source_progress_before_failure'] for e in episodes))
    report = dict(dataset_episodes=80, strict_lifted_starts=43,
                  strict_lifted_and_axis_similar_starts=41, fallback_starts=37,
                  initial_fields_exact=paired_initial_fields, all_static_trajectories_exact=True,
                  protocol='native eval/xjz_test.sh thresholds; one environment per run, 170 g, friction 2.2, DDIM4, seed44 fixed noise, guide2 scale50, insert1, exec2/exec1, fixed native wrist, stop at first native failure',
                  groups=aggregate, pairwise=pairwise)
    (P / 'single_aggregate_results.json').write_text(json.dumps(report, indent=2) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = {'direct': '#5b6770', 'guide2_exec2': '#e0842c', 'guide2_exec1': '#2371b4'}
    grid = np.linspace(0, 1, 101)
    plot_groups = [(label, groups[label]) for label in ('all80', 'lifted43', 'fallback37')]
    for ax, (label, episodes) in zip(axes, plot_groups):
        for method, title in METHODS:
            progress = np.asarray([results[method][e]['source_progress_before_failure']
                                   for e in episodes])
            ax.step(grid, [(progress >= p).mean() for p in grid], where='post',
                    label=title, linewidth=2, color=colors[method])
        ax.set_title(f'{label} (n={len(episodes)})')
        ax.set_xlabel('Fraction of source actions reached')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.25)
    axes[0].set_ylabel('No native failure so far')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(P / 'single_failure_free_progress.png', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(groups=aggregate, pairwise=pairwise), indent=2))


if __name__ == '__main__':
    main()
