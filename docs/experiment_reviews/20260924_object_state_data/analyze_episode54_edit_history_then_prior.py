#!/usr/bin/env python3
"""Compare continuing edit with a post-flip autonomous-prior branch."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from reference_resampling import interpolate_large_jumps

P = Path(__file__).resolve().parent
R = P/'reference_turn_baseline_20260926'
EDIT = R/'m170_mu20_guidance4_vs_edit015_20260928/episode_54/edit015'
PRIOR = R/'m170_mu20_episode54_edit_history_then_prior_20260928'
CASE = R/'qualified_comparison/episode_54/reference_full.npz'
OUT = PRIOR
BRANCH_LAST = 101       # displayed action 102; common state after this row
FIRST_AUTONOMOUS = 102  # displayed action 103


def load(folder):
    return (json.loads((folder/'trace.json').read_text()),
            json.loads((folder/'summary.json').read_text()),
            json.loads((folder/'grasp_metrics.json').read_text())['frames'])


def first_separation(trace, geometry):
    separated = []
    for row, geo in zip(trace, geometry):
        force = float(np.linalg.norm(row['object_contact_force']))
        separated.append((geo['mesh_vertex_gap_m'] > .005 and force < .05) or
                         geo['mesh_vertex_gap_m'] > .02)
    return next((i for i in range(len(separated)-2)
                 if all(separated[i:i+3])), None)


def effective(row, geo):
    return (geo['near_contact_link_count'] >= 2 and
            geo['mesh_vertex_gap_m'] < .008 and
            np.linalg.norm(row['object_contact_force']) > .1)


def main():
    edit, es, eg = load(EDIT)
    prior, ps, pg = load(PRIOR)
    assert len(edit) == len(prior) == 732
    assert es['mass_kg'] == ps['mass_kg'] == .17
    assert es['friction'] == ps['friction'] == 2.0

    # State, commands, and velocities must be exactly common through action 102.
    common_end = 60 + BRANCH_LAST + 1
    exact_fields = {}
    max_delta = {}
    for field in ('object_pose', 'q', 'command', 'executed_target',
                  'relative_position', 'object_velocity'):
        a = np.asarray([x[field] for x in edit[:common_end]])
        b = np.asarray([x[field] for x in prior[:common_end]])
        exact_fields[field] = bool(np.array_equal(a, b))
        max_delta[field] = float(np.max(np.abs(a-b)))
    assert all(exact_fields.values())
    force_delta = float(np.max(np.abs(
        np.asarray([x['object_contact_force'] for x in edit[:common_end]]) -
        np.asarray([x['object_contact_force'] for x in prior[:common_end]]))))
    assert force_delta <= 4e-6

    progress = np.load(EDIT/'reference_progress.npy')
    with np.load(CASE) as z:
        reference_pose = z['object_pose_wrist'][0]
        reference_actions, _ = interpolate_large_jumps(z['hand_target_rad'], .1)
        reference_actions = reference_actions[0]
    # Use the same positive right axis as the earlier verified post-flip audit.
    start_ref = int(np.floor(progress[BRANCH_LAST]))
    end_ref = 143
    delta = (Rotation.from_matrix(reference_pose[end_ref, :3, :3]) *
             Rotation.from_matrix(reference_pose[start_ref, :3, :3]).inv()).as_rotvec()
    axis_wrist = delta/np.linalg.norm(delta)
    axis_world = Rotation.from_quat(es['initial_wrist'][3:]).apply(axis_wrist)

    def curve(trace):
        rows = trace[60+BRANCH_LAST:60+612]
        rotations = Rotation.from_quat(np.asarray([x['object_pose'][3:] for x in rows]))
        increments = np.degrees(
            (rotations[1:]*rotations[:-1].inv()).as_rotvec() @ axis_world)
        return np.r_[0., np.cumsum(increments)]

    edit_curve, prior_curve = curve(edit), curve(prior)
    edit_sep, prior_sep = first_separation(edit, eg), first_separation(prior, pg)
    assert edit_sep == 60+189 and prior_sep is None

    def last_effective_action(trace, geometry):
        indices = [row['index'] for row, geo in zip(trace, geometry)
                   if row['phase'] == 'action' and row['index'] >= BRANCH_LAST and
                   effective(row, geo)]
        return max(indices)

    edit_last = last_effective_action(edit, eg)
    prior_last = last_effective_action(prior, pg)
    assert edit_last == 186 and prior_last == 611
    prior_hold_effective = int(sum(effective(row, geo) for row, geo in zip(prior, pg)
                                   if row['phase'] == 'hold'))
    assert prior_hold_effective == 60

    def command_alignment(trace):
        commands = np.asarray([x['command'] for x in trace if x['phase'] == 'action'])
        cosines = []
        for j in range(FIRST_AUTONOMOUS, edit_last, 2):
            issued = commands[j+2]-commands[j]
            wanted = reference_actions[j+2]-reference_actions[j]
            denom = np.linalg.norm(issued)*np.linalg.norm(wanted)
            if denom > 1e-12:
                cosines.append(float(issued@wanted/denom))
        return cosines

    e_cos, p_cos = command_alignment(edit), command_alignment(prior)

    def rotation_path(trace, end_action):
        rotations = Rotation.from_quat(np.asarray([
            x['object_pose'][3:] for x in trace[60+BRANCH_LAST:60+end_action+1]]))
        increments = np.degrees((rotations[1:]*rotations[:-1].inv()).as_rotvec())
        signed = increments @ axis_world
        return dict(full_3d_path_deg=float(np.linalg.norm(increments, axis=1).sum()),
                    target_axis_absolute_path_deg=float(np.abs(signed).sum()),
                    target_axis_net_deg=float(signed.sum()))
    result = dict(
        episode=54, mass_kg=.17, friction=2.0,
        branch=dict(common_through_displayed_action_step=102,
                    first_autonomous_displayed_action_step=103,
                    state_at_branch='after action 102',
                    pure_prior_input=('same four observation frames; action-history channels '
                                      'contain previous three issued edit targets'),
                    pure_prior_future_reference_steps=0),
        prebranch_validation=dict(exact_fields=exact_fields,
                                  max_abs_delta_by_field=max_delta,
                                  contact_force_max_abs_delta_N=force_delta,
                                  same_seed=44),
        continued_edit=dict(
            last_effective_contact_displayed_action_step=edit_last+1,
            first_geometric_separation_displayed_action_step=edit[edit_sep]['index']+1,
            effective_contact_seconds_after_branch=(edit_last-BRANCH_LAST)/30,
            target_right_rotation_at_last_contact_deg=float(edit_curve[edit_last-BRANCH_LAST]),
            maximum_net_target_right_rotation_deg=float(edit_curve.max()),
            command_reference_cosine_mean=float(np.mean(e_cos)),
            command_reference_cosine_median=float(np.median(e_cos)),
            command_reference_positive_blocks=int(sum(x > 0 for x in e_cos)),
            command_reference_blocks=len(e_cos),
            physical_rotation_path_until_last_contact=rotation_path(edit, edit_last)),
        edit_history_then_pure_prior=dict(
            last_effective_contact_displayed_action_step=prior_last+1,
            first_geometric_separation_displayed_action_step=None,
            contact_right_censored=True,
            effective_contact_seconds_after_branch_at_least=(prior_last-BRANCH_LAST+60)/30,
            terminal_hold_effective_steps=prior_hold_effective,
            target_right_rotation_at_edit_last_contact_time_deg=float(prior_curve[edit_last-BRANCH_LAST]),
            target_right_rotation_at_action_end_deg=float(prior_curve[-1]),
            maximum_net_target_right_rotation_deg=float(prior_curve.max()),
            maximum_opposite_rotation_deg=float(-prior_curve.min()),
            command_reference_cosine_mean=float(np.mean(p_cos)),
            command_reference_cosine_median=float(np.median(p_cos)),
            command_reference_positive_blocks=int(sum(x > 0 for x in p_cos)),
            command_reference_blocks=len(p_cos),
            physical_rotation_path_until_action_end=rotation_path(prior, prior_last)),
        conclusion=('Pure prior retained the bulb much longer but did not continue the '
                    'reference right turn: its net right rotation never exceeded the '
                    'branch state and it rotated mainly in the opposite direction.'),
        definitions=dict(
            positive_rotation='recorded reference right-turn axis from progress 80 to 143',
            effective_contact='>=2 near links, mesh gap <8mm, object force >0.1N',
            geometric_separation=('first of 3 consecutive steps with gap >5mm and force '
                                  '<0.05N, or gap >20mm')))
    (OUT/'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')

    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    x = np.arange(len(edit_curve))/30
    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    ax.plot(x, edit_curve, linewidth=2, label='继续 edit（有未来 reference）')
    ax.plot(x, prior_curve, linewidth=2, label='保留 edit 历史后 pure prior')
    ax.axhline(0, color='.6', linewidth=1)
    ax.axvline((edit_last-BRANCH_LAST)/30, color='tab:blue', linestyle=':',
               label='edit 最后有效接触')
    ax.set_xlabel('分叉后的时间（秒）')
    ax.set_ylabel('沿 reference 右转方向的累计角度（度）')
    ax.set_title('Episode 54 · 170g / 摩擦2.0 · 同状态分叉')
    ax.grid(alpha=.2); ax.legend(loc='best')
    fig.savefig(OUT/'right_turn_comparison.png', dpi=180)
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
