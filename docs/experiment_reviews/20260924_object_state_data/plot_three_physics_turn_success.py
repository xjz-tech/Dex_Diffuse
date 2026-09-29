#!/usr/bin/env python3
"""Stable-turn counts and post-turn retention for three physics settings."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

P = Path(__file__).resolve().parent
OUT = (P/'reference_turn_baseline_20260926' /
       'three_physics_turn_success_chart_20260928')

SETTINGS = [
    dict(label='44g / 摩擦 1.1', success=[4, 4, 4]),
    dict(label='170g / 摩擦 2.0', success=[4, 3, 4]),
    dict(label='130g / 摩擦 2.4', success=[4, 3, 4]),
]
METHOD_KEYS = ['direct', 'guidance', 'edit']
METHODS = ['replay actions', 'guidance\nscale50 / g4e2', 'replay edit\n0.15 / DDIM4']
COLORS = ['#4C78A8', '#F58518', '#54A24B']
LIGHT_COLORS = ['#AFC8E1', '#FBCB9C', '#ABD4A7']


def folders():
    root = P/'reference_turn_baseline_20260926'
    archived = json.loads((root/'adaptive010_direct_edit015_guidance4_comparison_20260928' /
                           'RESULTS.json').read_text())
    m44 = {(x['episode'], x['method']): Path(x['folder']) for x in archived}
    result = [m44]
    for base in ('m170_mu20_guidance4_vs_edit015_20260928',
                 'm130_mu24_guidance4_vs_edit015_20260928'):
        group = {}
        for episode in (76, 34, 54, 2):
            for method, name in (('direct', 'direct_interp'),
                                 ('guidance', 'guidance4'),
                                 ('edit', 'edit015')):
                group[(episode, method)] = root/base/f'episode_{episode:02d}'/name
        result.append(group)
    return result


def rollout_length(folder):
    """Actual control steps from action start to geometric separation."""
    trace = json.loads((folder/'trace.json').read_text())
    geometry = json.loads((folder/'grasp_metrics.json').read_text())['frames']
    separated = []
    for row, geo in zip(trace, geometry):
        force = float(np.linalg.norm(row['object_contact_force']))
        separated.append((geo['mesh_vertex_gap_m'] > .005 and force < .05) or
                         geo['mesh_vertex_gap_m'] > .02)
    loss = next(i for i in range(len(separated)-2) if all(separated[i:i+3]))
    assert trace[loss]['phase'] == 'action'
    return dict(steps=int(trace[loss]['index']+1),
                separation_action_step=int(trace[loss]['index']+1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    total_success = np.sum([setting['success'] for setting in SETTINGS], axis=0)
    success_panels = SETTINGS + [dict(label='总共（三组）', success=total_success.tolist())]
    fig, axes = plt.subplots(1, 4, figsize=(19, 6.4))
    x = np.arange(3)
    for panel_index, (ax, setting) in enumerate(zip(axes, success_panels)):
        success = np.asarray(setting['success'])
        ax.bar(x, success, width=.66, color=COLORS, alpha=.92,
               edgecolor=COLORS, linewidth=1.2)
        for i, s in enumerate(success):
            denominator = 12 if panel_index == 3 else 4
            offset = .2 if panel_index == 3 else .08
            ax.text(i, s+offset, f'{s}/{denominator}', ha='center', va='bottom',
                    fontsize=13, fontweight='bold')
        ax.set_title(setting['label'], fontsize=16, fontweight='bold', pad=12)
        ax.set_xticks(x, METHODS, fontsize=11)
        if panel_index == 3:
            ax.set_ylim(0, 13.1); ax.set_yticks([0, 3, 6, 9, 12])
        else:
            ax.set_ylim(0, 4.65); ax.set_yticks(range(5))
        ax.grid(axis='y', alpha=.2, linewidth=.8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('成功 episode 数（共4个）', fontsize=13)
    fig.suptitle('横抓灯泡：翻转成功次数',
                 fontsize=18, fontweight='bold', y=.97)
    fig.text(.5, .025,
             '成功定义：分离前至少1个动作步达到竖直、悬空且有效接触。前三组各4个episode；总共为12个case。',
             ha='center', fontsize=11, color='#444444')
    fig.subplots_adjust(left=.05, right=.99, bottom=.20, top=.83, wspace=.16)
    png = OUT/'turn_success_three_physics.png'
    svg = OUT/'turn_success_three_physics.svg'
    fig.savefig(png, dpi=200, facecolor='white')
    fig.savefig(svg, facecolor='white')
    plt.close(fig)

    # Second figure: all episodes, from the first action to the first
    # three-step-confirmed geometric separation marker.
    retention_data = []
    folder_groups = folders()
    retention_panels = [(setting, [group]) for setting, group in zip(SETTINGS, folder_groups)]
    retention_panels.append((dict(label='总共（三组）'), folder_groups))
    fig, axes = plt.subplots(1, 4, figsize=(19, 6.4), sharey=True)
    for ax, (setting, groups) in zip(axes, retention_panels):
        means = []
        details = []
        for method in METHOD_KEYS:
            values = []
            rows = []
            for group_index, group in enumerate(groups):
                for episode in (76, 34, 54, 2):
                    value = rollout_length(group[(episode, method)])
                    values.append(value['steps'])
                    rows.append(dict(setting_index=group_index, episode=episode, **value))
            assert len(values) == 4*len(groups)
            means.append(float(np.mean(values)))
            details.append(rows)
        bars = ax.bar(x, means, width=.66, color=COLORS, alpha=.92)
        for i, (mean, rows) in enumerate(zip(means, details)):
            ax.text(i, mean+8, f'{mean:.1f}步\n{mean/30:.2f}s',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_title(setting['label'], fontsize=16, fontweight='bold', pad=12)
        ax.set_xticks(x, METHODS, fontsize=11)
        ax.grid(axis='y', alpha=.2, linewidth=.8)
        ax.spines[['top', 'right']].set_visible(False)
        retention_data.append(dict(setting=setting['label'], means=means,
                                   episodes_by_method=dict(zip(METHOD_KEYS, details))))
    axes[0].set_ylabel('从动作开始到几何分离的实际控制步数', fontsize=13)
    ymax = max(max(x['means']) for x in retention_data)
    axes[0].set_ylim(0, max(350, ymax+55))
    fig.suptitle('从动作开始到结束：平均执行步数',
                 fontsize=18, fontweight='bold', y=.97)
    fig.text(.5, .025,
             '从第1个动作计至首次连续3步确认的几何分离；前三组各计入4个episode，总共直接对12个case取平均。',
             ha='center', fontsize=11, color='#444444')
    fig.subplots_adjust(left=.05, right=.99, bottom=.20, top=.83, wspace=.08)
    hold_png = OUT/'post_turn_retention_steps_three_physics.png'
    hold_svg = OUT/'post_turn_retention_steps_three_physics.svg'
    fig.savefig(hold_png, dpi=200, facecolor='white')
    fig.savefig(hold_svg, facecolor='white')
    plt.close(fig)

    # Third figure: for each method and physics setting, show the minimum and
    # maximum rollout length across the four episodes.
    extrema_data = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4), sharey=True)
    episode_ids = (76, 34, 54, 2)
    xx = np.arange(len(METHOD_KEYS))
    for ax, setting, group in zip(axes, SETTINGS, folder_groups):
        lows, highs, rows = [], [], []
        for method in METHOD_KEYS:
            values = {episode: rollout_length(group[(episode, method)])['steps']
                      for episode in episode_ids}
            low_episode = min(values, key=values.get)
            high_episode = max(values, key=values.get)
            lows.append(values[low_episode]); highs.append(values[high_episode])
            rows.append(dict(method=method, values_by_episode=values,
                             minimum_steps=values[low_episode],
                             minimum_episode=low_episode,
                             maximum_steps=values[high_episode],
                             maximum_episode=high_episode))
        width = .34
        ax.bar(xx-width/2, lows, width, color=LIGHT_COLORS, edgecolor=COLORS,
               linewidth=1.2, label='最低')
        ax.bar(xx+width/2, highs, width, color=COLORS, edgecolor=COLORS,
               linewidth=1.2, label='最高')
        for i, row in enumerate(rows):
            ax.text(i-width/2, row['minimum_steps']+6,
                    f"{row['minimum_steps']}",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(i+width/2, row['maximum_steps']+6,
                    f"{row['maximum_steps']}",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title(setting['label'], fontsize=16, fontweight='bold', pad=12)
        ax.set_xticks(xx, METHODS, fontsize=10)
        ax.set_ylim(0, 440)
        ax.grid(axis='y', alpha=.2, linewidth=.8)
        ax.spines[['top', 'right']].set_visible(False)
        extrema_data.append(dict(setting=setting['label'], methods=rows))
    axes[0].set_ylabel('从动作开始到几何分离的实际控制步数', fontsize=13)
    axes[0].legend(loc='upper left', frameon=False, fontsize=11)
    fig.suptitle('每种方法：4个 Episodes 中的最低与最高执行步数',
                 fontsize=18, fontweight='bold', y=.97)
    fig.text(.5, .025,
             '每种方法统计4个Episode中的最低与最高值；步数计至首次连续3步确认的几何分离。',
             ha='center', fontsize=11, color='#444444')
    fig.subplots_adjust(left=.06, right=.99, bottom=.20, top=.83, wspace=.08)
    extrema_png = OUT/'episode_min_max_steps_three_physics.png'
    extrema_svg = OUT/'episode_min_max_steps_three_physics.svg'
    fig.savefig(extrema_png, dpi=200, facecolor='white')
    fig.savefig(extrema_svg, facecolor='white')
    plt.close(fig)
    payload = dict(
        methods=['same_rule_interpolated_direct',
                 'guidance_ddim4_scale50_guide4_exec2',
                 'edit015_ddim4_exec2'],
        episodes=[76, 34, 54, 2], settings=SETTINGS,
        total_turn_success=total_success.tolist(),
        flip_success_definition=('at least one pre-separation action step meets '
                                 'vertical, airborne, and effective-contact criteria'),
        stable_definition='longest continuous qualifying block >=30 steps',
        post_turn_retention=dict(
            definition=('actual control steps from action start to first '
                        'three-step-confirmed geometric separation'),
            successful_episodes_only=False, all_four_episodes=True, control_hz=30,
            data=retention_data),
        per_episode_method_extrema=extrema_data,
        source=str(P/'reference_turn_baseline_20260926' /
                   'm130_mu24_guidance4_vs_edit015_20260928/REPORT.md'))
    (OUT/'chart_data.json').write_text(json.dumps(payload, ensure_ascii=False,
                                                  indent=2)+'\n')
    print(png)


if __name__ == '__main__':
    main()
