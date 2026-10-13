"""Summarize audited ten-seed sweep against the previous three seeds."""
import json
from pathlib import Path
import statistics

from run_random10_cross_model import OUT, SEEDS
from run_h8_cross_model_comparison import EPISODES, METHODS, PHYSICS


def retained_steps(row):
    if row['first_separation_action_step'] is not None:
        return row['first_separation_action_step']
    summary = json.loads((Path(row['folder']) / 'summary.json').read_text())
    return summary['steps']['action']


def main():
    rows = []
    for seed in SEEDS:
        for model in ('1b', '10b', 'h8'):
            for physics in PHYSICS:
                for episode in EPISODES:
                    for method in METHODS:
                        path = OUT / f'seed{seed}' / model / physics / f'episode_{episode:02d}' / method / 'audit.json'
                        rows.append(json.loads(path.read_text()))
    assert len(rows) == 720
    index = {(r['seed'], r['model'], r['physics'], r['episode'], r['method']): r for r in rows}
    assert len(index) == 720
    assert all(r['validation']['initial_all_fields_exact'] and
               r['validation']['settle_motion_exact'] and
               r['validation']['settle_contact_force_max_delta_N'] < 1e-5 for r in rows)
    old = json.loads((OUT.parent / '20260929_h8_object_state_data' / 'audited_results.json').read_text())
    assert len(old) == 216
    (OUT / 'audited_results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    names = {'1b': '旧1B', '10b': '10B', 'h8': 'h8'}
    methods = {'guidance4': 'guidance', 'edit015': 'replay edit'}
    lines = [
        '# 10 个预选随机噪声 seed：横抓灯泡翻转对照', '',
        '同四个 episode × 三组物理参数。每个模型、方法有 120 条轨迹。prior 噪声 seed 在运行前用 `random.Random(20260930).sample(range(1000, 1000000), 10)` 固定；环境初态 seed42 不变。配置见 [manifest.json](manifest.json)。每条轨迹均完成原始 reference 尾段及60步保持，并审计导入初态、60步静置与接触几何。', '',
        '对照的同规则插值 reference direct 只有12条独立轨迹：普通翻转12/12，连续≥30步稳定翻转9/12，首次几何分离前平均158.7个实际控制步。下表的步数是从动作起点到首次连续3步几何分离；未发生分离者在观察尾段截尾。', '',
        '| 模型 | 方法 | 新10 seed 普通翻转 | 新10 seed 稳定翻转 | 新10 seed 平均步数 | 旧3 seed 稳定翻转 | 旧3 seed 平均步数 |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for model in ('1b', '10b', 'h8'):
        for method in METHODS:
            new_group = [r for r in rows if (r['model'], r['method']) == (model, method)]
            old_group = [r for r in old if (r['model'], r['method']) == (model, method)]
            lines.append(f'| {names[model]} | {methods[method]} | '
                         f'{sum(r["ordinary_turn"] for r in new_group)}/120 | '
                         f'{sum(r["stable_turn"] for r in new_group)}/120 | '
                         f'{statistics.mean(retained_steps(r) for r in new_group):.1f} | '
                         f'{sum(r["stable_turn"] for r in old_group)}/36 | '
                         f'{statistics.mean(retained_steps(r) for r in old_group):.1f} |')
    lines += ['', '## 按物理参数', '',
              '| 质量／摩擦 | 模型 | 方法 | 普通翻转 | 稳定翻转 | 平均保持步数 |',
              '|---|---|---|---:|---:|---:|']
    for physics in PHYSICS:
        for model in ('1b', '10b', 'h8'):
            for method in METHODS:
                group = [r for r in rows if (r['physics'], r['model'], r['method']) ==
                         (physics, model, method)]
                assert len(group) == 40
                lines.append(f'| {physics} | {names[model]} | {methods[method]} | '
                             f'{sum(r["ordinary_turn"] for r in group)}/40 | '
                             f'{sum(r["stable_turn"] for r in group)}/40 | '
                             f'{statistics.mean(retained_steps(r) for r in group):.1f} |')
    lines += ['', '## 按噪声 seed', '',
              '| seed | 旧1B guidance | 旧1B edit | 10B guidance | 10B edit | h8 guidance | h8 edit |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        cells = []
        for model in ('1b', '10b', 'h8'):
            for method in METHODS:
                group = [r for r in rows if (r['seed'], r['model'], r['method']) == (seed, model, method)]
                assert len(group) == 12
                cells.append(f'{sum(r["stable_turn"] for r in group)}/12; '
                             f'{statistics.mean(retained_steps(r) for r in group):.1f}步')
        lines.append('| ' + str(seed) + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 配对比较', '',
              '以下比较在相同 seed、episode 和物理参数下逐条配对。稳定翻转的胜／平／负按布尔结果计算；长度差为“左方法减右方法”的平均实际控制步数。', '',
              '| 左方法 | 右方法 | 稳定翻转胜／平／负 | 平均长度差 |',
              '|---|---|---:|---:|']
    comparisons = [
        (('h8', method), (other, method))
        for method in METHODS for other in ('1b', '10b')
    ] + [((model, 'edit015'), (model, 'guidance4'))
         for model in ('1b', '10b', 'h8')]
    for left, right in comparisons:
        pairs = []
        for seed in SEEDS:
            for physics in PHYSICS:
                for episode in EPISODES:
                    a = index[(seed, left[0], physics, episode, left[1])]
                    b = index[(seed, right[0], physics, episode, right[1])]
                    pairs.append((a, b))
        assert len(pairs) == 120
        wins = sum(a['stable_turn'] > b['stable_turn'] for a, b in pairs)
        losses = sum(a['stable_turn'] < b['stable_turn'] for a, b in pairs)
        ties = len(pairs) - wins - losses
        delta = statistics.mean(retained_steps(a) - retained_steps(b) for a, b in pairs)
        lines.append(f'| {names[left[0]]} {methods[left[1]]} | '
                     f'{names[right[0]]} {methods[right[1]]} | '
                     f'{wins}／{ties}／{losses} | {delta:+.1f} |')
    lines += ['', '逐条审计结果见 [audited_results.json](audited_results.json)。'
              '同一 episode × 物理条件在各模型、方法和 seed 下配对；10个seed是新的采样噪声，不是10批新环境初态，因此120条并非120个独立抓握初态。h8未来动作窗口5步，旧1B/10B为9步，跨模型差异不能单独归因于规模。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:16]))


if __name__ == '__main__':
    main()
