"""Summarize the completed paired Object_state_data sweep."""
import json
from pathlib import Path
import statistics

from run_h8_cross_model_comparison import O, PHYSICS

MODEL_NAMES = {'1b': '旧 1B', '10b': '10B', 'h8': 'h8'}
METHOD_NAMES = {'guidance4': 'guidance 4 / scale50 / exec2',
                'edit015': 'replay edit 0.15 / DDIM4 / exec2'}


def main():
    rows = json.loads((O / 'audited_results.json').read_text())
    index = {(r['seed'], r['model'], r['physics'], r['episode'], r['method']): r for r in rows}
    assert len(rows) == len(index) == 216
    assert set(r['seed'] for r in rows) == {44, 45, 46}
    lines = [
        '# h8、旧 1B、10B：横抓灯泡 reference guidance 与 replay edit',
        '',
        '四个固定合格横抓起点为 episode 76/f114、34/f79、54/f110、2/f103；三组物理参数为 44g/摩擦1.1、170g/2.0、130g/2.4。每组分别运行 prior 噪声 seed44、45、46；仿真初态与环境 seed42 固定，且各方法逐字段一致。原始 reference 已独立通过横转竖资格检查。',
        '',
        '三模型均使用 EMA、DDIM4、exec2、相邻 reference 的 22 关节最大目标跳变严格 >0.1rad 时插入一个中点；guidance 用 scale50 和前 4 步 reference，edit 请求噪声0.15（实际0.153397），局部去噪时刻 8→5→3→0。h8 的模型窗口是历史3＋未来5步；旧1B、10B为历史3＋未来9步，因此跨模型对比包含窗口长度差异。',
        '',
        '翻转成功指首次几何分离前至少一个动作步竖直误差≤30°、悬空且有有效接触；稳定翻转要求连续≥30个这样的控制步。保持步数从动作起点计到首次连续3步几何分离；没有在观察尾段分离的记为该段截尾。该物理接触指标独立于原生 failure。',
        '',
        '| 模型 | 方法 | 普通翻转 | ≥30 步稳定翻转 | 平均观察动作步数* | 平均原始 reference 进度* |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for model in ('1b', '10b', 'h8'):
        for method in ('guidance4', 'edit015'):
            group = [r for r in rows if r['model'] == model and r['method'] == method]
            mean_steps = statistics.mean(r['first_separation_action_step'] for r in group)
            mean_progress = statistics.mean(r['first_separation_reference_progress'] for r in group)
            lines.append(f"| {MODEL_NAMES[model]} | {METHOD_NAMES[method]} | "
                         f"{sum(r['ordinary_turn'] for r in group)}/36 | "
                         f"{sum(r['stable_turn'] for r in group)}/36 | {mean_steps:.1f} | {mean_progress:.1f} |")
    lines += ['', '*到完整 reference 尾段仍未几何分离的轨迹在尾段截尾；上表平均值为观察窗内的截断平均，不是无界真实保持时间。',
              '', '## 每组物理参数下的稳定翻转', '',
              '| 质量／摩擦 | 模型 | guidance | replay edit |', '|---|---|---:|---:|']
    for physics in PHYSICS:
        for model in ('1b', '10b', 'h8'):
            g = [r for r in rows if r['model'] == model and r['physics'] == physics and r['method'] == 'guidance4']
            e = [r for r in rows if r['model'] == model and r['physics'] == physics and r['method'] == 'edit015']
            lines.append(f"| {physics} | {MODEL_NAMES[model]} | {sum(r['stable_turn'] for r in g)}/12 | {sum(r['stable_turn'] for r in e)}/12 |")
    lines += ['', '## 按 prior 噪声 seed', '',
              '| seed | 模型 | guidance 稳定翻转 | replay edit 稳定翻转 |', '|---:|---|---:|---:|']
    for seed in (44, 45, 46):
        for model in ('1b', '10b', 'h8'):
            g = [r for r in rows if r['seed'] == seed and r['model'] == model and r['method'] == 'guidance4']
            e = [r for r in rows if r['seed'] == seed and r['model'] == model and r['method'] == 'edit015']
            lines.append(f"| {seed} | {MODEL_NAMES[model]} | {sum(r['stable_turn'] for r in g)}/12 | {sum(r['stable_turn'] for r in e)}/12 |")
    lines += ['', 'seed44 的 10B 复用此前完整存档并逐条重新审计；其它组合由本轮新跑。每个 episode 都使用自己的横抓初态，60步静置与原始 direct 对照相同；没有按 prior 表现筛选起点。所有轨迹完整执行 reference 尾段并另加60步保持，不以几何分离提前终止控制。', '']
    (O / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[:21]))


if __name__ == '__main__':
    main()
