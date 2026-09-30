"""Descriptive, paired domain-transfer analysis; no population significance claims."""
from pathlib import Path
import json
import numpy as np

r = Path(__file__).resolve().parents[1]
summary = json.loads((r / 'evaluation_summary.json').read_text())
assert summary['complete']
names = [x['method'] for x in summary['models']]
rows = summary['all_results']
by_method = {n: sorted([x for x in rows if x['method'] == n], key=lambda x: x['case']) for n in names}
times = {n: np.minimum([x['seconds'] for x in by_method[n]], 400.) for n in names}
cases = by_method['ordinary_1b']
base = times['ordinary_1b']
details = {}
for n in names:
    t = times[n]
    delta = t - base
    details[n] = {
        'paired_gain_mean_s': float(delta.mean()),
        'paired_gain_median_s': float(np.median(delta)),
        'gain_over20_count': int((delta > 20).sum()),
        'loss_over20_count': int((delta < -20).sum()),
        'rescued_baseline_under20_to_atleast20_count': int(((base < 20) & (t >= 20)).sum()),
        'harmed_baseline_atleast20_to_under20_count': int(((base >= 20) & (t < 20)).sum()),
        'time_bins_counts_0_20_40_80_400_censored': [
            int((t < 20).sum()), int(((t >= 20) & (t < 40)).sum()),
            int(((t >= 40) & (t < 80)).sum()), int(((t >= 80) & (t < 400)).sum()),
            int((t >= 400).sum())],
        'by_initialization': {},
        'by_policy_seed': {},
    }
    for init in ['demo079', 'demo082', 'demo094', 'real_pose50']:
        mask = np.array([x['initialization'] == init for x in cases])
        details[n]['by_initialization'][init] = {
            'n': int(mask.sum()), 'capped_mean_s': float(t[mask].mean()),
            'paired_mean_gain_s': float(delta[mask].mean())}
    for seed in [8, 19, 25]:
        mask = np.array([x['policy_seed'] == seed for x in cases])
        details[n]['by_policy_seed'][str(seed)] = {
            'n': int(mask.sum()), 'capped_mean_s': float(t[mask].mean()),
            'paired_mean_gain_s': float(delta[mask].mean())}

factorial = []
for mass in [.044, .240]:
    for mu in [1.312, 2.572]:
        mask = np.array([x['mass_kg'] == mass and x['object_friction'] == mu for x in cases])
        means = {n: float(t[mask].mean()) for n, t in times.items()}
        light = (times['light_low'] + times['light_high']) / 2
        heavy = (times['heavy_low'] + times['heavy_high']) / 2
        low = (times['light_low'] + times['heavy_low']) / 2
        high = (times['light_high'] + times['heavy_high']) / 2
        matched = ('light' if mass == .044 else 'heavy') + ('_low' if mu == 1.312 else '_high')
        factorial.append({
            'mass_g': mass * 1000, 'mu': mu, 'n': int(mask.sum()), 'means': means,
            'heavy_vs_light_training_mean_s': float((heavy[mask] - light[mask]).mean()),
            'high_vs_low_friction_training_mean_s': float((high[mask] - low[mask]).mean()),
            'matched_specialist': matched,
            'matched_specialist_vs_balanced_s': float((times[matched][mask] - times['balanced'][mask]).mean()),
            'best_new_guide': max([n for n in names if n not in ['ordinary_1b', 'old10k']], key=lambda n: means[n]),
        })

# Same initial q/targets/prior noise: compare the first command before physical feedback.
initial = np.load(r / 'evaluation/ordinary_1b/initial_state.npz')
q0 = initial['q']
raw0 = {n: np.load(r / 'evaluation' / n / 'first_actions.npz')['raw'] for n in names}
first_commands = {}
for n, raw in raw0.items():
    first_commands[n] = {
        'mean_abs_target_minus_q_rad': float(np.abs(raw - q0).mean()),
        'mean_abs_target_minus_ordinary_rad': float(np.abs(raw - raw0['ordinary_1b']).mean()),
        'per_initialization': {},
    }
    for init in ['demo079', 'demo082', 'demo094', 'real_pose50']:
        mask = np.array([x['initialization'] == init for x in cases])
        first_commands[n]['per_initialization'][init] = float(np.abs(raw[mask] - raw0['ordinary_1b'][mask]).mean())

# Before any arm fails: same number of scored steps for every arm in each case.
common_steps = np.minimum(np.min([[x['steps'] for x in by_method[n]] for n in names], axis=0), 60)
early = {}
for n in names:
    z = np.load(r / 'evaluation' / n / 'rollout.npz')
    q = z['q'][:60]; raw = z['raw'][:60]
    qprev = np.concatenate([q0[None], q[:-1]])
    aprev = np.concatenate([initial['target'][None], raw[:-1]])
    metrics = []
    for i, count in enumerate(common_steps):
        metrics.append({
            'case': i, 'common_steps': int(count),
            'mean_abs_target_gap_rad': float(np.abs(raw[:count, i] - qprev[:count, i]).mean()),
            'mean_abs_target_step_rad': float(np.abs(raw[:count, i] - aprev[:count, i]).mean()),
            'mean_abs_q_step_rad': float(np.abs(q[:count, i] - qprev[:count, i]).mean()),
            'mean_target_error_m': float(z['error_m'][:count, i].mean()),
        })
    early[n] = {'per_case': metrics, 'mean_over_cases': {
        k: float(np.mean([v[k] for v in metrics])) for k in metrics[0] if k not in ['case', 'common_steps']}}

result = {'paired_outcomes': details, 'factorial': factorial, 'first_commands': first_commands,
          'common_early_window': early,
          'limitations': ['Only four diagnostic initializations and one training seed.',
                          '400 seconds is censoring, not failure.',
                          'Commands and target errors do not measure contact forces or prove a mechanism.',
                          'All new sets are success-selected matched segments; old10k has additional collection confounds.']}
attempts = sorted((r / 'interrupted_attempts').glob('ordinary_1b*'))
if attempts:
    old = attempts[-1]
    horizon = json.loads((old / 'progress.json').read_text())['sim_seconds']
    prev_rows = json.loads((old / 'partial_results.json').read_text())
    prev = np.full(48, horizon)
    for row in prev_rows:
        prev[row['case']] = min(row['seconds'], horizon)
    current = np.minimum(base, horizon)
    same_init = np.load(old / 'initial_state.npz')
    assert all(np.array_equal(initial[k], same_init[k]) for k in initial.files)
    assert (old / 'physical_parameters.json').read_bytes() == (r / 'evaluation/ordinary_1b/physical_parameters.json').read_bytes()
    assert np.array_equal(np.load(old / 'first_actions.npz')['raw'], raw0['ordinary_1b'])
    result['baseline_repeatability_at_common_censoring'] = {
        'horizon_seconds': horizon,
        'same_initial_state_physics_and_first_command': True,
        'mean_absolute_duration_difference_s': float(np.abs(current - prev).mean()),
        'duration_difference_over20_count': int((np.abs(current - prev) > 20).sum()),
        'old_minus_current_per_case_s': (prev - current).tolist(),
        'note': 'One interrupted repeat, censored at its recorded progress. This quantifies observed run variability, not a confidence interval or its cause.'}
    result['limitations'].append('A same-input baseline repeat had different later outcomes; small pairwise duration differences are not decisive.')
(r / 'transfer_analysis.json').write_text(json.dumps(result, indent=2))
lines = ['# 训练域到测试域的迁移', '',
         '下表是400秒截断保持均值，每格12个成对案例。训练域对应的是实际保留数据；只对本轮诊断初态成立。', '',
         '| 测试物理条件 | 普通1B | 旧10k | 轻/低 | 轻/高 | 重/低 | 重/高 | 均衡 |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for x in factorial:
    lines.append(f"| {x['mass_g']:.0f}g / μ{x['mu']} | " + ' | '.join(f"{x['means'][n]:.2f}" for n in names) + ' |')
lines += ['', '| 方法 | 相比普通1B均值增益/s | 提升超过20s | 下降超过20s |', '|---|---:|---:|---:|']
for n, x in details.items():
    lines.append(f"| {n} | {x['paired_gain_mean_s']:+.2f} | {x['gain_over20_count']}/48 | {x['loss_over20_count']}/48 |")
lines += ['', '初始动作差只比较相同观察和噪声下的输出；早期指标在每个案例所有方法共同存活的前至多60步计算，避免各方法存活时间不同导致窗口不可比。原始数值见transfer_analysis.json。这些指标不能单独证明接触、抓握力或滑移机制。']
(r / 'transfer_report.md').write_text('\n'.join(lines))
print(json.dumps({'factorial': factorial, 'paired_outcomes': details}, indent=2))
