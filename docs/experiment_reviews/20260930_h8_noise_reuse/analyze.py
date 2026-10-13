"""Post-run paired physical/outcome and plan-continuity audit; no simulation."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from experiment import (HERE, SOURCE, ARMS, SEEDS, PHYSICS, EPISODES, baseline,
                        destination, check_summary, read, write, sha, episodes_for)
sys.path.insert(0, str(SOURCE))
import analyze_h8_cross_model_comparison as legacy_audit


def noise_audit(predictions, rho):
    noise = np.asarray([p['raw_noise'] for p in predictions], dtype=np.float32)
    xi = np.asarray([p['innovation'] for p in predictions], dtype=np.float32)
    assert noise.shape == xi.shape == (len(predictions), 8, 22)
    assert np.isfinite(noise).all() and np.isfinite(xi).all()
    assert np.array_equal(noise[0], xi[0])
    assert np.array_equal(noise[:, :3], xi[:, :3])
    assert np.array_equal(noise[:, 6:], xi[:, 6:])
    expected = (np.float32(rho) * noise[:-1, 5:8]
                + np.float32(math.sqrt(1 - rho ** 2)) * xi[1:, 3:6])
    np.testing.assert_allclose(noise[1:, 3:6], expected, rtol=1e-6, atol=1e-6)
    if rho == 0:
        assert np.array_equal(noise, xi)
    return dict(raw_noise_mean=float(noise.mean()), raw_noise_variance=float(noise.var()),
                overlap_empirical_correlation=float(np.corrcoef(
                    noise[1:, 3:6].ravel(), noise[:-1, 5:8].ravel())[0, 1]))


def continuity(predictions, cutoff):
    """Only aligned, valid, pre-separation windows; no end padding in metrics."""
    overlap, residual, boundary = [], [], []
    for prev, curr in zip(predictions, predictions[1:]):
        j = curr['reference_index']
        assert j == prev['reference_index'] + 2
        n = min(3, curr['valid_future_steps'], max(0, cutoff - j))
        if n <= 0:
            continue
        a, b = np.asarray(prev['full_plan_rad']), np.asarray(curr['full_plan_rad'])
        ra, rb = np.asarray(prev['reference_rad']), np.asarray(curr['reference_rad'])
        assert np.array_equal(ra[2:2+n], rb[:n])
        overlap.extend((b[:n] - a[2:2+n]).ravel())
        residual.extend(((b[:n] - rb[:n]) - (a[2:2+n] - ra[2:2+n])).ravel())
        boundary.extend((b[0] - a[1]).ravel())
    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values)))) if values else None
    return dict(overlap_plan_rmse_rad=rmse(overlap),
                overlap_residual_rmse_rad=rmse(residual),
                executed_boundary_rmse_rad=rmse(boundary),
                aligned_joint_values=len(overlap))


def load_case(seed, physics, episode, arm):
    folder = destination(seed, physics, episode, arm)
    assert read(folder / 'complete.json')['manifest_sha256'] == sha(HERE / 'manifest.json')
    s = read(folder / 'summary.json')
    check_summary(s, seed, physics, episode)
    assert s['prior']['editor']['rho'] == ARMS[arm]
    assert s['prior']['editor']['noise_protocol'] == 'time_aligned_raw_gaussian_v1'
    # Reuse the exact archived physical metric and initialization audit. This
    # adapter only redirects destination; it does not alter criteria or raw refs.
    original = legacy_audit.destination
    legacy_audit.destination = lambda *_: folder
    try:
        a = legacy_audit.audit(seed, 'h8', physics, episode, 'edit015')
    finally:
        legacy_audit.destination = original
    predictions = read(folder / 'predictions.json')
    nstats = noise_audit(predictions, ARMS[arm])
    trace = read(folder / 'trace.json')
    old_folder = baseline(seed, physics, episode)
    with np.load(old_folder / 'initial_state.npz') as old, np.load(folder / 'initial_state.npz') as new:
        assert set(old.files) == set(new.files) and all(np.array_equal(old[k], new[k]) for k in old.files)
    old_trace = read(old_folder / 'trace.json')
    for before, after in zip(old_trace[:60], trace[:60]):
        for key in ('q', 'object_pose', 'executed_target', 'relative_position', 'vertical_error_deg'):
            assert before[key] == after[key]
    for before, after in zip(old_trace[60:62], trace[60:62]):
        for key in ('q', 'object_pose', 'executed_target', 'relative_position'):
            np.testing.assert_allclose(before[key], after[key], rtol=0, atol=1e-6)
    actions = [t for t in trace if t['phase'] == 'action']
    assert len(predictions) == (len(actions) + 1) // 2
    for pred in predictions:
        j = pred['reference_index']
        for k in range(min(2, len(actions) - j)):
            assert np.array_equal(pred['full_plan_rad'][k], actions[j+k]['command'])
    separation = a['first_separation_action_step']
    cutoff = (separation - 1 if a['first_separation_phase'] == 'action' else len(actions))
    cmd = np.asarray([t['command'] for t in actions])
    issued = np.asarray([t['executed_target'] for t in actions])
    result = dict(a, arm=arm, retained_steps=separation or len(actions),
        action_steps=len(actions), pre_separation_steps=cutoff,
        action_tail_censored=a['first_separation_phase'] != 'action',
        executed_target_command_rmse_rad=float(np.sqrt(np.mean((issued - cmd) ** 2))),
        predicted_x0_clip_fraction_mean=float(np.mean([p['predicted_x0_clip_fraction'] for p in predictions])),
        **nstats, **continuity(predictions, cutoff))
    write(folder / 'noise_reuse_audit.json', result)
    return result, predictions


def average(values):
    valid = [v for v in values if v is not None]
    return statistics.mean(valid) if valid else None


def summarize(rows):
    result = []
    for arm in ['L_archived', *ARMS]:
        group = [r for r in rows if r['arm'] == arm]
        result.append(dict(arm=arm, count=len(group),
            stable_turns=sum(r['stable_turn'] for r in group),
            ordinary_turns=sum(r['ordinary_turn'] for r in group),
            mean_retained_steps=average(r['retained_steps'] for r in group)))
    return result


def main():
    manifest = read(HERE / 'manifest.json')
    rows = list(manifest['baseline_rows'])
    plans, lookup = {}, {}
    for seed in SEEDS:
        for physics in PHYSICS:
            for episode in episodes_for(physics):
                for arm in ARMS:
                    key = seed, physics, episode, arm
                    r, predictions = load_case(*key)
                    rows.append(r)
                    plans[key], lookup[key] = predictions, r
                # Paired arms must consume identical innovations, even rho=1.
                b = plans[seed, physics, episode, 'B']
                for arm in ARMS:
                    x = plans[seed, physics, episode, arm]
                    assert len(x) == len(b)
                    assert all(p['innovation'] == q['innovation'] for p, q in zip(b, x))
                    assert b[0]['full_plan_rad'] == x[0]['full_plan_rad']
                # Audit historical first prediction without rerunning the old
                # fixed-slot protocol: same imported state/reference/first noise.
                archived_actions = [r for r in read(baseline(seed, physics, episode) / 'trace.json')
                                    if r['phase'] == 'action']
                np.testing.assert_allclose(np.asarray(b[0]['full_plan_rad'][:2]),
                    np.asarray([r['command'] for r in archived_actions[:2]]), rtol=0, atol=1e-6)
    assert len(rows) == 275
    pairs = []
    idx = {(r['seed'], r['physics'], r['episode'], r['arm']): r for r in rows}
    for seed in SEEDS:
        for physics in PHYSICS:
            for episode in episodes_for(physics):
                for arm in ARMS:
                    for control in ('B', 'L_archived'):
                        if arm == control:
                            continue
                        key = seed, physics, episode
                        a, b = idx[key + (arm,)], idx[key + (control,)]
                        pair = dict(seed=seed, physics=physics, episode=episode,
                            arm=arm, control=control,
                            stable_delta=int(a['stable_turn']) - int(b['stable_turn']),
                            retained_steps_delta=a['retained_steps'] - b['retained_steps'])
                        if control == 'B':
                            cutoff = min(a['pre_separation_steps'], b['pre_separation_steps'])
                            ac, bc = continuity(plans[key + (arm,)], cutoff), continuity(plans[key + (control,)], cutoff)
                            for metric in ('overlap_plan_rmse_rad', 'executed_boundary_rmse_rad'):
                                pair[metric + '_delta_common_prefix'] = (
                                    None if ac[metric] is None else ac[metric] - bc[metric])
                            pair['common_pre_separation_steps'] = cutoff
                        pairs.append(pair)
    write(HERE / 'results.json', dict(rows=rows, paired=pairs,
        overall=summarize(rows), by_seed={s: summarize([r for r in rows if r['seed'] == s]) for s in SEEDS},
        by_physics={p: summarize([r for r in rows if r['physics'] == p]) for p in PHYSICS},
        by_episode={e: summarize([r for r in rows if r['episode'] == e]) for e in EPISODES}))
    fields = sorted(set().union(*(r.keys() for r in pairs)))
    with (HERE / 'paired.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)
    lines = ['# h8 raw Gaussian 噪声复用结果', '',
        '5 个噪声 seed × 11 个direct合格条件；每组 55 条。ep2×170g/2.0未通过direct资格，预先排除。', '',
        '| 组 | 普通翻转 | 稳定翻转 ≥30步 | 平均观察动作步数 |',
        '|---|---:|---:|---:|']
    for r in summarize(rows):
        lines.append(f'| {r["arm"]} | {r["ordinary_turns"]}/55 | {r["stable_turns"]}/55 | {r["mean_retained_steps"]:.2f} |')
    lines += ['', '| 比较 | 稳定翻转胜/平/负 | 平均步数差 | 配对共同有效前缀 overlap RMSE差 |',
              '|---|---:|---:|---:|']
    for arm in ARMS:
        for control in ('B', 'L_archived'):
            group = [r for r in pairs if (r['arm'], r['control']) == (arm, control)]
            if not group:
                continue
            wins = sum(r['stable_delta'] > 0 for r in group)
            losses = sum(r['stable_delta'] < 0 for r in group)
            mech = average(r.get('overlap_plan_rmse_rad_delta_common_prefix') for r in group)
            label = '未存完整plan' if mech is None else f'{mech:+.6f} rad'
            lines.append(f'| {arm} − {control} | {wins}/{55-wins-losses}/{losses} | '
                         f'{average(r["retained_steps_delta"] for r in group):+.2f} | {label} |')
    lines += ['', '逐seed/物理参数/episode及原生failure见 results.json，逐配对见 paired.csv。',
        '步数沿用历史一基索引与尾段截尾口径；另保存零基分离前步数及截尾标志。',
        'L是历史固定槽位噪声，不是独立重采B。overlap变化只是不连续性代理，包含闭环观测变化；',
        '不能仅据更平滑断言消除了多模态切换，也不能用本实验代替受控扰动恢复能力验证。',
        '55条不是55个独立初态。几何分离和原生failure分别报告；数值结果仍需核看实际录像。', '']
    (HERE / 'REPORT.md').write_text('\n'.join(lines))
    print('Audited 220 new runs, compared with 55 archived records.')


if __name__ == '__main__':
    main()
