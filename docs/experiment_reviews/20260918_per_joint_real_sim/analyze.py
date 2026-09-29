"""Offline analysis of archived identical-command real/sim responses. No simulator or robot calls."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
WT = ROOT / '.worktrees/sim-real-holding-comparison'
SRC = WT / 'reports/sim_real_holding_20260917/validated'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(d):
    a = np.abs(d)
    return dict(n=len(d), mae_deg=a.mean(0).tolist(), bias_real_minus_sim_deg=d.mean(0).tolist(),
                median_abs_deg=np.median(a, axis=0).tolist(), p95_abs_deg=np.quantile(a, .95, axis=0).tolist(),
                rmse_deg=np.sqrt(np.mean(d*d, axis=0)).tolist(), max_abs_deg=a.max(0).tolist(),
                overall_joint_time_mae_deg=float(a.mean()),
                mean_frame_max_deg=float(a.max(1).mean()))


def main():
    manifest = json.loads((SRC / 'manifest.json').read_text())
    groups, runs, checks, hashes = {}, {}, {}, {}
    names = None
    for m in manifest:
        if m['mode'] != 'replay':
            continue
        source = WT / m['source']
        rows = [json.loads(s) for s in source.read_text().splitlines()]
        session = next(r for r in rows if r['event'] == 'session' and r['stage'] == 'inference')
        if names is None:
            names = session['policy_joint_names']
        assert names == session['policy_joint_names']
        rows = [r for r in rows if r.get('stage') == 'inference' and not r.get('synthetic')]
        states = {r['last_command_id']: r for r in rows if r['event'] == 'state' and r.get('kind') == 'policy'}
        cmds = [r for r in rows if r['event'] == 'command' and r.get('kind') == 'policy' and r['command_id'] in states]
        z = np.load(SRC / m['filename'])
        n = m['steps']
        assert n == len(cmds) == len(z['time'])
        assert z['action'].shape == z['real_qpos_end'].shape == z['qpos_end'].shape == (n, 22)
        raw_a = np.asarray([r['target_policy_rad'] for r in cmds])
        raw_q = np.asarray([states[r['command_id']]['qpos_policy_rad'] for r in cmds])
        assert np.array_equal(raw_a, z['real_action']) and np.array_equal(raw_q, z['real_qpos_end'])
        command_error = np.max(np.abs(z['action'] - raw_a))
        assert command_error < 1e-6
        assert np.isfinite(z['qpos_end']).all() and np.all(np.diff(z['time']) > 0)
        initial = np.asarray(next(r for r in rows if r['event'] == 'inference_input')['observation'])[0, -1, :22]
        assert np.max(np.abs(initial - m['initial_q'])) < 1e-6
        t0 = cmds[0]['latest_state_read_end_ns']
        raw_t = np.asarray([(states[c['command_id']]['read_end_ns'] - t0) * 1e-9 for c in cmds])
        time_error = np.max(np.abs(raw_t - z['time']))
        assert time_error <= m['physics_dt'] / 2 + 1e-8
        d = np.rad2deg(raw_q - z['qpos_end'])
        # Same-command residual identity; do not subtract absolute tracking-error averages.
        np.testing.assert_allclose(d, np.rad2deg((raw_a-z['qpos_end'])-(raw_a-raw_q)), atol=1e-12)
        group = ('bulb' if m['bulb'] else 'empty') + ('_hold' if m['hold'] else '_nohold')
        for key in [group, 'bulb_all' if m['bulb'] else 'empty_all']:
            groups.setdefault(key, []).append(d)
        runs[m['name']] = dict(**stats(d), seconds=float(z['time'][-1]), group=group,
                              first_2s=stats(d[z['time'] <= 2]), after_2s=stats(d[z['time'] > 2]),
                              real_target_tracking_mae_deg=np.rad2deg(np.abs(raw_a-raw_q).mean(0)).tolist(),
                              sim_target_tracking_mae_deg=np.rad2deg(np.abs(raw_a-z['qpos_end']).mean(0)).tolist())
        checks[m['name']] = dict(samples=n, command_max_diff_rad=float(command_error),
                                max_time_quantization_error_s=float(time_error), raw_jsonl_matches=True)
        hashes[str(source.relative_to(ROOT))] = sha(source)
        hashes[str((SRC / m['filename']).relative_to(ROOT))] = sha(SRC / m['filename'])

    # Check against the simulator's configured joint order, without importing its runtime.
    config = Path('/home/carus/Program/dex-controller/maniptrans_envs/lib/envs/dexhands/sharpa.py')
    tree = ast.parse(config.read_text())
    orders = [ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Attribute) and t.attr == 'dof_names' for t in n.targets)
              and isinstance(n.value, ast.List)]
    assert names in orders
    hashes[str(config)] = sha(config)
    aggregates = {g: stats(np.concatenate(ds)) for g, ds in groups.items()}
    result = dict(definition='delta = real_qpos_end - sim_replay_qpos_end; angular units degrees',
                  source_branch='origin/experiments/real-holding-20260917', source_commit='af4dd93d9a68ad58e844d8a8e474c16d72ade001',
                  replay_directory=str(SRC), joint_names=names, groups=aggregates, runs=runs,
                  checks=checks, sha256=hashes,
                  limitations=['Historical isolated position-drive replay; not native xjz DP task evaluation. No new rollout performed.',
                               'Real wrist/object poses and initial velocities were not measured; sim initial velocities zero.',
                               'Bulb groups include whole recordings, including changing/lost contact. No failure cutoff or holding-time comparison.',
                               'Time-weighting: equal completed-cycle samples; samples serially correlated, six runs only.',
                               'Read-end timestamps approximate sensor observation time; internal hardware latency unknown.'])
    (OUT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    empty, bulb = aggregates['empty_all'], aggregates['bulb_all']
    lines = ['# 同一命令下的逐关节实机—仿真偏差', '',
             '取真实下发命令的历史 replay，逐完成动作周期对齐，计算 Δ=q_real−q_sim。MAE=mean(|Δ|)，正的有符号偏差表示实机角度更大；P95是绝对差的时间分位数，不是置信区间。不能用两边各自跟踪误差的绝对值均值相减代替本指标。', '',
             f'数据：空手2轮共{empty["n"]}个周期，初始放灯泡4轮共{bulb["n"]}个周期；全部22个手关节。平均每关节绝对差：空手{empty["overall_joint_time_mae_deg"]:.3f}°，灯泡组{bulb["overall_joint_time_mae_deg"]:.3f}°。按周期等权合并。', '',
             '## 逐关节结果（度）', '',
             '| 关节（省略right_） | 空手MAE | 空手P95 | 空手有符号偏差 | 灯泡MAE | 灯泡P95 | 灯泡有符号偏差 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for j, name in enumerate(names):
        v = [empty['mae_deg'][j], empty['p95_abs_deg'][j], empty['bias_real_minus_sim_deg'][j],
             bulb['mae_deg'][j], bulb['p95_abs_deg'][j], bulb['bias_real_minus_sim_deg'][j]]
        lines.append('| '+name.removeprefix('right_')+' | '+' | '.join(f'{x:.2f}' for x in v)+' |')
    lines += ['', '## 各轮与模式敏感性', '',
              '| 运行 | 周期数 | 时长s | 全关节MAE | 前2秒MAE | 2秒后MAE |', '|---|---:|---:|---:|---:|---:|']
    for name, r in runs.items():
        lines.append(f'| {name} | {r["n"]} | {r["seconds"]:.2f} | {r["overall_joint_time_mae_deg"]:.2f}° | {r["first_2s"]["overall_joint_time_mae_deg"]:.2f}° | {r["after_2s"]["overall_joint_time_mae_deg"]:.2f}° |')
    lines += ['', 'holding表示推理等待时下发保持目标；这里每个模式都与其自己的真机命令完整序列对齐，包含中间的holding命令。各轮接触条件不同，不据此推断holding的因果影响。', '',
              '| 关节 | 空手无hold MAE | 空手hold MAE | 灯泡无hold MAE | 灯泡hold MAE | 灯泡4轮MAE范围 |', '|---|---:|---:|---:|---:|---:|']
    for j, name in enumerate(names):
        v=[aggregates[g]['mae_deg'][j] for g in ['empty_nohold','empty_hold','bulb_nohold','bulb_hold']]
        rv=[r['mae_deg'][j] for r in runs.values() if r['group'].startswith('bulb')]
        lines.append('| '+name.removeprefix('right_')+' | '+' | '.join(f'{x:.2f}°' for x in v)+f' | {min(rv):.2f}–{max(rv):.2f}° |')
    lines += ['', '## 解释与限制', '',
              '- 空手组是较少受物体接触混杂的执行响应基线，仍包含驱动、时延、摩擦、关节约束、腕姿假设及读数时序差异。',
              '- 灯泡组只表示这些记录对应的同命令响应差，不能当作精确硬件标定：真实腕部和灯泡位姿未记录，仿真摆放由静态验证选择。轨迹内接触可能改变或丢失，因此“灯泡组”不等于全程同样接触。',
              '- 本次仅离线分析已有replay文件，没有新跑仿真。历史replay使用独立位置驱动物理回放（1/600秒），关闭任务重置/随机化并直接执行已下发目标；不是xjz_test原生DP评估。未重新采用旧5cm初始化位移判据，也不比较保持时长。不能把数值直接推广到当前原生随机化评估。',
              '- 每周期后的实测角与仿真角对齐；日志read_end不是已校准的传感器曝光/编码器时间。原样命令核验误差小于1e-6rad；累计时间量化误差不超过0.8334ms。',
              '- 这是动态响应差，不是同一静态目标充分收敛后的零点偏差。不能将MAE直接加到命令上；有符号均值也可能随姿态、速度、接触而变化。',
              '- 六轮数据有限、样本有时间相关性；未将周期数当作独立实验数构造置信区间。若要估计可用于补偿的偏差，需要多姿态静态保持、正反向扫动、不同速度的同命令实机/仿真配对记录。', '',
              '## 来源与复算', '',
              f'- [历史回放说明]({SRC}/README.md)',
              f'- [原样回放实现]({WT}/eval/compare_real_sim_holding.py)',
              '- [完整数值与文件哈希](results.json)', '- [分析脚本](analyze.py)',
              '- 复算：`/home/carus/miniforge3/envs/dp/bin/python '+str(OUT/'analyze.py')+'`', '',
              '![逐关节同命令偏差](joint_gap.png)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
    y = np.arange(22)
    for ax, (key, title) in zip(axes, [('empty_all','Empty hand (2 runs)'), ('bulb_all','Initially holding bulb (4 runs)')]):
        s = aggregates[key]
        ax.barh(y, s['mae_deg'], height=.64, color='#3577ad', label='Mean absolute difference')
        ax.scatter(s['p95_abs_deg'], y, marker='|', s=100, color='#bf532c', label='95th percentile')
        ax.set_title(title)
        ax.set_xlabel('Absolute real - sim joint angle (degrees)')
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=8)
    axes[0].set_yticks(y, [n.removeprefix('right_') for n in names]); axes[0].invert_yaxis()
    fig.suptitle('Identical recorded commands, aligned end-of-cycle measurements', fontsize=14)
    fig.text(.5,.015,'Archived replay only. Bulb/wrist poses are approximations; these are dynamic gaps, not fixed offsets.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0,.035,1,.965)); fig.savefig(OUT/'joint_gap.png', dpi=170); plt.close(fig)
    print(json.dumps(dict(groups={k:dict(n=v['n'],mae_deg=v['overall_joint_time_mae_deg']) for k,v in aggregates.items()}, checks=checks), indent=2))


if __name__ == '__main__':
    main()
