"""Reaudit archived matched interpolation direct, SDEdit and guide4/exec2."""
import json
from pathlib import Path
import numpy as np
from compare_corrected_rollouts import PROTOCOL
from reference_resampling import interpolate_large_jumps

P = Path(__file__).resolve().parent
R = P / 'reference_turn_baseline_20260926'
O = R / 'adaptive010_direct_edit015_guidance4_comparison_20260928'
EPS = (76, 34, 54, 2)
METHODS = ('direct', 'edit', 'guidance')
LABELS = dict(direct='同规则插值 direct', edit='编辑0.15 / DDIM4 / exec2（9步reference）',
              guidance='传统guidance / DDIM4 / scale50 / guide4 exec2')


def audit(ep, method):
    case = R / f'qualified_comparison/episode_{ep:02d}'
    folders = dict(direct=case / 'direct_adaptive010_m044_mu11',
        guidance=case / 'guide4exec2_adaptive010_ddim4_scale50_m044_mu11',
        edit=R / f'four_reference_edit_noise_ddim_sweep_20260928/episode_{ep:02d}/edit_015_ddim4_seed44')
    f = folders[method]
    s = json.loads((f / 'summary.json').read_text())
    t = json.loads((f / 'trace.json').read_text())
    geo = json.loads((f / 'grasp_metrics.json').read_text())['frames']
    raw = case / 'direct_m044_mu11'
    bt = json.loads((raw / 'trace.json').read_text())
    source = np.load(case / 'reference_full.npz')['hand_target_rad']
    ref, progress = interpolate_large_jumps(source, .1)
    ref = ref[0]
    assert np.array_equal(np.load(f / 'reference_progress.npy'), progress)
    assert s['steps'] == s['intended_steps'] == dict(settle=60, action=len(ref), hold=60)
    assert len(t) == len(geo) == len(ref)+120
    assert s['mass_kg'] == .044 and s['friction'] == 1.1 and s['object_size_multiplier'] == 1
    assert s['settle_target_source'] == 'qpos' and s['native_protocol'] == PROTOCOL
    assert s['reference_interpolation'] == 0 and s['reference_interpolation_threshold'] == .1
    assert s['reference_interpolation_equal_jump'] is None
    assert not s['stop_on_native_failure'] and s['action_limit'] is None
    assert all(s[k] is None for k in ('vertical_scale_switch_angle_deg', 'vertical_scale_after', 'vertical_scale_trigger'))
    with np.load(raw / 'initial_state.npz') as a, np.load(f / 'initial_state.npz') as b:
        assert set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    keys = [k for k in bt[0] if k != 'object_contact_force']
    assert all(all(x[k] == y[k] for k in keys) for x, y in zip(bt[:60], t[:60]))
    fd = float(np.max(np.abs(np.asarray([x['object_contact_force'] for x in bt[:60]])-
                             np.asarray([x['object_contact_force'] for x in t[:60]]))))
    assert fd < 1e-5
    actions = [x for x in t if x['phase'] == 'action']
    cmd = np.asarray([x['command'] for x in actions])
    if method == 'direct':
        assert s['mode'] == 'direct' and np.array_equal(cmd.astype(np.float32), ref.astype(np.float32))
        saved = json.loads((f / 'm044_mu11_result.json').read_text())
    else:
        assert s['mode'] == 'guided' and s['execution_steps'] == 2 and s['prior']['ddim'] == 4
        assert s['prior']['reference_interpolation_threshold'] == .1
        pred = json.loads((f / 'predictions.json').read_text())
        assert [x['reference_index'] for x in pred] == list(range(0, len(ref), 2))
        scale = 0 if method == 'edit' else 50
        assert all(x['guidance_scale'] == scale for x in pred)
        assert s['guidance_scale'] == s['prior']['guidance_scale'] == scale
        if method == 'edit':
            e = s['prior']['editor']
            assert e['algorithm'] == 'reference_initialized_ddim' and e['requested_noise_ratio'] == .15
            assert e['future_reference_steps'] == 9 and e['timesteps'] == [8,5,3,0]
            assert all(x['history_mask_max_error'] == 0 and x['seeds'] == [44] for x in pred)
            saved = json.loads((f / 'analysis.json').read_text())
        else:
            assert s['prior']['guidance_steps'] == 4
            saved = json.loads((f / 'm044_mu11_result.json').read_text())
    flags = []
    for row, g in zip(t, geo):
        assert row['phase'] == g['phase'] and row['index'] == g['index']
        force = float(np.linalg.norm(row['object_contact_force']))
        flags.append((g['mesh_vertex_gap_m'] > .005 and force < .05) or g['mesh_vertex_gap_m'] > .02)
    loss = next((i for i in range(len(flags)-2) if all(flags[i:i+3])), None)
    assert loss is not None and t[loss]['phase'] == 'action'
    longest = current = 0
    for i, (row, g) in enumerate(zip(t, geo)):
        good = (i < loss and row['phase'] == 'action' and row['vertical_error_deg'] <= 30 and
            g['mesh_table_clearance_m'] > .08 and g['near_contact_link_count'] >= 2 and
            g['mesh_vertex_gap_m'] < .008 and np.linalg.norm(row['object_contact_force']) > .1)
        current = current+1 if good else 0
        longest = max(longest, current)
    j = t[loss]['index']
    sep = float(progress[j]); stable = longest >= 30
    assert sep == saved['first_separation']['original_reference_progress' if method == 'edit' else 'reference_action_number']
    assert stable == saved['stable_turn' if method == 'edit' else 'completed_turn']
    assert longest == saved['longest_vertical_contact_steps' if method == 'edit' else 'longest_vertical_contact_control_steps']
    return dict(episode=ep, method=method, folder=str(f), first_separation_reference_progress=sep,
        first_separation_actual_action_step=j+1, longest_vertical_contact_steps=longest, stable_turn=stable,
        original_action_count=source.shape[1], expanded_action_count=len(ref),
        command_reference_rmse_before_separation_rad=float(np.sqrt(np.mean((cmd[:j]-ref[:j])**2))),
        first_native_failure=s['first_native_failure'],
        validation=dict(initial_all_fields_exact=True, settle_state_exact=True,
            settle_contact_force_max_delta_N=fd, interpolation_sequence_exact=True,
            protocol_exact=True, full_tail=True, cached_geometry_metric_recomputed=True))


def main():
    O.mkdir(parents=True, exist_ok=True)
    results = [audit(ep, method) for method in METHODS for ep in EPS]
    rows = []
    for method in METHODS:
        group = [x for x in results if x['method'] == method]
        cells = [f"{x['first_separation_reference_progress']:g} {'✓' if x['stable_turn'] else '✗'}" for x in group]
        rows.append('| ' + LABELS[method] + ' | ' + ' | '.join(cells) + f" | {sum(x['stable_turn'] for x in group)}/4 |")
    report = '''# 相同>0.1 rad插值：direct、0.15编辑、guide4/exec2

本次统一复核12条已有完整仿真存档，没有重新运行仿真。对每条重新读取初态、summary、trace、插值进度和逐帧几何证据，重新计算几何分离与稳定翻转，结果均与原存档一致。

共同条件：各自源帧76/f114、34/f79、54/f110、2/f103，44g、摩擦1.1、尺寸1、固定wrist、30Hz；60步静置、完整reference尾段、60步保持。原始相邻22关节最大绝对跳变严格>0.1 rad时插一个中点。展开后动作数315/560/612/668，三组每episode完全相同。初态全部字段、60步静置运动状态、原生评估协议一致，净接触力数值差<1e-5 N。推理方法沿用10B EMA、固定噪声seed44，环境seed42。

编辑的“4”指DDIM4，沿用原9步reference编辑：过去3步实际目标+未来9步reference，噪声请求0.15，实际0.153397，局部时刻8→5→3→0，无额外MSE guidance，每次执行2步后reference前进2步。没有使用前4步reference+旧计划尾段版本。

传统guidance沿用既有DDIM4、scale50、guide4/exec2：从prior噪声采样，用前4个reference引导，每次执行2步。编辑使用9步reference、传统guidance只引导4步，因此是用户指定的两套配置比较，不能将差异单独归因于编辑算法；两者去噪起点和时间表也不同。Direct逐步执行插值目标，不调用prior；其存档execution_steps=1不改变实际目标序列或30Hz控制时序，也不存在重规划节奏。

下表数字为首次连续3步几何分离时的**原始reference进度**。✓表示分离前连续≥30个实际控制步距竖直≤30°、离桌>8cm、至少2个近接触link、网格间距<8mm、净物体接触力>0.1N。分离条件为“网格间距>5mm且接触力<0.05N，或网格间距>2cm”，连续3步确认。独立几何/接触指标不替代原生failure，不代表完成真机放置任务。

| 方法 | Episode76 | 34 | 54 | 2 | 稳定翻转 |
|---|---:|---:|---:|---:|---:|
''' + '\n'.join(rows) + '''

## 实际物理步与稳定竖直接触

| 方法 | Episode | 首次分离实际动作步 | 最长连续竖直接触步 |
|---|---:|---:|---:|
'''
    for x in results:
        report += f"| {LABELS[x['method']]} | {x['episode']} | {x['first_separation_actual_action_step']} | {x['longest_vertical_contact_steps']} |\n"
    report += '''
编辑相对同规则direct：76的保持进度略短，34相同，54和2更长；稳定翻转从3/4变为4/4，补上54。传统guidance在34、2保持更久，但54仍未稳定翻转。编辑在54只有33步连续竖直接触，刚超过30步标准，不能称为很强的稳定裕度。此结论限四个既有合格初态、单个prior种子，不能估计总体成功率。未录新视频，未写Downloads。
'''
    (O / 'RESULTS.json').write_text(json.dumps(results, indent=2)+'\n')
    (O / 'REPORT.md').write_text(report)
    print('\n'.join(rows))
    print('ALL 12 ARCHIVED RUNS REAUDITED')


if __name__ == '__main__':
    main()
