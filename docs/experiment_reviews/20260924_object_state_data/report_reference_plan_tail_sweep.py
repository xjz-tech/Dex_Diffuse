"""Compare completed reference-prefix editing runs against archived baselines."""
import json
from run_four_reference_plan_tail_sweep import O, R, EPISODES, RATIOS, STEPS, folder
from report_all_reference_edit_noise_ddim_sweep import result as old_result


def main():
    status = json.loads((O / 'run_status.json').read_text())
    results = json.loads((O / 'RESULTS.json').read_text())
    assert len(status) == len(results) == 24
    assert all(x[-1] in ('complete', 'existing') for x in status)
    old_baselines = json.loads((R / 'four_episode_reference_edit_adaptive010_video_20260928/comparison.json').read_text())
    compact, rows = [], []
    for key, label in [('raw', '原速 reference'), ('guide', '传统 prior：DDIM4 / scale25 / guide2 exec1')]:
        group = [next(x for x in old_baselines if x['method'] == key and x['episode'] == ep) for ep in EPISODES]
        cells = [f"{x['first_separation_reference_progress']:g} {'✓' if x['stable_turn'] else '✗'}" for x in group]
        rows.append('| ' + label + ' | ' + ' | '.join(cells) + f" | {sum(x['stable_turn'] for x in group)}/4 |")
    for ratio in RATIOS:
        for steps in STEPS:
            for version, getter in [('原9步 reference', old_result), ('4步 reference + 旧计划尾段', lambda ep, r, s: json.loads((folder(ep, r, s) / 'analysis.json').read_text()))]:
                group = [getter(ep, ratio, steps) for ep in EPISODES]
                cells, episodes = [], []
                for ep, x in zip(EPISODES, group):
                    sep = x['first_separation']
                    progress = None if sep is None else sep['original_reference_progress']
                    cells.append(('未分离' if progress is None else f'{progress:g}') + (' ✓' if x['stable_turn'] else ' ✗'))
                    episodes.append(dict(episode=ep, first_separation_reference_progress=progress,
                        physical_action_step=None if sep is None else sep['control_step'],
                        stable_turn=x['stable_turn'], longest_vertical_contact_steps=x['longest_vertical_contact_steps'],
                        command_reference_rmse_before_separation_rad=x['command_reference_rmse_before_separation_rad'],
                        mean_inference_seconds=x['mean_inference_seconds']))
                wins = sum(x['stable_turn'] for x in group)
                rows.append(f'| {version} / {ratio:.2f} / DDIM{steps} | ' + ' | '.join(cells) + f' | {wins}/4 |')
                compact.append(dict(version=version, noise_ratio=ratio, local_ddim_steps=steps,
                    total_network_calls=steps,
                    stable_turn_count=wins, episodes=episodes))
    report = """# 方案B：前4步reference + 上一轮计划的对齐尾段

## 方法

12步动作窗口由过去3步实际下发目标、前4步reference和5步初始化尾段组成。第一轮尾段重复当前第4个reference；后续每轮只执行2步，reference在插值后的序列上前进2步。上一轮预测9步计划先对齐到当前时刻，再抽取尾段。因此，新窗口第4、5、6、7、8号位置对应上一轮计划第6、7、8、8、8号动作（零起始索引），最后两格以旧计划末项补齐。不能直接拼旧2～6号，那会让尾段时间错位4步。

整个12步窗口一起前向加噪，再用局部DDIM4或6去噪一次。每个去噪时刻仅把history3固定为其对应的前向加噪值，最后恢复干净history。前4个reference和后5个尾段均可被prior修改；尾段只作SDEdit初值，不是硬约束，也不是严格无条件生成。观察history4×66保持干净。无额外MSE guidance，scale=0。

每轮仅4/6次网络调用，与旧9步reference编辑一致，没有额外补全调用。实际输出保存完整9步，其中前2步交给仿真，完整计划留作下一轮初始化。模型这轮不读取第5个以后的数据集reference。

| 请求噪声比 | 实际噪声比 | 编辑起点 | DDIM4局部时刻 | DDIM6局部时刻 |
|---|---:|---:|---|---|
| 0.10 | 0.105623 | 5 | 5→3→2→0 | 5→4→3→2→1→0 |
| 0.15 | 0.153397 | 8 | 8→5→3→0 | 8→6→5→3→2→0 |
| 0.20 | 0.201754 | 11 | 11→7→4→0 | 11→9→7→4→2→0 |

## 可比条件与核验

沿用已由原始reference验收的四条起点76/f114、34/f79、54/f110、2/f103。每条采用自身初态，44g、摩擦1.1、尺寸1、固定wrist、30Hz，60步静置后执行完整reference尾段并保持60步。原始相邻目标22个关节最大绝对跳变严格>0.1 rad时插一个中点。原始/展开目标数分别为239/315、441/560、462/612、492/668。环境seed42、prior固定噪声seed44，其余原生评估协议不变。

本次新跑24个方案B rollout。原9步reference编辑、原速reference、传统prior行复用同条件已核验存档，没有声称重新运行这些基线。传统prior为DDIM4、scale25、guide2/exec1；编辑各行exec2。原速reference不插值。刚才额外4次网络调用补全尾段的试验已因用户切换方案停止，部分结果单独存档，不混入本表。

验证包括：明确的时间索引哨兵检查、拒收9步reference及错误的cache进度/episode、未来第5步后的reference不影响本轮输入、history噪声匹配、实际网络调用数、固定seed复现、原编辑器默认输出未改变。24组逐轮核验尾段与上一轮完整计划指定位置逐值相同、实际下发command与计划前2步逐值相同。初始化所有字段和60步静置运动状态与同episode原始reference一致，净接触力浮点差<1e-5 N，记录含首实际物理步和完整尾段。详见validation.json、RESULTS.json和各组analysis.json。

## 结果

数字为首次连续3步满足几何分离条件时的**原始reference动作进度**，不是实际物理步数。✓表示分离前连续至少30个实际动作步满足竖直偏差≤30°、离桌>8cm、至少2个近接触link、网格间距<8mm、净物体接触力>0.1N。分离判定为连续3步满足“网格间距>5mm且净物体接触力<0.05N，或网格间距>2cm”。这些独立接触/几何指标不替代原生failure，不证明完成真机放置。

| 方法 / 噪声 / 局部DDIM | Episode76 | 34 | 54 | 2 | 稳定翻转 |
|---|---:|---:|---:|---:|---:|
""" + '\n'.join(rows) + """

只覆盖4个合格起点、单个prior种子，不能据此估计泛化成功率。各组实际分离控制步、连续竖直接触步数、动作误差和推理用时见SUMMARY.json。没有录视频或写入Downloads。
"""
    diagnostics = json.loads((O / 'TRACE_DIAGNOSTICS.json').read_text())['results']
    selected = [x for x in diagnostics if x['noise_ratio'] == .15 and x['ddim_steps'] == 4]
    assert len(selected) == 4
    diagnostic_rows = []
    for x in selected:
        end = x['tail_diagnostics'][-1]
        assert end['reference_index'] == 58
        diagnostic_rows.append(f"| {x['episode']} | {x['old_command_reference_rmse_rad']:.4f} | {x['new_command_reference_rmse_rad']:.4f} | {end['reference_tail_boundary_max_abs_rad']:.4f} |")
    report += '''
## 观察与局限

方案B六种配置均为1/4稳定翻转，只有episode2通过；原9步reference对应配置为3/4、3/4、4/4、3/4、2/4、4/4。Episode2部分配置保持进度延长，例如0.10/DDIM4从184到209.5、0.10/DDIM6从153到205，但不能补偿其他episode失去稳定翻转。

0.15/DDIM4的共同前60个控制步诊断如下。动作RMSE跨这60步的全部22关节计算；拼接差是reference第4步与初始化tail第1步之间的最大单关节差，取零起始展开进度58，属于**规划输入窗口内部**，不等于实际下发的相邻两步跳变。

| Episode | 原9步reference：command RMSE(rad) | 方案B：command RMSE(rad) | 进度58拼接最大单关节差(rad) |
|---|---:|---:|---:|
''' + '\n'.join(diagnostic_rows) + '''

第一轮hold初始化的拼接差为0；后续旧计划尾段逐渐偏离当前近期reference，且近期实际输出误差增大。该观察提示拼接不连续可能影响整段去噪，但尚未通过单独固定尾段/平滑拼接的消融证明因果。不能据此断言所有“只给4步reference”的方法都无效。尾段本来没有跟踪远期reference的义务，离开远期reference本身也不是失败判据；任务结果仍以翻转及接触分析为准。
'''
    (O / 'REPORT.md').write_text(report)
    (O / 'SUMMARY.json').write_text(json.dumps(compact, indent=2) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
