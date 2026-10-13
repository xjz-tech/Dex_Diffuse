"""Compare completed reference-prefix editing runs against archived baselines."""
import json
from run_four_reference4_free5_sweep import O, R, EPISODES, RATIOS, STEPS, folder
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
            for version, getter in [('原9步 reference', old_result), ('4步 reference + 自由5步', lambda ep, r, s: json.loads((folder(ep, r, s) / 'analysis.json').read_text()))]:
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
                    total_network_calls=steps + (4 if version.startswith('4步') else 0),
                    stable_turn_count=wins, episodes=episodes))
    report = '''# 只编辑前4步 reference，后5步由 prior 自由生成

## 方法

网络的12步动作窗口为：过去3步实际下发目标 + 近期4步reference + 自由5步。模型始终只接收4个未来reference；第5个以后的数据集目标不进入这次预测。前4步提供可修改的动作意图，非最终输出硬约束。执行前2步后，reference在插值后的序列上前进2步，再重新规划。

首先从t=99的标准高斯噪声开始，用99→75→50→25四次网络调用补全自由尾段，每个时刻把过去3步和reference4步固定为相应的前向加噪值。最后一次从25降到编辑起点。到编辑起点后只继续固定过去3步，reference4步和自由5步共同去噪。后5步在低噪声起点已是条件补全的带噪轨迹，不是未经匹配的纯高斯噪声。

局部DDIM4/6分别另需4/6次调用，总共8/10次。旧9步reference编辑只需4/6次，因此这次是方案效果对照，不是相同算力的单变量比较。补全阶段需要前4步硬条件，局部编辑阶段允许其改变；没有MSE梯度引导，scale=0。

| 请求噪声比 | 实际噪声比 | 编辑起点 | DDIM4局部时刻 | DDIM6局部时刻 |
|---|---:|---:|---|---|
| 0.10 | 0.105623 | 5 | 5→3→2→0 | 5→4→3→2→1→0 |
| 0.15 | 0.153397 | 8 | 8→5→3→0 | 8→6→5→3→2→0 |
| 0.20 | 0.201754 | 11 | 11→7→4→0 | 11→9→7→4→2→0 |

## 可比条件与核验

沿用已由原始reference验收的四条起点76/f114、34/f79、54/f110、2/f103。每条采用自身初态，44g、摩擦1.1、尺寸1、固定wrist、30Hz，60步静置后执行完整reference尾段并保持60步。最大单关节相邻跳变严格>0.1 rad时插一个中点。原始/展开目标数分别为239/315、441/560、462/612、492/668。环境seed42、prior噪声seed44，其余原生评估协议不变。

本次新跑24个4+5编辑rollout。原9步编辑、原速reference、传统prior行复用同条件存档，没有声称重新运行这些基线。传统prior行使用DDIM4、scale25、guide2/exec1；编辑各行使用exec2，故不能将差异单独归因于算法。原速reference不插值。

验证包括：未来第5步后的reference修改不影响输入窗口；拒收9步reference；所有补全/编辑时刻的已知部分正确匹配训练噪声；前4步确实可修改；DDIM显式跨时刻转移解析核验；seed复现。24组均检查完整初态所有字段、60步静置、首实际控制步存档、reference推进和完整尾段执行。相对各自原速reference，初态和静置运动状态一致；净接触力浮点差<1e-5 N。详见validation.json、RESULTS.json和各组analysis.json。

## 结果

数字为首次连续3步满足几何分离条件时的**原始reference动作进度**，不是实际物理步数。✓表示分离前连续至少30个实际动作步满足竖直偏差≤30°、离桌>8cm、至少2个近接触link、网格间距<8mm、净物体接触力>0.1N。分离判定为连续3步满足“网格间距>5mm且净物体接触力<0.05N，或网格间距>2cm”。这些独立接触/几何指标不替代原生failure，也不证明完成真机放置。

| 方法 / 噪声 / 局部DDIM | Episode76 | 34 | 54 | 2 | 稳定翻转 |
|---|---:|---:|---:|---:|---:|
''' + '\n'.join(rows) + '''

这里只是4个合格起点、单个prior种子上的方案对照，不能据此估计泛化成功率。后5步自由生成仍会通过整段网络预测影响前4步；自由度增大不保证翻转和接触更好。各组的实际分离控制步、连续竖直接触步数、动作误差和推理用时保存在SUMMARY.json。未录视频，未写Downloads。
'''
    (O / 'REPORT.md').write_text(report)
    (O / 'SUMMARY.json').write_text(json.dumps(compact, indent=2) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
