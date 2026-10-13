"""Compare no-interpolation .15/DDIM4 editing with existing matched baselines."""
import json
from run_four_reference_edit015_nointerp import O, R, EPISODES, folder
from report_all_reference_edit_noise_ddim_sweep import result as interpolated_result


def main():
    status = json.loads((O / 'run_status.json').read_text())
    results = json.loads((O / 'RESULTS.json').read_text())
    assert len(status) == len(results) == 4
    archived = json.loads((R / 'four_episode_reference_edit_adaptive010_video_20260928/comparison.json').read_text())
    baseline = [next(x for x in archived if x['method'] == 'raw' and x['episode'] == ep) for ep in EPISODES]
    rows = ['| 原速 reference | ' + ' | '.join(f"{x['first_separation_reference_progress']:g} {'✓' if x['stable_turn'] else '✗'}" for x in baseline) + f" | {sum(x['stable_turn'] for x in baseline)}/4 |"]
    details, groups = [], []
    for label, getter in [('原编辑0.15/DDIM4，>0.1 rad插值', lambda ep: interpolated_result(ep, .15, 4)),
                          ('原编辑0.15/DDIM4，不插值', lambda ep: next(x for x in results if x['episode'] == ep))]:
        group = [getter(ep) for ep in EPISODES]
        cells = []
        for ep, x in zip(EPISODES, group):
            sep = x['first_separation']
            cells.append(('未分离' if sep is None else f"{sep['original_reference_progress']:g}") + (' ✓' if x['stable_turn'] else ' ✗'))
            details.append(dict(method=label, episode=ep, first_separation=sep,
                stable_turn=x['stable_turn'], longest_vertical_contact_steps=x['longest_vertical_contact_steps'],
                min_vertical_error_before_separation=x['min_vertical_error_before_separation'],
                command_reference_rmse_before_separation_rad=x['command_reference_rmse_before_separation_rad'],
                first_native_failure=x['first_native_failure']))
        rows.append('| ' + label + ' | ' + ' | '.join(cells) + f" | {sum(x['stable_turn'] for x in group)}/4 |")
        groups.append((label, group))
    report = '''# 原9步reference编辑：0.15 / DDIM4，取消插值

本轮新跑episode76、34、54、2各一次，不插值。原速reference与>0.1 rad插值编辑行复用相同条件的已核验存档。只有本轮取消插值，不使用上一轮的4步reference+旧计划尾段。

过去3步实际下发动作 + 未来9步原始reference，一起按原方案加噪；观察history4×66保持干净，每个去噪时刻固定正确加噪的history3，未来9步均可被编辑。噪声系数请求0.15、实际0.153397，起点t8，局部DDIM时刻8→5→3→0，eta0、无MSE guidance。每次执行前2步，原始reference也前进2步。仍为每次4个网络调用，没有额外补全。

相同各自初态76/f114、34/f79、54/f110、2/f103，44g、摩擦1.1、尺寸1、固定wrist、30Hz、环境seed42、prior固定噪声seed44。60步静置、完整reference尾段、60步保持及原生评估协议不变。不插值时实际动作数239/441/462/492，原插值版315/560/612/668。取消插值也取消了中点带来的额外物理执行时间；9步窗口覆盖的原始动作范围相应改变，本对照衡量整套取消插值后的效果。

所有新运行均核验：初态所有字段与各自原始reference完全相同，60步静置运动状态完全相同，净接触力浮点差<1e-5 N；服务端和仿真均未插值；算法确认为原reference_initialized_ddim；预测索引0、2、4…，完整尾段执行，history mask误差0，原生failure协议一致。详见RESULTS.json及每组analysis.json。

数字为首次连续3步满足几何分离条件时的**原始reference动作进度**，不是native failure，也不是插值版的实际控制步数。✓代表分离前连续至少30个实际动作步满足距竖直≤30°、离桌>8cm、至少2个近接触link、网格间距<8mm、净物体接触力>0.1N。几何分离条件为“网格间距>5mm且净物体接触力<0.05N，或网格间距>2cm”，需连续3步。

| 方法 | Episode76 | 34 | 54 | 2 | 稳定翻转 |
|---|---:|---:|---:|---:|---:|
''' + '\n'.join(rows) + '''

## 实际步数核对

| 方法 | Episode | 首次几何分离实际动作步 | 最长连续竖直接触实际步 |
|---|---:|---:|---:|
'''
    for label, group in groups:
        for ep, x in zip(EPISODES, group):
            sep = x['first_separation']
            report += f"| {label} | {ep} | {None if sep is None else sep['control_step']} | {x['longest_vertical_contact_steps']} |\n"
    report += '\n只覆盖四个既有合格起点及一个prior种子，不作为总体成功率估计。没有录制新视频或写入Downloads。\n'
    (O / 'REPORT.md').write_text(report)
    (O / 'COMPARISON.json').write_text(json.dumps(details, indent=2)+'\n')
    print('\n'.join(rows))
    for d in details:
        print(d['method'], d['episode'], 'longest stable steps', d['longest_vertical_contact_steps'], 'minimum vertical error', d['min_vertical_error_before_separation'])


if __name__ == '__main__':
    main()
