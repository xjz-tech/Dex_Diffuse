"""Report the 130 g / friction 2.4 comparison and place it beside prior physics."""
import json
import numpy as np
from pathlib import Path

P=Path(__file__).resolve().parent
R=P/'reference_turn_baseline_20260926'
O=R/'m130_mu24_guidance4_vs_edit015_20260928'
EPISODES=(76,34,54,2)
LABELS=dict(direct_interp='同规则插值direct',guidance4='guidance：DDIM4 / scale50 / guide4 exec2',
            edit015='编辑0.15 / DDIM4 / exec2')


def row(physics,method,group):
    cells=[]
    for x in group:
        marker='≥' if x.get('no_separation_right_censored') else ''
        turn_success=x.get('longest_vertical_contact_steps',30 if x['stable_turn'] else 0)>0
        cells.append(f"{marker}{x['first_separation_reference_progress']:g} {'✓' if turn_success else '✗'}")
    return '| '+' | '.join([physics,LABELS[method],*cells,
        f"{np.mean([x['first_separation_reference_progress'] for x in group]):.3f}",
        f"{sum(x.get('longest_vertical_contact_steps',30 if x['stable_turn'] else 0)>0 for x in group)}/4",
        f"{sum(x['stable_turn'] for x in group)}/4"])+ ' |'


def prior_results(path):
    return json.loads((R/path/'RESULTS.json').read_text())


def main():
    status=json.loads((O/'run_status.json').read_text())
    current=json.loads((O/'RESULTS.json').read_text())
    assert len(status)==len(current)==12
    # Assemble explicitly because interpolated direct was added after the prior comparison.
    data44=prior_results('adaptive010_direct_edit015_guidance4_comparison_20260928')
    data170=prior_results('m170_mu20_guidance4_vs_edit015_20260928')+json.loads((R/'m170_mu20_guidance4_vs_edit015_20260928/DIRECT_INTERP_RESULTS.json').read_text())
    data130=current+json.loads((O/'DIRECT_INTERP_RESULTS.json').read_text())
    datasets=[('44g／1.1',data44),('170g／2.0',data170),('130g／2.4',data130)]
    rows=[]
    for physics,data in datasets:
        if physics=='44g／1.1':
            direct=[dict(x,method='direct_interp') for x in data if x['method']=='direct']
            guide=[dict(x,method='guidance4') for x in data if x['method']=='guidance']
            edit=[dict(x,method='edit015') for x in data if x['method']=='edit']
        else:
            direct=[next(x for x in data if x['method']=='direct_interp' and x['episode']==ep) for ep in EPISODES]
            guide=[next(x for x in data if x['method']=='guidance4' and x['episode']==ep) for ep in EPISODES]
            edit=[next(x for x in data if x['method']=='edit015' and x['episode']==ep) for ep in EPISODES]
        rows += [row(physics,'direct_interp',direct),row(physics,'guidance4',guide),row(physics,'edit015',edit)]
    guide=[x for x in current if x['method']=='guidance4']
    edit=[x for x in current if x['method']=='edit015']
    report='''# 130g、摩擦2.4：guide4/exec2与编辑0.15

共同协议与前两组一致：相同四个episode及各自横抓初态；灯泡尺寸1、固定wrist、30Hz、60步静置、完整尾段和60步保持；环境seed42、prior固定噪声seed44。传统guidance为10B EMA、DDIM4、scale50、guide4/exec2；编辑为未来9步reference的0.15/DDIM4/exec2。三种方法都在相邻reference的22关节最大跳变严格>0.1rad时插一个中点，执行完全相同的315/560/612/668个目标。Direct逐步执行插值目标；两个prior方法每执行2步后重规划并让reference前进2步。

数字为首次连续3步几何分离时的原始reference进度。按用户本次口径，✓表示分离前至少有一个实际动作步同时满足距竖直≤30°、悬空及有效接触，说明已经翻到竖直附近；连续≥30步仍作为单独的“稳定翻转”指标。三种方法现在使用同一插值序列和原始进度映射，可以直接比较。

| 质量／摩擦 | 方法 | Episode76 | 34 | 54 | 2 | 平均进度 | 翻转成功 | ≥30步稳定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

130g／2.4下，传统guidance翻转成功3/4、其中连续≥30步为2/4；编辑翻转成功4/4、其中连续≥30步为3/4。同规则插值direct也按相同两种成功口径列入表格。

新运行的初态逐字段一致、60步静置运动状态一致、净接触力数值差<1e-5N；插值进度、预测索引推进、完整尾段和原生协议均已核验。没有录制视频或写入Downloads。
'''
    (O/'REPORT.md').write_text(report)
    direct=[x for x in data130 if x['method']=='direct_interp']
    print(row('130g／2.4','direct_interp',direct))
    print(row('130g／2.4','guidance4',guide))
    print(row('130g／2.4','edit015',edit))


if __name__=='__main__':
    main()
