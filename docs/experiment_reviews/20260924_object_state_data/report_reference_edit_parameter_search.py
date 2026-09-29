"""Publish the completed precommitted parameter search without cherry-picking."""
import json
from run_reference_edit_parameter_search import O, R, EPISODES


def cell(x):
    value = x['first_separation_reference_progress']
    suffix = '≥' if x['no_separation_right_censored'] else ''
    return f"{suffix}{value:g} {'✓' if x['stable_turn'] else '✗'}"


def main():
    scores=[]
    for stage in (1,2,3):
        batch=json.loads((O/f'stage{stage}_scores.json').read_text())
        assert len(batch['new_scores']) == {1:5,2:6,3:2}[stage]
        scores += batch['new_scores']
    validation=json.loads((O/'validation_results.json').read_text())
    grid=json.loads((O/'noise_grid.json').read_text())
    rows=[]
    for stage, start, stop in [(1,0,5),(2,5,11),(3,11,13)]:
        for x in scores[start:stop]:
            c=x['config'];rr=x['episode_results']
            cells=[cell(y) for y in rr]
            rows.append(f"| {stage} | t{c['t']} ({c['ratio']:.4f}) | {c['ddim']} | {c['exec']} | "+' | '.join(cells)+f" | {x['average_original_progress']:.3f} | {x['stable_count']}/4 |")
    top=sorted(scores,key=lambda x:(-x['stable_count'],-x['average_original_progress'],x['config']['t'],x['config']['ddim'],x['config']['exec']))[0]
    target=190.875
    qualifying=[x for x in scores if x['stable_count']==4 and x['average_original_progress']>=target]
    text='''# 原9步reference加噪编辑：噪声、执行步长、DDIM搜索

固定四个已经由原速reference验收横转竖的起点：76/f114、34/f79、54/f110、2/f103。所有条目均为各自初态、44g、摩擦1.1、尺寸1、固定wrist、30Hz，原始相邻手部22关节最大跳变严格>0.1rad时插一个中点；完整reference尾段和60步保持。原生任务容忍阈值、目标更新和failure逻辑保持不变。数据集 reference 的未来9步与最近3步实际下发历史拼成12步SDEdit窗口，history按正确训练噪声约束，未来9步可编辑；无额外MSE guidance。环境seed42，第一轮参数搜索prior种子44，10B EMA。

用户目标为四条**首次连续3步几何分离对应的原始reference进度**均值≥190.875，并保留编辑现有4/4稳定翻转。190.875是原同规则插值传统guidance（DDIM4、scale50、guide4/exec2、seed44）112／244.5／154／253的均值。原编辑0.15/DDIM4/exec2为153／197／151／183，均值171.000，稳定翻转4/4。均值只对固定同四条episode计算，单元格数字不是实际物理步数。

筛选规则在任何新结果出炉前写入PLAN.md：阶段1仅DDIM4/exec2扫扩散t6～t10；阶段2按稳定翻转数优先、平均进度其次选两个起点，各扫exec1、3、4；阶段3从此前全部结果按同规则选两个配置测试DDIM6。t8/DDIM4/exec2和t8/DDIM6/exec2沿用已核验的同配置存档，其余11组共44条为本轮新仿真。所有配置与失败都留在表中，不按单条episode结果筛选。

✓要求首次分离前连续≥30实际动作步满足距竖直≤30°、悬空>8cm、至少2个近接触link、网格间距<8mm、净物体接触力>0.1N。几何分离需连续3步“网格间距>5mm且接触力<0.05N，或间距>2cm”。原生failure另存，绝不当作独立物理分离。每条新仿真核验初态全部字段、60步静置运动状态、插值序列、预测reference索引推进、完整尾段和原生协议。

| 阶段 | 噪声起点（实际噪声比） | DDIM | exec | 76 | 34 | 54 | 2 | 平均原始进度 | 稳定翻转 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

## 结果判断

共13组配置，t6～t10对应训练噪声比{', '.join(f'{grid[str(t)]:.6f}' for t in range(6,11))}。最佳排序配置为t{top['config']['t']}、DDIM{top['config']['ddim']}、exec{top['config']['exec']}：平均进度{top['average_original_progress']:.3f}，稳定翻转{top['stable_count']}/4。达到预设seed44目标的配置数：{len(qualifying)}。\n'''
    if validation['status']=='no_seed44_candidate':
        text+='\n无seed44配置同时达到平均进度与4/4稳定翻转，按预设规则停止；没有运行追加种子。不能从该有限网格推断其他参数也无法达到目标。\n'
    else:
        assert validation['status']=='complete'
        text+='\n有seed44候选达到目标；按预设规则将排名最高的至多两组追加seed45、46，传统guidance也用相同种子配对重跑。\n\n'
        text+='| 候选 | seed44 编辑/传统均值，翻转 | seed45 编辑/传统均值，翻转 | seed46 编辑/传统均值，翻转 | 三种子编辑/传统均值 |\n|---|---|---|---|---|\n'
        for x in validation['summary']:
            c=x['config'];cells=[]
            for seed in (44,45,46):
                v=x['byseed'][str(seed)]
                cells.append(f"{v['editor_average']:.2f}/{v['guidance_average']:.2f}，{v['editor_stable']}/4 vs {v['guidance_stable']}/4")
            text+=f"| t{c['t']}/DDIM{c['ddim']}/exec{c['exec']} | "+' | '.join(cells)+f" | {x['three_seed_editor_average']:.2f}/{x['three_seed_guidance_average']:.2f} |\n"
        text+='\n三种子的均值是每个种子同四条episode平均后再取平均。若候选只在seed44达标，不能宣称跨种子稳定优于传统guidance。\n'
    text+='\n全程没有新增视频或写入Downloads。所有逐条结果、实际控制步数及验证字段见stage1/2/3_results.json、对应scores.json和validation_results.json。\n'
    (O/'REPORT.md').write_text(text)
    print('REPORT WRITTEN',O/'REPORT.md',flush=True)
    for x in scores:
        print('ROW',x['config']['t'],x['config']['ddim'],x['config']['exec'],x['average_original_progress'],x['stable_count'],flush=True)


if __name__=='__main__':
    main()
