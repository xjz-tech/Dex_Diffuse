"""Report the matched 170 g / friction 2.0 comparison."""
import json
import numpy as np
from run_m170_mu20_guidance4_vs_edit015 import O, EPISODES

LABELS=dict(raw='原速reference（资格检查）',
            guidance4='传统guidance：DDIM4 / scale50 / guide4 exec2',
            edit015='编辑0.15 / DDIM4 / exec2（未来9步reference）')


def main():
    status=json.loads((O/'run_status.json').read_text())
    results=json.loads((O/'RESULTS.json').read_text())
    assert len(status)==len(results)==12
    rows=[]
    for method in ('raw','guidance4','edit015'):
        group=[next(x for x in results if x['episode']==ep and x['method']==method) for ep in EPISODES]
        cells=[]
        for x in group:
            marker='≥' if x['no_separation_right_censored'] else ''
            cells.append(f"{marker}{x['first_separation_reference_progress']:g} {'✓' if x['stable_turn'] else '✗'}")
        avg=float(np.mean([x['first_separation_reference_progress'] for x in group]))
        rows.append('| '+LABELS[method]+' | '+' | '.join(cells)+f" | {avg:.3f} | {sum(x['stable_turn'] for x in group)}/4 |")
    raw=[x for x in results if x['method']=='raw']
    qualified=[x['episode'] for x in raw if x['stable_turn']]
    qualified_guidance=[x for x in results if x['method']=='guidance4' and x['episode'] in qualified]
    qualified_edit=[x for x in results if x['method']=='edit015' and x['episode'] in qualified]
    report='''# 170g、摩擦2.0：guide4/exec2与编辑0.15

将上一轮44g、摩擦1.1改为灯泡质量170g、手与灯泡摩擦系数2.0。四个episode继续使用各自既定横抓起点76/f114、34/f79、54/f110、2/f103，不按本轮结果换起点。物体尺寸1、固定wrist、30Hz、60步静置、完整动作尾段和60步保持；环境seed42、prior固定噪声seed44。原生任务的目标更新、容忍阈值、failure逻辑保持原协议，failure不作为物理分离判据。

物理参数改变后，先以未经修改、未经插值的原速reference检查起点是否仍能稳定完成横转竖。两种待比较方法使用完全相同的>0.1rad选择性插值：相邻保存目标的22关节最大绝对差严格>0.1rad时插一个中点，展开动作数为315/560/612/668；每次执行2步后reference也前进2步。

传统guidance沿用用户上一轮所指配置：10B EMA、DDIM4、scale50、前4步reference引导、exec2。编辑沿用原9步reference方案：历史3步实际下发目标加未来9步reference；请求噪声比0.15，实际0.153397，局部DDIM时刻8→5→3→0，未来9步允许prior修改，无额外MSE guidance，exec2。

数字为首次连续3步满足几何分离条件时的**原始reference进度**；≥表示完整尾段中未检测到分离，因此只按reference末端右删失计数。✓要求分离前连续至少30个实际动作步满足距竖直≤30°、离桌>8cm、至少2个近接触link、网格间距<8mm且净物体接触力>0.1N。几何分离为连续3步满足“网格间距>5mm且接触力<0.05N，或间距>2cm”。原速reference没有插值，只用于资格检查，其平均进度不与插值方法直接比较。

| 方法 | Episode76 | 34 | 54 | 2 | 平均原始进度 | 稳定翻转 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

原速reference在新物理条件下通过的episode：{qualified}，共{len(qualified)}/4。若少于4条，两个prior方法的全四条结果仍完整列出，但方法优劣只能作为固定四起点上的诊断；不能把原速reference失效的条目算作“已验收翻转任务”的成功率。

在仍通过原速reference资格检查的{len(qualified)}条上，传统guidance平均进度{np.mean([x['first_separation_reference_progress'] for x in qualified_guidance]):.3f}、稳定翻转{sum(x['stable_turn'] for x in qualified_guidance)}/{len(qualified)}；编辑0.15平均进度{np.mean([x['first_separation_reference_progress'] for x in qualified_edit]):.3f}、稳定翻转{sum(x['stable_turn'] for x in qualified_edit)}/{len(qualified)}。全四条固定起点作为诊断时，传统guidance平均136.625、2/4；编辑平均173.625、4/4。

## 实际控制步与连续竖直接触

| 方法 | Episode | 首次分离实际动作步 | 最长连续竖直接触步 |
|---|---:|---:|---:|
'''
    for method in ('guidance4','edit015'):
        for ep in EPISODES:
            x=next(x for x in results if x['episode']==ep and x['method']==method)
            report+=f"| {LABELS[method]} | {ep} | {x['first_separation_actual_action_step']} | {x['longest_vertical_contact_steps']} |\n"
    report+='''
每组的初态快照逐字段相同，60步静置运动状态相同；净接触力数值差<1e-5N。质量和摩擦均在刷新PhysX后写入手关节、目标、物体根位姿及零速度，并保存首个实际物理步。完整性检查覆盖插值进度、prediction reference_index、history噪声约束、完整尾段和原生协议。未录新视频，未写Downloads。
'''
    (O/'REPORT.md').write_text(report)
    print('\n'.join(rows))


if __name__=='__main__':
    main()
