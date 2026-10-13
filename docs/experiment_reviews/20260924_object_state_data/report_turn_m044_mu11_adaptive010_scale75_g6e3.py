"""Report DDIM4 scale75 guide6/exec3 alongside existing qualified baselines."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import json
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926';O=R/'m044_mu11_adaptive010_ddim4_scale75_g6e3_20260928';EPS=(76,34,54,2)
def read(name):return json.loads((R/name/'RESULTS.json').read_text())
def rows(data,mode='guided'):
 rr=[next(r for r in data if r['episode']==ep and r['mode']==mode) for ep in EPS]
 vals=[r['first_separation']['reference_action_number'] if r['first_separation'] else None for r in rr]
 return rr,vals
def main():
 raw=read('m044_mu11_adaptive010_ddim_scale_20260927');prev=read('m044_mu11_adaptive010_ddim4_scale50_g4e2_20260928');new=json.loads((O/'RESULTS.json').read_text())
 combos=[('原速 reference',raw,'raw'),('同规则插值 direct',raw,'direct'),('scale25 guide2/exec1',raw,'guided'),('scale50 guide4/exec2',prev,'guided'),('scale75 guide6/exec3',new,'guided')]
 lines=['# 44g / μ1.1 / >0.1插值：DDIM4 scale75 guide6/exec3','', '新配置与既有四组使用相同的episode76/f114、34/f79、54/f110、2/f103。10B EMA、DDIM4、固定wrist、30Hz、环境seed42、prior噪声44、原尺寸、质量44g、手与物体摩擦1.1。相邻22关节最大目标跳变严格>0.1rad时插入一个中点；每执行3个插值目标，reference前进3个。每次重新预测时用接下来的6个reference目标作guidance；scale全程75。60步静置、完整动作尾段、60步末目标保持。四条已事先由原速reference通过稳定翻转验收，不按本次guide结果筛选。','', '| 配置 | ep76 | ep34 | ep54 | ep2 | 平均原始进度 | 稳定翻转 |','|---|---:|---:|---:|---:|---:|---:|']
 for name,data,mode in combos:
  rr,vals=rows(data,mode);assert all(v is not None for v in vals)
  mean=(sum(Decimal(str(v)) for v in vals)/4).quantize(Decimal('.1'),rounding=ROUND_HALF_UP)
  cells=[f"{v:g} {'✓' if r['completed_turn'] else '✗'}" for v,r in zip(vals,rr)]
  lines.append('| '+' | '.join([name]+cells+[str(mean),f"{sum(r['completed_turn'] for r in rr)}/4"])+ ' |')
 rr,_=rows(new);prevrr,_=rows(prev)
 assert all(a['inserted_steps']==b['inserted_steps'] and a['executed_action_steps']==b['executed_action_steps'] for a,b in zip(rr,prevrr))
 lines+=['','数字是首次持续几何/接触分离对应的原始reference进度，插值中点记半步；该分离需连续3个控制步确认，是独立分析指标，不是原生环境failure。✓要求分离前距竖直≤30°、悬空且接触条件连续至少30个实际控制步。','', '| Episode | 插入中点 | 实际动作步数 | scale50 guide4/exec2 最长竖直接触 | scale75 guide6/exec3 最长竖直接触 |','|---|---:|---:|---:|---:|']
 for a,b in zip(prevrr,rr):lines.append(f"| {b['episode']} | {b['inserted_steps']} | {b['executed_action_steps']} | {a['longest_vertical_contact_control_steps']} | {b['longest_vertical_contact_control_steps']} |")
 oldmean=sum(r['first_separation']['reference_action_number'] for r in prevrr)/4;newmean=sum(r['first_separation']['reference_action_number'] for r in rr)/4
 lines+=['',f'平均保持进度从{oldmean:.3f}变为{newmean:.3f}（{100*(newmean/oldmean-1):+.1f}%）。此对比同时改变scale、guidance窗口和每次执行步数，不能将差异单独归因于scale。结果仅限这四条起点与固定seed；未新增录像核验。','', '新运行的初态快照与原速reference逐字段相同，60步静置除物体净接触力允许<1e-5N差外逐字段一致；原生协议、完整动作和末尾保持、插值progress映射、prior reference_index逐3步推进均在分析中核验。未写入Downloads。']
 (O/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
