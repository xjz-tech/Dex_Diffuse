"""Paired >0.1 interpolation DDIM4 scale25/50, fixed 44g and mu1.1."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import json
P=Path(__file__).resolve().parent
ROOT=P/'reference_turn_baseline_20260926'
OUT=ROOT/'m044_mu11_adaptive010_ddim4_scale50_20260928'
EPS=(76,34,54,2)
def main():
 old=json.loads((ROOT/'m044_mu11_adaptive010_ddim_scale_20260927/RESULTS.json').read_text());new=json.loads((OUT/'RESULTS.json').read_text())
 lines=['# 44g / 摩擦1.1 / >0.1插值：DDIM4 scale25与scale50','',
 '仅将固定guidance scale从25改为50。其余均保持：10B EMA、DDIM4、guide2/exec1、相邻22关节最大目标跳变严格>0.1 rad时插1个中点，44g、手和灯泡摩擦1.1、原尺寸、固定wrist、30Hz、环境seed42、prior固定噪声44。各自源帧76/f114、34/f79、54/f110、2/f103；60静置+完整reference尾段+60末目标保持。沿用已通过原速reference验收的四条起点，不重新选例子。','',
 '| 方式 | 76 | 34 | 54 | 2 | 平均原始进度 | 稳定翻转 |','|---|---:|---:|---:|---:|---:|---:|']
 for label,data,mode in [('原速reference',old,'raw'),('同规则插值direct',old,'direct'),('DDIM4 / scale25',old,'guided'),('DDIM4 / scale50',new,'guided')]:
  rr=[next(r for r in data if r['episode']==ep and r['mode']==mode) for ep in EPS];vals=[r['first_separation']['reference_action_number'] for r in rr]
  cells=[f"{v:g} {'✓' if r['completed_turn'] else '✗'}" for v,r in zip(vals,rr)]
  mean=(sum(Decimal(str(v)) for v in vals)/4).quantize(Decimal('.1'),rounding=ROUND_HALF_UP)
  lines.append('| '+' | '.join([label]+cells+[str(mean),f"{sum(r['completed_turn'] for r in rr)}/4"])+ ' |')
 lines+=['','数字为首次持续几何/接触分离对应的原始reference进度；不是native failure或停止条件。✓要求在首次分离前，距竖直≤30°并连续至少30个控制步满足悬空、几何接近和接触判据，口径沿用之前表格。未新增录像核验，结果限这四条起点与seed。','',
 '| Episode | 插入中点数 | 实际动作控制步数 | scale25最长连续竖直接触步数 | scale50最长连续竖直接触步数 |','|---|---:|---:|---:|---:|']
 for ep in EPS:
  a=next(r for r in old if r['episode']==ep and r['mode']=='guided');b=next(r for r in new if r['episode']==ep)
  assert a['inserted_steps']==b['inserted_steps'] and a['executed_action_steps']==b['executed_action_steps']
  lines.append(f"| {ep} | {b['inserted_steps']} | {b['executed_action_steps']} | {a['longest_vertical_contact_control_steps']} | {b['longest_vertical_contact_control_steps']} |")
 oldmean=sum(r['first_separation']['reference_action_number'] for r in old if r['mode']=='guided')/4;newmean=sum(r['first_separation']['reference_action_number'] for r in new)/4
 lines+=['',f'平均保持进度变化：{oldmean:.3f}→{newmean:.3f}（{100*(newmean/oldmean-1):+.1f}%）。实际动作控制步数不含60静置+60保持。','',
 '所有新运行的初态快照与既有原速reference逐值相同；前60步静置记录状态一致，物体净接触力允许<1e-5 N差；原生协议、执行完整性、插值progress映射、prior reference_index逐步推进、scale50固定不切换均核验。视频/Downloads未新增。','',
 '不要混淆之前>0.12插值的scale50结果（3/4）与本次>0.1插值；这次唯一比较变量是同>0.1规则下的scale。']
 (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
