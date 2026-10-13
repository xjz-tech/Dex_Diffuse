"""Paired >0.1 interpolation DDIM4 scale25/50, fixed 44g and mu1.1."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import json
P=Path(__file__).resolve().parent
ROOT=P/'reference_turn_baseline_20260926'
OUT=ROOT/'m044_mu11_adaptive010_ddim4_scale50_g4e2_20260928'
EPS=(76,34,54,2)
def main():
 old=json.loads((ROOT/'m044_mu11_adaptive010_ddim_scale_20260927/RESULTS.json').read_text());new=json.loads((ROOT/'m044_mu11_adaptive010_ddim4_scale50_20260928/RESULTS.json').read_text());g4=json.loads((OUT/'RESULTS.json').read_text())
 lines=['# 44g / 摩擦1.1 / >0.1插值：DDIM4 scale25、scale50与guide4/exec2','',
 '两项新增：①guide2/exec1下固定scale25→50；②scale50下guide2/exec1→guide4/exec2。guide4/exec2同时改变引导窗口和每次执行长度，不能将差异单独归因其中一个。共同配置：10B EMA、DDIM4、相邻22关节最大目标跳变严格>0.1 rad时插1个中点，44g、手和灯泡摩擦1.1、原尺寸、固定wrist、30Hz、环境seed42、prior固定噪声44。各自源帧76/f114、34/f79、54/f110、2/f103；60静置+完整reference尾段+60末目标保持。沿用已通过原速reference验收的四条起点，不重新选例子。','',
 '| 方式 | 76 | 34 | 54 | 2 | 平均原始进度 | 稳定翻转 |','|---|---:|---:|---:|---:|---:|---:|']
 for label,data,mode in [('原速reference',old,'raw'),('同规则插值direct',old,'direct'),('scale25 / guide2 exec1',old,'guided'),('scale50 / guide2 exec1',new,'guided'),('scale50 / guide4 exec2',g4,'guided')]:
  rr=[next(r for r in data if r['episode']==ep and r['mode']==mode) for ep in EPS];vals=[r['first_separation']['reference_action_number'] for r in rr]
  cells=[f"{v:g} {'✓' if r['completed_turn'] else '✗'}" for v,r in zip(vals,rr)]
  mean=(sum(Decimal(str(v)) for v in vals)/4).quantize(Decimal('.1'),rounding=ROUND_HALF_UP)
  lines.append('| '+' | '.join([label]+cells+[str(mean),f"{sum(r['completed_turn'] for r in rr)}/4"])+ ' |')
 lines+=['','数字为首次持续几何/接触分离对应的原始reference进度；不是native failure或停止条件。✓要求在首次分离前，距竖直≤30°并连续至少30个控制步满足悬空、几何接近和接触判据，口径沿用之前表格。未新增录像核验，结果限这四条起点与seed。','',
 '| Episode | 插入中点数 | 实际动作控制步数 | scale25最长连续竖直接触步数 | scale50 guide2 exec1最长连续竖直接触步数 | scale50 guide4 exec2最长连续竖直接触步数 |','|---|---:|---:|---:|---:|---:|']
 for ep in EPS:
  a=next(r for r in old if r['episode']==ep and r['mode']=='guided');b=next(r for r in new if r['episode']==ep);c=next(r for r in g4 if r['episode']==ep)
  assert a['inserted_steps']==b['inserted_steps']==c['inserted_steps'] and a['executed_action_steps']==b['executed_action_steps']==c['executed_action_steps']
  lines.append(f"| {ep} | {b['inserted_steps']} | {b['executed_action_steps']} | {a['longest_vertical_contact_control_steps']} | {b['longest_vertical_contact_control_steps']} | {c['longest_vertical_contact_control_steps']} |")
 oldmean=sum(r['first_separation']['reference_action_number'] for r in old if r['mode']=='guided')/4;newmean=sum(r['first_separation']['reference_action_number'] for r in new)/4
 g4mean=sum(r['first_separation']['reference_action_number'] for r in g4)/4
 lines+=['',f'guide2/exec1下，scale25→50平均保持进度：{oldmean:.3f}→{newmean:.3f}（{100*(newmean/oldmean-1):+.1f}%）。scale50下，guide2/exec1→guide4/exec2：{newmean:.3f}→{g4mean:.3f}（{100*(g4mean/newmean-1):+.1f}%）；新增配置相对scale25 guide2/exec1为{100*(g4mean/oldmean-1):+.1f}%。三组稳定翻转均为3/4，episode54均未通过。实际动作控制步数不含60静置+60保持。','',
 '所有新运行的初态快照与既有原速reference逐值相同；前60步静置记录状态一致，物体净接触力允许<1e-5 N差；原生协议、执行完整性、插值progress映射、prior reference_index逐步推进、scale50固定不切换均核验。视频/Downloads未新增。','',
 '不要混淆之前>0.12插值的scale50结果（3/4）与本次>0.1插值；本次三组guide均采用同一>0.1规则。guide4/exec2每执行2个重采样目标，reference_index前进2，末尾不足2个只执行剩余步；末尾窗口padding不增加真实reference长度。']
 (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
