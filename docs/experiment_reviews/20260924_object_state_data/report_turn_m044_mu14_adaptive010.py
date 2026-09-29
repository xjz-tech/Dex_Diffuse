"""Paired friction 1.1/1.4 report; fixed >0.10 DDIM4 scale25 protocol."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import json
P=Path(__file__).resolve().parent
ROOT=P/'reference_turn_baseline_20260926'
OUT=ROOT/'m044_mu14_adaptive010_ddim_scale_20260927'
EPS=(76,34,54,2)

def main():
 old=json.loads((ROOT/'m044_mu11_adaptive010_ddim_scale_20260927/RESULTS.json').read_text())
 new=json.loads((OUT/'RESULTS.json').read_text())
 lines=['# 摩擦1.1与1.4：44g，>0.1插值，DDIM4/scale25','',
 '唯一指定变化是手和灯泡的滑动摩擦系数1.1→1.4；44g、原尺寸、各episode自身起点76/f114、34/f79、54/f110、2/f103、固定wrist、30Hz、环境seed42、固定prior噪声44保持一致。10B EMA、DDIM4、scale25、guide2/exec1。每次执行1个重采样目标，reference也前进1个；最大相邻22关节跳变严格>0.1 rad时插1个中点。60步静置+完整尾段+60步末目标保持。','',
 '| 方式 | 摩擦 | 76 | 34 | 54 | 2 | 平均原始进度 | 稳定翻转 |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
 for mode,label in [('raw','原速reference'),('direct','同规则插值direct'),('guided','DDIM4 / scale25')]:
  for mu,data in [(1.1,old),(1.4,new)]:
   rows=[next((r for r in data if r['episode']==ep and r['mode']==mode),None) for ep in EPS]
   cells=[];vals=[]
   for r in rows:
    if r is None:cells.append('未运行：raw基线未通过');continue
    loss=r['first_separation'];v=loss['reference_action_number'] if loss else None
    cells.append((f'{v:g}' if v is not None else '尾段未分离')+(' ✓' if r['completed_turn'] else ' ✗'))
    if v is not None:vals.append(Decimal(str(v)))
   mean=str((sum(vals)/4).quantize(Decimal('.1'),rounding=ROUND_HALF_UP)) if len(vals)==4 else '—'
   lines.append('| '+' | '.join([label,str(mu)]+cells+[mean,f"{sum(bool(r and r['completed_turn']) for r in rows)}/{sum(r is not None for r in rows)}"])+ ' |')
 common=[ep for ep in EPS if any(r['episode']==ep and r['mode']=='guided' for r in new)]
 if common:
  means=[]
  for data in (old,new):
   rr=[next(r for r in data if r['episode']==ep and r['mode']=='guided') for ep in common]
   means.append(sum(r['first_separation']['reference_action_number'] for r in rr)/len(rr))
  lines += ['',f'仅比较两种摩擦都运行guide的相同episode {common}：平均原始进度 {means[0]:.2f}→{means[1]:.2f}，变化{100*(means[1]/means[0]-1):+.1f}%。没有将不同episode集合的均值直接比较。']
 lines+=['','✓/✗是稳定翻转判据：首次持续分离前，距竖直≤30°并连续至少30个控制步满足悬空、几何接近及接触条件。进度是沿用此前几何/接触判据的首次持续分离位置，映射回原始reference，不是native failure或额外终止规则。分离需连续3步确认。','',
 '本轮先运行各episode的摩擦1.4原速reference，再按原有判据验收；未通过则保留原始结果且跳过该条后续guide，不更换起点或挑新episode。结果仅代表这四条固定起点及seed。', '',
 '## 静置与初态核验','',
 '| Episode | 摩擦1.1首步 mm / ° | 摩擦1.4首步 mm / ° | 摩擦1.1静置60步 mm / ° | 摩擦1.4静置60步 mm / ° |','|---|---:|---:|---:|---:|']
 for ep in EPS:
  case=ROOT/f'qualified_comparison/episode_{ep:02d}'
  ts=[json.loads((case/f'direct_m044_mu{mu}/trace.json').read_text()) for mu in ('11','14')]
  cells=[f"{t[i]['displacement_from_import_m']*1000:.2f} / {t[i]['rotation_from_import_deg']:.2f}" for i in (0,59) for t in ts]
  lines.append('| '+' | '.join([str(ep)]+cells)+' |')
 lines+=['', '所有新组的导入快照相对摩擦1.1只允许hand_friction与object_friction改变；其余字段逐值一致。同一摩擦下，raw/direct/guide导入字段一致，60步静置状态一致（object_contact_force允许<1e-5 N差）。跨摩擦的静置后位姿不要求一致，因此结果涵盖摩擦变化对初始化重平衡及后续动作的整体影响。','',
 '原生评估协议、完整执行长度、原始动作锚点、插值进度映射、reference递进和固定guidance参数均核验。新增原速reference留有实际录像作为基线证据；未制作对比视频，未写Downloads。详见RESULTS.json、manifest.json及各episode对应m044_mu14运行目录。']
 (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
