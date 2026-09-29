"""Report the >0.10 experiment alongside >0.12 and exact0.18."""
import json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from reference_resampling import interpolate_equal_jumps, interpolate_large_jumps

P = Path(__file__).resolve().parent
ROOT = P / 'reference_turn_baseline_20260926'
OUT = ROOT / 'm044_mu11_adaptive010_ddim_scale_20260927'
EPS = (76, 34, 54, 2)

def main():
    old = json.loads((ROOT / 'm044_mu11_adaptive012_ddim_scale_20260927/RESULTS.json').read_text())
    new = json.loads((OUT / 'RESULTS.json').read_text())
    exact = json.loads((ROOT / 'm044_mu11_exact018_ddim_scale_20260927/RESULTS.json').read_text())
    rows = [('原速 reference', old, 'raw'),
            ('>0.12 插值 direct', old, 'direct'),
            ('仅0.18 插值 direct', exact, 'direct'),
            ('>0.10 插值 direct', new, 'direct'),
            ('>0.12 插值 DDIM4 / scale25', old, 'guided'),
            ('仅0.18 插值 DDIM4 / scale25', exact, 'guided'),
            ('>0.10 插值 DDIM4 / scale25', new, 'guided')]
    lines = ['# 44g / 摩擦1.1：>0.10跳变插值，DDIM4 / scale25', '',
             '灯泡沿用原尺寸（actor scale 0.987453997），不是上一轮×1.1诊断。四条使用各自原起点：76/f114、34/f79、54/f110、2/f103。10B EMA、DDIM4、scale25、guide2/exec1；环境seed42、固定prior噪声44。每次执行1步后reference也前进1个重采样后的目标。60步固定q静置、完整reference尾段、60步末目标保持。质量44g、手和灯泡滑动摩擦1.1。未录视频、未写Downloads。', '',
             '本次在相邻22关节目标的最大绝对差严格大于0.10 rad时插1个中点。之前两组分别为>0.12 rad与仅等于0.18 rad（容差1e-6）。原始动作和全部原始锚点不变。', '',
             '| 运行方式 | Episode76 | 34 | 54 | 2 | 平均原始进度 | 稳定翻转 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for label, data, mode in rows:
        rr = [next(r for r in data if r['episode']==ep and r['mode']==mode and
                   (mode!='guided' or (r['ddim_steps']==4 and r['guidance_scale']==25))) for ep in EPS]
        vals = [r['first_separation']['reference_action_number'] for r in rr]
        cells = [f"{v:g}（{'通过' if r['completed_turn'] else '未通过'}）" for v,r in zip(vals,rr)]
        mean = (sum(Decimal(str(v)) for v in vals)/4).quantize(Decimal('.1'),rounding=ROUND_HALF_UP)
        lines.append('| '+ ' | '.join([label]+cells+[str(mean),f"{sum(r['completed_turn'] for r in rr)}/4"])+ ' |')
    lines += ['', '进度为沿用之前几何/接触判据的首次持续分离位置，换算到原始reference动作编号；不是native failure，也不是新增的终止逻辑。稳定翻转同旧表：距竖直≤30°，在首次分离前连续至少30个控制步满足悬空、几何接近与接触要求。此次未新增独立录像核验。结果仅适用于这4条原始direct已通过翻转的条件子集与该seed。', '',
              '| Episode | 原始动作数 | >0.12插入数 | >0.10插入数 | 增加 | 本次实际动作控制步数 |',
              '|---|---:|---:|---:|---:|---:|']
    total_old=total_new=0
    for ep in EPS:
        a=np.load(ROOT/f'qualified_comparison/episode_{ep:02d}/reference_full.npz')['hand_target_rad']
        n=a.shape[1]; b,_=interpolate_large_jumps(a,.12); c,_=interpolate_large_jumps(a,.10)
        old_n=b.shape[1]-n;new_n=c.shape[1]-n
        total_old+=old_n;total_new+=new_n
        lines.append(f'| {ep} | {n} | {old_n} | {new_n} | {new_n-old_n} | {c.shape[1]} |')
    lines += ['',f'总插入数：{total_old}→{total_new}，增加{total_new-total_old}（{100*(total_new/total_old-1):.1f}%）。实际控制步数不含60静置+60末目标保持。', '',
              '核验：新组27项初态与该episode原速reference逐值相同；60步静置各已记录状态字段一致，object_contact_force允许<1e-5 N数值差；native协议完全相同。sim与prior两侧插值进度表一致，prediction reference_index逐步递增且scale固定25。完整尾段与hold均执行完成。原尺寸和默认qpos静置目标均核验；新增加的qd字段不影响与旧记录已有字段的比较。', '',
              '详见 manifest.json、RESULTS.json，以及 qualified_comparison 下各episode的 direct_adaptive010_m044_mu11 / guide1_adaptive010_ddim4_scale25_m044_mu11。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
