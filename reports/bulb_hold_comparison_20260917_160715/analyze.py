"""Compare actual holding flags in the two bulb TXT logs; offline only."""
import json
import math
from pathlib import Path
import re
import statistics

OUT=Path(__file__).resolve().parent

def stats(values):
    values=sorted(values)
    x=(len(values)-1)*.95
    p95=values[int(x)]+(values[math.ceil(x)]-values[int(x)])*(x-int(x))
    return dict(n=len(values),mean_rad=statistics.mean(values),mean_deg=math.degrees(statistics.mean(values)),median_deg=math.degrees(statistics.median(values)),p95_deg=math.degrees(p95),max_deg=math.degrees(values[-1]))

def analyze(path,holding,limit=None):
    text=path.read_text()
    assert f'[real] hold during inference: {holding} ' in text
    entries=[]
    for kind,chunk,step,delta in re.findall(r'\[cmd \d+ kind=(\w+) chunk=(\d+) chunk_step=([\d-]+)\] delta_max_rad=([\w.]+)',text):
        entries.append(dict(kind=kind,chunk=int(chunk),step=step,delta=float(delta)))
    policy=[r for r in entries if r['kind']=='policy' and (limit is None or r['chunk']<limit)]
    chunks={r['chunk'] for r in policy}
    groups={'within':[r['delta'] for r in policy if r['step']=='1'],
            'first':[r['delta'] for r in policy if r['step']=='0' and r['chunk']>0]}
    if holding:
        groups['hold']=[r['delta'] for r in entries if r['kind']=='inference_hold' and r['chunk'] in chunks and r['chunk']>0]
    else:
        assert not any(r['kind']=='inference_hold' for r in entries)
    return dict(commands=len(policy),chunks=len(chunks),metrics={k:stats(v) for k,v in groups.items()})

def main():
    off=analyze(OUT/'bulb_no_hold_160715.txt',0)
    on=analyze(OUT/'bulb_hold_152240.txt',1)
    matched=analyze(OUT/'bulb_hold_152240.txt',1,off['chunks'])
    result=dict(no_hold=off,hold=on,hold_first_matching_chunks=matched)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['放灯泡真机推理：holding / 不holding对比',
           '最新记录2026-09-17 16:07:15，来源eval/real/latest_run.txt，已归档为bulb_no_hold_160715.txt。',
           '用户说明本次放灯泡；用户称其holding，但TXT和JSONL实际均为hold_during_inference=0，且无inference_hold指令，故按不holding统计。',
           '对照记录2026-09-17 15:22:40，来源eval/real/latest_run.1.txt，已归档为bulb_hold_152240.txt；此前用户确认放灯泡，holding=1。',
           '两次均obs_4-66.ckpt、TensorRT FP32、fused DDIM、4步推理、chunk=2、30Hz、ROTATE初始化。',
           '指标=max_j |相邻下发目标角度差|；排除首个policy指令、启动hold和退出hold。holding末尾无后续policy的hold也排除。',
           '最新记录248条policy指令、124个chunk；最后一条指令已发送但控制等待被Ctrl-C中断，故统计指令跳变时保留，不能视作完成了248个控制周期。',
           '旧holding记录608条policy指令、304个chunk。', '',
           '| 条件 | 指令切换 | 样本数 | 平均幅度 | 中位数 | P95 | 最大值 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for label,data in [('不holding（最新）',off),('holding（此前）',on)]:
        for key,desc in [('within','chunk内：第0步→第1步'),('hold','上一chunk末步→hold'),('first','hold→下一chunk首步' if label.startswith('holding') else '上一chunk末步→下一chunk首步')]:
            if key not in data['metrics']:continue
            s=data['metrics'][key]
            lines.append(f"| {label} | {desc} | {s['n']} | {s['mean_deg']:.2f}°（{s['mean_rad']:.4f} rad） | {s['median_deg']:.2f}° | {s['p95_deg']:.2f}° | {s['max_deg']:.2f}° |")
    ratio=on['metrics']['first']['mean_deg']/off['metrics']['first']['mean_deg']
    lines.extend(['',f'全记录比较：holding边界首步平均跳变为不holding的{ratio:.3f}倍；不holding均值降低{(1-1/ratio)*100:.2f}%。',
                  f"不holding跨chunk/内部均值比={off['metrics']['first']['mean_deg']/off['metrics']['within']['mean_deg']:.3f}。",
                  '', '相同记录长度核查：仅取旧holding的前124个chunk（0..123）：'])
    for key,s in matched['metrics'].items():
        lines.append(f"  {key}: n={s['n']}, mean={s['mean_deg']:.4f}°, median={s['median_deg']:.4f}°, P95={s['p95_deg']:.4f}°, max={s['max_deg']:.4f}°")
    lines.append(f"相同长度时holding首步/不holding边界均值比={matched['metrics']['first']['mean_deg']/off['metrics']['first']['mean_deg']:.3f}。")
    lines.extend(['', '两次是不同时间的闭环运行，不是同轨迹配对重放；灯泡放置、接触及动作阶段可能不同，不能仅凭本表判定全部差异由holding引起。',
                  '旧holding TXT没有完整22维目标，无法算出其跳过hold的纯策略跨chunk差值。'])
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
