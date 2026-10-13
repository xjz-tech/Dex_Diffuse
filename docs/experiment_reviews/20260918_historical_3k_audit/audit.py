import json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
suite=ROOT/'eval/hold_runs/20260912_fresh_noise_3000_seed42_8_19_25'
out=Path(__file__).resolve().parent
runs=[];physics={}
for seed in [42,8,19,25]:
 blocks=[]
 for prior in ['1B','mixed']:
  for scale in [0,25]:
   p=suite/f'seed{seed}_{prior}_scale{scale}'
   records=[json.loads(l) for l in next(p.glob('*.jsonl')).read_text().splitlines() if l]
   assert len(records)==3000 and {v['env'] for v in records}==set(range(3000)) and all(v['episode']==0 for v in records)
   lengths=np.array([v['length'] for v in records]);t=lengths/30
   timeout=sum(v['reason']=='timeout' for v in records)
   assert all(v['length']==12000 for v in records if v['reason']=='timeout')
   shortest=[v for v in records if v['length']==lengths.min()]
   runs.append(dict(seed=seed,prior=prior,scale=scale,n=len(records),min_s=float(t.min()),p10_s=float(np.percentile(t,10)),median_s=float(np.median(t)),capped_mean_s=float(t.mean()),max_observed_s=float(t.max()),timeout=timeout,timeout_pct=timeout/30,under1s=int((t<1).sum()),shortest=shortest,source=str(p)))
   text=(p/'console.log').read_text();start=text.index('[DR-DEBUG]');block=text[start:text.index('\n\n',start)];blocks.append(block)
   if prior=='1B' and scale==0:
    physics[str(seed)]={};section=''
    for line in block.splitlines()[1:]:
     if line.startswith('  ') and not line.startswith('    '):section=line.strip().rstrip(':');physics[str(seed)][section]={}
     elif 'min=' in line:
      m=re.match(r'\s*(\w+): min=(\S+)\s+max=(\S+)\s+mean=(\S+)\s+std=(\S+)\s+n=(\d+)',line)
      key,lo,hi,mean,std,n=m.groups();physics[str(seed)][section][key]=dict(min=float(lo),max=float(hi),mean=float(mean),std=float(std),n=int(n))
 assert len(set(blocks))==1,seed
(out/'results.json').write_text(json.dumps(dict(suite=str(suite),time_conversion='length / 30; same nominal conversion as original episode_stats.py',initial_dr_summary_equal_within_seed=True,physics=physics,runs=runs),indent=2))
lines=['# 历史 3000 环境四 seed 实验核查','',f'来源：{suite}','','每配置 3000 个环境的首轮，4 seed × 2 prior × 2 scale = 48000 条记录。时间按原报告 length/30 换算；400 秒未失败记为右删失。scale0 是历史 guided-DDIM 路径，不能冒称当前普通 1B 采样器。','', '| seed | prior | scale | 最短/s | P10/s | 中位/s | 截断均值/s | 满400s | <1s数量 |','|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for x in runs:lines.append(f"| {x['seed']} | {x['prior']} | {x['scale']} | {x['min_s']:.3f} | {x['p10_s']:.2f} | {x['median_s']:.2f} | {x['capped_mean_s']:.2f} | {x['timeout_pct']:.2f}% | {x['under1s']} |")
lines+=['','## 物体随机化实测汇总','','同一 seed 的四个模型配置初始 DR 汇总完全相同（汇总相同本身不证明逐环境完全一致）。以下质量为随机化传入物理引擎的参数日志，不采用旧质量缓存。','','| seed | 最轻/g | 最重/g | 均值/g | 标准差/g |','|---:|---:|---:|---:|---:|']
for seed,v in physics.items():
 m=v['manip_obj/rigid_body_properties']['mass'];lines.append('| '+seed+' | '+' | '.join(f'{m[k]*1000:.3f}' for k in ['min','max','mean','std'])+' |')
lines+=['','逐 seed 摩擦等完整 min/max/mean/std 见 results.json。历史日志未保存逐环境物体质量与刚度、阻尼、尺寸的完整映射，无法可靠给最短案例附上精确物理参数。原始日志亦出现质量更新缓存缺陷警告；随机外力计算所用缓存应与实际物理质量区分。','','## 最短案例','']
for x in runs:lines.append(f"- seed{x['seed']} {x['prior']} scale{x['scale']}: {x['min_s']:.3f}s，env={[v['env'] for v in x['shortest']]}，结束原因={sorted(set(v['reason'] for v in x['shortest']))}")
(out/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
