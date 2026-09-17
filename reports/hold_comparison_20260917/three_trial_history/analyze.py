"""Recompute the empty-hand comparison from complete acknowledged command vectors."""
import csv
import json
import math
import re
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
LABELS = {
    'within': 'chunk内策略步0→步1',
    'policy_boundary': '纯策略跨chunk末步→首步（跳过hold）',
    'last_to_hold': '上一chunk末步→hold',
    'first_command': 'chunk首步相对前一实际指令（holding时为hold→首步）',
    'boundary_peak': '每个边界实际指令跳变峰值（holding取两次切换较大值）',
}

def jump(a, b):
    assert len(a) == len(b) and all(math.isfinite(x) for x in a+b)
    return max(abs(x-y) for x,y in zip(a,b))

def summarize(values):
    if not values:
        return None
    ordered = sorted(values)
    i = (len(ordered)-1)*.95
    p95 = ordered[int(i)] + (ordered[math.ceil(i)]-ordered[int(i)])*(i-int(i))
    return dict(n=len(values), mean_rad=statistics.mean(values),
                mean_deg=math.degrees(statistics.mean(values)),
                median_deg=math.degrees(statistics.median(values)),
                p95_deg=math.degrees(p95), max_deg=math.degrees(max(values)))

def metrics(commands):
    result = {key: [] for key in LABELS}
    policy = None
    previous = None
    pending_hold = None
    for row in commands:
        if row['kind'] not in ('policy', 'inference_hold'):
            continue
        target = row['target_policy_rad']
        if row['kind'] == 'inference_hold':
            if policy is not None:
                pending_hold = jump(target, policy['target_policy_rad'])
                result['last_to_hold'].append(pending_hold)
        else:
            if policy is not None:
                direct = jump(target, policy['target_policy_rad'])
                if row['chunk'] == policy['chunk']:
                    result['within'].append(direct)
                else:
                    result['policy_boundary'].append(direct)
                    actual = jump(target, previous['target_policy_rad'])
                    result['first_command'].append(actual)
                    result['boundary_peak'].append(max(actual, pending_hold or 0.0))
            policy = row
            pending_hold = None
        previous = row
    return {k:summarize(v) for k,v in result.items()}

def verify_fixture():
    # Two 2-joint chunks: within jumps 1 and 2, direct boundary 2,
    # inserted hold returns by 4 and resumes by 6. Startup/exit excluded.
    rows = [dict(kind='policy',chunk=0,target_policy_rad=[0.,0.]),
            dict(kind='policy',chunk=0,target_policy_rad=[1.,0.]),
            dict(kind='inference_hold',chunk=1,target_policy_rad=[-3.,0.]),
            dict(kind='policy',chunk=1,target_policy_rad=[3.,0.]),
            dict(kind='policy',chunk=1,target_policy_rad=[5.,0.]),
            dict(kind='exit_hold',chunk=2,target_policy_rad=[99.,0.])]
    got = metrics(rows)
    for key, expected in [('within',1.5),('policy_boundary',2.),('last_to_hold',4.),
                          ('first_command',6.),('boundary_peak',6.)]:
        assert got[key]['mean_rad'] == expected, (key,got[key])
    without = metrics([r for r in rows if r['kind'] != 'inference_hold'])
    assert without['first_command']['mean_rad'] == 2.
    assert without['last_to_hold'] is None

def main():
    verify_fixture()
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert len(manifest) == 6, 'Need all six trials'
    results=[]
    for run in manifest:
        text = Path(run['log']).read_text()
        records=[]
        with Path(run['log']).with_suffix('.jsonl').open() as source:
            for line in source:
                row=json.loads(line)
                if row['stage']=='inference' and row['event'] in (
                    'command','session','session_end','exception','command_error','state_error'):
                    records.append(row)
        assert not any(r['event'] in ('exception','command_error','state_error') for r in records)
        session=next(r for r in records if r['event']=='session')
        args=session['args']
        assert args['hold_during_inference']==bool(run['hold']) and args['seed']==run['seed']
        assert args['max_steps']==600 and args['action_chunk_steps']==2 and args['hz']==30
        assert args['tensorrt'] and args['trt_precision']=='fp32'
        commands=[r for r in records if r['event']=='command']
        policies=[r for r in commands if r['kind']=='policy']
        assert len(policies)==600
        assert [(r['chunk'],r['chunk_step']) for r in policies]==[(i//2,i%2) for i in range(600)]
        assert sum(r['kind']=='inference_hold' for r in commands)==300*run['hold']
        assert all(r['status']=='acknowledged' and len(r['target_policy_rad'])==22 for r in commands)
        assert records[-1]['event']=='session_end' and records[-1]['steps_executed']==600
        assert records[-1]['chunks']==300 and commands[-1]['kind']=='exit_hold'
        assert '[done] steps=600 chunks=300' in text and '[cleanup] SharpA held' in text
        assert '[debug] recording failed' not in text and '[experiment]' in text
        # Cross-check the human-readable log against independently parsed vectors.
        logged = [float(value) for value in re.findall(r'\[cmd [^\n]*?delta_max_rad=([0-9.]+)', text)]
        vector_jumps = [jump(b['target_policy_rad'], a['target_policy_rad'])
                        for a,b in zip(commands, commands[1:])]
        assert len(logged) == len(vector_jumps)
        assert all(abs(a-b) <= 0.00000051 for a,b in zip(logged, vector_jumps))
        calculated=metrics(commands)
        assert calculated['within']['n']==300 and calculated['policy_boundary']['n']==299
        if run['hold']:
            assert calculated['last_to_hold']['n']==299
        results.append(dict(**run, metrics=calculated))
    groups={}
    for hold in (0,1):
        trials=[r for r in results if r['hold']==hold]
        groups[str(hold)]={}
        for key in LABELS:
            stats=[r['metrics'][key] for r in trials if r['metrics'][key]]
            if stats:
                groups[str(hold)][key]=dict(
                    mean_of_trial_means_deg=statistics.mean(s['mean_deg'] for s in stats),
                    sd_of_trial_means_deg=statistics.stdev(s['mean_deg'] for s in stats),
                    mean_of_trial_p95_deg=statistics.mean(s['p95_deg'] for s in stats),
                    maximum_deg=max(s['max_deg'] for s in stats))
    (OUT/'results.json').write_text(json.dumps(dict(runs=results,groups=groups),ensure_ascii=False,indent=2)+'\n')
    lines=['空手无灯泡：holding vs 不 holding 真机对照实验',
           '日期：2026-09-17；用户已移走灯泡。',
           '每组3轮，每轮600动作/300chunk；配对种子50、51、52；顺序off/on, on/off, off/on。',
           '每轮ROTATE复位；TensorRT FP32，FUSED_DDIM=1，DDIM 4步，chunk=2，30Hz，无插值。',
           '保持原启动脚本的软件限幅关闭设置；结束时保持实测位置。',
           '跳变定义=max_j |target_next[j]-target_previous[j]|，是指令角度差，不是实测位移或速度。',
           '纯策略跨chunk跳过hold，实际边界峰值则取末步→hold与hold→首步两次变化的较大值。',
           '排除初始化、首个策略动作和退出hold；每轮内部300个、跨chunk299个样本。',
           '三轮等权平均；±为三轮均值的样本标准差，不是置信区间。',
           '每轮完整参数、22维目标和传感器记录见同名TXT/JSONL。', '']
    for run in results:
        block=[f"轮次{run['pair']} seed={run['seed']} holding={run['hold']}"]
        for key,stats in run['metrics'].items():
            if stats:
                block.append(f"  {LABELS[key]}: n={stats['n']}; mean={stats['mean_deg']:.6f} deg ({stats['mean_rad']:.8f} rad); median={stats['median_deg']:.6f}; P95={stats['p95_deg']:.6f}; max={stats['max_deg']:.6f} deg")
        lines.extend(block+[''])
        log=Path(run['log'])
        original=log.read_text().split('\n[analysis] 离线完整向量统计',1)[0]
        log.write_text(original+'\n[analysis] 离线完整向量统计（单位deg；跳变=max绝对关节差；排除初始化/首动作/退出hold）\n'+'\n'.join(block)+'\n')
    lines.append('三轮平均结果：')
    for hold in (0,1):
        lines.append(f'holding={hold}')
        for key,stats in groups[str(hold)].items():
            lines.append(f"  {LABELS[key]}: {stats['mean_of_trial_means_deg']:.6f} ± {stats['sd_of_trial_means_deg']:.6f} deg; 平均P95={stats['mean_of_trial_p95_deg']:.6f}; 全部最大={stats['maximum_deg']:.6f} deg")
    lines.append('\n对照比值 holding / 不holding：')
    for key in ('within','policy_boundary','first_command','boundary_peak'):
        ratio=groups['1'][key]['mean_of_trial_means_deg']/groups['0'][key]['mean_of_trial_means_deg']
        lines.append(f'  {LABELS[key]}: {ratio:.6f}x')
    off=groups['0']['first_command']['mean_of_trial_means_deg']
    on=groups['1']['first_command']['mean_of_trial_means_deg']
    lines.extend(['', f'结论：本次三组空手实验，holding实际边界首步平均跳变为{on:.4f}°，不holding为{off:.4f}°；holding增大{(on/off-1)*100:.2f}%。',
                  '纯策略跨chunk差值两种条件均约6°，且接近chunk内部；这组数据中的边界额外跳变主要来自hold指令切换。'])
    lines.extend(['', '解读范围：仅限本次空手、此checkpoint与参数、三组配对种子；闭环状态会因holding改变，不能当作相同轨迹的离线重放。',
                  '采样角度变化不能直接解释为机械瞬时位移或速度；边界处实际时间间隔也不同。',
                  '日志保留策略已改为永久保留全部编号历史，不再只保留10轮。'])
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':
    main()
