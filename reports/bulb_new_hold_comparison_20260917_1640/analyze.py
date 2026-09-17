"""Only compare consecutive policy targets across chunk boundaries; omit all holds."""
import json
import math
from pathlib import Path
import re
import statistics

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]

def summarize(values):
    a=sorted(values)
    k=(len(a)-1)*.95
    p95=a[int(k)]+(a[math.ceil(k)]-a[int(k)])*(k-int(k))
    return dict(n=len(a),mean_rad=statistics.mean(a),mean_deg=math.degrees(statistics.mean(a)),
                median_deg=math.degrees(statistics.median(a)),p95_deg=math.degrees(p95),max_deg=math.degrees(max(a)))

def delta(a,b):
    assert len(a)==len(b)==22 and all(math.isfinite(x) for x in a+b)
    return max(abs(x-y) for x,y in zip(a,b))

def main():
    files=[('不holding',ROOT/'reports/bulb_no_hold_20260917_1637/bulb_no_hold_163706.txt'),
           ('不holding',ROOT/'reports/bulb_no_hold_20260917_1637/bulb_no_hold_163749.txt')]
    files += [('新holding',ROOT/r['txt']) for r in json.loads((OUT/'manifest.json').read_text())]
    runs=[]
    for condition,txt in files:
        commands=[]; sessions=[]; inputs=[]; ends=[]; init=[]; model=None; hold_states=0
        with txt.with_suffix('.jsonl').open() as source:
            for line in source:
                r=json.loads(line)
                if r['stage']=='initialization' and r['event']=='command':init.append(r['target_real_rad'])
                if r['stage']!='inference':continue
                if r['event']=='command':commands.append(r)
                if r['event']=='session':sessions.append(r)
                if r['event']=='model':model=r
                if r['event']=='inference_input':inputs.append(r)
                if r['event']=='session_end':ends.append(r)
                if r['event']=='state' and r.get('kind')=='inference_hold':hold_states+=1
                assert r['event'] not in ('command_error','state_error'),r
        args=sessions[0]['args'];expected=condition=='新holding'
        assert args['hold_during_inference']==expected and args['seed']==50
        assert args['hz']==30 and args['action_chunk_steps']==2 and args['trt_precision']=='fp32' and args['tensorrt']
        assert args['disable_joint_limits'] and args['disable_step_clamp'] and args['inference_steps']==4
        assert ends and commands[-1]['kind']=='exit_hold' and all(r['status']=='acknowledged' for r in commands)
        assert hold_states==0
        policies=[r for r in commands if r['kind']=='policy']
        assert [(r['chunk'],r['chunk_step']) for r in policies]==[(i//2,i%2) for i in range(len(policies))]
        holds=[r for r in commands if r['kind']=='inference_hold']
        assert bool(holds)==expected
        if expected:
            by_chunk={r['chunk']:r for r in inputs}
            for h in holds:
                inp=by_chunk[h['chunk']]
                assert inp['monotonic_ns']<h['monotonic_ns']
                assert delta(inp['observation'][0][-1][:22],h['target_policy_rad'])<1e-7
        values=[]
        for a,b in zip(policies,policies[1:]):
            if b['chunk']!=a['chunk']:
                assert a['chunk_step']==1 and b['chunk_step']==0 and b['chunk']==a['chunk']+1
                values.append(delta(a['target_policy_rad'],b['target_policy_rad']))
        text=txt.read_text()
        logged=[float(v) for v in re.findall(r'\[cmd [^\n]*?delta_max_rad=([0-9.]+)',text)]
        actual=[delta(a['target_policy_rad'],b['target_policy_rad']) for a,b in zip(commands,commands[1:])]
        assert len(logged)==len(actual) and all(abs(a-b)<.00000051 for a,b in zip(logged,actual))
        started=re.search(r'^\[real\] started: (.+)$',text,re.M).group(1)
        runs.append(dict(condition=condition,started=started,txt=str(txt.relative_to(ROOT)),jsonl=str(txt.with_suffix('.jsonl').relative_to(ROOT)),
                         seed=args['seed'],sent_policy_commands=len(policies),completed_steps=ends[-1]['steps_executed'],
                         checkpoint_sha256=model['checkpoints']['prior']['sha256'],source_sha256=model['source_sha256'],
                         initialization_target_real_rad=init[-1],metrics=summarize(values),boundary_deltas_rad=values))
    assert len({r['checkpoint_sha256'] for r in runs})==1
    assert all(r['initialization_target_real_rad']==runs[0]['initialization_target_real_rad'] for r in runs)
    assert len({r['source_sha256']['eval/real/inference_real.py'] for r in runs})==1
    common=min(r['metrics']['n'] for r in runs)
    groups={}
    for condition in ('不holding','新holding'):
        selected=[r for r in runs if r['condition']==condition]
        groups[condition]=dict(pooled=summarize([x for r in selected for x in r['boundary_deltas_rad']]),
            mean_of_run_means_deg=statistics.mean(r['metrics']['mean_deg'] for r in selected),
            matched_prefix=summarize([x for r in selected for x in r['boundary_deltas_rad'][:common]]))
    for r in runs:r['matched_prefix_metrics']=summarize(r['boundary_deltas_rad'][:common])
    result=dict(runs=runs,groups=groups,matched_boundary_count_per_run=common)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['放灯泡：最新两次新holding vs 前两次不holding',
           '物体条件由用户确认；holding开关由TXT和JSONL交叉核对；新holding已核对固定obs先于hold、目标来自同一qpos，边界不额外读取。',
           '共同参数：seed=50，chunk=2，30Hz，DDIM=4步，TensorRT FP32，软件限幅关闭；四次checkpoint哈希、inference_real.py哈希、初始化目标均一致。',
           '统一统计口径：max_j |下一chunk第一个策略目标[j] − 上一chunk最后策略目标[j]|。跳过所有hold和exit_hold，首个chunk不计；角度差不是实际位移或速度。',
           '按已确认下发的policy指令统计，包含中断前最后一条已下发但可能未执行完一个周期的指令。', '',
           '| 模式 | 开始时间 | 边界样本数 | 平均幅度 | 中位数 | P95 | 最大值 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in runs:
        s=r['metrics'];lines.append(f"| {r['condition']} | {r['started'][11:19]} | {s['n']} | {s['mean_deg']:.2f}°（{s['mean_rad']:.4f} rad） | {s['median_deg']:.2f}° | {s['p95_deg']:.2f}° | {s['max_deg']:.2f}° |")
    lines.extend(['', '合并各条件两次记录（每个边界等权；不是两轮均值等权）：',
                  '| 模式 | 边界样本数 | 平均幅度 | 中位数 | P95 | 最大值 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for condition,g in groups.items():
        s=g['pooled'];lines.append(f"| {condition} | {s['n']} | {s['mean_deg']:.4f}° | {s['median_deg']:.4f}° | {s['p95_deg']:.4f}° | {s['max_deg']:.4f}° |")
    off=groups['不holding']['pooled']['mean_deg'];on=groups['新holding']['pooled']['mean_deg']
    lines.append(f'合并平均角度差：新holding相对不holding变化{(on/off-1)*100:+.2f}%。')
    lines.append('每轮均值等权：'+ '；'.join(f"{k}={v['mean_of_run_means_deg']:.4f}°" for k,v in groups.items()))
    lines.extend(['',f'相同长度核查：每次只取前{common}个边界（chunk 0→1到{common-1}→{common}），每种条件合并{2*common}个边界：'])
    for condition,g in groups.items():
        s=g['matched_prefix'];lines.append(f"  {condition}: mean={s['mean_deg']:.4f}°, median={s['median_deg']:.4f}°, P95={s['p95_deg']:.4f}°, max={s['max_deg']:.4f}°")
    lines.extend(['', '每种条件仅两次，且为不同时间的闭环运行；统计是描述性对比，不能排除灯泡接触、起始实测姿态及轨迹差异。', '', '数据来源及完整执行步数：'])
    for r in runs:lines.append(f"  {r['condition']} {r['started']}: {r['txt']}；同名JSONL；已下发{r['sent_policy_commands']}条policy，完整执行{r['completed_steps']}步。")
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
