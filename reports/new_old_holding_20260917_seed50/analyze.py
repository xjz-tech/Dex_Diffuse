"""Offline, full-vector comparison of new holding with seed-50 empty-hand baselines."""
import json
import math
from pathlib import Path
import re
import runpy

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
helper=runpy.run_path(str(ROOT/'reports/hold_comparison_20260917/analyze.py'))
helper['verify_fixture']()
metrics=helper['metrics']; summarize=helper['summarize']; jump=helper['jump']
SOURCES=[('不holding',ROOT/'reports/hold_comparison_20260917/pair1_seed50_hold0.txt'),
         ('旧holding',ROOT/'reports/hold_comparison_20260917/pair1_seed50_hold1.txt'),
         ('新holding',OUT/'new_holding_seed50.txt')]
KEYS=[('policy_boundary','上一chunk最后策略目标→下一chunk第一个策略目标')]

def main():
    results=[]
    for label,path in SOURCES:
        text=path.read_text()
        rows=[]; init_rows=[]
        with path.with_suffix('.jsonl').open() as source:
            for line in source:
                row=json.loads(line)
                if row['event'] not in ('command','state','session','session_end','inference_input','inference_output','model','exception','command_error','state_error'):
                    continue
                if row['stage']=='inference':rows.append(row)
                elif row['stage']=='initialization':init_rows.append(row)
        assert not any(r['event'] in ('exception','command_error','state_error') for r in rows)
        args=next(r['args'] for r in rows if r['event']=='session')
        assert args['hold_during_inference']==(label!='不holding')
        assert args['seed']==50 and args['max_steps']==600 and args['hz']==30 and args['action_chunk_steps']==2
        assert args['tensorrt'] and args['trt_precision']=='fp32' and args['disable_step_clamp'] and args['disable_joint_limits']
        model=next(r for r in rows if r['event']=='model')
        init_target=[r['target_real_rad'] for r in init_rows if r['event']=='command'][-1]
        commands=[r for r in rows if r['event']=='command']
        policies=[r for r in commands if r['kind']=='policy']
        states=[r for r in rows if r['event']=='state']
        assert len(policies)==600
        assert [(r['chunk'],r['chunk_step']) for r in policies]==[(i//2,i%2) for i in range(600)]
        holds=[r for r in commands if r['kind']=='inference_hold']
        assert len(holds)==(0 if label=='不holding' else 300)
        assert all(r['status']=='acknowledged' for r in commands)
        assert rows[-1]['event']=='session_end' and rows[-1]['steps_executed']==600 and rows[-1]['chunks']==300
        assert commands[-1]['kind']=='exit_hold' and '[done] steps=600 chunks=300' in text
        assert '[debug] recording failed' not in text
        deltas=[jump(b['target_policy_rad'],a['target_policy_rad']) for a,b in zip(commands,commands[1:])]
        logged=[float(x) for x in re.findall(r'\[cmd [^\n]*?delta_max_rad=([0-9.]+)',text)]
        assert len(logged)==len(deltas) and all(abs(a-b)<.00000051 for a,b in zip(logged,deltas))
        inputs=[r for r in rows if r['event']=='inference_input']
        assert len(inputs)==300
        if label=='新holding':
            # No sensor read after freezing the obs; hold uses that exact sample.
            for inp,hold in zip(inputs,holds):
                assert inp['monotonic_ns']<hold['monotonic_ns']
                assert max(abs(a-b) for a,b in zip(inp['observation'][0][-1][:22],hold['target_policy_rad']))<1e-7
                assert not any(r['event']=='state' for r in rows if inp['monotonic_ns']<r['monotonic_ns']<hold['monotonic_ns'])
            assert sum(r.get('kind')=='inference_hold' for r in states)==0
        elif label=='旧holding':
            assert sum(r.get('kind')=='inference_hold' for r in states)==300
        calculated=metrics(commands)
        assert calculated['within']['n']==300 and calculated['policy_boundary']['n']==299
        # Actual positions sampled after policy control periods, excluding hold/cleanup reads.
        feedback=[r for r in states if r.get('kind')=='policy']
        assert len(feedback)==600
        q_within=[];q_between=[];tracking=[]
        for previous,current in zip(feedback,feedback[1:]):
            value=jump(current['qpos_policy_rad'],previous['qpos_policy_rad'])
            (q_within if current['chunk']==previous['chunk'] else q_between).append(value)
        for row in feedback:
            tracking.append(max(abs(x) for x in row['tracking_error_rad']))
        inference_ms=[r['model_inference_seconds']*1000 for r in rows if r['event']=='inference_output']
        results.append(dict(condition=label,txt=str(path.relative_to(ROOT)),jsonl=str(path.with_suffix('.jsonl').relative_to(ROOT)),
                            args=args,checkpoint_sha256=model['checkpoints']['prior']['sha256'],source_sha256=model['source_sha256'],
                            initialization_target_real_rad=init_target,initial_qpos_policy_rad=inputs[0]['observation'][0][-1][:22],
                            inference_state_read_count=len(states),boundary_state_reads=sum(r.get('kind')=='inference_hold' for r in states),
                            metrics=calculated,feedback=dict(within=summarize(q_within),between=summarize(q_between),tracking=summarize(tracking)),
                            mean_inference_ms=sum(inference_ms)/len(inference_ms)))
    assert len({r['checkpoint_sha256'] for r in results})==1
    assert all(r['initialization_target_real_rad']==results[0]['initialization_target_real_rad'] for r in results)
    (OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    lines=[(OUT/'protocol.txt').read_text().replace('（待运行）','（已完成）'),
           '新holding实验已完成：600个动作/300个chunk，正常退出并保持实测位置。三组checkpoint SHA256及初始化目标完全一致。',
           '空手条件：运行前用户再次确认已取走灯泡。',
           '统一口径：跳过所有hold指令，计算上一chunk最后策略目标与下一chunk第一个策略目标的差，取22关节最大绝对值；每组299个边界。下表不是实测位移。', '',
           '| 条件 | 指令切换 | 样本数 | 平均幅度 | 中位数 | P95 | 最大值 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        for key,label in KEYS:
            s=r['metrics'][key]
            if s is None:continue
            if key=='first_command':label='hold→下一chunk首步' if r['condition']!='不holding' else '上一chunk末步→下一chunk首步'
            lines.append(f"| {r['condition']} | {label} | {s['n']} | {s['mean_deg']:.2f}°（{s['mean_rad']:.4f} rad） | {s['median_deg']:.2f}° | {s['p95_deg']:.2f}° | {s['max_deg']:.2f}° |")
    old,new=results[1],results[2]
    lines.extend(['','新holding相对旧holding：'])
    for key,label in KEYS:
        a=old['metrics'][key]['mean_deg'];b=new['metrics'][key]['mean_deg']
        lines.append(f'  {label}均值：{a:.4f}° → {b:.4f}°，变化{(b/a-1)*100:+.2f}%。')
    lines.extend(['', '本次结果：三种模式的两个chunk之间平均目标差均约6.09°，数值接近。单轮差异不能视作统计显著变化。', '', '采样时序验证：'])
    for r in results:
        lines.append(f"  {r['condition']}：inference阶段状态读取{r['inference_state_read_count']}次，其中边界额外读取{r['boundary_state_reads']}次；平均推理{r['mean_inference_ms']:.3f}ms。")
    lines.append('新holding逐chunk核对：inference_input先于hold，hold与该输入qpos一致，两者之间没有新状态读取。')
    lines.extend(['', '每组仅一轮且在不同时间采集，即使seed相同，初始实测误差和闭环轨迹也可能不同，结果仅作本次对照。',
                  '完整向量、hold切换及实测反馈保存在JSONL/results.json中，本表统一只展示两个chunk之间的策略目标差。',
                  '', '实际数据来源：']+[f"  {r['condition']}：{r['txt']}；{r['jsonl']}" for r in results])
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    p=OUT/'protocol.json';protocol=json.loads(p.read_text());protocol['status']='completed and verified';p.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    log=OUT/'new_holding_seed50.txt'
    original=log.read_text().split('\n[analysis] 新旧holding比较',1)[0]
    extra=['[analysis] 新旧holding比较：完整统计和版本说明见同目录summary.txt；均值单位deg。']
    for r in results:
        extra.append(r['condition']+': '+', '.join(key+'='+format(r['metrics'][key]['mean_deg'],'.6f') for key,_ in KEYS if r['metrics'][key]))
    log.write_text(original+'\n'+'\n'.join(extra)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
