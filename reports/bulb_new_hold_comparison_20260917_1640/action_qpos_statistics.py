"""Compare action gaps and tracking gaps with explicit sample timing; offline only."""
import json
import math
from pathlib import Path
import statistics

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
LABELS={
 'action_between':'action间：跨chunk末步→首步（跳过hold）',
 'action_within':'action间：chunk内第0步→第1步',
 'action_qpos_between':'跨chunk首步：新action−最近实测qpos',
 'action_qpos_within':'chunk内第1步：新action−最近实测qpos',
 'end_tracking':'完成动作周期末：该action−周期末实测qpos',
}

def quantile(a,p):
    a=sorted(a);i=(len(a)-1)*p
    return a[int(i)]+(a[math.ceil(i)]-a[int(i)])*(i-int(i))

def stat(a):
    return dict(n=len(a),mean_deg=math.degrees(statistics.mean(a)),
                **{name:math.degrees(quantile(a,p)) for name,p in [('p25_deg',.25),('median_deg',.5),('p75_deg',.75),('p95_deg',.95)]},
                max_deg=math.degrees(max(a)))

def difference(a,b):
    assert len(a)==len(b)==22 and all(math.isfinite(x) for x in a+b)
    return max(abs(x-y) for x,y in zip(a,b))

def main():
    assert quantile([0,1,2,3,4],.25)==1 and quantile([0,1,2,3,4],.75)==3
    prior=json.loads((OUT/'results.json').read_text())
    grouped={};details=[]
    for source in prior['runs']:
        condition=source['condition']
        rows=[]
        with (ROOT/source['jsonl']).open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['stage']=='inference' and row['event'] in ('command','state','session_end'):rows.append(row)
        policy=[r for r in rows if r['event']=='command' and r['kind']=='policy']
        samples={k:[] for k in LABELS};ages={'between':[],'within':[]};peaks={}
        for a,b in zip(policy,policy[1:]):
            phase='within' if a['chunk']==b['chunk'] else 'between'
            aa=difference(a['target_policy_rad'],b['target_policy_rad'])
            aq=difference(b['target_policy_rad'],b['latest_qpos_policy_rad'])
            assert abs(aq-max(abs(x) for x in b['tracking_before_rad']))<1e-10
            samples['action_'+phase].append(aa)
            samples['action_qpos_'+phase].append(aq)
            age=(b['send_start_ns']-b['latest_state_read_end_ns'])/1e6
            assert age>=0;ages[phase].append(age)
            if 'action_qpos_'+phase not in peaks or aq>peaks['action_qpos_'+phase]['delta_rad']:
                j=max(range(22),key=lambda j:abs(b['target_policy_rad'][j]-b['latest_qpos_policy_rad'][j]))
                peaks['action_qpos_'+phase]=dict(delta_rad=aq,chunk=b['chunk'],chunk_step=b['chunk_step'],joint_index=j,
                    qpos_rad=b['latest_qpos_policy_rad'][j],action_rad=b['target_policy_rad'][j],sample_age_ms=age)
        feedback=[r for r in rows if r['event']=='state' and r.get('kind')=='policy']
        end=next(r for r in reversed(rows) if r['event']=='session_end')
        assert len(feedback)==end['steps_executed']
        for r in feedback:
            v=difference(r['target_policy_rad'],r['qpos_policy_rad'])
            assert abs(v-max(abs(x) for x in r['tracking_error_rad']))<1e-10
            samples['end_tracking'].append(v)
        assert stat(samples['action_between'])['n']==source['metrics']['n']
        assert abs(stat(samples['action_between'])['mean_deg']-source['metrics']['mean_deg'])<1e-10
        result=dict(condition=condition,started=source['started'],source_txt=source['txt'],source_jsonl=source['jsonl'],
                    metrics={k:stat(v) for k,v in samples.items()},largest_gaps=peaks,
                    qpos_sample_age_ms={k:dict(mean=statistics.mean(v),max=max(v)) for k,v in ages.items()})
        details.append(result)
        if condition not in grouped:grouped[condition]={k:[] for k in LABELS}
        for key,values in samples.items():grouped[condition][key].extend(values)
    groups={c:{k:stat(v) for k,v in samples.items()} for c,samples in grouped.items()}
    (OUT/'action_qpos_statistics.json').write_text(json.dumps(dict(runs=details,pooled=groups),ensure_ascii=False,indent=2)+'\n')
    lines=['最近四次放灯泡：qpos-action差与action-action差的分位数',
           '不holding：16:37:06、16:37:49；新holding：16:40:11、16:40:56；每种条件合并两次记录，每个样本等权。',
           '每个样本先计算22个关节绝对角度差的最大值，再沿时间求分位数；不是把全部关节差混在一起求分位数。',
           '分位数采用排序后位置(n-1)*p线性插值；单位均为度。',
           'action-qpos：本次下发策略目标减该指令之前最近一次实测qpos。chunk首步qpos在推理前采样，通常已过去约一次推理的时间，并非指令下发瞬间再次测得的位置。',
           '不holding期间手可能继续移动，因此这个差不能直接称为动作发送瞬间的真实跟踪误差。',
           'action-action：仅相邻策略目标，不包含hold或exit_hold。动作前差统计排除每轮首个策略动作；已下发但中断周期未完成的末条策略动作仍计入。',
           '周期末跟踪误差另列：用完成该动作周期后读取的qpos与该动作目标相比，只统计已完成周期。', '',
           '| 模式 | 指标 | 样本数 | 平均 | P25 | 中位数 | P75 | P95 | 最大值 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key,label in LABELS.items():
        for condition,g in groups.items():
            s=g[key]
            lines.append(f"| {condition} | {label} | {s['n']} | "+' | '.join(f"{s[k]:.2f}°" for k in ('mean_deg','p25_deg','median_deg','p75_deg','p95_deg','max_deg'))+' |')
    lines.extend(['','采样年龄（最近实测qpos返回至策略指令开始发送）：'])
    for r in details:
        a=r['qpos_sample_age_ms'];lines.append(f"  {r['condition']} {r['started']}: 跨chunk均值{a['between']['mean']:.3f}ms，chunk内均值{a['within']['mean']:.3f}ms。")
    lines.extend(['','来源（TXT同名JSONL提供完整向量）：']+[r['source_jsonl'] for r in details])
    (OUT/'action_qpos_statistics.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
