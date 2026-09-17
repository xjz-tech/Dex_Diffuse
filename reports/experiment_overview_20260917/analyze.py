"""Recompute all archived successful inference trials using explicit metrics."""
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
METRICS={'within':'chunk内action→action','between':'跨chunk action→action',
         'new_action_qpos':'跨chunk新action−最近qpos','end_error':'动作周期末action−qpos'}

def stat(x):
    x=np.asarray(x,float)
    return dict(n=len(x),mean_deg=float(np.rad2deg(x.mean())),
                median_deg=float(np.rad2deg(np.median(x))),
                p95_deg=float(np.rad2deg(np.quantile(x,.95))),max_deg=float(np.rad2deg(x.max())))

def diff(a,b):
    assert len(a)==len(b)==22
    return float(np.max(np.abs(np.asarray(a)-b)))

def main():
    sources=[]
    for pair,seed in enumerate((50,51,52),1):
        for hold in (0,1):
            group='空手原初始化/'+('旧holding' if hold else '不holding')
            sources.append((group,f'seed={seed}',f'reports/hold_comparison_20260917/pair{pair}_seed{seed}_hold{hold}.jsonl'))
    sources.append(('空手原初始化/新holding','seed=50','reports/new_old_holding_20260917_seed50/new_holding_seed50.jsonl'))
    sources.append(('放灯泡/早期不holding','16:07:15','reports/bulb_hold_comparison_20260917_160715/bulb_no_hold_160715.jsonl'))
    for stamp in ('163706','163749'):
        sources.append(('放灯泡/近期不holding',stamp,f'reports/bulb_no_hold_20260917_1637/bulb_no_hold_{stamp}.jsonl'))
    for stamp in ('164011','164056'):
        sources.append(('放灯泡/新holding',stamp,f'reports/bulb_new_hold_comparison_20260917_1640/bulb_new_hold_{stamp}.jsonl'))
    for i in (1,2,3):
        sources.append((f'空手数据集初始化{i}/不holding','seed=50',f'reports/dataset_initialization_no_hold_20260917/dataset_pose_{i}_seed50.jsonl'))
    results=[];groups={}
    for group,label,source in sources:
        rows=[]
        for line in (ROOT/source).open():
            r=json.loads(line)
            if r['stage']=='inference' and r['event'] in ('command','state','session','session_end'):rows.append(r)
        policy=[r for r in rows if r['event']=='command' and r.get('kind')=='policy']
        assert all(r['status']=='acknowledged' for r in policy)
        samples={k:[] for k in METRICS}
        ages=[]
        for a,b in zip(policy,policy[1:]):
            key='within' if a['chunk']==b['chunk'] else 'between'
            samples[key].append(diff(a['target_policy_rad'],b['target_policy_rad']))
            if key=='between':
                samples['new_action_qpos'].append(diff(b['target_policy_rad'],b['latest_qpos_policy_rad']))
                ages.append((b['send_start_ns']-b['latest_state_read_end_ns'])/1e6)
        for r in rows:
            if r['event']=='state' and r.get('kind')=='policy':
                samples['end_error'].append(diff(r['target_policy_rad'],r['qpos_policy_rad']))
        end=next(r for r in reversed(rows) if r['event']=='session_end')
        assert len(samples['end_error'])==end['steps_executed']
        result=dict(group=group,label=label,jsonl=source,txt=str(Path(source).with_suffix('.txt')),
                    policy_commands=len(policy),completed_steps=end['steps_executed'],
                    qpos_age_ms_mean=float(np.mean(ages)),metrics={k:stat(v) for k,v in samples.items()})
        results.append(result)
        dst=groups.setdefault(group,{k:[] for k in METRICS})
        for k,v in samples.items():dst[k].extend(v)
    aggregate={g:{k:stat(v) for k,v in data.items()} for g,data in groups.items()}
    (OUT/'results.json').write_text(json.dumps(dict(runs=results,pooled=aggregate),ensure_ascii=False,indent=2)+'\n')
    lines=['截至2026-09-17：所有已归档对照实验统一口径汇总',
           '共15次有完整JSONL的推理实验，另有1次放灯泡旧holding仅有TXT。归档副本不重复计数；两次初始化失败未开始推理，单独留档、不计入。',
           '每个角度样本=max_j(abs(差值[j]))，即22关节中最大绝对差；下面均值是对这些最大值沿时间平均，不是22关节的平均误差。',
           'action→action仅相邻policy目标，跳过holding；新action−qpos使用发送前最近实测值，跨chunk时约已过去一次推理时间。',
           '周期末误差使用完成该动作周期后的qpos，是评估一个控制周期后是否跟上目标的指标。',
           '所有指令差排除首个policy动作及退出hold；中断前已下发但未完成周期的末指令计入action差，周期末误差只计完成周期。',
           '同组按样本汇总；历史seed51/52在此作为已有记录展示，不代表新增运行。不同组时长、种子和接触状态可能不同。','',
           '| 条件 | 轮数 | chunk内action差均值 | 跨chunk action差均值 | 跨chunk新action−qpos均值 | 周期末误差均值 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for g,m in aggregate.items():
        n=sum(r['group']==g for r in results)
        lines.append(f'| {g} | {n} | '+' | '.join(f"{m[k]['mean_deg']:.2f}°" for k in METRICS)+' |')
    lines += ['','原seed50空手单轮对照：']
    for r in results:
        if r['label']=='seed=50' and r['group'].startswith('空手原初始化'):
            lines.append(r['group']+'：'+', '.join(f"{METRICS[k]}均值={r['metrics'][k]['mean_deg']:.2f}°" for k in METRICS))
    lines += ['','放灯泡旧holding（15:22:40，仅TXT，608policy动作）：',
              'chunk内action差：n304，均值5.78°，最大7.77°。',
              '上一action→hold(qpos)：n303，均值9.46°，最大39.22°。',
              'hold(qpos)→下一action：n303，均值12.83°，最大40.78°。',
              '无法还原跳过hold的跨chunk action差，也无法从此TXT还原全部周期末误差。不能将40.78°当作纯action跨chunk变化。',
              '来源：reports/hold_comparison_20260917/bulb_holding_original.txt；其副本不重复统计。','',
              '逐轮完整指标（均值/中位数/P95/最大值，单位度）：']
    for r in results:
        lines += ['',f"{r['group']} {r['label']}；已下发{r['policy_commands']}、完成{r['completed_steps']}步；跨chunk qpos平均年龄{r['qpos_age_ms_mean']:.3f}ms。",r['txt'],r['jsonl']]
        for k,label in METRICS.items():
            s=r['metrics'][k];lines.append(f"  {label}：n={s['n']}；"+'/'.join(f'{s[v]:.4f}' for v in ('mean_deg','median_deg','p95_deg','max_deg')))
    lines += ['','初始化替换实验（每种一轮600步；<=0.09rad=5.15662°）：',
              '逐关节相邻action差占比：原83.69%；数据集1=83.41%，2=84.65%，3=83.84%；三组汇总83.97%。',
              '整步22关节都达标：原4.01%；数据集1=4.01%，2=8.68%，3=3.51%；三组合并5.40%。',
              '这些是action→action分布，不能用于声称换初始化减小了action−qpos误差。详情及向量见reports/dataset_initialization_no_hold_20260917/summary.txt。','',
              '解释边界：目前纯action跨chunk变化与chunk内相近；放灯泡时，目标与实际位置的差及尾部偏差更值得关注。',
              '新action−旧qpos包含新目标本身的变化，不能全部归因于跟踪滞后；周期末误差表明执行完周期仍存在偏差。',
              'holding使实际指令序列变成上一action→实测qpos→新action，新增切换包含目标与实测位置偏差；这不等于证明模型预测的跨chunk action突变。',
              '这些独立闭环记录不能单独确定偏差来自接触、关节约束、驱动响应还是时延，不能证明holding是全部误差的原因。']
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:33]))

if __name__=='__main__':main()
