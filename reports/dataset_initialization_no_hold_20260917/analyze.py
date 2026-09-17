"""Offline action-increment statistics; never connects to hardware."""
import json
import math
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
THRESHOLD = .09

def stats(delta):
    delta = np.asarray(delta, dtype=float).reshape(-1, 22)
    assert len(delta) and np.isfinite(delta).all()
    peak = delta.max(axis=1)
    return dict(transitions=len(delta), joint_samples=delta.size,
                joint_within_count=int((delta <= THRESHOLD).sum()),
                joint_within_pct=float((delta <= THRESHOLD).mean()*100),
                all_joints_within_count=int((peak <= THRESHOLD).sum()),
                all_joints_within_pct=float((peak <= THRESHOLD).mean()*100),
                peak_mean_deg=float(np.rad2deg(peak.mean())),
                peak_percentiles_deg=dict(zip(('p25','median','p75','p95'),np.rad2deg(np.quantile(peak,[.25,.5,.75,.95])).tolist())),
                peak_max_deg=float(np.rad2deg(peak.max())),
                per_joint_within_pct=((delta <= THRESHOLD).mean(axis=0)*100).tolist())

def read_run(path, label, pose=None):
    records=[]
    for line in path.open():
        r=json.loads(line)
        if r['event'] in ('session','command','state','session_end','exception','command_error','model'):
            records.append(r)
    assert not [r for r in records if r['event'] in ('exception','command_error')]
    inference=[r for r in records if r['stage']=='inference']
    session=next(r for r in inference if r['event']=='session')
    model=next(r for r in inference if r['event']=='model')
    policy=[r for r in inference if r['event']=='command' and r['kind']=='policy']
    end=next(r for r in reversed(inference) if r['event']=='session_end')
    assert len(policy)==end['steps_executed']==600
    assert all(r['status']=='acknowledged' for r in policy)
    assert not [r for r in inference if r['event']=='command' and r['kind']=='inference_hold']
    assert sum(r.get('limit_clips',0)+r.get('step_clips',0) for r in policy)==0
    assert len([r for r in inference if r['event']=='state' and r.get('kind')=='policy'])==600
    args=session['args']
    assert args['hold_during_inference'] is False and args['disable_step_clamp'] and args['disable_joint_limits']
    init=[r for r in records if r['stage']=='initialization' and r['event']=='command' and r['kind']=='initial_pose'][-1]
    init_state=[r for r in records if r['stage']=='initialization' and r['event']=='state' and r['kind']=='initial_pose'][-1]
    if pose is not None:
        assert np.array_equal(init['target_policy_rad'],pose['qpos_policy_rad'])
        assert np.array_equal(init['target_real_rad'],pose['qpos_real_rad'])
    target=np.asarray([r['target_policy_rad'] for r in policy])
    delta=np.abs(np.diff(target,axis=0))
    between=np.asarray([a['chunk']!=b['chunk'] for a,b in zip(policy,policy[1:])])
    assert between.sum()==299 and (~between).sum()==300
    groups={'all':delta,'within_chunk':delta[~between], 'between_chunks':delta[between],
            'first_100_actions':delta[:99], 'remaining_500_actions':delta[99:]}
    first_qpos=next(r['qpos_policy_rad'] for r in inference if r['event']=='state')
    result=dict(label=label, txt=str(path.with_suffix('.txt').relative_to(ROOT)),jsonl=str(path.relative_to(ROOT)),
                args=args, checkpoint=model['checkpoints'], source_sha256=model['source_sha256'],
                init_target_policy_rad=init['target_policy_rad'], init_target_real_rad=init['target_real_rad'],
                init_final_qpos_policy_rad=init_state['qpos_policy_rad'],
                init_max_error_rad=float(np.max(abs(np.asarray(init_state['qpos_policy_rad'])-init['target_policy_rad']))),
                inference_first_qpos_policy_rad=first_qpos,
                metrics={k:stats(v) for k,v in groups.items()})
    return result, groups

def main():
    poses=json.loads((OUT/'selected_poses.json').read_text())
    baseline, _=read_run(ROOT/'reports/hold_comparison_20260917/pair1_seed50_hold0.jsonl','原随意初始化')
    runs=[baseline]; arrays=[]
    for i,pose in enumerate(poses['poses'],1):
        r,g=read_run(OUT/(pose['name']+'_seed50.jsonl'),f'数据集初始化{i}',pose)
        assert r['args']==baseline['args'], 'Inference arguments differ from historical baseline'
        assert r['checkpoint']==baseline['checkpoint'], 'Checkpoint identity changed'
        runs.append(r);arrays.append(g)
    pooled={k:stats(np.concatenate([a[k] for a in arrays])) for k in arrays[0]}
    payload=dict(threshold_rad=THRESHOLD, threshold_deg=math.degrees(THRESHOLD),
                 selection=poses, runs=runs, dataset_pooled=pooled,
                 limitations='历史基线，非同一时刻配对重跑；每个初始化仅一轮；真机测量/时序存在差异。初始化选择受可到位筛选影响。')
    (OUT/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    lines=['数据集三个初始化 vs 原随意初始化：空手、不holding',
           '共同配置：seed=50；每组600个policy动作、300个chunk；chunk=2；30Hz；TensorRT FP32/fused DDIM4；同一checkpoint。',
           '0.09rad = %.6f°。统计的是相邻policy action的绝对差，不是绝对关节位置，也不是action-qpos误差。'%math.degrees(THRESHOLD),
           '逐关节比例：所有时间×22关节的abs(action[t,j]-action[t-1,j])<=0.09的占比。',
           '整步比例：一次切换中22关节最大绝对变化<=0.09的占比，与之前max_joint统计口径一致。',
           '包含阈值端点，直接使用JSONL全精度数值，不四舍五入后比较。',
           '排除初始化、初始状态→首个policy动作和退出hold；每轮599次切换，其中chunk内300次、跨chunk299次。',
           'DISABLE_STEP_CLAMP=1、DISABLE_JOINT_LIMITS=1，所有记录clips=0。0.09rad仅为统计阈值，不启用限幅。',
           '初始化到位容差0.1rad与限位无关；每个最终姿态均通过原有到位检查，完整向量见下文。','',
           '| 初始化 | 范围 | 切换数 | 逐关节≤0.09rad | 整步最大≤0.09rad | 最大关节差均值 | P25 | 中位数 | P75 | 最大值 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    entries=[(r['label'],r['metrics']) for r in runs]+[('数据集三组汇总',pooled)]
    for label,metrics in entries:
        for key,name in [('all','全部'),('within_chunk','chunk内'),('between_chunks','跨chunk')]:
            s=metrics[key];p=s['peak_percentiles_deg']
            lines.append(f"| {label} | {name} | {s['transitions']} | {s['joint_within_pct']:.2f}% | {s['all_joints_within_pct']:.2f}% | {s['peak_mean_deg']:.2f}° | {p['p25']:.2f}° | {p['median']:.2f}° | {p['p75']:.2f}° | {s['peak_max_deg']:.2f}° |")
    base=baseline['metrics']['all'];agg=pooled['all']
    lines += ['',f"数据集三组相对原初始化：逐关节达标比例变化{agg['joint_within_pct']-base['joint_within_pct']:+.2f}个百分点；整步达标比例变化{agg['all_joints_within_pct']-base['all_joints_within_pct']:+.2f}个百分点。",
              '三组样本数相同，汇总比例等于三组比例的算术平均；不会跨实验末首动作计算差分。',
              '早期/后期（前100个动作内部99次切换；其余500次切换含第100→101动作）：']
    for r in runs:
        a=r['metrics']['first_100_actions'];b=r['metrics']['remaining_500_actions']
        lines.append(f"  {r['label']}：前期逐关节{a['joint_within_pct']:.2f}%/整步{a['all_joints_within_pct']:.2f}%；后期逐关节{b['joint_within_pct']:.2f}%/整步{b['all_joints_within_pct']:.2f}%。")
    lines += ['','初始化来源及选择说明：',poses['dataset_root'],poses['criteria'],poses['dataset_identity_note'],
              '姿态为数据集episode第0步robot/qpos原值，经policy→SDK关节重排后下发，未裁剪数值。并非对整个数据集均匀随机采样，不保证代表整体。',
              '数据集无图像/接触信息，本次不判断是否已抓到灯泡或朝向；用户确认灯泡已取走。',
              '原初始化为用户此前描述的随意挑选、看起来能抓灯泡的动作；使用历史日志实际下发向量，不以当前源文件反推。',
              '前两次候选在初始化阶段未达0.1rad容差，均未开始推理；见failed_initialization_attempt/及failed_initialization_attempt_2/。',
              '因此增加食指侧摆abs<=0.18rad、小指MCP_FE<=1.4rad选行条件；这只是数据选择，不是运行时限位。',
              'robot_init.py默认初始化未修改；本次通过报告目录initialize_pose.py在单独初始化进程中覆盖ROTATE槽位。',
              '注意：新旧运行推理参数和checkpoint身份一致，源代码版本有差异（旧基线采于holding改动前，二者均未启用holding）。详见results.json源代码哈希。',
              '每个初始化仅一轮，使用历史空手基线；这只能描述本次观测，不能据此证明换初始化导致普遍改善。','',
              '关节顺序（SDK/real）：'+json.dumps(poses['real_joint_names'],ensure_ascii=False)]
    for i,r in enumerate(runs):
        lines += ['',r['label'], 'TXT：'+r['txt'], '完整向量来源JSONL：'+r['jsonl'],
                  '初始化目标SDK顺序(rad)：'+json.dumps(r['init_target_real_rad']),
                  f"到位检查最后一次最大误差：{r['init_max_error_rad']:.8f} rad"]
        if i:
            p=poses['poses'][i-1]
            lines.append(f"H5：{p['shard']}，row={p['row']}（从0计），episode={p['episode_id']}，env={p['env_id']}，step={p['step']}。")
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:32]))

if __name__=='__main__':main()
