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
    manifest = [r for r in manifest if r['pair'] == 1 and r['seed'] == 50]
    assert len(manifest) == 2 and {r['hold'] for r in manifest} == {0, 1}, 'Need one seed-50 trial per condition'
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
    context=json.loads((OUT/'comparison_context.json').read_text())
    (OUT/'results.json').write_text(json.dumps(dict(runs=results, selection='one empty-hand trial per condition; pair=1; seed=50', comparison_context=context),ensure_ascii=False,indent=2)+'\n')
    lines=['真机对比：空手不holding / 空手holding / 放灯泡holding（每种条件各1轮）',
           '版本说明：本报告holding记录均为旧holding（推理前额外读qpos并刷新obs）；新holding统一采样后的seed=50空手实验，见reports/new_old_holding_20260917_seed50/summary.txt。',
           '空手采用已有配对第1轮，seed=50，各600动作/300chunk；放灯泡采用用户确认的旧记录，共608动作/304chunk。本次仅重新统计，没有再次运行硬件。',
           '每轮ROTATE复位；TensorRT FP32，FUSED_DDIM=1，DDIM 4步，chunk=2，30Hz，无插值。',
           '跳变=max_j |相邻指令角度差|；排除初始化、首个策略动作及退出hold。',
           '其他轮次保留为历史，不纳入本次对比。原三轮汇总保存在three_trial_history目录。',
           '固定推理输入与随机状态有助于复现推理；真机闭环仍可能因实测反馈不同而产生不同轨迹。', '']
    lines.extend(['数据来源（以下路径相对于项目根目录）', '',
                  '| 实验条件 | 使用的TXT文件 | 用途 |',
                  '| --- | --- | --- |'])
    for source in context['txt_data_sources']:
        lines.append(f"| {source['condition']} | {source['path']} | {source['use']} |")
    lines.extend(['', '空手第2、3轮TXT不纳入本报告；summary.txt为汇总输出，不是原始数据来源。', ''])
    rows={0:[('within','chunk 内：第 0 步 → 第 1 步'),
             ('policy_boundary','边界：上一 chunk 末步 → 下一 chunk 首步')],
          1:[('within','chunk 内：第 0 步 → 第 1 步'),
             ('last_to_hold','边界：上一 chunk 末步 → hold'),
             ('first_command','边界：hold → 下一 chunk 首步'),
             ('policy_boundary','纯策略：上一 chunk 末步 → 下一 chunk 首步（跳过 hold）')]}
    for run in sorted(results,key=lambda r:r['hold']):
        lines.extend(['空手 holding='+str(run['hold'])+'；来源：'+Path(run['log']).name,
                      '| 指令切换 | 样本数 | 平均幅度 | 中位数 | P95 | 最大值 |',
                      '| --- | ---: | ---: | ---: | ---: | ---: |'])
        for key,label in rows[run['hold']]:
            v=run['metrics'][key]
            lines.append(f"| {label} | {v['n']} | {v['mean_deg']:.2f}°（{v['mean_rad']:.4f} rad） | {v['median_deg']:.2f}° | {v['p95_deg']:.2f}° | {v['max_deg']:.2f}° |")
        lines.append('')
    lines.extend(['放灯泡 holding=1；来源：'+context['bulb_source']+'（用户确认有灯泡）', '',
                  '| 指令切换 | 样本数 | 平均幅度 | 中位数 | P95 | 最大值 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for key,label in rows[1][:3]:
        v=context['bulb_metrics'][key]
        lines.append(f"| {label} | {v['n']} | {v['mean_deg']:.2f}°（{v['mean_rad']:.4f} rad） | {v['median_deg']:.2f}° | {v['p95_deg']:.2f}° | {v['max_deg']:.2f}° |")
    lines.extend(['', '放灯泡记录仅有TXT最大关节差值，无法还原跳过hold后的完整22维纯策略跨chunk差值。',
                  '此处只比较已记录的三个条件，没有放灯泡且不holding的数据。', ''])
    by_hold={r['hold']:r['metrics'] for r in results}
    off=by_hold[0]['first_command']['mean_deg']
    on=by_hold[1]['first_command']['mean_deg']
    lines.append(f'空手单轮对照结论：holding实际边界首步均值{on:.4f}°，不holding为{off:.4f}°；holding增大{(on/off-1)*100:.2f}%。')
    lines.append('以上是指令角度跳变，不能直接解释为机械运动速度。日志继续保留全部历史。')
    empty=by_hold[1]['first_command']
    bulb=context['bulb_metrics']['first_command']
    lines.append(f"同为holding，放灯泡首步平均跳变{bulb['mean_deg']:.2f}°，空手{empty['mean_deg']:.2f}°；P95分别为{bulb['p95_deg']:.2f}°和{empty['p95_deg']:.2f}°。这是两次不同运行的描述性比较。")
    lines.extend(['', '初始化设置与完整关节数据',
                  context['initialization_selection_note'],
                  context['sampled_backup_verification_note'],
                  '当前配置及空手两轮JSONL确认：INIT_POSE=ROTATE；50步/2.0秒；到位容差0.1 rad；等待超时3.0秒；轮询0.1秒；skip_franka=True，仅初始化SharpA手。',
                  context['target_provenance'],
                  'robot_init.py顶部HAND_READY_JOINTS=zeros是默认占位；main根据ROTATE复制HAND_ROTATE_JOINTS。注释中的采样初始姿态未启用。',
                  '下表按policy顺序排列；SDK序号为从0开始的原始硬件顺序；所有角度单位rad。实测值取各TXT的[initial-state]，保留日志的6位小数精度。', '',
                  '| policy序号 | SDK序号 | 关节 | 当前ROTATE目标 | 空手不holding起始实测 | 空手holding起始实测 | 放灯泡holding起始实测 |',
                  '| ---: | ---: | --- | ---: | ---: | ---: | ---: |'])
    for i,name in enumerate(context['policy_joint_names']):
        j=context['real_joint_names'].index(name)
        values=[context['current_rotate_target_real_rad'][j]]+[context['conditions'][key]['initial_qpos_policy_rad'][i] for key in ('empty_no_hold','empty_hold','bulb_hold')]
        lines.append(f"| {i} | {j} | {name} | "+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines.extend(['', '当前ROTATE目标原始精度（SDK顺序，rad）：',
                  json.dumps(context['current_rotate_target_real_rad']),
                  '对应SDK关节顺序：', json.dumps(context['real_joint_names']),
                  '初始化来源及结构化数据：comparison_context.json。'])
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':
    main()
