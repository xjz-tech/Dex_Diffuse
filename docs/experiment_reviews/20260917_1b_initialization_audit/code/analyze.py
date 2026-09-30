import json,sys,hashlib
from pathlib import Path
import numpy as np
out=Path(sys.argv[1]).resolve();cfg=json.loads((out/'config.json').read_text());n=cfg['trials_per_group']
labels={'native_bulb':'原生姿态／有灯泡','native_empty':'原生姿态／空手','real_bulb':'真机手型／有灯泡','real_empty':'真机手型／空手'}
z=np.load(out/'rollouts.npz');a=z['action'];raw=z['raw'];active=z['active'];drop=z['drop_steps'];init=np.load(out/'initial_state.npz');static=np.load(out/'static_check.npz')
assert a.shape==raw.shape==z['qpos'].shape and a.shape[2]==22
assert np.isfinite(a).all() and np.isfinite(raw).all()
def stats(x):
    x=np.asarray(x)
    return {'n':int(x.size),'mean':float(x.mean()),'median':float(np.median(x)),'p95':float(np.quantile(x,.95)),'max':float(x.max()),'over_009_fraction':float((x>.090001).mean())} if x.size else {'n':0}
def deltas(arr,ids,cap):
    end=min(len(arr),cap);dif=np.abs(np.diff(arr[:end,ids],axis=0));valid=active[1:end,ids]&active[:end-1,ids]
    mx=dif.max(axis=2)
    within=np.broadcast_to((np.arange(1,end)%2==1)[:,None],valid.shape)
    return {'max_joint':stats(mx[valid]),'per_joint':stats(dif[valid].ravel()),'within':stats(mx[valid&within]),'between':stats(mx[valid&~within])}
groups=[]
for gi,g in enumerate(cfg['groups']):
    ids=np.arange(gi*n,(gi+1)*n);bulb=g.endswith('_bulb')
    durations=np.where(drop[ids]>=0,(drop[ids]+1)/30,12000/30)
    item={'group':g,'raw_first20s':deltas(raw,ids,600),'sent_first20s':deltas(a,ids,600),'sent_until_stop':deltas(a,ids,len(a)),
      'static_1s_displacement_m':static['displacement_m'][ids].tolist(),
      'clipped_joint_fraction':float((abs(raw[:,ids]-a[:,ids])[active[:,ids]]>1e-7).mean()),
      'tracking_error_rad':stats(np.max(abs(a[:,ids]-z['qpos'][:,ids]),axis=2)[active[:,ids]])}
    if bulb:
        item.update(hold_seconds=durations.tolist(),drop_steps=drop[ids].tolist(),survived_400s=int((drop[ids]<0).sum()),truncated_mean_hold_seconds=float(durations.mean()),truncated_median_hold_seconds=float(np.median(durations)))
        keep=static['displacement_m'][ids]<.05
        valid_ids=ids[keep]
        item['static_valid_subset']={'criterion':'static 1s object center displacement < 5cm','count':int(keep.sum()),'trial_indices':np.flatnonzero(keep).tolist(),'hold_seconds':durations[keep].tolist(),'sent_first20s':deltas(a,valid_ids,600) if len(valid_ids) else None}
    item['trials']=[]
    for k,i in enumerate(ids):
        row={'trial':k,'seed':cfg['seeds'][k],'first20s':deltas(a,[i],600),'static_displacement_m':float(static['displacement_m'][i])}
        if bulb:row['hold_seconds']=float(durations[k]);row['censored']=bool(drop[i]<0)
        item['trials'].append(row)
    groups.append(item)
result={'metric':'abs consecutive targets per joint, max across 22 joints then time average; exclude initial transition and post-drop steps; first20s includes failure step','config':cfg,'groups':groups}
(out/'results.json').write_text(json.dumps(result,indent=2))
lines=['# 1B 模型：初始手型与空转动作差对照','','模型为 obs_4-66.ckpt，EMA、DDIM4、每次执行2步、30Hz、无引导、无holding，无额外真机推理延时。四组各16轮；同序号四组复用相同扩散噪声。','','原生组从16条均匀覆盖150条演示编号的轨迹按仿真原有随机帧初始化规则取姿态，保留手与物体配对。真机组循环使用四轮归档的实测手型和各自通过静态验证的仿真灯泡摆放；腕部／灯泡摆放是假设，不能视作真机接触的精确复现。','','为隔离初始姿态和有无接触，关闭随机外力、域随机化、观测噪声、自动重置。沿用模型预测的绝对目标，使用仿真常规物理关节范围裁剪，不加0.09rad单步限幅。此诊断不是此前带随机扰动的3000环境完整任务评测。','','## 动作差（rad）','','指标是每步22关节最大相邻目标差的时间统计，不是所有关节平均。比较前20秒，发生脱离则只计至脱离步；排除初始化到首动作和脱离后的动作，因此不同组有效样本数不同。','','| 条件 | 原始预测均值 | 下发目标均值 | 下发中位数 | 下发P95 | 下发最大值 | 步超0.09占比 |','|---|---:|---:|---:|---:|---:|---:|']
for x in groups:
    s=x['sent_first20s']['max_joint'];r=x['raw_first20s']['max_joint']
    lines.append(f"| {labels[x['group']]} | {r['mean']:.6f} | {s['mean']:.6f} | {s['median']:.6f} | {s['p95']:.6f} | {s['max']:.6f} | {s['over_009_fraction']:.2%} |")
lines+=['','## 保持时长','','首次灯泡中心相对初始位置偏移>5cm作为脱离初始抓持区域的诊断阈值；它不是旋转成功率或常规任务全部终止条件。时间按物理步累计，最长400秒，未触发的按400秒截断。','','| 条件 | 400秒仍未触发 | 截断平均时长 | 截断中位时长 | 每轮时长（秒，+表示截断） |','|---|---:|---:|---:|---|']
for x in groups:
    if 'hold_seconds' not in x:continue
    times=', '.join(f"{t:.2f}"+('+' if d<0 else '') for t,d in zip(x['hold_seconds'],x['drop_steps']))
    lines.append(f"| {labels[x['group']]} | {x['survived_400s']}/16 | {x['truncated_mean_hold_seconds']:.2f} | {x['truncated_median_hold_seconds']:.2f} | {times} |")
lines+=['','静态检查子集（仅按推理前静态1秒的中心偏移<5cm筛选，未按策略表现筛选）：']
for x in groups:
    if 'static_valid_subset' in x:
        v=x['static_valid_subset'];lines.append(f"- {x['group']}: {v['count']}/16轮通过，保持秒数：{v['hold_seconds']}。")
lines+=['','## 验证与限制','','- 模型在相同日志观测与噪声上的输出已对照原真机记录，误差见model_validation.json。','- 初始22关节数值逐项核对；保存完整初态、动作、反馈、物体状态、终止步和配置。','- 四轮真机手型各重复四个噪声种子；这16轮不是16种独立真机接触条件。','- 静态保持1秒检查另存static_check.npz；随后精确恢复初态开始推理。','- 原生随机初始化包含静态持物就会脱离的状态，不能把本表解释成所有原生状态都是稳定抓持，也不能据此声称真机初始手型优于原生初始化。','- 输出包含原始预测和经物理关节范围裁剪的下发目标。','- 所有实验均在仿真执行，没有操作硬件。','']
lines+=['## 本次结论','','空手时两种初始化的平均最大关节目标差均约0.101rad；有灯泡且统计至脱离时均约0.08rad。这组样本中，空手与有接触的差异比两种初始化之间的差异大；0.105rad不是模型的固定步幅。两种条件仍有不少步超过训练轨迹约0.09rad的范围。','','真机手型组中位保持38.55秒、平均39.70秒、最长96.67秒，所有16轮都在上限400秒前触发阈值。因此这些时长没有右截断。这个结论仅对应记录的四种仿真接触摆放与无扰动诊断条件。','','附加校验见validation.json：独立重算每轮首次5cm越界步、有效步掩码、第一步预测与离线重算的一致性。early_late.json保留前100步和后500步的动作差；空手后500步依然约0.101rad，并非只由首几个动作造成。','']
(out/'report.md').write_text('\n'.join(lines))
print('\n'.join(lines))
