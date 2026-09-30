"""Summarize serial experiments using identical physical environment slots."""
import sys,json
from pathlib import Path
import numpy as np
out=Path(sys.argv[1]).resolve();partial='--partial' in sys.argv
names=['ordinary_1b','guided_scale0','guide10k_scale25'];labels=['普通1B','guided采样器 scale=0','10k guide scale=25']
def stats(x):
    x=np.asarray(x)
    return {'n':int(x.size),'mean':float(x.mean()),'median':float(np.median(x)),'p95':float(np.quantile(x,.95)),'max':float(x.max()),'over009_fraction':float(np.mean(x>.090001))} if x.size else {'n':0}
def action_stats(z,ids,limit=600):
    a=z['action'];active=z['active'];end=min(len(a),limit)
    delta=np.abs(np.diff(a[:end,ids],axis=0));valid=active[1:end,ids]&active[:end-1,ids]
    return {'max_joint_delta':stats(delta.max(axis=2)[valid]),'per_joint_delta':stats(delta[valid].ravel()),'end_tracking':stats(np.max(abs(a[:end,ids]-z['qpos'][:end,ids]),axis=2)[active[:end,ids]])}
groups=[];runs={};validations=[]
old=out.parent/'20260917_1b_initialization_audit';old_initial=np.load(old/'initial_state.npz')
for name,label in zip(names,labels):
    folder=out/name;source=folder/'rollouts.npz';complete=source.exists()
    if not complete:source=folder/'partial.npz'
    if not source.exists():
        if partial:continue
        raise FileNotFoundError(source)
    if not partial:assert json.loads((folder/'completion.json').read_text())['complete']
    z=np.load(source);runs[name]=z;cfg=json.loads((folder/'config.json').read_text())
    cap=len(z['action'])/30;ids=np.arange(32,48);d=z['drop_steps'][ids];times=np.where(d>=0,(d+1)/30,cap)
    ini=np.load(folder/'initial_state.npz')
    for key in ['q','wrist','object']:assert np.array_equal(ini[key],old_initial[key])
    assert json.loads((folder/'static_validation.json').read_text())['max_displacement_difference_from_baseline_m']==0
    for i in ids:
        hits=np.flatnonzero(np.linalg.norm(z['object'][:,i,:3]-ini['object'][i,:3],axis=1)>.05)
        assert (int(hits[0]) if len(hits) else -1)==z['drop_steps'][i]
        if z['drop_steps'][i]>=0:
            ds=int(z['drop_steps'][i]);assert z['active'][:ds+1,i].all() and not z['active'][ds+1:,i].any()
    expect=np.load(folder/'initial_predictions.npz')[{'ordinary_1b':'ordinary','guided_scale0':'scale0','guide10k_scale25':'scale25'}[name]]
    initial_error=float(abs(z['raw'][:2,32:48].transpose(1,0,2)-expect).max());assert initial_error<1e-5
    validations.append({'arm':name,'saved_initial_state_exact':True,'static_response_exact_match':True,'recomputed_drop_steps_match':True,'initial_prediction_error_rad':initial_error,'max_joint_clip_rad':float(abs(z['raw']-z['action'])[z['active']].max())})
    group={'name':name,'label':label,'complete':complete,'observation_cap_s':cap,'alive':int((d<0).sum()),'mean_capped_hold_s':float(times.mean()),'median_capped_hold_s':float(np.median(times)),'min_capped_hold_s':float(times.min()),'max_capped_hold_s':float(times.max()),'hold_seconds':times.tolist(),'censored':(d<0).tolist(),'first20s':action_stats(z,ids),'trials':[]}
    for k,i in enumerate(ids):
        event_xyz=(z['object'][int(d[k]),i,:3]-ini['object'][i,:3]).tolist() if d[k]>=0 else None
        group['trials'].append({'seed':50+k,'pose_group':k%4+1,'seconds':float(times[k]),'censored':bool(d[k]<0),'drop_step':int(d[k]),'event_displacement_xyz_m':event_xyz,'first20s':action_stats(z,[i])})
    groups.append(group)
byname={g['name']:g for g in groups};comparison={}
for name,left,right in [('guide_vs_ordinary',names[0],names[2]),('guide_vs_scale0',names[1],names[2]),('scale0_vs_ordinary',names[0],names[1])]:
    if left not in byname or right not in byname:continue
    lt=np.array(byname[left]['hold_seconds']);rt=np.array(byname[right]['hold_seconds']);delta=rt-lt;paired=[];pooled=[[],[]]
    for k in range(16):
        stop=min(600,int(round(lt[k]*30)),int(round(rt[k]*30)));x=[]
        for j,armname in enumerate([left,right]):
            values=np.max(abs(np.diff(runs[armname]['action'][:stop,32+k],axis=0)),axis=1)
            pooled[j].extend(values.tolist());x.append(stats(values))
        paired.append({'seed':50+k,'steps':stop,'left':x[0],'right':x[1]})
    comparison[name]={'improved_capped':int((delta>1e-6).sum()),'worsened_capped':int((delta< -1e-6).sum()),'equal_capped':int((abs(delta)<=1e-6).sum()),'mean_capped_difference_s':float(delta.mean()),'per_seed_delta_s':delta.tolist(),'common_prefix_action_metrics':paired,'pooled_common_prefix':{'left':stats(pooled[0]),'right':stats(pooled[1])}}
result={'partial':partial,'event_definition':'first bulb root-position displacement >0.05m from initial center; not confirmed loss of grasp','groups':groups,'paired_comparisons':comparison}
(out/('partial_summary.json' if partial else 'results.json')).write_text(json.dumps(result,indent=2))
(out/('partial_validation.json' if partial else 'validation.json')).write_text(json.dumps(validations,indent=2))
lines=['# 真机初始手型：1B + 10k guide 对照','','每组16轮，使用上一轮归档的4组实测手型与仿真灯泡摆放，各分配4个扩散采样seed（50–65）。三组逐项相同的关节角、腕部和灯泡初态，复用逐chunk的1B高斯噪声；guide另用seed+100000。','','三组在独立的新仿真进程中依次运行，均使用相同的物理环境槽位32–47。初始状态数组逐元素一致，静态1秒持物的物体偏移与旧基线逐元素一致。不同槽位的预检查已隔离至pilot_different_slots，不纳入结论。','','配置：DDIM4/4，30Hz，执行2步，引导前2步，scale25，不holding，无额外推理等待、随机外力或域随机化；沿用物理关节范围，无单步限幅。首次物体根节点位置（记录中的object xyz）相对初态偏移>5cm记为阈值事件，最长400秒。它不是旋转任务成功率，也不等同于真正掉落。seed=65的guide回放显示，触发阈值时灯泡看起来仍在指间；seed=63的0.20秒事件则伴随5.82cm向下位移。','','生产guided DDIM的scale0仍重算裁剪后epsilon，与普通DDIM不同，因此补跑普通1B与guided scale0两个控制组。对比普通1B表示整条guide推理路径的效果；对比scale0更能隔离引导本身的影响。','','| 组 | 未触发5cm阈值 | 平均阈值时间(s) | 中位阈值时间(s) | 前20秒目标差均值(rad) | 控制步超0.09比例 |','|---|---:|---:|---:|---:|---:|']
for x in groups:
    m=x['first20s']['max_joint_delta'];lines.append(f"| {x['label']} | {x['alive']}/16 | {x['mean_capped_hold_s']:.2f} | {x['median_capped_hold_s']:.2f} | {m.get('mean',float('nan')):.5f} | {m.get('over009_fraction',float('nan')):.2%} |")
lines+=['','动作差为每步22关节最大相邻目标差的时间统计；排除初始目标→首动作和触发阈值后的动作。只比较前20秒，先触发阈值则在该步截断，因此有效样本数不完全相同。JSON另存逐seed共同前缀指标。','','| seed | 初态组 | 普通1B(s) | guided scale0(s) | 10k guide(s) |','|---:|---:|---:|---:|---:|']
for k in range(16):
    vals=[f"{byname[name]['hold_seconds'][k]:.2f}"+('+' if byname[name]['censored'][k] else '') if name in byname else '待完成' for name in names]
    lines.append('| %d | %d | %s |'%(50+k,k%4+1,' | '.join(vals)))
lines+=['','“+”表示截至观察上限仍未触发阈值，是右截断，不能当作真实掉落时刻。',('当前为中间结果。' if partial else '本次运行已结束。'),'','4组灯泡摆放是仿真假设，未精确标定真机腕部／物体位姿；本组16轮是小样本诊断。旧实验基线仅作历史参照，主对比使用此次同时补跑的控制组。不同batch数下FP32推理与接触动力学可能放大微小数值差异，相同seed并不意味着整条闭环轨迹逐位相同。','']
if comparison:
    lines+=['## 成对比较','','动作差使用每个seed两组共同的有效时间前缀（最多20秒），因此每一行左右样本数相同。不同对比行的共同区间不一定相同。','','| 比较：加入guide前→后 | 阈值时间变长/变短 | 共同前缀动作差均值(rad) | 均值降幅 |','|---|---:|---:|---:|']
    for key,label in [('guide_vs_ordinary','普通1B → 10k guide'),('guide_vs_scale0','guided scale0 → 10k guide')]:
        if key not in comparison:continue
        c=comparison[key];a=c['pooled_common_prefix']['left']['mean'];b=c['pooled_common_prefix']['right']['mean']
        lines.append(f"| {label} | {c['improved_capped']} / {c['worsened_capped']} | {a:.6f} → {b:.6f} | {1-b/a:.2%} |")
    lines+=['']
if (out/'action_spike_audit.json').exists():
    spike=json.loads((out/'action_spike_audit.json').read_text())
    lines+=['## 动作尖峰复查','','seed=59在17.60秒、动作下标15出现0.418553rad相邻目标差。固定该时刻的完整观测与prior噪声重新推理，原guide输出逐元素精确复现（最大误差0）。同一个关节：','','| 固定观测下的推理路径 | 两步目标差(rad) |','|---|---:|']
    for key,label in [('ordinary_1b','普通1B'),('guided_scale0','guided scale0'),('guide10k_scale25','10k guide scale25'),('guide_reference','10k guide自身的参考目标')]:
        lines.append(f"| {label} | {spike[key]['joint_delta_rad']:.6f} |")
    lines+=['','该局部反事实对比说明这个尖峰由guide参考轨迹引入；它不是三个独立rollout在17.60秒的直接对比。详见action_spike_audit.json。guide减小平均步幅，不代表保证时间平滑或0.09rad限幅。','']
lines+=['## 状态回放','','[seed=65三路回放](video_seed65/replay.mp4)；[seed=63普通1B与guide回放](preview_ordinary_1b_seed63/replay.mp4)。视频根据保存的关节角、物体位姿重建，未重新运行策略；达到5cm阈值的画面冻结，不是物体在现实中停止运动。','']
(out/('partial_report.md' if partial else 'report.md')).write_text('\n'.join(lines))
print('\n'.join(lines))
