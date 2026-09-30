import sys,json
from pathlib import Path
import numpy as np
out=Path(sys.argv[1]).resolve();cfg=json.loads((out/'config.json').read_text())
partial='--partial' in sys.argv;source=out/('partial.npz' if partial else 'rollouts.npz')
z=np.load(source);ini=np.load(out/'initial_state.npz');a=z['action'];raw=z['raw'];active=z['active'];drop=z['drop_steps'];T=len(a);cap=T/30
labels=['普通1B','guided采样器 scale=0','10k guide scale=25']
def stats(x):
    x=np.asarray(x)
    return {'n':int(x.size),'mean':float(x.mean()),'median':float(np.median(x)),'p95':float(np.quantile(x,.95)),'max':float(x.max()),'over009_fraction':float(np.mean(x>.090001))} if x.size else {'n':0}
def action_stats(ids,limit=600):
    end=min(T,limit);delta=np.abs(np.diff(a[:end,ids],axis=0));valid=active[1:end,ids]&active[:end-1,ids]
    return {'max_joint_delta':stats(delta.max(axis=2)[valid]),'per_joint_delta':stats(delta[valid].ravel()),'end_tracking':stats(np.max(abs(a[:end,ids]-z['qpos'][:end,ids]),axis=2)[active[:end,ids]])}
groups=[]
for g in range(3):
    ids=np.arange(g*16,(g+1)*16);d=drop[ids];times=np.where(d>=0,(d+1)/30,cap)
    group={'name':cfg['groups'][g],'label':labels[g],'alive':int((d<0).sum()),'mean_capped_hold_s':float(times.mean()),'median_capped_hold_s':float(np.median(times)),'hold_seconds':times.tolist(),'censored':(d<0).tolist(),'first20s':action_stats(ids),'trials':[]}
    for k,i in enumerate(ids):
        group['trials'].append({'seed':50+k,'pose_group':k%4+1,'seconds':float(times[k]),'censored':bool(d[k]<0),'drop_step':int(d[k]),'first20s':action_stats([i])})
    groups.append(group)
comparison={}
for name,left,right in [('guide_vs_ordinary',0,2),('guide_vs_scale0',1,2),('scale0_vs_ordinary',0,1)]:
    lt=np.array(groups[left]['hold_seconds']);rt=np.array(groups[right]['hold_seconds']);delta=rt-lt
    paired=[]
    for k in range(16):
        stop=min(600,int(round(lt[k]*30)),int(round(rt[k]*30)));x=[]
        for gi in [left,right]:x.append(stats(np.max(abs(np.diff(a[:stop,gi*16+k],axis=0)),axis=1)))
        paired.append({'seed':50+k,'steps':stop,'left':x[0],'right':x[1]})
    comparison[name]={'improved':int((delta>1e-6).sum()),'worsened':int((delta< -1e-6).sum()),'equal_capped':int((abs(delta)<=1e-6).sum()),'mean_capped_difference_s':float(delta.mean()),'per_seed_delta_s':delta.tolist(),'common_prefix_action_metrics':paired}
result={'partial':partial,'observation_cap_s':cap,'groups':groups,'paired_comparisons':comparison}
(out/('partial_summary.json' if partial else 'results.json')).write_text(json.dumps(result,indent=2))
lines=['# 真机初始手型：1B + 10k guide 对照','','每组16轮，使用上一轮归档的4组实测手型与仿真灯泡摆放，各分配4个扩散采样seed（50–65）。三组逐项相同的关节角、腕部和灯泡初态，复用逐chunk的1B高斯噪声；guide另用seed+100000。','','配置：DDIM4/4，30Hz，执行2步，引导前2步，scale25，不holding，无额外推理等待、随机外力或域随机化；沿用物理关节范围，无单步限幅。首次灯泡中心相对初始位置偏移>5cm视为脱离初始抓持区域，最长400秒。它不是旋转任务成功率。','','由于生产guided DDIM的scale0仍重算裁剪后epsilon，与普通DDIM不同，本次同时补跑普通1B与guided scale0两个控制组。对比普通1B表示整条guide推理路径的收益；对比scale0更能隔离引导本身的影响。','','| 组 | 观察上限仍未脱离 | 截断平均保持秒 | 截断中位保持秒 | 前20秒目标差均值(rad) | 控制步超0.09比例 |','|---|---:|---:|---:|---:|---:|']
for x in groups:
    m=x['first20s']['max_joint_delta'];lines.append(f"| {x['label']} | {x['alive']}/16 | {x['mean_capped_hold_s']:.2f} | {x['median_capped_hold_s']:.2f} | {m.get('mean',float('nan')):.5f} | {m.get('over009_fraction',float('nan')):.2%} |")
lines+=['','动作差为每步22关节最大相邻目标差的时间统计；排除初始目标→首动作和脱离后的动作。只比较前20秒，先脱离则在终止步截断，因此组间有效样本数不完全相同。JSON另存逐seed共同前缀指标。','','| seed | 初态组 | 普通1B(s) | guided scale0(s) | 10k guide(s) |','|---:|---:|---:|---:|---:|']
for k in range(16):
    vals=[f"{x['hold_seconds'][k]:.2f}"+('+' if x['censored'][k] else '') for x in groups]
    lines.append('| %d | %d | %s |'%(50+k,k%4+1,' | '.join(vals)))
lines+=['','“+”表示截至观察上限仍未触发阈值，是右截断，不能当作真实失效时刻。',f'当前观察上限：{cap:.2f}秒。'+('当前为中间结果。' if partial else '本次运行已结束。'),'','4组灯泡摆放是仿真假设，未精确标定真机腕部／物体位姿；本组16轮是小样本诊断。原生示范初始化和空手不在本次比较范围。旧实验基线仅作历史参照，主对比使用此次同时补跑的控制组。','']
(out/('partial_report.md' if partial else 'report.md')).write_text('\n'.join(lines))
print('\n'.join(lines))
