"""Combine new sweep with previously completed DDIM4 runs at 0.10/0.20/0.35."""
import json
from run_four_reference_edit_noise_ddim_sweep import O,R,EPISODES,folder
from run_four_episode_reference_edit_adaptive010_video import folder as previous_folder
RATIOS=(.10,.15,.20,.25,.30,.35)
OLDER=(.10,.20,.35)

def result(ep,ratio,steps):
 if ratio in OLDER and steps==4:
  path=previous_folder(ep,f'edit{int(round(ratio*100)):03d}',44)
 else:path=folder(ep,ratio,steps)
 a=json.loads((path/'analysis.json').read_text())
 assert a['seed']==44 and a['editor']['inference_steps']==steps
 assert a['editor']['requested_noise_ratio']==ratio
 return a

def main():
 assert (O/'run_status.json').exists() and (O/'extra_run_status.json').exists()
 previous=json.loads((R/'four_episode_reference_edit_adaptive010_video_20260928/comparison.json').read_text())
 baselines=[]
 for key,label in [('raw','原速 reference'),('guide','传统 prior 引导：DDIM4 / scale25 / guide2 exec1')]:
  group=[next(x for x in previous if x['method']==key and x['episode']==ep) for ep in EPISODES]
  cells=[f"{x['first_separation_reference_progress']:g}"+(' ✓' if x['stable_turn'] else ' ✗') for x in group]
  baselines.append('| '+label+' | — | — | '+' | '.join(cells)+f" | {sum(int(x['stable_turn']) for x in group)}/4 |")
 rows=[];compact=[]
 for ratio in RATIOS:
  for steps in (4,6):
   rr=[result(ep,ratio,steps) for ep in EPISODES]
   m=rr[0]['editor'];ts=m['timesteps'];actual=m['actual_noise_ratio']
   assert all(a['editor']['timesteps']==ts and a['editor']['actual_noise_ratio']==actual for a in rr)
   cells=[];items=[]
   for ep,a in zip(EPISODES,rr):
    loss=a['first_separation'];p=None if loss is None else loss['original_reference_progress'];physical=None if loss is None else loss['control_step']
    cells.append(f'{p:g}'+(' ✓' if a['stable_turn'] else ' ✗') if p is not None else '未分离')
    items.append(dict(episode=ep,stable_turn=a['stable_turn'],first_separation_reference_progress=p,first_separation_physical_action_step=physical,longest_vertical_contact_steps=a['longest_vertical_contact_steps'],cmd_reference_rmse_before_separation_rad=a['command_reference_rmse_before_separation_rad']))
   wins=sum(int(a['stable_turn']) for a in rr)
   rows.append(f'| {ratio:.2f} / {steps} | {actual:.4f} | {"→".join(map(str,ts))} | '+' | '.join(cells)+f' | {wins}/4 |')
   compact.append(dict(requested_noise_ratio=ratio,actual_noise_ratio=actual,ddim_steps=steps,timesteps=ts,stable_turn_count=wins,episodes=items))
 text='''# Reference 加噪编辑：六档噪声、DDIM4与DDIM6

相同四条已用原始reference验收翻转的episode：76/f114、34/f79、54/f110、2/f103。每条使用自己的横抓初态和reference尾段；44g、摩擦1.1、尺寸1、固定wrist、30Hz、60步静置、完整尾段加60步保持、seed44。其余原生评估协议不变。此前的0.10/0.20/0.35 DDIM4结果已核验并复用，本次新跑0.15/0.25/0.30的DDIM4与六档的DDIM6，本次新增36个rollout，合并已有12个DDIM4结果，共48个同口径结果。

对原始相邻reference目标22个关节最大绝对跳变严格>0.1 rad插一个中点。过去3步实际下发目标与未来9步reference构成12步轨迹。按训练噪声表选最接近请求噪声比的起始时刻，从该时刻**局部**取4或6个去噪时刻；同一噪声比的DDIM4和DDIM6起点相同。eta0、无额外MSE guidance，执行前2步后reference前进2步。这里的DDIM6不是100//6的全程网格。

表中原速reference按原始30Hz目标直接执行；传统prior引导使用相同>0.1 rad插值，但DDIM4、scale25、guide2/exec1。动作编辑均为exec2，因此传统引导行是既有配置对照，不单独归因于prior算法。\n\n单元格数字：首次连续3步满足几何分离条件时的**原始reference动作进度**；✓代表分离前至少连续30个实际动作步竖直、悬空且有接触，✗则未通过。它是独立的接触/几何分析，不等同原生环境failure。不同episode的绝对进度不宜直接求平均。

| 请求噪声比 / DDIM | 实际噪声比 | 局部去噪时刻 | 76 | 34 | 54 | 2 | 稳定翻转 |
|---|---:|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(baselines+rows)+'''

本轮新运行的各组 initial_state 与同episode原速reference逐字段一致，60步静置的运动状态一致，净接触力数值差逐组单独核验。初态同组一致和原速reference通过只保证这组仿真条件下的可比性；不是毫米级真实抓握复现。详细结果见SUMMARY_ALL.json、新运行的RESULTS.json与RESULTS_EXTRA.json及每组analysis.json。未录新视频，未写Downloads。
'''
 (O/'REPORT_ALL.md').write_text(text)
 (O/'SUMMARY_ALL.json').write_text(json.dumps(compact,indent=2)+'\n')
 print('\n'.join(rows))
if __name__=='__main__':main()
