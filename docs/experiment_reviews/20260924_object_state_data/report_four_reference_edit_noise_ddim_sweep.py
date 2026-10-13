"""Summarize the four-episode paired noise/DDIM sweep."""
import json,statistics
from run_four_reference_edit_noise_ddim_sweep import O,R,EPISODES,RATIOS,STEPS,folder

def text_progress(value):
 return '未分离' if value is None else f'{value:g}'

def main():
 results=json.loads((O/'RESULTS.json').read_text())
 assert len(results)==len(EPISODES)*len(RATIOS)*len(STEPS)
 index={(r['episode'],r['requested_noise_ratio'],r['ddim_steps']):r for r in results}
 assert len(index)==len(results)
 lines=['# Reference 加噪编辑：噪声比与 DDIM4/6 对照','',
  '同一四条已用原速reference验收翻转的episode：76/f114、34/f79、54/f110、2/f103。每条使用自己的横抓初态和reference尾段，44g、摩擦1.1、物体尺寸1、固定wrist、30Hz、60步静置、完整尾段加60步保持。seed44。原生环境阈值、目标更新与失败逻辑未更改。', '',
  '相邻reference目标的22关节最大绝对跳变严格>0.1 rad时插一个中点。过去3步实际下发目标+未来9步reference构成12步轨迹，在训练噪声日程表中选最接近请求噪声比的起点，局部等间隔选4或6个去噪时刻；eta0、无额外MSE guidance，执行前2步后reference前进2步。DDIM4和DDIM6保持相同起点噪声强度。', '',
  '表格为首次连续三步几何分离时的原始reference动作进度（✓为分离前连续至少30实际动作步竖直悬空接触）。这项物理/接触分析与原生failure独立；不同episode的进度不直接相加比较。', '',
  '| 请求噪声比 / DDIM | 起始训练时刻 / 实际噪声比 | 时间表 | 76 | 34 | 54 | 2 | 稳定翻转 |',
  '|---|---|---|---:|---:|---:|---:|---:|']
 compact=[]
 for ratio in RATIOS:
  for steps in STEPS:
   rs=[index[(ep,ratio,steps)] for ep in EPISODES]
   meta=rs[0]['editor'];ts=meta['timesteps'];actual=meta['actual_noise_ratio']
   assert all(x['editor']['timesteps']==ts and x['editor']['actual_noise_ratio']==actual for x in rs)
   cells=[]
   for r in rs:
    loss=r['first_separation'];progress=None if loss is None else loss['original_reference_progress']
    cells.append(f'{text_progress(progress)} '+('✓' if r['stable_turn'] else '✗'))
   good=sum(int(x['stable_turn']) for x in rs)
   row=f'| {ratio:.2f} / {steps} | t={ts[0]}, {actual:.4f} | {"→".join(map(str,ts))} | '+ ' | '.join(cells)+f' | {good}/4 |'
   lines.append(row)
   compact.append(dict(requested_noise_ratio=ratio,ddim_steps=steps,actual_noise_ratio=actual,timesteps=ts,stable_turn_count=good,episodes=[dict(episode=r['episode'],stable_turn=r['stable_turn'],separation_reference_progress=None if r['first_separation'] is None else r['first_separation']['original_reference_progress'],separation_physical_action_step=None if r['first_separation'] is None else r['first_separation']['control_step'],longest_vertical_contact_steps=r['longest_vertical_contact_steps']) for r in rs]))
 lines.extend(['','逐条机器可读结果见RESULTS.json；各rollout的summary、trace、predictions、initial_state、grasp_metrics、analysis保留在本目录。实验只改变请求噪声比与局部去噪步数；导入初态逐字段及60步静置物理状态逐值核对。未录视频，未写Downloads。',''])
 (O/'REPORT.md').write_text('\n'.join(lines)+'\n')
 (O/'COMPACT.json').write_text(json.dumps(compact,indent=2)+'\n')
 print('\n'.join(lines[8:8+len(compact)]))
if __name__=='__main__':main()
