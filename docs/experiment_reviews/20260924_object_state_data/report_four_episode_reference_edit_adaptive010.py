import json
from run_four_episode_reference_edit_adaptive010_video import O,R,folder
EPISODES=[76,34,54,2]
METHODS=[('raw','原速 reference'),('direct','>0.1 插值 direct'),('guide','传统 DDIM4 / scale25 / guide2 exec1'),('edit010','加噪编辑 0.10 / exec2'),('edit020','加噪编辑 0.20 / exec2'),('edit035','加噪编辑 0.35 / exec2')]
rows=[];all_results=[]
for method,label in METHODS:
 cells=[];success=0
 for ep in EPISODES:
  case=R/f'qualified_comparison/episode_{ep:02d}'
  if method.startswith('edit'):
   result=json.loads((folder(ep,method,44)/'analysis.json').read_text());loss=result['first_separation'];progress=loss['original_reference_progress'] if loss else None;ok=result['stable_turn'];physical=None if loss is None else loss['control_step'];longest=result['longest_vertical_contact_steps']
  else:
   name={'raw':'direct_m044_mu11','direct':'direct_adaptive010_m044_mu11','guide':'guide1_adaptive010_ddim4_scale25_m044_mu11'}[method]
   result=json.loads((case/name/'m044_mu11_result.json').read_text());loss=result['first_separation'];progress=loss['reference_action_number'] if loss else None;ok=result['completed_turn'];physical=None if loss is None else loss['control_step'];longest=result['longest_vertical_contact_control_steps']
  cells.append(f'{progress:g}'+(' ✓' if ok else ' ✗') if progress is not None else '未分离');success+=int(ok)
  all_results.append(dict(method=method,episode=ep,first_separation_reference_progress=progress,physical_action_step=physical,stable_turn=ok,longest_vertical_contact_steps=longest))
 rows.append('| '+label+' | '+' | '.join(cells)+f' | {success}/4 |')
report='''# 四个 episode：reference 加噪编辑 vs 传统 guidance

使用此前原速 reference 已通过翻转的相同四条条件子集：76/f114、34/f79、54/f110、2/f103。没有按编辑结果筛选。44g、摩擦1.1、尺寸1、固定 wrist、30Hz；各自60步静置、完整尾段、60步末尾保持。原生环境阈值/目标更新/失败逻辑保持既定协议。

编辑输入和传统guidance都按原始reference相邻22关节最大绝对跳变严格>0.1 rad插一个中点。四条原始动作数239/441/462/492，插入76/119/150/176，实际执行315/560/612/668。只在相邻reference之间插值，初始化到第一个动作不额外插值。

编辑：10B EMA，历史3个已执行目标固定，未来9个插值后的reference加训练对应噪声，4次DDIM去噪，执行2步后reference前进2步，无额外MSE guidance。噪声请求值0.1/0.2/0.35为归一化扩散噪声比，不是rad，实际时间表[5,3,2,0]/[11,7,4,0]/[20,13,7,0]。

传统对照：10B EMA、DDIM4、scale25、guide2/exec1，reference每次前进1步，使用此前同配置录像，逐步核对其cmd/q/object pose/竖直角/native failure与存档评估一致。编辑seed44；这是各episode单seed对比，不能当多seed成功率。

表格值为首次连续三步几何分离的起始原始reference进度；✓表示此前完成至少30个实际动作步的连续竖直悬空接触，✗未达该标准。几何分离是独立分析，不能与native failure混称。

| 方法 | Episode 76 | 34 | 54 | 2 | 稳定翻转 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

各组导入初态逐字段相同；静置运动轨迹一致。零编辑完整仿真与同规则direct匹配，净接触力允许<1e-5 N数值误差，其余存档字段精确一致。episode54复用上轮已经核验的编辑录像，其余三条新录。

合成视频依次为76、34、54、2。上排原速reference、插值direct、传统guide2/exec1；下排弱/中/强加噪编辑。全为仿真正面，显示物体XYZ轴。各段按原始reference进度对齐，原速画面在插值中点重复，因此各组物理时间不同。包含FK导入瞬间、60步静置、完整尾段和2秒保持。不保存到Downloads。
'''
(O/'README.md').write_text(report);(O/'comparison.json').write_text(json.dumps(all_results,indent=2)+'\n');print('\n'.join(rows))
