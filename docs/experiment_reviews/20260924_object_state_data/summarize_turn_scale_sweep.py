from pathlib import Path
import json,time,shutil
import numpy as np
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926';C=R/'qualified_comparison';OUT=R/'scale45_55';EPS=[76,34,54,2]
end=time.monotonic()+2400
while not (OUT/'new_results.json').exists():
 if time.monotonic()>end:raise TimeoutError('analysis did not finish')
 time.sleep(2)
new=json.load(open(OUT/'new_results.json'));old=json.load(open(R/'RESULTS.json'));baseline={r['episode']:r for r in old['results']};allrows=[]
for ep in EPS:
 for exe in [2,1]:
  for sc in [45,50,55]:
   if sc==50:
    b=baseline[ep][f'guide{exe}'];row=dict(episode=ep,execution_steps=exe,scale=50,first_separation=b['first_separation'],completed_turn=b['held_turn']['completed'],longest_vertical_contact_control_steps=b['held_turn']['longest_consecutive_control_steps'],vertical_contact_reference_interval=b['held_turn']['reference_interval'],first_native_failure=b['first_native_failure'],reused=True)
   else:row=dict(next(r for r in new if r['episode']==ep and r['execution_steps']==exe and r['scale']==sc),reused=False)
   folder=C/f'episode_{ep:02d}'/(f'guide{exe}' if sc==50 else f'guide{exe}_scale{sc}');ret=json.load(open(folder/'retention.json'));trace=json.load(open(folder/'trace.json'))
   sensitivity={}
   for gap in [.003,.005,.008,.01]:
    flags=[(g['mesh_vertex_gap_m']>gap and g['object_contact_force_norm_N']<.05) or g['mesh_vertex_gap_m']>.02 for g in ret['frames']]
    idx=next((i for i in range(len(flags)-2) if all(flags[i:i+3])),None)
    sensitivity[str(gap)]=1+trace[idx]['index']/2 if idx is not None and trace[idx]['phase']=='action' else None
   row['sensitivity_reference_action_number']=sensitivity
   row['reference_action_at_loss']=row['first_separation']['reference_action_number'] if row['first_separation'] else None
   allrows.append(row)
lookup={(r['episode'],r['execution_steps'],r['scale']):r for r in allrows};stats=[]
for exe in [2,1]:
 for sc in [45,50,55]:
  rows=[lookup[ep,exe,sc] for ep in EPS];vals=[r['reference_action_at_loss'] for r in rows];oldvals=[lookup[ep,exe,50]['reference_action_at_loss'] for ep in EPS]
  assert all(v is not None for v in vals+oldvals),'right-censoring requires explicit report'
  changes=np.array(vals)-oldvals
  stats.append(dict(execution_steps=exe,scale=sc,completed_turns=sum(r['completed_turn'] for r in rows),later_than_scale50=int(sum(changes>0)),earlier_than_scale50=int(sum(changes<0)),same_as_scale50=int(sum(changes==0)),paired_delta_reference_steps=changes.tolist(),mean_reference_step_at_loss=float(np.mean(vals)),median_reference_step_at_loss=float(np.median(vals)),mean_paired_delta_reference_steps=float(np.mean(changes)),later_than_direct=sum(v>baseline[ep]['direct']['first_separation']['reference_action_number'] for ep,v in zip(EPS,vals))))
report=dict(episodes=EPS,manifest=json.load(open(OUT/'manifest.json')),results=allrows,summary=stats,interpretation='same4 direct-turn-qualified starts and single fixed seed/noise; conditional paired scale sweep, not population significance; compare source reference progress, not expanded simulation-step counts; native failure separate')
(OUT/'RESULTS.json').write_text(json.dumps(report,indent=2))
lines=['# scale45 / 50 / 55：同4个可完成原始翻转的起点\n','只改变guide scale。76/f114、34/f79、54/f110、2/f103保持原样；10B EMA/DDIM4、guide2、插1中点、exec1/2、170g、摩擦2.2、原生固定wrist、seed42/固定噪声44不变。原始direct和scale50复用已验证记录，45/55新增16次完整回放。用户取消视频后停止录制，剩余实验使用no-video，没有合成新视频。\n','下表为首次物理分离分析标记对应的**原始reference动作编号（1开始）**，越大表示保持到更晚的reference；.5为插值中点。该指标沿用上轮的接触力/几何定义，不是native failure；新no-video结果不声称有新录像独立确认。\n']
for exe in [2,1]:
 lines += [f'## guide2 / exec{exe}\n','| episode | 原始direct | scale45 | scale50 | scale55 |','|---:|---:|---:|---:|---:|']
 for ep in EPS:
  vals=[lookup[ep,exe,sc]['reference_action_at_loss'] for sc in [45,50,55]];d=baseline[ep]['direct']['first_separation']['reference_action_number'];lines.append(f'| {ep} | {d:g} | '+' | '.join(f'{v:g}' for v in vals)+' |')
 lines+=['\n| scale | 完成翻转 | 比scale50保持更晚/更早/相同 | 比direct更晚 | 4条平均分离进度 |','|---:|---:|---:|---:|---:|']
 for st in [s for s in stats if s['execution_steps']==exe]:lines.append(f'| {st["scale"]} | {st["completed_turns"]}/4 | {st["later_than_scale50"]}/{st["earlier_than_scale50"]}/{st["same_as_scale50"]} | {st["later_than_direct"]}/4 | {st["mean_reference_step_at_loss"]:.2f} |')
lines += ['\n## 翻转与分离的分析口径\n','完成翻转：首次分离前，距竖直≤30°，连续至少30控制步，同时灯泡网格离桌>8cm、至少2个手部link净接触力>0.05N且网格顶点距离<8mm、物体合接触力>0.1N。保持更久与完成翻转分别统计，不能互相替代。\n','分离：全部手collision顶点至灯泡collision顶点最近距离>5mm且物体净接触力<0.05N，或最近顶点距离>2cm，连续3控制步满足，取第1步。网格距离使用保守包围球/AABB加速，距离在3cm封顶；优化结果已与已有ep76/direct完整359步逐字段核对一致。3/5/8/10mm阈值敏感性保存于JSON；半步至一步的小差异不作显著改善结论。\n','## 成对核验\n','16个新实验均检查27项初态完全一致、60步静置位姿/关节/命令/手接触力完全一致；物体净接触力浮点归约差异<1e-5N。全部原生failure配置一致，guide窗口按实际exec前进，每个完整reference尾段和60步hold都完成，未按物理分析标记提前结束或重置。真实源state、obj和actions不变，reference文件SHA256见manifest。\n','## 解释范围\n','仅4个经过direct翻转验收的起点、固定环境seed和prior噪声；不同episode的尾段长度不同，平均值只是这4个配对case的描述统计，不能推广成数据集成功率或统计显著性。原生fixed wrist与未独立标定外参限制沿用上轮。\n']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
for name in ['RESULTS.md','RESULTS.json']:shutil.copy2(OUT/name,Path('/home/carus/Downloads')/('Object_state_data_scale45_50_55_'+name))
print(json.dumps(dict(rows=[dict(episode=ep,exec=exe,scales=[lookup[ep,exe,sc]['reference_action_at_loss'] for sc in [45,50,55]],turns=[lookup[ep,exe,sc]['completed_turn'] for sc in [45,50,55]]) for exe in [2,1] for ep in EPS],stats=stats),indent=2),flush=True)
