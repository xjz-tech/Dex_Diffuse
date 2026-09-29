from pathlib import Path
import json,time,shutil,subprocess
import numpy as np
import cv2
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926';C=R/'qualified_comparison';D=Path('/home/carus/Downloads');EPS=[76,34,54,2];MODES=['direct','guide2','guide1']
for ep in EPS:
 deadline=time.monotonic()+1800
 while not all((C/f'episode_{ep:02d}'/m/'retention.json').exists() for m in MODES):
  if time.monotonic()>deadline:raise TimeoutError(str(ep))
  time.sleep(2)
rows=[];audit=[]
for ep in EPS:
 f=C/f'episode_{ep:02d}';baseline=json.load(open(f/'baseline_verified.json'));init=np.load(f/'direct/initial_state.npz');t0=json.load(open(f/'direct/trace.json'));ref=np.load(f/'reference_full.npz');ix=ref['source_state_frame_indices'][0];raw=np.load(f'/home/carus/Data/Object_state_data/episode_{ep}/action.npy');states=np.load(f'/home/carus/Data/Object_state_data/episode_{ep}/state.npy');assert np.array_equal(ref['hand_target_rad'][0],raw[ix[:-1],9:]);assert np.array_equal(ref['hand_qpos_rad'][0],states[ix,9:]);row=dict(episode=ep,baseline=baseline)
 for m in MODES:
  sub=f/m;init2=np.load(sub/'initial_state.npz');s=json.load(open(sub/'summary.json'));t=json.load(open(sub/'trace.json'));rr=json.load(open(sub/'retention.json'));loss=rr['first_separation'];assert all(np.array_equal(init[k],init2[k]) for k in init.files);keys=[k for k in t0[0] if k!='object_contact_force'];assert all(all(a[k]==b[k] for k in keys) for a,b in zip(t0[:60],t[:60]));cfmax=float(np.max(np.abs(np.array([r['object_contact_force'] for r in t0[:60]])-np.array([r['object_contact_force'] for r in t[:60]]))));assert cfmax<1e-5;assert s['steps']==s['intended_steps'];assert s['action_limit'] is None;assert s['mass_kg']==.17 and s['friction']==2.2
  if m!='direct':
   preds=json.load(open(sub/'predictions.json'));assert [p['reference_index'] for p in preds]==list(range(0,s['steps']['action'],s['execution_steps']))
  cur=[];best=[]
  for ti,(state,geo) in enumerate(zip(t,rr['frames'])):
   ok=(state['phase']=='action' and ti<loss['trace_index'] and state['vertical_error_deg']<=30 and geo['mesh_table_clearance_m']>.08 and geo['near_contact_link_count']>=2 and geo['mesh_vertex_gap_m']<.008 and geo['object_contact_force_norm_N']>.1)
   if ok:
    cur.append(state['index']+1)
    if len(cur)>len(best):best=cur.copy()
   else:cur=[]
  turn=dict(completed=len(best)>=30,longest_consecutive_control_steps=len(best),block_control_steps=[best[0],best[-1]] if best else None,reference_interval=[1+(best[0]-1)/(1+s['reference_interpolation']),1+(best[-1]-1)/(1+s['reference_interpolation'])] if best else None)
  row[m]=dict(first_separation=loss,first_native_failure=s['first_native_failure'],action_steps=s['steps']['action'],held_turn=turn)
  audit.append(dict(episode=ep,method=m,all27_initial_fields_exact=True,all60_settle_fields_except_object_contact_force_exact=True,max_object_contact_force_difference_N=cfmax,full_reference_and_hold=True,source_state_and_actions_exact=True))
 rows.append(row)
wins={m:sum(r[m]['first_separation']['reference_action_number']>r['direct']['first_separation']['reference_action_number'] for r in rows) for m in ['guide2','guide1']}
result=dict(selected_episodes=EPS,baseline_requirement='own horizontal source state, raw reference unchanged30Hz, settled angle65..115, action angle<=30 for30 consecutive steps held in air; geometry/contact+video validated',selected_conditionally_on_direct_success=True,selection=json.load(open(C/'selection_final.json')),results=rows,guide_later_separation_count=wins,paired_audit=audit)
(R/'RESULTS.json').write_text(json.dumps(result,indent=2));print('RESULTS',[(r['episode'],[r[m]['first_separation']['reference_action_number'] for m in MODES]) for r in rows],flush=True)
lines=['# 修正：先通过原始reference翻转基线，再做guidance对照\n','## 上一轮的问题\n','上一轮把“相似横抓＋静置稳定”误当成翻转初态合格。旧50/f111、76/f94、70/f73、40/f93的direct，在脱手前距竖直30°内分别为0、5、1、8步，均没有稳定完成横转竖。写入数值等于数据，并不能证明碰撞/接触和接下来的动作都可复现。因此不能把上一轮4/4保持更久直接解释为翻转能力改善。\n','## 这次怎样修正\n','保持0.17kg、摩擦2.2、原生固定wrist、30Hz、同一资产/seed/控制器不变。先在旧4条各自的真实横抓段测试15个起点（见candidates.json和screen_results.json），所有源q、obj、reference同时按对应源帧切片，动作数值不改、无插值。\n','ep76由f94改到f114，通过；两帧仍都是横抓（真机角度74.0°与70.8°），但手关节差异RMSE0.3054rad、灯泡相对手腕平移差1.58cm、旋转差23.34°，不是同一个接触构型。新起点的direct稳定完成翻转。50、70、40在本次已测起点中未通过全部门槛；这不等于证明它们没有任何可行起点。ep40/f122能接近竖直，但静置后已转到54.4°，不满足横抓起点要求，因此拒绝。\n','保留修正后的76；另外从已有direct轨迹显示可完成翻转的候选中，以seed20260926打乱顺序（34、54、2、57、30、35、63），取首先经过新鲜完整回放＋接触/几何/视频核验的34、54、2补足4条。57也做了备用direct回放但未进入4条对比。未按guide结果筛选。**这是direct基线合格的条件子集，不是对全数据集的随机成功率估计。**\n','## 原始动作翻转验收\n','验收要求：动作开始时距竖直65–115°；真实源帧仍横抓且估计离桌至少1cm；仿真静置抓稳。随后原始actions直接执行，灯泡距竖直≤30°且持续至少30控制步仍悬空握住。用全部hand collision网格/灯泡几何、近物接触link、物体合接触力及录像核验；不是“瞬间掠过竖直”或落到桌面。30°是事先声明的本轮近竖直阈值，不声称精确0°。\n','| Episode | 自身源起点 | 仿真动作起点角度 | 连续近竖直且接触的原始动作区间 | 持续时间 |','|---:|---:|---:|---:|---:|']
for r in rows:
 b=r['baseline'];lines.append(f'| {r["episode"]} | {b["start"]} | {b["settle_angle"]:.1f}° | {b["block"][0]}–{b["block"][1]} | {b["longest_vertical_contact_steps"]/30:.2f}s |')
lines+=['\n每条上述区间前30步进一步逐帧核验手/物体网格距离及至少2个近物接触link；灯泡网格距桌面至少18.8cm。完整区间的角度、净力和近物手部接触也持续记录。初态和验收末帧对照：qualified_comparison/baseline_proof.jpg。\n','## 重跑guidance：按原始reference编号比较\n','10B EMA/DDIM4、scale50、guide2，原始目标间插1中点，分别exec2/exec1。下表是首次连续3步几何分离标记对应的原始动作编号，1开始；.5表示插值中点，越大表示保持到更晚的reference进度。\n','| Episode | 原始direct | guide2/exec2 | guide2/exec1 |','|---:|---:|---:|---:|']
for r in rows:lines.append(f'| {r["episode"]} | '+ ' | '.join(f'{r[m]["first_separation"]["reference_action_number"]:g}' for m in MODES)+' |')
lines+=['\n另外核对guide是否也完成翻转（≤30°且连续30控制步保持近物接触、悬空，必须在首次脱手前）：\n','| Episode | direct翻转 | exec2翻转 | exec1翻转 |','|---:|---:|---:|---:|']
for r in rows:lines.append(f'| {r["episode"]} | '+' | '.join(('通过' if r[m]['held_turn']['completed'] else '未通过')+f'（最长{r[m]["held_turn"]["longest_consecutive_control_steps"]}步）' for m in MODES)+' |')
lines += [f'\n本轮exec2有{wins["guide2"]}/4条更晚分离，exec1有{wins["guide1"]}/4条更晚分离。旧4条的结果不可搬到这轮，guide不保证优于原始reference。数值为整个guide＋插值放慢方案，不单独证明prior因果收益。\n','物理分离分析沿用前轮：手/灯泡collision顶点间隙>5mm且物体合接触力<0.05N，或间隙>2cm，连续3个控制步满足时取第1步；分析不影响控制，视频另行复核。native failure仍按原生目标流记录，不能等同物理脱手。\n','## 核验与限制\n','每episode三组27项初态逐值相同，60步静置的位姿/关节/指令/手接触记录逐值一致，物体净接触力归约差异<1e-5N。源state/actions逐值核验，guide调用索引按实际exec滑动；全部reference尾段及60步hold执行完，没有按失败停止或重置。原生xjz_test.sh协议和全部阈值保留。\n','现在证明的是这4个具体仿真起点的**原始动作可完成近竖直翻转**。TCP→手根、动捕→网格外参仍未独立标定，没有据此宣称完整初始化映射已经校准，也未修好旧50/70/40的所有可能起点。固定wrist且无插座，不代表完成真机放置。\n','## 文件\n','Downloads/Object_state_data_reference_turn_verified.mp4：四条原始动作翻转验收短片；实际控制30Hz、动作画面放慢2倍，前100步，最后画面明确标注定格2秒。完整后续动作未隐藏，见下面对照。\n','Downloads/Object_state_data_reference_turn_qualified_comparison.mp4：四条完整reference/direct/guide三组对照，含原始真机画面、导入/静置、全尾段、末尾2秒；左上原始仿真。\n','Downloads/Object_state_data_reference_turn_initial_and_success.jpg：源初态、仿真横抓起点、原始动作翻转后抓握证据。\n','EPISODE53_REPLAY_LESSONS.md和项目AGENTS.md已加入“静置抓稳不足，先验收direct能翻转”的规则。']
(R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
for name in ['RESULTS.md','RESULTS.json']:shutil.copy2(R/name,D/('Object_state_data_reference_turn_'+name))
paths=[C/f'episode_{ep:02d}/episode_{ep:02d}_comparison.mp4' for ep in EPS]
for p in paths:
 deadline=time.monotonic()+1800
 while not p.with_name('video_manifest.json').exists():
  if time.monotonic()>deadline:raise TimeoutError(str(p))
  time.sleep(2)
(R/'concat.txt').write_text(''.join("file '"+str(p)+"'\n" for p in paths))
ff='/home/carus/miniforge3/envs/dp/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2';out=D/'Object_state_data_reference_turn_qualified_comparison.mp4'
subprocess.run([ff,'-y','-v','error','-f','concat','-safe','0','-i',str(R/'concat.txt'),'-c','copy','-movflags','+faststart',str(out)],check=True)
cap=cv2.VideoCapture(str(out));count=0
while True:
 ok,fr=cap.read()
 if not ok:break
 assert fr.shape==(1152,1280,3);count+=1
expected=sum(json.load(open(p.with_name('video_manifest.json')))['frames'] for p in paths);assert count==expected
cap.release();(R/'final_video_verification.json').write_text(json.dumps(dict(file=str(out),decoded_frames=count,fps=30,duration_s=count/30,episodes=EPS),indent=2));print('FINAL VIDEO',out,count,count/30,flush=True)
