"""Audit online A/C references, original initialization and real-time six-panel video."""
import json,hashlib,subprocess,shutil
from pathlib import Path
import numpy as np
import cv2,imageio_ffmpeg
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
METHODS=['direct','guidance','edit'];LABELS={'direct':'Direct','guidance':'Guidance DDIM4','edit':'Edit DDIM4 / noise 0.15'}

def read(path):return json.loads(path.read_text())
def load(d,m,reviews):
 f=P/'runs'/f'{d}_{m}';rows=[json.loads(x) for x in (f/'trajectory.jsonl').read_text().splitlines()]
 assert len(rows)==600 and [x['step'] for x in rows]==list(range(1,601))
 manifest=read(f/'manifest.json');model=read(f/'model.json')
 assert manifest['replay'] is None and manifest['guidance_steps']==9 and manifest['execution_steps']==2 and manifest['required_plan_steps']==16
 assert manifest['data_indices']=='000-149' and model['prior_ddim_steps']==4 and not model['fixed_noise'] and model['noise_seed']==42
 expected=dict(failure_obj_pos_thres_m=.05,failure_tip_pos_thres_m=.1,failure_obj_rot_thres_deg=180.,invalid_obj_pos_thres_m=.15,failure_tolerance_scale=10000.,fixed_tolerance_steps=20000,traj_steps_limit=12000,reset_on_reach_goal=False,cross_trajectory_goal_prob=.3)
 assert manifest['protocol']==expected
 with np.load(f/'initial_state.npz') as current,np.load(ROOT/'.worktrees/Astra-controller/outputs/astra_left_sequence/v2_s5_pause100/initial_state.npz') as original:
  assert set(current.files)==set(original.files)
  for k in current.files:np.testing.assert_array_equal(current[k],original[k],err_msg=f.name+':'+k)
  mass=float(current['object_mass'][0]);mu=float(current['object_friction'][0,0]);fields=len(current.files)
  movement=float(np.linalg.norm(np.asarray(rows[0]['state']['object_position'])-current['object'][0,:3]))
 assert movement<.01
 insertions=0;plan_jumps=[];actual_refs=[];last=None;phases=[]
 for i,step in enumerate(range(0,600,8),1):
  req=read(f/f'request_{i:04d}.json');rsp=read(f/f'response_{i:04d}.json')
  assert req['step']==step and rsp['online_feedback_step']==step and 'replay_source' not in rsp
  if step:
   for k in ['qpos','target_before','object_position','object_xyzw']:
    np.testing.assert_array_equal(req['state'][k],rows[step-1]['state'][k],err_msg=k)
  raw=np.asarray(rsp['raw_actions']);target=np.asarray(rsp['actions']);expanded=[];origins=[]
  prev=np.asarray(req['state']['target_before']) if last is None else last
  for j,q in enumerate(raw):
   if np.max(np.abs(q-prev))>.09:expanded.append((prev+q)/2);origins.append(True)
   expanded.append(q);origins.append(False);prev=q
  np.testing.assert_array_equal(target,np.asarray(expanded[:16]));last=target[7]
  insertions+=sum(origins[:8]);plan_jumps.append(rsp['interpolation']['raw_max_jump_rad'])
  actual_refs.extend(target[:8])
  phases.extend([rsp.get('action_phases',[rsp['phase']]*16)[x['raw_index']] for x in rsp['interpolation']['plan_origins'][:8]])
  for off in [0,2,4,6]:
   s=step+off;meta=read(f/f'prediction_{s:06d}.json')
   assert meta['reference_hold_padding_steps']==0
   with np.load(f/f'prediction_{s:06d}.npz') as z:
    np.testing.assert_array_equal(z['reference'][0],target.astype(np.float32)[off:off+9])
    if m=='direct':np.testing.assert_array_equal(z['prediction'][0],target.astype(np.float32)[off:off+2])
   if m=='edit':
    assert meta['edit_stats']['history_mask_max_error']==0 and meta['reference_editor']['inference_steps']==4
    assert meta['reference_editor']['noise_mode']=='fresh_generator_stream'
   if m=='guidance':
    assert np.isfinite(meta['mse_before']) and np.isfinite(meta['mse_after'])
    if d=='left':
     scale=5
     for interval in read(f/'guidance_schedule.json')['intervals']:
      if interval['start_step']<=s<interval['end_step']:scale=interval['scale']
     assert meta['guidance_scale_used']==scale
 np.testing.assert_array_equal(np.asarray(actual_refs,dtype=np.float32),np.asarray([r['astra_reference'] for r in rows],dtype=np.float32))
 if m=='edit':assert model['reference_editor']['requested_noise_ratio']==.15 and model['guidance_scale']==0
 if m=='guidance':assert model['guidance_scale']==(5 if d=='left' else 25) and model['reference_editor'] is None
 review=reviews[f.name];cut=review['last_confirmed_held_step'];sign=1 if d=='left' else -1
 angles=[sign*r['twist_degrees'] for r in rows[:cut]]
 summary=dict(direction=d,method=m,steps=600,control_dt=manifest['control_dt'],mass_kg=mass,object_friction=mu,
  initial_fields_identical_to_A=fields,first_control_step_object_translation_m=movement,
  first_native_failure=next((r['step'] for r in rows if r['failure']),None),physical_review=review,
  held_max_requested_deg=max([0.]+angles),held_net_requested_deg=angles[-1],
  online_plans=75,verified_prediction_windows=300,executed_midpoints=insertions,max_raw_reference_jump_rad=max(plan_jumps),
  noise_seed=model['noise_seed'],fixed_noise=False,reference_editor=model['reference_editor'])
 (f/'summary.json').write_text(json.dumps(summary,indent=2))
 return f,summary,rows,phases

def panel(data,step):
 f,s,rows,phases=data;d=s['direction'];m=s['method'];cut=s['physical_review']['last_confirmed_held_step'];sep=s['physical_review'].get('first_confirmed_separated_step',601)
 im=cv2.imread(str(f/'frames'/f'{step:06d}.png'));assert im is not None
 # Preserve complete scene below two old text lines, including floor and fallen bulb.
 im=cv2.resize(im[80:],(640,437));out=np.zeros((528,640,3),np.uint8);out[61:498]=im
 title=d.upper()+' | '+LABELS[m]
 cv2.putText(out,title,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),1,cv2.LINE_AA)
 label='g=5/100' if d=='left' else 'g=25'
 if m!='guidance':label='online reference'
 cv2.putText(out,f'{step*s["control_dt"]:05.2f} / 20.00 s | 1x | {label} | {phases[step-1]}',(12,47),cv2.FONT_HERSHEY_SIMPLEX,.47,(205,220,230),1,cv2.LINE_AA)
 status='DROPPED (commands continue)' if step>=sep else ('LOSS WINDOW' if step>cut else 'IN OBSERVATION')
 sign=1 if d=='left' else -1;angle=sign*rows[min(step,cut)-1]['twist_degrees']
 cv2.putText(out,f'{status} | held turn {angle:+.1f} deg',(12,519),cv2.FONT_HERSHEY_SIMPLEX,.49,(50,175,255) if step>cut else (170,230,170),1,cv2.LINE_AA)
 return out

def main():
 reviews=read(P/'physical_reviews.json');data={d:[load(d,m,reviews) for m in METHODS] for d in ['left','right']}
 name='astra_ac_online_direct_guidance_edit_1x.mp4';video=P/name;dt=data['left'][0][1]['control_dt']
 process=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1056','-r',str(1/dt),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
 for step in range(1,601):
  frame=cv2.vconcat([cv2.hconcat([panel(x,step) for x in data[d]]) for d in ['left','right']]);process.stdin.write(frame.tobytes())
  if step in [1,150,300,600]:cv2.imwrite(str(P/f'comparison_{step:03d}.jpg'),frame)
 process.stdin.close();assert process.wait()==0
 cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);n=0
 while cap.read()[0]:n+=1
 cap.release();assert n==600 and abs(n/fps-600*dt)<.002
 result=dict(summaries=[x[1] for d in data for x in data[d]],video=dict(path=str(video),frames=n,fps=fps,playback_duration_s=n/fps,simulation_duration_s=600*dt,speed_ratio=600*dt/(n/fps),added_freeze_frames=0))
 (P/'analysis.json').write_text(json.dumps(result,indent=2))
 table=['|方法|左转观察结果|右转观察结果|左转握持阶段最大转角|右转握持阶段最大转角|','|---|---|---|---:|---:|']
 def hold(s):
  r=s['physical_review'];return '20秒截止仍握住' if r['action']=='held_at_cap' else f"{r['last_confirmed_held_step']*dt:.2f}–{r['first_confirmed_separated_step']*dt:.2f}秒脱手"
 for i,m in enumerate(METHODS):
  l=data['left'][i][1];r=data['right'][i][1];table.append(f"|{LABELS[m]}|{hold(l)}|{hold(r)}|{l['held_max_requested_deg']:.2f}°|{r['held_max_requested_deg']:.2f}°|")
 report='''# A / C 方法在线控制：Direct、Guidance、Edit\n\n本次重新运行六组，而非拼接旧视频。上排左转，下排右转；各排从左至右 Direct / Guidance / Edit。600实际控制步、约20秒、1×；掉落后仍实际模拟和下发动作至20秒，无补定格。\n\n'''+ '\n'.join(table)+'''\n\n按用户选定的旧 A、C 配置：10B EMA、DDIM4、guide9/exec2，环境seed42、采样seed42、每窗口新噪声；质量262.26166g、物体摩擦1.05999994。全部27项初始化字段与旧A逐值相同，也与旧C相同；未使用上一轮170g/2.2覆盖。\n\n左转采用A的15°指腹几何推进上限、IK增量上限0.15rad、0内收、每达到累计半圈附近暂停60控制步。Guidance转动scale5、暂停100；20秒内继续后续半圈，不在原A第三次暂停后结束。右转沿C记录的推进/接触恢复参数阶段及其控制步时序，使用当前手指与灯泡位姿重算16步目标；含早期负向扫动、向内恢复和显式收拢，未加后来周期换指。C结束后继续其末段负向扫动。右转阶段切换沿原时间表，几何目标是在线反馈，并非每阶段都由新接触事件触发。\n\nAstra在此指当前助手编写并恢复的解析URDF Jacobian反馈规则，与当时实现一致；没有每8步调用外部LLM。三组各自每8实际步读取本组状态、重算16目标，reference会随状态分叉。这是完整在线控制方法比较，不是完全相同数值reference的消融。逐窗口审计当前反馈、16/9/2切片、直接组精确执行reference和Edit历史mask。\n\nEdit为reference加噪去噪：请求noise ratio0.15，实际0.153397，DDIM时刻[8,5,3,0]，guidance0。为沿用A/C每次新噪声配置，本轮Edit也使用seed42连续生成器流；区别于上一轮固定seed44每窗口重复同一噪声。Guidance与Edit的去噪起点及历史约束不同，属于方法差异。\n\n沿用用户要求：原始相邻reference任一关节跳变>0.09rad时插一个中点，包括重规划边界；扩展后取16步完整计划，实际执行前8步后重规划，逐项记录实际执行的中点数。只插一次，不宣称插后所有差值都≤0.09rad。该处理及20秒续转是相对于历史A/C的新修改，本次不是旧轨迹逐帧复现。\n\n原生协议保留：示范000–149，位置0.05m、指尖0.1m、旋转180°、立即位置0.15m、FailureToleranceScale10000、fixedToleranceSteps20000、resetOnReachGoal=false、跨轨迹目标概率0.3、traj_steps_limit12000。依据用户持续录制20秒要求，记录native failure后抑制重置继续运行；不改阈值与目标更新。脱手区间由实际录像独立审阅，握持转角截止到最后确认仍握住的帧，掉落后转角不计入。\n\n单一seed的探索对比，不能作为普遍方法成功率或排序。\n\n'''+f'[六宫格实录]({name})\n'
 (P/'RESULTS.md').write_text(report)
 # Single decoder, actual simulation playback duration, explicit 1x button.
 html='''<!doctype html><meta charset="utf-8"><title>A/C 在线左右转 · 六组实录</title><style>body{margin:24px;background:#141922;color:#edf2fa;font:17px system-ui}h1{font-size:26px}video{width:100%;max-height:80vh;background:#000}p{line-height:1.6;color:#b8c7dc}button{padding:10px 18px;background:#a8d5ff;border:0;border-radius:7px;cursor:pointer}a{color:#a8d5ff}</style><h1>A/C 方法在线左右转：Direct / Guidance / Edit</h1><p>上排左转，下排右转；从左到右 Direct → Guidance → Edit（DDIM4，noise 0.15）。原 A/C 物理参数：262g、摩擦1.06。各组自行在线生成 reference，全部实录20秒。</p><video id="movie" controls playsinline preload="metadata" src="VIDEO"></video><p id="status">点击播放 · 1×真实仿真速度</p><button id="restart">从头播放（1×）</button><p>左转 Guidance：转动5／暂停100；右转：25。相邻 reference 跳变&gt;0.09rad插一次中点。掉落后继续模拟，无补定格。</p><p><a href="VIDEO" download>下载六宫格录像</a> · <a href="RESULTS.md">实验记录</a></p><script>const v=document.querySelector('#movie');function u(){document.querySelector('#status').textContent=`${v.paused?'暂停':'播放中'} · ${v.currentTime.toFixed(2)} / ${Number.isFinite(v.duration)?v.duration.toFixed(2):'--'}秒 · ${v.playbackRate}倍速`;}['loadedmetadata','timeupdate','play','pause','ratechange'].forEach(e=>v.addEventListener(e,u));document.querySelector('#restart').onclick=()=>{v.currentTime=0;v.playbackRate=1;v.play()};</script>'''.replace('VIDEO',name)
 (P/'index.html').write_text(html)
 hashes={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'run.py',P/'left_planner.py',P/'right_original_parameters.json',P/'analyze.py',P/'source/eval/astra_model_server.py',P/'source/eval/reference_action_editor.py']}
 (P/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
 dest=Path('/home/carus/Downloads/astra_ac_online_20s_20260928');dest.mkdir(exist_ok=True)
 for f in [name,'index.html','RESULTS.md','analysis.json']:shutil.copy2(P/f,dest/f)
 print('\n'.join(table));print(video)
if __name__=='__main__':main()
