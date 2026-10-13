"""Add scale50 / guide4 / exec2 / DDIM4 to the existing fixed-reference table."""
import json,hashlib,subprocess,shutil
from pathlib import Path
import cv2,numpy as np
from analyze import P,FFMPEG,load,panel
METHOD='guidance50_g4_ddim4'

def interval(s):
 r=s['physical_review'];dt=s['control_dt']
 return '20.00秒仍握住' if r['action']=='held_at_cap' else f"{r['last_confirmed_held_step']*dt:.2f}–{r['first_confirmed_separated_step']*dt:.2f}"

def main():
 reviews=json.loads((P/'physical_reviews.json').read_text())
 with np.load(P/'runs/left_direct/initial_state.npz') as z:base={k:z[k].copy() for k in z.files}
 data=[load(d,METHOD,reviews,base) for d in ['left','right']]
 saved=json.loads((P/'source_hashes.json').read_text());hashes={}
 for d,(f,s,rows) in zip(['left','right'],data):
  key=d+'_reference.npz';hashes[key]=hashlib.sha256((P/key).read_bytes()).hexdigest();assert hashes[key]==saved[key]
  manifest=json.loads((f/'manifest.json').read_text());model=json.loads((f/'model.json').read_text())
  assert manifest['guidance_steps']==4 and manifest['execution_steps']==2
  assert model['guidance_steps']==4 and model['guidance_scale']==50 and model['prior_ddim_steps']==4
  old_model=json.loads((P/'runs'/f'{d}_edit015_ddim4/model.json').read_text())
  assert model['checkpoint']==old_model['checkpoint'] and model['checkpoint_info']==old_model['checkpoint_info']
 name='guidance50_guide4_exec2_ddim4_left_right_1x.mp4';video=P/name;dt=data[0][1]['control_dt']
 process=subprocess.Popen([str(FFMPEG),'-nostdin','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x510','-r',str(1/dt),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
 for step in range(1,601):
  frame=cv2.hconcat([panel(*item,step,d,METHOD) for item,d in zip(data,['left','right'])]);process.stdin.write(frame.tobytes())
  if step in [1,300,600]:cv2.imwrite(str(P/f'guidance50_g4_{step:03d}.jpg'),frame)
 process.stdin.close();assert process.wait()==0
 cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);n=0
 while cap.read()[0]:n+=1
 cap.release();assert n==600 and abs(n/fps-600*dt)<.002
 result=dict(summaries=[s for f,s,r in data],unchanged_reference_hashes=hashes,
  video=dict(path=str(video),frames=n,fps=fps,simulation_duration_s=600*dt,video_duration_s=n/fps,speed_ratio=600*dt/(n/fps),frozen_frames_added=0),
  source_hashes={k:hashlib.sha256((P/k).read_bytes()).hexdigest() for k in ['run.py','analyze.py','analyze_guidance50_g4.py','source/eval/astra_model_server.py','source/eval/inference_dp_controller.py']})
 (P/'guidance50_g4_ddim4_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 row=f'|Astra|4|Guidance，scale50 / guide4 / exec2|{interval(data[0][1])}|{data[0][1]["held_max_requested_deg"]:.2f}°|{interval(data[1][1])}|{data[1][1]["held_max_requested_deg"]:.2f}°|'
 table=P/'COMPARISON_TABLE.md';t=table.read_text().replace('**0.15（新增）**','0.15').replace('新加两组沿用新对称循环reference','noise0.15/DDIM4 两组沿用新对称循环reference')
 t=t.replace('# 含 noise0.15 / DDIM4 的合并表','# 含 noise0.15 / DDIM4 与 Guidance scale50 / guide4 / exec2 的合并表')
 lines=[x for x in t.splitlines() if not x.startswith('|新对称循环|4|**Guidance scale50') and not x.startswith('| Astra | 4 | Guidance') and not x.startswith('|Astra|4|Guidance')]
 end=max(i for i,x in enumerate(lines) if x.startswith('|'));lines.insert(end+1,row);t='\n'.join(lines)+'\n'
 marker='\n## 新增 Guidance：scale50 / guide4 / exec2\n'
 if marker in t:t=t.split(marker)[0]
 details=marker+'\nGuide4表示引导未来4步reference；DDIM采样4步、每次执行2步。10B EMA、关节MSE梯度引导、scale50全程不变，无Edit加噪编辑。沿用170g、物体摩擦2.2、环境seed42、固定noise44、新对称循环reference、相邻>0.09rad插一次中点规则和600实际控制步/20秒上限。Edit使用未来9步reference，本行Guidance引导未来4步；两者采样方式也不同，本行不是仅改变scale的单变量比较。\n\n'
 for f,s,rows in data:
  native='无' if s['first_native_failure_step'] is None else 'step'+str(s['first_native_failure_step'])
  details+=f"- {s['direction']}：{interval(s)}；原生首次failure：{native}；握持最大指定方向净转角{s['held_max_requested_deg']:.2f}°，握持截止净转角{s['held_net_requested_deg']:.2f}°。\n"
 details+='\n两组27项初态与现有Direct/Edit对照逐值一致，左右reference文件哈希与DDIM6批次相同。600个4步引导窗口逐值核验，全部未补末目标；首实际控制步灯泡位移<2mm。guide4只改变传给Guidance的未来窗口长度，实际reference仍按每次执行2步连续前进。保留原生失败阈值与目标更新，failure独立记录；按既有20秒指令抑制重置，脱手后继续实录。\n\n'
 details+=f'[新增Guidance左右同步录像]({name})：600帧、约30fps、20.00004秒、1×，无补定格。\n'
 table.write_text(t+details)
 report=P/'RESULTS.md';old=report.read_text();section='\n## 补充 scale50 / guide4 / exec2\n'
 if section in old:old=old.split(section)[0]
 report.write_text(old+section+'\n'+row+'\n\n见[更新后的完整合并表](COMPARISON_TABLE.md)。\n')
 html='''<!doctype html><meta charset="utf-8"><title>Guidance scale50 / guide4 / exec2</title><style>body{margin:24px;background:#141922;color:#edf2fa;font:17px system-ui}h1{font-size:25px}video{width:100%;max-height:78vh;background:#000}p{line-height:1.6;color:#c4d1df}a{color:#9ed5ff}button{padding:10px 18px;background:#a8d5ff;border:0;border-radius:7px}</style><h1>新增：Guidance scale50 · guide4 · exec2 · DDIM4</h1><p>左侧左转，右侧右转。170g／物体摩擦2.2，固定noise44、环境seed42，与原表同一新对称循环reference。</p><video id="movie" controls playsinline preload="metadata" src="VIDEO"></video><p id="status">完整20秒 · 1×实录</p><button id="restart">从头播放（1×）</button><p><a href="COMPARISON_TABLE.md">完整合并表</a> · <a href="VIDEO" download>下载新增录像</a></p><script>const v=document.querySelector('#movie');function u(){document.querySelector('#status').textContent=`${v.paused?'暂停':'播放中'} · ${v.currentTime.toFixed(2)} / ${Number.isFinite(v.duration)?v.duration.toFixed(2):'--'}秒 · ${v.playbackRate}倍速`;}['loadedmetadata','timeupdate','play','pause','ratechange'].forEach(e=>v.addEventListener(e,u));document.querySelector('#restart').onclick=()=>{v.currentTime=0;v.playbackRate=1;v.play()};</script>'''.replace('VIDEO',name)
 (P/'guidance50_g4.html').write_text(html)
 dest=Path('/home/carus/Downloads/astra_symmetric_ddim6_20s_20260928');dest.mkdir(exist_ok=True)
 for name in [name,'guidance50_g4.html','COMPARISON_TABLE.md','RESULTS.md','guidance50_g4_ddim4_analysis.json']:shutil.copy2(P/name,dest/name)
 print(row)
 for _,s,_ in data:print(json.dumps(s,ensure_ascii=False))
if __name__=='__main__':main()
