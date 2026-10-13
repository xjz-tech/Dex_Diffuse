from pathlib import Path
import json,sys,subprocess
import numpy as np,cv2,imageio_ffmpeg
r=Path(__file__).resolve().parents[1];results=[];pairs=[]
for seed in [25,19,42]:
 loaded={}
 for arm in ['ordinary_1b','guide10k']:
  p=r/f'seed{seed}_{arm}'
  if not (p/'result.json').exists():continue
  result=json.loads((p/'result.json').read_text());config=json.loads((p/'config.json').read_text());physical=json.loads((p/'physical_parameters.json').read_text())
  assert result['complete'] and config['prior_ddim_steps']==config['guide_ddim_steps']==4
  assert abs(physical['mass_kg']-.044)<1e-7 and physical['scale']==1
  assert config['overrides']['invalidObjPosThres']==.15 and config['overrides']['FailureToleranceScale']==10000
  z=np.load(p/'rollout.npz');assert len(z['q'])==result['steps']
  cap=cv2.VideoCapture(str(p/'live_raw.mp4'));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==result['steps']
  for idx in [0,result['steps']//2,result['steps']-1]:
   cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,img=cap.read();assert ok
  cap.release()
  if not (p/'live.mp4').exists():subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-i',str(p/'live_raw.mp4'),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(p/'live.mp4')],check=True)
  result.update(initialization=config['initialization'],physical=physical,video=str(p/'live.mp4'));results.append(result);loaded[arm]=p
 if len(loaded)==2:
  a=np.load(loaded['ordinary_1b']/'initial_state.npz');b=np.load(loaded['guide10k']/'initial_state.npz');equal={k:bool(np.array_equal(a[k],b[k])) for k in a.files};assert all(equal.values())
  assert (loaded['ordinary_1b']/'physical_parameters.json').read_text()==(loaded['guide10k']/'physical_parameters.json').read_text()
  pairs.append({'seed':seed,'initial_state_equal':equal,'physical_equal':True})
(r/'summary.json').write_text(json.dumps({'complete':len(results)==6,'results':results,'pairs':pairs},indent=2))
lines=['# 44克、原始尺寸灯泡视频实验','','质量44g，scale=1，对应模型横向包围盒7.4663cm（非理想球体直径）；所有物理质量从引擎读回验证。原生xjz判据；prior/guide DDIM4、执行2步，guide scale25引导前2步，约30Hz，最多400秒。普通1B对照10k guide，无scale0。','','默认初始化：前述demo079/082/094的历史快速失败案例，继承q、腕部、物体位姿、目标帧以及手部质量/摩擦/Kp/Kd、物体表面属性、外力概率。仅物体质量和尺度指定为44g/1，重新计算物体惯量，并将外力质量缓存同步44g。使用新的单环境随机轨迹，不能直接与原3000环境旧scale0时长作因果比较。','','| demo | seed | 方法 | 结束时间/s | 原因 | 摩擦 |','|---:|---:|---|---:|---|---:|']
for x in results:lines.append(f"| {x['initialization'].get('demo','?')} | {x['seed']} | {x['arm']} | {'≥' if x['reason']=='timeout' else ''}{x['seconds']:.2f} | {x['reason']} | {x['physical']['friction']:.3f} |")
lines+=['','## 实际运行视频','']
for x in results:lines.append(f"- [{x['seed']} {x['arm']}](seed{x['seed']}_{x['arm']}/live.mp4)")
for seed in [25,19,42]:
 if (r/f'paired_seed{seed}/comparison.mp4').exists():lines.append(f'- [seed{seed} 并排视频](paired_seed{seed}/comparison.mp4)')
lines+=['','短于10秒的并排视频标注0.25倍慢放，其余为正常速度；较短一侧结束后停在已标注的末帧。原生failure为失败代理，未独立自动标注接触丢失。','']
(r/'report.md').write_text('\n'.join(lines))
print('completed',len(results), 'pairs',len(pairs),flush=True)
for x in results:print(x['seed'],x['arm'],x['seconds'],x['reason'],flush=True)
