from pathlib import Path
import json,numpy as np,cv2
r=Path(__file__).resolve().parents[1];old=r.with_name('20260918_bulb44g_size100');rows=[];checks=[]
for seed in [25,19,42]:
 for arm in ['ordinary_1b','guide10k']:
  d=r/f'seed{seed}_{arm}';p=old/d.name
  a=np.load(d/'initial_state.npz');b=np.load(p/'initial_state.npz')
  ra=np.load(d/'rollout.npz');rb=np.load(p/'rollout.npz');assert np.array_equal(ra['raw'][:2],rb['raw'][:2])
  equality={k:bool(np.array_equal(a[k],b[k])) for k in a.files};assert all(equality.values()),equality
  phys=json.loads((d/'physical_parameters.json').read_text());prev=json.loads((p/'physical_parameters.json').read_text())
  assert abs(phys['friction']-2.572)<1e-6
  for k in phys:
   if k!='friction':assert phys[k]==prev[k],(k,phys[k],prev[k])
  cfg=json.loads((d/'config.json').read_text());oldcfg=json.loads((p/'config.json').read_text())
  for k in cfg:
   if k!='physical_override':assert cfg[k]==oldcfg[k],k
  assert (d/'model_manifest.json').read_text()==(p/'model_manifest.json').read_text()
  assert json.loads((d/'first_step_wrist.json').read_text())['quaternion_delta_norm']<1e-4
  x=json.loads((d/'result.json').read_text());y=json.loads((p/'result.json').read_text())
  rows.append({'seed':seed,'arm':arm,'demo':cfg['initialization']['demo'],'old_friction':prev['friction'],'new_friction':phys['friction'],'old_seconds':y['seconds'],'new_seconds':x['seconds'],'old_reason':y['reason'],'new_reason':x['reason']})
  checks.append({'seed':seed,'arm':arm,'initial_state_and_rng_equal_to_previous':equality,'only_physical_difference':'friction','model_and_protocol_equal':True,'first_inference_identical':True})
  cap=cv2.VideoCapture(str(d/'live.mp4'));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==x['steps']
  for i in [0,x['steps']//2,x['steps']-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,img=cap.read();assert ok
  cap.release()
 m=json.loads((r/f'paired_seed{seed}/metadata.json').read_text());cap=cv2.VideoCapture(str(r/f'paired_seed{seed}/comparison.mp4'));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==m['output_frames']
 for i in [0,m['output_frames']//2,m['output_frames']-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,img=cap.read();assert ok
 cap.release()
(r/'comparison.json').write_text(json.dumps({'rows':rows,'checks':checks,'all_9_videos_verified':True},indent=2))
lines=['# 44g灯泡统一摩擦系数2.572','', '与上一轮相比，仅将灯泡各碰撞形状的摩擦系数改为2.572（引擎读回确认）。保留44g、scale1（横向包围盒约7.47cm）、同一历史初始化与seed，逐项核对初始状态及CUDA随机状态一致。手部摩擦和其余物理参数保持一致。','', '沿用xjz_test原生判据；DDIM4/4、执行2步、guide scale25引导前2步、400秒观察上限。400秒是删失观察，不代表恰好在400秒失败。每次参数设置后先进行1步不计分同步，再恢复完整初始状态。','', '| demo / seed | 方法 | 原摩擦系数 | 原时长/s | μ=2.572时长/s |','|---|---|---:|---:|---:|']
for x in rows:
 before=('≥' if x['old_reason']=='timeout' else '')+f"{x['old_seconds']:.2f}";after=('≥' if x['new_reason']=='timeout' else '')+f"{x['new_seconds']:.2f}"
 lines.append(f"| {x['demo']:03d} / {x['seed']} | {x['arm']} | {x['old_friction']:.3f} | {before} | {after} |")
lines+=['','## 实际运行录像','']
for seed in [25,19,42]:lines.append(f'- [seed{seed} 并排录像：左1B、右guide](paired_seed{seed}/comparison.mp4)')
lines+=['','较短一侧结束后停留在标注的末帧；短片如慢放会显式标注。所有9个录像核对帧数和首、中、末帧解码。原生failure为环境失败代理。每个条件仅一次配对运行，不能当作总体成功率。']
(r/'report.md').write_text('\n'.join(lines));print(json.dumps(rows,indent=2))
