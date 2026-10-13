import sys,json,hashlib,subprocess
from pathlib import Path
import numpy as np
import cv2,imageio_ffmpeg

root=Path(sys.argv[1]).resolve();partial='--partial' in sys.argv
results=[];pairs=[];media=[]
for seed in [50]:
    loaded={}
    for arm in ['ordinary_1b','guide10k']:
        p=root/f'seed{seed}_{arm}'
        if not (p/'result.json').exists():
            if partial:continue
            raise FileNotFoundError(p/'result.json')
        r=json.loads((p/'result.json').read_text());cfg=json.loads((p/'config.json').read_text())
        assert r['complete'] and cfg['prior_ddim_steps']==cfg['guide_ddim_steps']==4
        assert cfg['overrides']['invalidObjPosThres']==.15 and cfg['overrides']['FailureToleranceScale']==10000
        z=np.load(p/'rollout.npz');assert len(z['q'])==r['steps']
        assert abs(r['seconds']-r['steps']*cfg['control_dt'])<1e-8
        cap=cv2.VideoCapture(str(p/'live_raw.mp4'));assert cap.isOpened()
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert n==r['steps'],(p,n,r['steps'])
        for index in [0,n//2,n-1]:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read();assert ok
        cap.release()
        r['video_frames']=n;r['video']=str(p/'live.mp4');results.append(r)
        loaded[arm]=p
        if not (p/'live.mp4').exists():
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-i',str(p/'live_raw.mp4'),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(p/'live.mp4')],check=True)
        media.append({'path':str(p/'live.mp4'),'frames':n,'fps':30,'duration':n/30})
    if len(loaded)==2:
        a=np.load(loaded['ordinary_1b']/'initial_state.npz');b=np.load(loaded['guide10k']/'initial_state.npz')
        checks={k:bool(np.array_equal(a[k],b[k])) for k in a.files};assert all(checks.values()),(seed,checks)
        old=np.load(root.parent/'20260917_1b_initialization_audit/initial_state.npz')
        idx=32+seed-50
        errors={k:float(np.max(abs(a[k][0]-old[k][idx]))) for k in ['q','wrist','object']}
        assert max(errors.values())<1e-6,errors
        pairs.append({'seed':seed,'initial_states_equal':checks,'max_error_from_archived_real_initial':errors})
result={'complete':len(results)==2,'results':results,'paired_validation':pairs,'media':media}
(root/('seed50_partial_summary.json' if partial else 'seed50_summary.json')).write_text(json.dumps(result,indent=2))
lines=['# xjz_test原生协议：seed=50实录对照','','DDIM4/4，执行2步，guide scale25/引导2步；相同真机初态和原生随机化，约30Hz，最多400秒。只比较普通1B与1B+10k guide。时间为仿真时间、原生环境首轮结束时间；failure包含相对当前目标的位置误差超过15cm、累计跟踪失败或数值异常。','','| seed | 普通1B | 1B+10k guide |','|---:|---:|---:|']
for seed in [50]:
    row=[]
    for arm in ['ordinary_1b','guide10k']:
        match=[r for r in results if r['seed']==seed and r['arm']==arm]
        row.append((f"≥{match[0]['seconds']:.2f}s (观察上限)" if match[0]['reason']=='timeout' else f"{match[0]['seconds']:.2f}s ({match[0]['reason']})") if match else '运行中')
    lines.append(f"| {seed} | {' | '.join(row)} |")
lines+=['','## 实录视频','','每个视频均来自实际运行中的摄像头，逐控制步采集，未重新推理或离线重建。','']
for r in results:
    name=f"seed{r['seed']}_{r['arm']}";lines.append(f"- [seed={r['seed']} {r['arm']}]({name}/live.mp4)")
lines+=['','seed=50是诊断案例，不足以估计总体成功率。它们与旧5cm初始化偏移诊断的判据、物理设置和随机化不同，不能把跨协议的时长差异归因于guide。每对初始q、qd、腕部、物体、目标、质量和GPU随机数状态逐元素核对一致。','']
(root/('seed50_partial_report.md' if partial else 'seed50_report.md')).write_text('\n'.join(lines))
print('\n'.join(lines))
