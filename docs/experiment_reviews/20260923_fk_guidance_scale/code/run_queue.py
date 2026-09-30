"""Resume-safe serial native-protocol evaluation and automatic reporting."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import cv2
import imageio_ffmpeg
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()

EXP=Path(__file__).resolve().parents[1]
ROOT=EXP.parents[2]
SEEDS=(42,8,19)
ARMS=[('joint',25)]+[('fingertip',s) for s in (25,50,100,200,500,1000)]
N=1024

def now(): return dt.datetime.now().astimezone().isoformat()
def save(name,value):
    p=EXP/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def records(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def validate(out):
    if not (out/'complete.json').exists():return False
    rows=records(out/'episodes.jsonl')
    return len(rows)==N and sorted(r['env'] for r in rows)==list(range(N)) and all(0<r['length']<=12000 and r['episode']==0 for r in rows)

def report():
    summary={};pairing={}
    for metric,scale in ARMS:
        name=f'{metric}_scale{scale}';perseed={};allrows=[];deltas=[]
        for seed in SEEDS:
            out=EXP/'runs'/f'{name}_seed{seed}'
            if not validate(out):continue
            rows=sorted(records(out/'episodes.jsonl'),key=lambda x:x['env'])
            values=np.array([r['length']/30 for r in rows])
            allrows.extend(rows)
            perseed[seed]={'mean_s':float(values.mean()),'median_s':float(np.median(values))}
            if metric=='fingertip':
                baseline=EXP/'runs'/f'joint_scale25_seed{seed}'
                if validate(baseline):
                    a=np.load(baseline/'initial_state.npz');b=np.load(out/'initial_state.npz')
                    fields={k:bool(np.array_equal(a[k],b[k])) for k in a.files}
                    pairing[out.name]={'all_fields_exact':all(fields.values()),'fields':fields}
                    if not all(fields.values()):raise RuntimeError(f'Pairing mismatch: {out.name}: {fields}')
                    base_rows=sorted(records(baseline/'episodes.jsonl'),key=lambda x:x['env'])
                    diff=values-np.array([r['length']/30 for r in base_rows]);deltas.extend(diff.tolist())
                    perseed[seed]['paired_mean_delta_vs_joint25_s']=float(diff.mean())
        if allrows:
            v=np.array([r['length']/30 for r in allrows])
            summary[name]={'n':len(v),'completed_seeds':list(perseed),'mean_s':float(v.mean()),'median_s':float(np.median(v)),
                'survival_20s':float((v>=20).mean()),'survival_80s':float((v>=80).mean()),'survival_400s':float((v>=400).mean()),
                'native_failure_fraction':sum(r['reason']=='failure' for r in allrows)/len(v),'per_seed':perseed}
            if deltas:
                delta=np.asarray(deltas)
                summary[name]['paired_vs_joint25']={'mean_delta_s':float(delta.mean()),'median_delta_s':float(np.median(delta)),
                    'improved_fraction':float((delta>0).mean()),'worsened_fraction':float((delta<0).mean()),'tied_fraction':float((delta==0).mean())}
    save('results.json',summary);save('pairing_validation.json',pairing)
    lines=['# 1B + 10k guide：FK 指尖 loss scale 扫描','',f'更新：{now()}','',
        '原生 xjz 协议；DDIM4/4，exec2，引导前2步；seed42/8/19，各1024个随机初态。首轮，12000控制步（400秒）封顶。',
        'FK loss = 反归一化关节角经 SharpA URDF FK 后，前2步×5指尖×XYZ 的均方误差（m²），无额外缩放、无关节夹紧。',
        '均值为400秒封顶观测时长（包含尚未失败的样本），不是仅失败样本的均值。原生 failure 是失败代理。',
        '所有组 fresh noise，prior seed=环境seed，guide seed=seed+100000；初态和物理参数逐字段核对。',
        '视频由 headless camera 在每次原生 env.step 后采集 env0 首轮，30fps，每个控制步一帧；通过帧数与首轮长度核验。','',
        '| Guidance | 完成seed | n | 平均秒 | 中位秒 | ≥20s | ≥80s | 达400s |','|---|---|---:|---:|---:|---:|---:|---:|']
    for name,s in summary.items():
        lines.append(f'| {name} | {s["completed_seeds"]} | {s["n"]} | {s["mean_s"]:.2f} | {s["median_s"]:.2f} | {s["survival_20s"]:.1%} | {s["survival_80s"]:.1%} | {s["survival_400s"]:.1%} |')
    lines.extend(['','| FK scale | 相对同期关节scale25平均变化(s) | 配对改善比例 | 配对变差比例 |','|---|---:|---:|---:|'])
    for name,s in summary.items():
        d=s.get('paired_vs_joint25')
        if d:lines.append(f'| {name} | {d["mean_delta_s"]:+.2f} | {d["improved_fraction"]:.1%} | {d["worsened_fraction"]:.1%} |')
    if len(summary)==len(ARMS) and all(len(s['completed_seeds'])==3 for s in summary.values()):
        best=max((k for k in summary if k.startswith('fingertip')),key=lambda k:summary[k]['mean_s'])
        lines.extend(['',f'本轮按400秒封顶平均保持时长，FK 最优为 {best}，{summary[best]["mean_s"]:.2f} 秒。三 seed 的描述性比较，尚未单独做最优 scale 的独立复验。'])
    else:lines.extend(['','实验仍在运行，上表仅列已完成组；全部三个 seed 完成后再比较各 scale。'])
    (EXP/'report.md').write_text('\n'.join(lines)+'\n')
    return summary

def main():
    state={'state':'running','started_at':now(),'completed':[],'planned_runs':21,'num_envs_per_seed':N}
    for seed in SEEDS:
        for metric,scale in ARMS:
            name=f'{metric}_scale{scale}_seed{seed}';out=EXP/'runs'/name
            out.mkdir(parents=True,exist_ok=True)
            if validate(out):
                state['completed'].append(name);report();continue
            if (out/'episodes.jsonl').exists():
                raise RuntimeError(f'Incomplete previous run requires inspection: {out}')
            state.update(current=name,current_started_at=now());save('status.json',state)
            env=os.environ.copy();env['NUM_ENV']=str(N);env['MAX_STEPS']='12000'
            with (out/'run.log').open('w') as log:
                p=subprocess.run(['bash',str(EXP/'code/run_arm.sh'),metric,str(scale),str(seed),str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            rows=records(out/'episodes.jsonl') if (out/'episodes.jsonl').exists() else []
            valid=p.returncode==0 and len(rows)==N and sorted(r['env'] for r in rows)==list(range(N)) and all(0<r['length']<=12000 and r['episode']==0 for r in rows)
            raw=out/'video/live_raw.mp4'
            if raw.exists():
                conversion=subprocess.run([FFMPEG,'-v','error','-y','-i',str(raw),'-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video/live.mp4')])
                valid=valid and conversion.returncode==0
            recording=json.loads((out/'video/recording.json').read_text()) if (out/'video/recording.json').exists() else {}
            env0=next((r for r in rows if r['env']==0),{})
            valid=valid and recording.get('frames')==env0.get('length') and bool(recording)
            videos=list((out/'video').glob('live.mp4'))
            valid=valid and bool(videos)
            video_checks=[]
            for video in videos:
                capture=cv2.VideoCapture(str(video))
                frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps=capture.get(cv2.CAP_PROP_FPS)
                readable,frame=capture.read()
                capture.release()
                video_checks.append({'path':str(video),'frames':frame_count,'fps':fps,'readable':bool(readable)})
                valid=valid and readable and frame_count==env0.get('length') and abs(fps-30)<1e-6
            if not valid:
                state.update(state='failed',returncode=p.returncode,finished_at=now());save('status.json',state);return 1
            save(str(out.relative_to(EXP)/'complete.json'),{'finished_at':now(),'returncode':p.returncode,'episodes':len(rows),'video_checks':video_checks})
            state['completed'].append(name);state.pop('current',None);save('status.json',state);report()
    state.update(state='complete',finished_at=now());save('status.json',state);report();return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        save('queue_error.json',{'time':now(),'error':repr(e)});raise
