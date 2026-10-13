"""Six synchronized actual-camera panels: left/right x direct/edit/guidance."""
import argparse,json,hashlib,subprocess,shutil
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import P,FFMPEG,load,panel,LABELS

def interval(s):
    r=s['physical_review'];dt=s['control_dt']
    if s['physical_drop_confirmed']:
        return f"{r['last_confirmed_held_step']*dt:.2f}–{r['first_confirmed_separated_step']*dt:.2f}秒"
    return '20秒截止仍握住'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scale',type=int,choices=[25,100],default=25);args=parser.parse_args()
    method=f'guidance{args.scale}_ddim4';methods=['direct','edit015_ddim4',method]
    LABELS['edit015_ddim4']='Edit noise 0.15 / DDIM4'
    reviews=json.loads((P/'physical_reviews.json').read_text())
    with np.load(P/'runs/left_direct/initial_state.npz') as z:baseline={k:z[k].copy() for k in z.files}
    data={d:[load(d,m,reviews,baseline) for m in methods] for d in ['left','right']}
    base_model=json.loads((data['left'][0][0]/'model.json').read_text())
    for d in data:
        for folder,summary,rows in data[d]:
            model=json.loads((folder/'model.json').read_text())
            assert model['checkpoint']==base_model['checkpoint'] and model['checkpoint_info']==base_model['checkpoint_info']
    tag=f'direct_edit015_guidance{args.scale}_ddim4'
    path=P/(tag+'_1x.mp4');dt=data['left'][0][1]['control_dt']
    command=[str(FFMPEG),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1020','-r',str(1/dt),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    for step in range(1,601):
        strips=[cv2.hconcat([panel(*item,step,d,m) for item,m in zip(data[d],methods)]) for d in ['left','right']]
        frame=cv2.vconcat(strips);process.stdin.write(frame.tobytes())
        if step in [1,300,600]:cv2.imwrite(str(P/(tag+f'_step{step:03d}.jpg')),frame)
    process.stdin.close();assert process.wait()==0
    c=cv2.VideoCapture(str(path));fps=c.get(cv2.CAP_PROP_FPS);count=0
    while c.read()[0]:count+=1
    c.release();assert count==600 and abs(count/fps-600*dt)<.002
    fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    for i,d in enumerate(['left','right']):
        for folder,s,rows in data[d]:
            cutoff=s['physical_review']['last_confirmed_held_step'];sign=1 if d=='left' else -1
            axes[i].plot(np.arange(1,cutoff+1)*dt,[sign*r['twist_degrees'] for r in rows[:cutoff]],label=LABELS[s['method']])
        axes[i].set(title=d,xlim=(0,20),xlabel='Simulation time (s)',ylabel='Requested-direction turn (deg)');axes[i].legend();axes[i].grid(alpha=.25)
    fig.savefig(P/(tag+'_curves.png'),dpi=150)
    hashes={d:hashlib.sha256((P/(d+'_reference.npz')).read_bytes()).hexdigest() for d in data}
    archived_hashes=json.loads((P/'source_hashes.json').read_text())
    for d,h in hashes.items():assert h==archived_hashes[d+'_reference.npz']
    result=dict(summaries=[s for d in data for _,s,_ in data[d]],video=dict(path=str(path),decoded_frames=count,fps=fps,
        video_duration_s=count/fps,simulation_duration_s=600*dt,speed_ratio=600*dt/(count/fps),layout='row1 left; row2 right; columns Direct / Edit noise0.15 DDIM4 / Guidance DDIM4 scale'+str(args.scale),frozen_frames_added=0),
        reference_hashes=hashes,source_hashes={k:hashlib.sha256((P/k).read_bytes()).hexdigest() for k in ['run.py','analyze.py','compare_guidance_edit.py','source/eval/astra_model_server.py','source/eval/inference_dp_controller.py','source/diffusion_policy/guidance/guided_ddim.py']})
    (P/(tag+'_analysis.json')).write_text(json.dumps(result,indent=2)+'\n')
    table=['|方法|左转脱手区间|右转脱手区间|左转握持时最大净转角|右转握持时最大净转角|','|---|---|---|---:|---:|']
    for j,m in enumerate(methods):
        l=data['left'][j][1];r=data['right'][j][1]
        table.append(f"|{LABELS[m]}|{interval(l)}|{interval(r)}|{l['held_max_requested_deg']:.2f}°|{r['held_max_requested_deg']:.2f}°|")
    report='# 新对称循环：Direct / Edit / Guidance 同屏对比\n\n'+ '\n'.join(table)+'\n\n'
    report+=f'使用同一套新对称循环reference，初态27字段逐值相同，170g、物体摩擦2.2、物理seed42；模型为同一10B EMA checkpoint。Edit为noise0.15（实际0.153397）、DDIM4、exec2。Guidance使用原Astra关节MSE梯度引导、scale{args.scale}、DDIM4、9步未来reference、exec2；模型从Gaussian prior起步，在4次采样中施加reference梯度。Guidance与Edit都固定noise seed44，每窗口复用噪声。没有附加guidance模型、残差clamp或动态scale。\n\n'
    report+='这是同一reference上的新补跑Guidance，不是拼接旧物理条件或旧左右reference的视频。旧的右转Guidance源实验用scale100和每窗口新噪声；本轮的scale选择和固定噪声配置在此显式记录。Edit具有已执行target的历史mask，Guidance保留既有算法；两者的加噪起点/反向采样时间点不同属于方法差异，并未声称相同扩散时刻。\n\n'
    report+='六组均完整执行600实际步，约20秒；脱手后继续动作、不重置。原生failure阈值与目标更新协议保持：demo000–149，位置0.05m、指尖0.1m、旋转180°、立即位置0.15m、FailureToleranceScale10000、fixedToleranceSteps20000、traj_steps_limit12000、resetOnReachGoal=false、跨轨迹0.3。native failure与录像脱手分开记录；抑制重置并继续至20秒来自用户明确要求。\n\n'
    report+='插值规则仍为reference相邻任一关节差>0.09rad插一个中点；当前共同reference最大差0.0375rad，因此实际新增0点。每个run的300个9步预测窗口与固定reference逐值相同，全部共1800窗口通过核验；Direct预测等于reference，Edit历史mask误差0，Guidance记录的MSE统计有限且正scale分支启用。首实际控制步灯泡位移<1.5mm。\n\n'
    report+='视频上排左转、下排右转，列顺序Direct / Edit / Guidance。600帧逐控制步同步、约30fps、20.00004秒、H.264、1倍速，无终帧冻结或补时。握持角度截止到各自最后确认仍握住的帧，后续下落旋转不计入。此为单一物理seed/噪声seed比较，不能外推普遍方法排序。\n\n'
    report+=f'[六组同屏视频]({path.name}) · [握持转角曲线]({tag}_curves.png)\n'
    (P/'GUIDANCE_VS_EDIT.md').write_text(report)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Direct / Edit / Guidance · 左右转同屏</title><style>body{margin:0;background:#14171d;color:#eef2f8;font:16px system-ui}main{max-width:1600px;margin:20px auto;padding:0 20px}h1{font-size:25px}p{color:#b9c4d4;line-height:1.6}video{width:100%;background:#000}a{color:#8bcbff}button{padding:10px 20px;font:inherit;background:#8bcbff;border:0;border-radius:7px;cursor:pointer}#status{font-variant-numeric:tabular-nums}</style><main><h1>Direct / Edit / Guidance · 左右转同屏</h1><p>上排左转，下排右转。每排从左到右：Direct → Edit（DDIM4，noise 0.15）→ Guidance（DDIM4，scale SCALE）。同初态、同对称循环 reference、170g / 摩擦2.2，完整20秒。</p><video id="movie" controls preload="auto" playsinline src="VIDEO"></video><p id="status">加载中</p><button id="restart">从头播放（1倍速）</button><p><a href="VIDEO" download>下载同屏录像</a> · <a href="GUIDANCE_VS_EDIT.md">实验结果</a></p></main><script>const v=document.querySelector('#movie');function u(){document.querySelector('#status').textContent=`${v.paused?'暂停':'播放中'} · ${v.currentTime.toFixed(2)} / ${Number.isFinite(v.duration)?v.duration.toFixed(2):'--'}秒 · ${v.playbackRate}倍速`;}['loadedmetadata','timeupdate','play','pause','ratechange'].forEach(e=>v.addEventListener(e,u));document.querySelector('#restart').onclick=()=>{v.currentTime=0;v.playbackRate=1;v.play()};</script></html>'''.replace('SCALE',str(args.scale)).replace('VIDEO',path.name)
    (P/'guidance_vs_edit.html').write_text(html)
    dest=Path('/home/carus/Downloads/astra_symmetric_ddim6_20s_20260928');dest.mkdir(exist_ok=True)
    for name in [path.name,'GUIDANCE_VS_EDIT.md','guidance_vs_edit.html',tag+'_curves.png']:shutil.copy2(P/name,dest/name)
    print(json.dumps(result,indent=2));print('\n'.join(table))

if __name__=='__main__':main()
