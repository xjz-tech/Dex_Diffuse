"""Audit the added paired DDIM4/noise0.15 runs and extend the comparison table."""
import json,hashlib,subprocess,shutil
from pathlib import Path
import cv2
import numpy as np
from analyze import P,FFMPEG,load,panel

def interval(s):
    r=s.get('physical_review',s.get('physical_drop_review'))
    dt=s.get('control_dt',1/30)
    if r and r['action']=='confirmed_drop':
        return f"{r['last_confirmed_held_step']*dt:.2f}–{r['first_confirmed_separated_step']*dt:.2f}"
    return f"{s['steps']*dt:.2f}秒仍握住"

def main():
    reviews=json.loads((P/'physical_reviews.json').read_text())
    with np.load(P/'runs/left_direct/initial_state.npz') as z: baseline={k:z[k].copy() for k in z.files}
    data=[load(d,'edit015_ddim4',reviews,baseline) for d in ['left','right']]
    for _,s,_ in data:
        assert s['reference_editor']['requested_noise_ratio']==.15
        assert s['reference_editor']['timesteps']==[8,5,3,0]
    path=P/'noise015_ddim4_left_right_1x.mp4';dt=data[0][1]['control_dt']
    command=[str(FFMPEG),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x510','-r',str(1/dt),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    for step in range(1,601):
        frame=cv2.hconcat([panel(*item,step,d,'edit015_ddim4') for item,d in zip(data,['left','right'])])
        process.stdin.write(frame.tobytes())
    process.stdin.close();assert process.wait()==0
    c=cv2.VideoCapture(str(path));fps=c.get(cv2.CAP_PROP_FPS);n=0
    while c.read()[0]:n+=1
    c.release();assert n==600 and abs(n/fps-600*dt)<.002
    before=json.loads((P/'source_hashes.json').read_text())
    for d in ['left','right']:
        key=d+'_reference.npz';assert hashlib.sha256((P/key).read_bytes()).hexdigest()==before[key]
    result=dict(summaries=[s for _,s,_ in data],video=dict(file=str(path),decoded_frames=n,fps=fps,duration_s=n/fps,speed_ratio=600*dt/(n/fps)),
        unchanged_reference_hashes={k:before[k] for k in ['left_reference.npz','right_reference.npz']},
        extension_source_hashes={k:hashlib.sha256((P/k).read_bytes()).hexdigest() for k in ['run.py','analyze.py','analyze_noise015_ddim4.py']})
    (P/'noise015_ddim4_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    new=json.loads((P/'analysis.json').read_text())['summaries']+[s for _,s,_ in data]
    table=['| Reference | DDIM | 方法 / noise | 左转脱手区间（秒） | 左转最大角度 | 右转脱手区间（秒） | 右转最大角度 |','|---|---:|---|---:|---:|---:|---:|']
    for label,batch,methods in [('Astra',new,['direct','edit015_ddim4'])]:
        index={(s['direction'],s['method']):s for s in batch}
        for m in methods:
            ddim='—' if m=='direct' else (4 if m=='edit015_ddim4' else 6)
            name={'direct':'Direct','edit010':'Edit，noise 0.10','edit020':'Edit，noise 0.20','edit035':'Edit，noise 0.35','edit015_ddim4':'Edit，noise 0.15'}[m]
            table.append(f"|{label}|{ddim}|{name}|{interval(index['left',m])}|{index['left',m]['held_max_requested_deg']:.2f}°|{interval(index['right',m])}|{index['right',m]['held_max_requested_deg']:.2f}°|")
    text='# Astra 左右转对比\n\n170g、物体摩擦2.2；同一套 Astra 对称循环 reference。区间来自实际录像，原生failure独立记录。环境seed42、固定noise44、exec2、600实际步至20秒。\n\n'+'\n'.join(table)+'\n\nnoise0.15/DDIM4与其他Edit行同时改变噪声比例和DDIM步数，不能单独归因于一个参数。\n\n'
    for _,s,_ in data:
        text+=f"- 新{s['direction']} 0.15/DDIM4：原生首次failure step{s['first_native_failure_step']}；握持时最大指定方向净转角{s['held_max_requested_deg']:.2f}°，握持截止净转角{s['held_net_requested_deg']:.2f}°。\n"
    text+='\n实际噪声比0.153397，4个DDIM时间点[8,5,3,0]。新增两组27项初态与已有对照逐值相同，600个推理窗口reference逐值核验，历史mask误差0。视频600帧、约30fps、20.00004秒、1倍速，脱手后继续循环并记录。\n\n[新增左右同步录像](noise015_ddim4_left_right_1x.mp4)\n'
    if not (P/'COMPARISON_TABLE.md').exists():
        (P/'COMPARISON_TABLE.md').write_text(text)
    report=P/'RESULTS.md';content=report.read_text()
    if 'COMPARISON_TABLE.md' not in content:report.write_text(content+'\n## 新增noise0.15、DDIM4\n\n'+text.split('\n\n',1)[1]+'\n[完整合并表](COMPARISON_TABLE.md)\n')
    page=P/'index.html';html=page.read_text()
    if 'noise015_ddim4_left_right_1x.mp4' not in html:
        html=html.replace('</main>','<h2>新增：noise 0.15 / DDIM4（新对称循环）</h2><video controls preload="metadata" playsinline src="noise015_ddim4_left_right_1x.mp4"></video><p>左侧左转，右侧右转；完整20秒，1倍速。<a href="noise015_ddim4_left_right_1x.mp4" download>下载新增录像</a> · <a href="COMPARISON_TABLE.md">完整合并表</a></p></main>')
        page.write_text(html)
    dest=Path('/home/carus/Downloads/astra_symmetric_ddim6_20s_20260928');dest.mkdir(exist_ok=True)
    for name in ['noise015_ddim4_left_right_1x.mp4','COMPARISON_TABLE.md','RESULTS.md','index.html']:shutil.copy2(P/name,dest/name)
    print(json.dumps(result,indent=2));print('\n'.join(table))

if __name__=='__main__':main()
