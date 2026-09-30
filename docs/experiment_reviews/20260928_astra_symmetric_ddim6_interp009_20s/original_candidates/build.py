from pathlib import Path
import hashlib,json,subprocess,shutil
import imageio_ffmpeg
here=Path(__file__).resolve().parent
root=here.parents[3]
old=root/'.worktrees/Astra-controller/outputs'
items=[
 ('A','左转：约14秒／549°','astra_left_sequence/v2_s5_pause100/rollout.mp4','DDIM4 · 转动 guidance 5／暂停 100 · 环境与噪声 seed 42。无显式换指循环。'),
 ('B','左转：转10秒停6秒／累计1313°','astra_left10_pause6/joint_mse_p100_e42_n45/left10s_pause6s.mp4','DDIM4 · 转动 guidance 5／暂停 100 · 环境 seed 42／噪声 45。无显式换指循环。'),
 ('C','早期右转：scale25','astra_halfturn/seed42_10b_right_plan16_s25_v1/rollout.mp4','DDIM4 · guidance 25 · seed 42。没有后来完整换指循环，但有拇指卸载与手指收拢恢复动作；接触阶段最远约52°，未完成半圈。'),
 ('D','无显式换指：左10秒→停9秒→右12秒','astra_gait_pause9/no_gait_e3577_n48/rollout.mp4','DDIM4 · guidance 25 · 环境 seed 3577／噪声 48。前段左转约95°、右转约65°，不是上面两条快速左转。')]
manifest=[];cards=[]
for key,title,relative,desc in items:
 src=old/relative;dest=here/(key+'.mp4')
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-y','-loglevel','error','-i',str(src),'-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
 manifest.append(dict(id=key,title=title,source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),description=desc))
 cards.append(f'<article><h2>{key} · {title}</h2><video controls playsinline preload="metadata" src="{key}.mp4"></video><p>{desc}</p></article>')
(here/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
(here/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Astra 原始录像核对</title><style>body{background:#141922;color:#eef2f9;font:17px system-ui;margin:28px}h1{font-size:27px}main{display:grid;grid-template-columns:1fr 1fr;gap:22px}article{background:#222a37;border-radius:12px;padding:18px}h2{font-size:20px;margin:0 0 14px}video{width:100%;max-height:420px;background:#000}p{line-height:1.6;color:#cbd6e6}</style><h1>Astra 原始录像核对</h1><p>均为旧实验原始录像，仅转为浏览器兼容格式；1×播放，无新动作、无补帧。A/B 是两段快速左转候选；C/D 用于核对你记忆中的右转。</p><main>'''+''.join(cards)+'''</main><script>document.querySelectorAll('video').forEach(v=>v.addEventListener('play',()=>{v.playbackRate=1;document.querySelectorAll('video').forEach(w=>{if(w!==v)w.pause()})}));</script>''')
print(here/'index.html')
