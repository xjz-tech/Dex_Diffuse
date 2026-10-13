"""Combine live videos; short episodes also receive a labelled 0.25x copy."""
import sys,json,subprocess
from pathlib import Path
import cv2,numpy as np,imageio_ffmpeg
root=Path(sys.argv[1]).resolve()
for seed in [25,19,42,50]:
    folders=[root/f'seed{seed}_{a}' for a in ['ordinary_1b','guide10k']]
    if not all((p/'result.json').exists() for p in folders):continue
    dest=root/f'paired_seed{seed}';dest.mkdir(exist_ok=True)
    if (dest/'comparison.mp4').exists():continue
    cap=[cv2.VideoCapture(str(p/'live_raw.mp4')) for p in folders]
    counts=[int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in cap]
    slow=max(counts)<300;repeat=4 if slow else 1
    writer=cv2.VideoWriter(str(dest/'raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1600,720));assert writer.isOpened()
    frames=[None,None]
    try:
        for i in range(max(counts)):
            for j in range(2):
                if i<counts[j]:
                    ok,frames[j]=cap[j].read();assert ok
            panel=np.zeros((720,1600,3),np.uint8)
            panel[40:,:800]=frames[0];panel[40:,800:]=frames[1]
            title=f'seed {seed} | 1B (left) vs 1B + 10k guide (right) | '+('0.25x slow motion' if slow else '1x')
            cv2.putText(panel,title,(15,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            for j in range(2):
                if i>=counts[j]:cv2.putText(panel,'EPISODE ENDED - LAST FRAME',(j*800+16,710),cv2.FONT_HERSHEY_SIMPLEX,.65,(90,130,255),2)
            for _ in range(repeat):writer.write(panel)
            if i in [0,max(counts)//2,max(counts)-1]:cv2.imwrite(str(dest/f'frame_{i:05d}.png'),panel)
    finally:
        writer.release()
        for c in cap:c.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-i',str(dest/'raw.mp4'),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'comparison.mp4')],check=True)
    (dest/'metadata.json').write_text(json.dumps({'source_frames':counts,'playback_speed':.25 if slow else 1.,'output_frames':max(counts)*repeat,'source':'live images from original two runs; shorter side holds labelled terminal frame'},indent=2))
    print(dest/'comparison.mp4',flush=True)
