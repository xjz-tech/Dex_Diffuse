"""Encode only compared camera frames; old longer source recordings stay intact."""
import json,subprocess,sys
from pathlib import Path
import imageio_ffmpeg
import cv2
h=Path(__file__).resolve().parents[1]
if len(sys.argv)>1:h=Path(sys.argv[1])
methods=json.loads((h/'protocol.json').read_text())['methods']
for p in (h/'runs').glob('*/summary.json'):
 r=json.loads(p.read_text())
 if r['phase']!='formal' or r['method'] not in methods:continue
 assert r['steps']<=900
 out=p.parent/'comparison_clip.mp4'
 if out.exists():
  c=cv2.VideoCapture(str(out));n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));fps=c.get(cv2.CAP_PROP_FPS);c.set(cv2.CAP_PROP_POS_FRAMES,max(0,n-1));ok,_=c.read();c.release()
  if n==r['steps'] and fps==30 and ok:continue
 tmp=out.with_name('comparison_clip.partial.mp4')
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-framerate','30','-start_number','1','-i',str(p.parent/'frames/%06d.png'),'-frames:v',str(r['steps']),'-an','-c:v','libx264','-preset','veryfast','-crf','20','-threads','2','-pix_fmt','yuv420p',str(tmp)],check=True)
 tmp.replace(out)
 print(r['label'],r['steps'],flush=True)
