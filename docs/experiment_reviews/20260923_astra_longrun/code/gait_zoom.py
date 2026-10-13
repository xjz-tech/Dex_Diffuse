"""Actual camera closeup with logged phase; no synthetic poses or interpolated frames."""
from pathlib import Path
import json,subprocess,sys
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
r=Path(sys.argv[1]);summary=json.loads((r/'summary.json').read_text());events=json.loads((r/'gait_events.json').read_text());out=r/'gait_closeup.mp4'
f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
p=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1440x800','-r','10','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','20','-threads','2','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE)
steps=list(range(1,summary['steps']+1,3))
if steps[-1]!=summary['steps']:steps.append(summary['steps'])
for step in steps:
 im=Image.open(r/'frames'/f'{step:06d}.png').convert('RGB');c=Image.new('RGB',(1440,800),'#101827');d=ImageDraw.Draw(c)
 c.paste(im,(0,50));c.paste(im.crop((300,150,640,490)).resize((480,480)),(960,50));e=next((e for e in reversed(events) if e['step']<step),events[0])
 d.text((16,10),f"{r.name} | ACTUAL recording | t={step/30:.2f}s",font=f,fill='white')
 notes=[f"phase: {e['phase']}",f"selected finger: {e['finger'] or '-'}",'Other finger references held in gait','Thumb reference retained in gait','Nominal unload: <=1mm (linearized)','Observed motion may differ.','1x simulation playback; 10fps samples.']
 for j,line in enumerate(notes):d.text((973,548+j*30),line,font=small,fill='#cbd5e1')
 p.stdin.write(c.tobytes())
 if step==40:c.save(r/'gait_closeup_preview.jpg')
p.stdin.close();assert p.wait()==0;print(out)
