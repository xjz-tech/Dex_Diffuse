"""Compose four native recordings in step-aligned panels; no synthesized frames."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
P=Path(__file__).resolve().parent
panels=[('native_n48_lambdam0p50000','noise48  lambda -0.500  final +12.92 deg'),
        ('native_n48_lambdam0p62500','noise48  lambda -0.625  final -11.91 deg'),
        ('native_n49_lambdap0p12500','noise49  lambda +0.125  final +8.64 deg'),
        ('native_n49_lambdam0p12500','noise49  lambda -0.125  final -5.15 deg')]
frames=[sorted((P/'runs'/n/'frames').glob('*.png')) for n,_ in panels]
assert len(set(map(len,frames)))==1
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
writer=imageio_ffmpeg.write_frames(str(P/'paired_direction_changes.mp4'),(960,784),fps=30,codec='libx264',quality=8,macro_block_size=16)
writer.send(None)
for k in range(len(frames[0])):
    canvas=Image.new('RGB',(960,784),'#15212e');d=ImageDraw.Draw(canvas)
    for i,(_,label) in enumerate(panels):
        x=(i%2)*480;y=(i//2)*392
        im=Image.open(frames[i][k]).convert('RGB').resize((480,360))
        canvas.paste(im,(x,y+32));d.text((x+6,y+6),label,fill='white',font=font)
    writer.send(canvas.tobytes())
writer.close()
print('frames',len(frames[0]))
