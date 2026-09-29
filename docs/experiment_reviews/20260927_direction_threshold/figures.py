import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent
rows=json.loads((P/'analysis.json').read_text())
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
colors={48:'#2766ba',49:'#bd533b',50:'#328061'}
for seed,col in colors.items():
    rs=sorted([r for r in rows if r['noise_seed']==seed],key=lambda r:r['lambda_direction'])
    x=np.array([r['lambda_direction'] for r in rs]);y=np.array([r['twist_after_initial8_deg'] for r in rs])
    full=np.array([not r['native_failure'] for r in rs])
    ax[0,0].plot(x[full],y[full],'o-',color=col,label=f'noise {seed}')
    if (~full).any():
        ax[0,0].scatter(x[~full],y[~full],marker='x',s=90,color=col,label='early failure (8.87 s)')
    ax[0,1].plot(x,[r['step9_twist_difference_deg'] for r in rs],'o-',color=col,label=f'noise {seed}')
    ax[1,0].plot(x,[1000*r['first_command_action1_delta_rms_rad'] for r in rs],'o-',color=col)
    ax[1,1].plot(x,[1000*r['first_command_delta_P_term_rms_Nm'] for r in rs],'o-',color=col)
for a in ax.flat:a.axvline(0,color='.6',lw=.7);a.grid(alpha=.2);a.set_xlabel('Reference contrast lambda (-1 right, +1 left)')
ax[0,0].axhspan(-5,5,color='.9');ax[0,0].axhline(0,color='.3',lw=.8)
ax[0,0].set_ylabel('Net twist after common 8 steps (deg)');ax[0,0].set_title('9.6 s closed loop: non-monotonic, noise-dependent');ax[0,0].legend()
ax[0,1].axhline(0,color='.3',lw=.8);ax[0,1].set_ylabel('First physical-step twist difference (deg)');ax[0,1].set_title('Same starting state: local response vs lambda=0')
ax[1,0].set_ylabel('First command difference RMS (mrad)');ax[1,0].set_title('Actual prior output, first action, 22 joints')
ax[1,1].set_ylabel('Kp * command difference RMS (mN m)');ax[1,1].set_title('Nominal proportional torque term; NOT bulb torque')
fig.suptitle('10B / DDIM4 / guide9 / scale25 / exec2; physical seed3577',fontsize=14)
fig.savefig(P/'direction_response.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained',sharey=True)
for a,seed in zip(axes,[48,49,50]):
    rs=sorted([r for r in rows if r['noise_seed']==seed and r['lambda_direction'] in [-1,-.5,0,.5,1]],key=lambda r:r['lambda_direction'])
    for r in rs:
        lines=[json.loads(s) for s in (Path(r['video']).parent/'trajectory.jsonl').read_text().splitlines()]
        y=np.array([v['twist_degrees'] for v in lines]);y-=y[7]
        line,=a.plot(np.arange(1,len(y)+1)/30,y,label=f"lambda {r['lambda_direction']:g}"+(' (early fail)' if r['native_failure'] else ''))
        if r['native_failure']:a.plot(len(y)/30,y[-1],'x',color=line.get_color(),ms=9)
    a.axhline(0,color='.4',lw=.7);a.set_title(f'Noise seed {seed}');a.set_xlabel('Physical time (s)');a.grid(alpha=.2);a.legend(fontsize=8)
axes[0].set_ylabel('Twist relative to step8 (deg)');fig.savefig(P/'rotation_traces.png',dpi=180);plt.close(fig)

font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
thumbs=[]
for r in rows:
    run=Path(r['video']).parent;frames=sorted((run/'frames').glob('*.png'))
    im=Image.open(frames[-1]).convert('RGB').resize((288,216))
    canvas=Image.new('RGB',(288,252),'#15212e');canvas.paste(im,(0,36));d=ImageDraw.Draw(canvas)
    d.text((5,3),f"n{r['noise_seed']} lambda {r['lambda_direction']:+g}  {r['twist_after_initial8_deg']:+.1f} deg",font=font,fill='white')
    d.text((5,19),f"tilt {r['max_axis_tilt_deg']:.1f} deg; native fail {r['native_failure']}",font=font,fill='white');thumbs.append(canvas)
grid=Image.new('RGB',(288*6,252*((len(thumbs)+5)//6)),(230,230,230))
for i,im in enumerate(thumbs):grid.paste(im,((i%6)*288,(i//6)*252))
grid.save(P/'final_frames.jpg')
