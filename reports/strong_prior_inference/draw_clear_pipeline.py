from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
fig, ax = plt.subplots(figsize=(15, 10))
fig.patch.set_facecolor('white')
ax.set(xlim=(0,15), ylim=(0,10))
ax.axis('off')
blue='#34678C'; green='#35715B'; amber='#A7762B'; purple='#75618C'; ink='#263746'

def box(x,y,w,h,title,body,color,fill):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.025,rounding_size=0.09',
                               linewidth=1.3,edgecolor=color,facecolor=fill,zorder=3))
    ax.text(x,y+h*.20,title,ha='center',va='center',weight='bold',fontsize=12,color=ink,zorder=4)
    ax.text(x,y-h*.17,body,ha='center',va='center',fontsize=10.5,color=ink,linespacing=1.5,zorder=4)

def line(points,color=ink,dashed=False):
    xs,ys=zip(*points)
    ax.plot(xs[:-1],ys[:-1],color=color,lw=1.4,ls=(0,(4,3)) if dashed else '-',zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=13,
                                linewidth=1.4,color=color,linestyle=(0,(4,3)) if dashed else '-',zorder=2))

def label(x,y,text,color=ink):
    ax.text(x,y,text,ha='center',va='center',fontsize=10,color=color,
            bbox=dict(facecolor='white',edgecolor='none',pad=2),zorder=5)

ax.text(7.5,9.65,'Dual-Guided Strong-Prior Inference',ha='center',fontsize=20,weight='bold',color=ink)
ax.text(1.15,9.15,'1   VISUAL TASK PLANNING',fontsize=11,weight='bold',color=blue)
box(2.8,8.35,3.2,1.05,'Visual observations','Front + wrist images\nEnd-effector and hand states',blue,'#EDF4FA')
box(7.25,8.35,3.4,1.05,'Visual Diffusion Policy','16 denoising iterations',blue,'#DFECF6')
box(11.65,8.35,3.2,1.05,'Cached DP plan','16 actions\nArm / EEF + hand',blue,'#EDF4FA')
line([(4.425,8.35),(5.525,8.35)],blue)
line([(8.975,8.35),(10.025,8.35)],blue)

ax.text(1.15,7.3,'2   HAND REPLANNING  ·  REPEAT FOUR TIMES PER DP PLAN',fontsize=11,weight='bold',color=green)
box(2.8,6.4,3.2,1.05,'Latest hand history','4 frames × 66 dimensions',green,'#EDF5F0')
box(7.25,6.4,3.4,1.05,'Weak hand policy','4 DDIM iterations\nKeep 2 future actions',amber,'#FCF1E1')
box(11.65,6.4,3.2,1.05,'DP hand reference','Sliding 9-action window',blue,'#EDF4FA')
line([(4.425,6.4),(5.525,6.4)],green)
line([(11.65,7.80),(11.65,6.95)],blue)

box(7.25,4.4,5.1,1.25,'Strong hand prior + dual guidance',
    '8 DDIM iterations from fixed noise\nBoth references guide every denoising iteration',green,'#DEEEE4')
line([(7.25,5.85),(7.25,5.05)],amber)
label(7.25,5.5,'Weak: 2 actions  ·  weight 25',amber)
line([(11.65,5.85),(11.65,4.65),(9.825,4.65)],blue)
label(11.65,5.45,'DP: 9 actions\nweight 100',blue)
line([(2.8,5.85),(2.8,4.4),(4.675,4.4)],green)
label(3.6,4.68,'Condition',green)

ax.text(1.15,3.4,'3   ACTION EXECUTION',fontsize=11,weight='bold',color=purple)
box(7.25,2.5,5.1,1.0,'Assemble the next 2 actions',
    'DP arm / EEF  +  strong-prior hand',purple,'#F0EBF6')
line([(7.25,3.75),(7.25,3.025)],green)
label(7.25,3.35,'First 2 hand actions',green)
line([(13.275,8.35),(14.1,8.35),(14.1,2.5),(9.825,2.5)],blue)
ax.text(14.30,5.3,'Corresponding DP arm / EEF actions',rotation=90,ha='center',va='center',fontsize=10,color=blue)
box(7.25,.95,5.1,1.0,'Franka + SharpA','Execute 2 actions  ·  target 30 Hz',purple,'#E7DEF0')
line([(7.25,1.975),(7.25,1.475)],purple)
line([(4.675,.95),(.85,.95),(.85,6.4),(1.175,6.4)],green,True)
ax.text(1.04,2.1,'Update hand history after each action',rotation=90,ha='center',va='bottom',fontsize=9.5,color=green)
line([(7.25,.425),(7.25,.17),(.25,.17),(.25,8.35),(1.175,8.35)],blue,True)
ax.text(.08,4.4,'Replan visual DP after 4 calls / 8 executed actions',rotation=90,ha='center',va='center',fontsize=9.5,color=blue)

fig.subplots_adjust(left=.03,right=.98,top=.98,bottom=.02)
for suffix in ('pdf','svg','png'):
    fig.savefig(OUT/f'pipeline_clear.{suffix}',dpi=200,facecolor='white')
plt.close(fig)
