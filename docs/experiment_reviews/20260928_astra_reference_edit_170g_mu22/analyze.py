"""Summarize physics-reviewed rollouts and compose synchronized camera video."""
import json
from pathlib import Path
import subprocess
from functools import lru_cache
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
METHODS=['direct','edit010','edit020','edit035']


def load(direction,method):
    folder=P/'runs'/(direction+'_'+method)
    summary=json.loads((folder/'summary.json').read_text())
    summary.setdefault('physical_drop_confirmed',summary['physical_drop_review'] is not None)
    summary.setdefault('observation_censored_without_confirmed_drop',not summary['physical_drop_confirmed'])
    rows=[json.loads(x) for x in (folder/'trajectory.jsonl').read_text().splitlines()]
    return folder,summary,rows


def check(folder,summary,rows):
    assert len(rows)==summary['steps']
    assert summary['mass_kg']>0.16999 and summary['mass_kg']<0.17001
    assert abs(summary['object_friction']-2.2)<1e-5
    first=np.load(folder/'initial_state.npz')['object'][0]
    initial_jump=float(np.linalg.norm(np.asarray(rows[0]['state']['object_position'])-first[:3]))
    assert initial_jump<.01,initial_jump
    summary['first_control_step_object_translation_m']=initial_jump
    summary['first_terminal_hold_step']=(json.loads((folder/'reference_tail_transition.json').read_text())['first_terminal_hold_step']
        if (folder/'reference_tail_transition.json').exists() else None)
    native=next((r['step'] for r in rows if r['failure']),None)
    assert native==summary['first_native_failure_step']
    if summary['physical_drop_confirmed']:
        r=summary['physical_drop_review'];assert 1<=r['last_confirmed_held_step']<r['first_confirmed_separated_step']<=len(rows)
        assert (folder/'frames'/f"{r['first_confirmed_separated_step']:06d}.png").exists()
    else:
        assert len(rows)==10000 and not summary['physical_drop_review']
    return summary


@lru_cache(maxsize=16)
def read_frame(path):
    return cv2.imread(path)


def panel(folder,summary,row,step,direction,method):
    actual=min(step,len(row));r=row[actual-1]
    f=folder/'frames'/f'{actual:06d}.png';im=read_frame(str(f));assert im is not None,f
    im=cv2.resize(im,(640,480));out=np.zeros((520,640,3),np.uint8);out[40:]=im
    sign=1 if direction=='left' else -1
    held_cutoff=(summary['physical_drop_review']['last_confirmed_held_step']
        if summary['physical_drop_confirmed'] else len(row))
    shown_angle=sign*row[min(actual,held_cutoff)-1]['twist_degrees']
    if step>len(row):
        status='DROP - FROZEN' if summary['physical_drop_confirmed'] else 'OBSERVED - FROZEN'
    elif step==len(row):
        status='DROP' if summary['physical_drop_confirmed'] else 'OBSERVATION LIMIT'
    else:
        status='RUNNING'
    cv2.putText(out,f'{direction} {method} | {status} | t={actual/30:.2f}s | held turn={shown_angle:+.1f}deg',
        (8,25),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1,cv2.LINE_AA)
    return out


def video(direction,data):
    # 3 recorded physics frames -> one video frame at 10fps preserves real time.
    n=max(len(r) for _,_,r in data)
    path=P/(direction+'_comparison.mp4')
    w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(1280,1040))
    assert w.isOpened()
    selected=list(range(1,n+1,3))
    if selected[-1]!=n:selected.append(n)
    for step in selected:
        images=[panel(folder,summary,rows,step,direction,method)
                for method,(folder,summary,rows) in zip(METHODS,data)]
        w.write(cv2.vconcat([cv2.hconcat(images[:2]),cv2.hconcat(images[2:])]))
    w.release()
    c=cv2.VideoCapture(str(path));count=0;last=None
    while True:
        ok,frame=c.read()
        if not ok:break
        last=frame;count+=1
    assert count==len(selected)
    c.release();cv2.imwrite(str(P/(direction+'_comparison_final.jpg')),last)
    return dict(path=str(path),fps=10,decoded_frames=count,actual_control_steps=n,duration_seconds=count/10)


def main():
    summaries=[];videos={};fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    for di,direction in enumerate(('left','right')):
        data=[load(direction,m) for m in METHODS]
        for method,(folder,summary,rows) in zip(METHODS,data):
            summary=check(folder,summary,rows);summaries.append(summary)
            cutoff=(summary['physical_drop_review']['last_confirmed_held_step']
                if summary['physical_drop_confirmed'] else len(rows))
            sign=1 if direction=='left' else -1
            t=np.arange(1,cutoff+1)/30
            angles=[sign*r['twist_degrees'] for r in rows[:cutoff]]
            axes[di].plot(t,angles,label=method)
            axes[di].scatter([t[-1]],[angles[-1]],s=18)
        axes[di].set(xlabel='Actual control time (s)',ylabel='Net requested-direction turn (deg)',title=direction)
        axes[di].grid(alpha=.25);axes[di].legend()
        videos[direction]=video(direction,data)
    fig.suptitle('Astra fixed references at bulb 170g / object friction 2.2\nCurves end at last visually reviewed held frame or observation limit')
    fig.savefig(P/'held_turn_curves.png',dpi=150)
    (P/'analysis.json').write_text(json.dumps(dict(summaries=summaries,videos=videos),indent=2)+'\n')
    print(json.dumps(videos,indent=2))


if __name__=='__main__':main()
