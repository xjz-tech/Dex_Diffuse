"""Front-view progress-aligned comparison of scale25 and scale50."""
import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

P=Path(__file__).resolve().parent
BASE=P/'corrected_direct_vs_reference'
OUT=BASE/'interpolation_1_to_5'
FONT=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--interpolation',type=int,choices=[0,1],default=1)
    args=parser.parse_args()
    if args.interpolation==1:
        out=OUT
        variants=[
            ('原始动作直接仿真',BASE/'direct','direct',0,False),
            ('guide2 / exec2 · scale25',OUT/'insert1_guide2_exec2','guided',1,True),
            ('guide2 / exec2 · scale50',OUT/'insert1_guide2_exec2_scale50','guided',1,True),
            ('guide9 / exec2 · scale25',OUT/'insert1_exec2','guided',1,False),
            ('guide9 / exec2 · scale50',OUT/'insert1_guide9_exec2_scale50','guided',1,True),
            ('guide9 / exec4 · scale25',OUT/'insert1_exec4','guided',1,False),
            ('guide9 / exec4 · scale50',OUT/'insert1_guide9_exec4_scale50','guided',1,True),
        ]
        output_stem='insert1_scale25_vs50'
    else:
        out=BASE/'nointerp_scale50'
        variants=[
            ('原始动作直接仿真',BASE/'direct','direct',0,False),
            ('guide2 / exec2 · scale25',BASE/'guided','guided',0,False),
            ('guide2 / exec2 · scale50',out/'guide2_exec2','guided',0,True),
            ('guide9 / exec2 · scale25',BASE/'guided9','guided',0,False),
            ('guide9 / exec2 · scale50',out/'guide9_exec2','guided',0,True),
            ('guide9 / exec4 · scale25',BASE/'guided9_exec4','guided',0,False),
            ('guide9 / exec4 · scale50',out/'guide9_exec4','guided',0,True),
        ]
        output_stem='nointerp_scale25_vs50'
    runs=[]
    for label,folder,video_stem,inserted,front_only in variants:
        file=folder/(f'{video_stem}_front.mp4' if front_only else f'{video_stem}_two_views.mp4')
        cap=cv2.VideoCapture(str(file));assert cap.isOpened(),file
        trace=json.loads((folder/'trace.json').read_text())
        runs.append(dict(label=label,cap=cap,trace=trace,inserted=inserted,
                         front_only=front_only,last=-1,frame=None))
    file=out/f'{output_stem}_front_progress.mp4'
    writer=imageio.get_writer(str(file),fps=30,codec='libx264',quality=8,macro_block_size=16)
    try:
        for shown in range(135):
            panels=[]
            for run in runs:
                length=75+74*run['inserted']
                j=shown*(run['inserted']+1) if shown<75 else length+shown-75
                while run['last']<j:
                    ok,frame=run['cap'].read();assert ok,(run['label'],j,run['last'])
                    run['frame']=frame;run['last']+=1
                raw=run['frame']
                front=raw[64:] if run['front_only'] else raw[64:,640:]
                assert front.shape==(480,640,3)
                front=cv2.resize(front,(480,360),interpolation=cv2.INTER_AREA)
                panel=Image.new('RGB',(480,408),(22,30,40));panel.paste(Image.fromarray(cv2.cvtColor(front,cv2.COLOR_BGR2RGB)),(0,48))
                row=run['trace'][60+j];d=ImageDraw.Draw(panel)
                d.text((8,0),run['label'],font=FONT,fill='white')
                d.text((8,24),f'仿真 {j/30:.2f}s  |  距竖直 {row["vertical_error_deg"]:.1f}°  |  failure {row["native_failure"]}',
                       font=FONT,fill=(255,180,165) if row['native_failure'] else (185,232,210))
                panels.append(np.asarray(panel))
            note=Image.new('RGB',(480,408),(30,35,43));d=ImageDraw.Draw(note)
            for i,s in enumerate(['均为仿真正面视角','按原始 reference 进度对齐','各组实际仿真时间见每格标题',
                                  '动作结束后仿真保持末目标2秒','灯泡170g、摩擦2.2','10B EMA · DDIM4 · 噪声44']):
                d.text((15,22+57*i),s,font=FONT,fill=(220,230,238))
            panels.append(np.asarray(note))
            frame=np.concatenate([np.concatenate(panels[:4],axis=1),np.concatenate(panels[4:],axis=1)],axis=0)
            writer.append_data(frame)
            if shown in (74,134):
                Image.fromarray(frame).save(out/f'{output_stem}_{"action_end" if shown==74 else "hold_end"}.png')
    finally:
        writer.close()
        for run in runs:run['cap'].release()
    cap=cv2.VideoCapture(str(file));n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(816,1920,3)
        n+=1
    cap.release();assert n==135
    (out/f'{output_stem}_video_verification.json').write_text(json.dumps(dict(
        frames=n,fps=30,duration_s=4.5,panels=[x[0] for x in variants],
        alignment='original reference action progress; per-panel simulation time is labeled'),indent=2)+'\n')
    print(file)


if __name__=='__main__':main()
