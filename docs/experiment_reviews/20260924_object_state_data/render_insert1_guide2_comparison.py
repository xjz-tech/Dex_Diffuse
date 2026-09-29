"""Front-view comparison of original direct actions and insert1 guidance variants."""
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
FONT=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
SMALL=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exec1',action='store_true')
    args=parser.parse_args()
    if args.exec1:
        specs=[
            ('原始动作直接仿真（无 prior）',BASE/'direct','direct',0,False),
            ('guide2 / exec2 · scale25',OUT/'insert1_guide2_exec2','guided',1,True),
            ('guide2 / exec2 · scale50',OUT/'insert1_guide2_exec2_scale50','guided',1,True),
            ('guide2 / exec1 · scale50',OUT/'insert1_guide2_exec1_scale50','guided',1,True),
        ]
        output_stem='insert1_guide2_exec1_vs_exec2'
    else:
        specs=[
            ('原始动作直接仿真（无 prior）',BASE/'direct','direct',0,False),
            ('插入1个 · guide2 / exec2',OUT/'insert1_guide2_exec2','guided',1,True),
            ('插入1个 · guide9 / exec2',OUT/'insert1_exec2','guided',1,False),
            ('插入1个 · guide9 / exec4',OUT/'insert1_exec4','guided',1,False),
        ]
        output_stem='insert1_direct_guide2_guide9'
    runs=[]
    for label,folder,video_stem,inserted,front_only in specs:
        file=folder/(f'{video_stem}_front.mp4' if front_only else f'{video_stem}_two_views.mp4')
        cap=cv2.VideoCapture(str(file));assert cap.isOpened(),file
        trace=json.loads((folder/'trace.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        runs.append(dict(label=label,cap=cap,trace=trace,summary=summary,
                         inserted=inserted,front_only=front_only,last=-1,frame=None))
    video=OUT/f'{output_stem}_front_progress.mp4'
    writer=imageio.get_writer(str(video),fps=30,codec='libx264',quality=8,macro_block_size=16)
    try:
        for shown in range(135):
            canvas=Image.new('RGB',(1280,1088),(15,20,26));draw=ImageDraw.Draw(canvas)
            for i,run in enumerate(runs):
                length=75+74*run['inserted']
                j=shown*(run['inserted']+1) if shown<75 else length+shown-75
                while run['last']<j:
                    ok,frame=run['cap'].read();assert ok,(i,j,run['last'])
                    run['last']+=1;run['frame']=frame
                raw=run['frame']
                front=raw[64:] if run['front_only'] else raw[64:,640:]
                assert front.shape==(480,640,3)
                x=(i%2)*640;y=(i//2)*544
                canvas.paste(Image.fromarray(cv2.cvtColor(front,cv2.COLOR_BGR2RGB)),(x,y+64))
                row=run['trace'][60+j]
                draw.rectangle((x,y,x+639,y+63),fill=(26,35,47))
                draw.text((x+8,y+1),run['label'],font=FONT,fill='white')
                phase='动作' if shown<75 else '末态保持'
                message=f'{phase} | 仿真 {j/30:.2f}s | 距竖直 {row["vertical_error_deg"]:.1f}° | failure {row["native_failure"]}'
                draw.text((x+8,y+35),message,font=SMALL,fill=(255,175,145) if row['native_failure'] else (187,231,211))
            frame=np.asarray(canvas)
            writer.append_data(frame)
            if shown in (74,134):
                still_stem=output_stem if args.exec1 else 'insert1_guide2_comparison'
                canvas.save(OUT/(f'{still_stem}_action_end.png' if shown==74 else f'{still_stem}_hold_end.png'))
    finally:
        writer.close()
        for run in runs:run['cap'].release()
    cap=cv2.VideoCapture(str(video));n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(1088,1280,3)
        n+=1
    cap.release();assert n==135
    verification_stem=output_stem if args.exec1 else 'insert1_guide2'
    (OUT/f'{verification_stem}_video_verification.json').write_text(json.dumps(dict(
        frames=n,fps=30,duration_s=4.5,
        panels=[s[0] for s in specs],
        alignment='original action progress; each panel shows its own elapsed simulation time',
        hold='60 real physics steps after each simulation finishes its action phase'),indent=2)+'\n')
    print(video)


if __name__=='__main__':main()
