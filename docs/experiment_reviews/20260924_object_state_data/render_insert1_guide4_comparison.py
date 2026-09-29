"""Front-view comparison of insert1 guide4 scales and execution lengths."""
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
FONT=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',21)
SMALL=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execution',type=int,choices=[2,4],default=2)
    args=parser.parse_args()
    if args.execution==2:
        variants=[
            ('原始动作直接仿真',BASE/'direct','direct',0,False),
            ('guide2 / exec2 · scale50',OUT/'insert1_guide2_exec2_scale50','guided',1,True),
            ('guide4 / exec2 · scale25',OUT/'insert1_guide4_exec2_scale25','guided',1,True),
            ('guide4 / exec2 · scale50',OUT/'insert1_guide4_exec2_scale50','guided',1,True),
            ('guide9 / exec2 · scale50',OUT/'insert1_guide9_exec2_scale50','guided',1,True),
        ]
    else:
        variants=[
            ('原始动作直接仿真',BASE/'direct','direct',0,False),
            ('guide4 / exec2 · scale25',OUT/'insert1_guide4_exec2_scale25','guided',1,True),
            ('guide4 / exec2 · scale50',OUT/'insert1_guide4_exec2_scale50','guided',1,True),
            ('guide4 / exec4 · scale25',OUT/'insert1_guide4_exec4_scale25','guided',1,True),
            ('guide4 / exec4 · scale50',OUT/'insert1_guide4_exec4_scale50','guided',1,True),
        ]
    runs=[]
    for label,folder,stem,inserted,front_only in variants:
        file=folder/(f'{stem}_front.mp4' if front_only else f'{stem}_two_views.mp4')
        cap=cv2.VideoCapture(str(file));assert cap.isOpened(),file
        runs.append(dict(label=label,cap=cap,
            trace=json.loads((folder/'trace.json').read_text()),
            inserted=inserted,front_only=front_only,last=-1,frame=None))
    path=OUT/f'insert1_guide4_exec{args.execution}_front_comparison.mp4'
    writer=imageio.get_writer(str(path),fps=30,codec='libx264',quality=8,macro_block_size=16)
    try:
        for shown in range(135):
            panels=[]
            for run in runs:
                length=75+74*run['inserted']
                j=shown*(run['inserted']+1) if shown<75 else length+shown-75
                while run['last']<j:
                    ok,frame=run['cap'].read();assert ok,(run['label'],j,run['last'])
                    run['last']+=1;run['frame']=frame
                raw=run['frame'];front=raw[64:] if run['front_only'] else raw[64:,640:]
                assert front.shape==(480,640,3)
                panel=Image.new('RGB',(640,544),(26,35,47))
                panel.paste(Image.fromarray(cv2.cvtColor(front,cv2.COLOR_BGR2RGB)),(0,64))
                row=run['trace'][60+j];d=ImageDraw.Draw(panel)
                d.text((10,1),run['label'],font=FONT,fill='white')
                d.text((10,34),f'仿真 {j/30:.2f}s | 距竖直 {row["vertical_error_deg"]:.1f}° | failure {row["native_failure"]}',
                    font=SMALL,fill=(255,180,165) if row['native_failure'] else (185,232,210))
                panels.append(np.asarray(panel))
            note=Image.new('RGB',(640,544),(30,35,43));d=ImageDraw.Draw(note)
            for k,line in enumerate(['插入1个目标，原始动作均保留','按原始 reference 进度对齐','每格标注实际仿真时间',
                                      '动作结束后各保持末目标2秒','灯泡170g，摩擦2.2','10B EMA · DDIM4 · 噪声44']):
                d.text((25,45+70*k),line,font=FONT,fill=(220,230,238))
            panels.append(np.asarray(note))
            frame=np.concatenate([np.concatenate(panels[:3],axis=1),np.concatenate(panels[3:],axis=1)],axis=0)
            writer.append_data(frame)
            if shown in (74,134):
                Image.fromarray(frame).save(OUT/(f'insert1_guide4_exec{args.execution}_action_end.png' if shown==74 else f'insert1_guide4_exec{args.execution}_hold_end.png'))
    finally:
        writer.close()
        for run in runs:run['cap'].release()
    cap=cv2.VideoCapture(str(path));n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(1088,1920,3)
        n+=1
    cap.release();assert n==135
    (OUT/f'insert1_guide4_exec{args.execution}_video_verification.json').write_text(json.dumps(dict(
        frames=135,fps=30,duration_s=4.5,panels=[x[0] for x in variants],
        alignment='original reference progress, with per-panel elapsed simulation time'),indent=2)+'\n')
    print(path)


if __name__=='__main__':main()
