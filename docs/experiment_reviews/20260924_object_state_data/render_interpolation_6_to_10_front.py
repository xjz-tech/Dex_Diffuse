"""Front-only comparison: direct original-action simulation and inserts 6–10."""
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

P = Path(__file__).resolve().parent
BASE = P/'corrected_direct_vs_reference'
OUT = BASE/'interpolation_6_to_10'


def title(panel, line1, line2, warning=False):
    header = np.full((48,480,3), (25,22,19), np.uint8)
    cv2.putText(header,line1,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.51,(255,255,255),1,cv2.LINE_AA)
    color=(120,160,255) if warning else (180,230,200)
    cv2.putText(header,line2,(8,39),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    return np.concatenate([header,panel],axis=0)


def main():
    for execute in [2,4]:
        runs=[]
        for inserted in [0,6,7,8,9,10]:
            folder = BASE/'direct' if inserted==0 else OUT/f'insert{inserted}_exec{execute}'
            video = folder/('direct_two_views.mp4' if inserted==0 else 'guided_front.mp4')
            cap=cv2.VideoCapture(str(video));assert cap.isOpened(),video
            trace=json.loads((folder/'trace.json').read_text())
            action_len=75+74*inserted
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= action_len+60
            runs.append(dict(inserted=inserted,cap=cap,trace=trace,action_len=action_len,last=-1,frame=None))
        writer=imageio.get_writer(str(OUT/f'guide9_exec{execute}_front_direct_original_insert6_10.mp4'),
                                  fps=30,codec='libx264',quality=8,macro_block_size=16)
        try:
            for shown in range(135):
                panels=[]
                for run in runs:
                    inserted=run['inserted'];action_len=run['action_len']
                    j=shown*(inserted+1) if shown<75 else action_len+shown-75
                    while run['last']<j:
                        ok,frame=run['cap'].read()
                        assert ok,(inserted,execute,j,run['last'])
                        run['last']+=1;run['frame']=frame
                    raw=run['frame']
                    front=raw[64:,640:] if inserted==0 else raw[64:]
                    assert front.shape==(480,640,3),front.shape
                    front=cv2.resize(front,(480,360),interpolation=cv2.INTER_AREA)
                    row=run['trace'][60+j]
                    line1='Original actions | direct sim' if inserted==0 else f'Insert {inserted} | guide9 exec{execute}'
                    line2=f'Sim {j/30:.2f}s | vertical {row["vertical_error_deg"]:.1f}deg | failure {int(row["native_failure"])}'
                    panels.append(title(front,line1,line2,row['native_failure']))
                frame=np.concatenate([np.concatenate(panels[:3],axis=1),
                                      np.concatenate(panels[3:],axis=1)],axis=0)
                writer.append_data(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                if shown in [74,134]:
                    cv2.imwrite(str(OUT/f'guide9_exec{execute}_{"action_end" if shown==74 else "plus_two_seconds"}.png'),frame)
        finally:
            writer.close()
            for run in runs:run['cap'].release()
        video=OUT/f'guide9_exec{execute}_front_direct_original_insert6_10.mp4'
        cap=cv2.VideoCapture(str(video));count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(816,1440,3),frame.shape
            count+=1
        cap.release();assert count==135,count
        (OUT/f'guide9_exec{execute}_video_verification.json').write_text(json.dumps({
            'frames':135,'fps':30,'duration_s':4.5,'size':[1440,816],
            'panels':['original recorded actions directly executed in simulation, no prior','insert6','insert7','insert8','insert9','insert10'],
            'alignment':'same original source action progress, with actual simulation time labeled per panel',
            'after_action':'60 physical hold steps for each simulation'},indent=2)+'\n')
        print(video)


if __name__=='__main__':main()
