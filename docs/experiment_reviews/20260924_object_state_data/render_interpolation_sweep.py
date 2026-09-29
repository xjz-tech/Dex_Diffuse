"""Render 0–5 inserted-reference variants aligned by original action progress."""
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

P = Path(__file__).resolve().parent
BASE = P/'corrected_direct_vs_reference'
OUT = BASE/'interpolation_1_to_5'


def main():
    for execute in [2, 4]:
        inputs = []
        for inserted in range(6):
            folder = (BASE/'guided9' if execute==2 else BASE/'guided9_exec4') if inserted==0 else OUT/f'insert{inserted}_exec{execute}'
            cap = cv2.VideoCapture(str(folder/'guided_two_views.mp4'))
            assert cap.isOpened(), folder
            trace = json.loads((folder/'trace.json').read_text())
            summary = json.loads((folder/'summary.json').read_text())
            assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= summary['steps']['action']+60
            inputs.append((inserted, cap, trace, summary))
        file = OUT/f'guide9_exec{execute}_progress_aligned.mp4'
        writer = imageio.get_writer(str(file), fps=30, codec='libx264', quality=7)
        last_read = [-1]*6
        last_frame = [None]*6
        try:
            for shown in range(135):
                panels = []
                for i,(inserted,cap,trace,summary) in enumerate(inputs):
                    action_len = summary['steps']['action']
                    j = shown*(inserted+1) if shown<75 else action_len+shown-75
                    while last_read[i]<j:
                        ok,frame = cap.read()
                        assert ok, (execute,inserted,j,last_read[i])
                        last_frame[i] = frame
                        last_read[i] += 1
                    row = trace[60+j]
                    panel = cv2.resize(last_frame[i], (640,272), interpolation=cv2.INTER_AREA)
                    header = np.zeros((49,640,3),dtype=np.uint8)
                    phase = 'action' if shown<75 else 'hold'
                    sim_time = (j if shown<75 else action_len+shown-75)/30
                    cv2.putText(header,f'insert {inserted} | guide9 exec{execute} | {phase} | sim {sim_time:.2f}s',
                        (8,20),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
                    color = (100,150,255) if row['native_failure'] else (185,240,205)
                    cv2.putText(header,f'original progress {min(shown+1,75)}/75 | vertical {row["vertical_error_deg"]:.1f} deg | failure {row["native_failure"]}',
                        (8,41),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
                    panels.append(np.concatenate([header,panel],axis=0))
                composite = np.concatenate([np.concatenate(panels[:3],axis=1),
                                            np.concatenate(panels[3:],axis=1)],axis=0)
                writer.append_data(cv2.cvtColor(composite,cv2.COLOR_BGR2RGB))
        finally:
            writer.close()
            for _,cap,_,_ in inputs: cap.release()
        cap = cv2.VideoCapture(str(file)); count=0
        while True:
            ok,_=cap.read()
            if not ok: break
            count+=1
        cap.release(); assert count==135, (file,count)
        (OUT/f'guide9_exec{execute}_video.json').write_text(json.dumps({
            'frames':count,'fps':30,'duration_s':4.5,
            'alignment':'original action progress; each panel shows its actual simulation time',
            'hold':'60 real simulation steps after each variant finishes its own action trajectory'},indent=2)+'\n')
        print(file)


if __name__ == '__main__':
    main()
