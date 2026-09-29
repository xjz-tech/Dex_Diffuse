"""Audit exact inputs, inspect terminal frames and compose real-time comparisons."""
import json
from pathlib import Path
import subprocess
import shutil
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
METHODS = ['direct', 'edit010', 'edit020', 'edit035']
FFMPEG = shutil.which('ffmpeg') or subprocess.check_output([
    '/home/carus/miniforge3/envs/dp/bin/python', '-c',
    'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'], text=True).strip()


def main():
    results = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for di, direction in enumerate(['left', 'right']):
        all_rows = []
        sheets = []
        for method in METHODS:
            p = HERE/'runs'/(direction+'_'+method)
            summary = json.loads((p/'summary.json').read_text())
            rows = [json.loads(line) for line in (p/'trajectory.jsonl').read_text().splitlines()]
            m = json.loads((p/'manifest.json').read_text())
            sign = 1 if direction == 'left' else -1
            angles = np.array([r['twist_degrees'] for r in rows])*sign
            rotations = Rotation.from_quat([r['state']['object_xyzw'] for r in rows])
            tilt = np.degrees(np.arccos(np.clip(rotations.apply([0, 1, 0])@np.array(m['axis_world']), -1, 1)))
            times = np.arange(1, len(rows)+1)*m['control_dt']
            axes[di, 0].plot(times, angles, label=method)
            axes[di, 1].plot(times, tilt, label=method)
            predictions = [json.loads(f.read_text()) for f in sorted(p.glob('prediction_*.json'))]
            assert all(d['reference_hold_padding_steps'] == 0 for d in predictions)
            summary.update(max_axis_tilt_deg=float(tilt.max()), no_reference_padding=True)
            if method != 'direct':
                summary['editor'] = json.loads((p/'model.json').read_text())['reference_editor']
                summary['mean_executed_edit_rmse_rad'] = float(np.mean([d['edit_stats']['executed_prefix_edit_rmse_rad'] for d in predictions]))
                summary['history_mask_max_error'] = max(d['edit_stats']['history_mask_max_error'] for d in predictions)
            results.append(summary)
            all_rows.append(rows)
            selected = [1, max(1,len(rows)-20), max(1,len(rows)-12), max(1,len(rows)-8), max(1,len(rows)-4), len(rows)]
            images=[]
            for step in selected:
                im=cv2.imread(str(p/'frames'/f'{step:06d}.png'))
                assert im is not None
                im=cv2.resize(im,(320,240))
                cv2.rectangle(im,(0,0),(320,30),(0,0,0),-1)
                cv2.putText(im,f'{method} step {step}',(5,21),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                images.append(im)
            sheets.append(cv2.hconcat(images))
        cv2.imwrite(str(HERE/(direction+'_terminal_review.jpg')),cv2.vconcat(sheets))
        for col, ylabel in enumerate(['Net requested-direction rotation (deg)', 'Tilt from initial bulb axis (deg)']):
            axes[di,col].set(title=direction, xlabel='Actual control time (s)', ylabel=ylabel, xlim=(0,10))
            axes[di,col].grid(alpha=.25);axes[di,col].legend()
        # Every panel is an actual camera frame, synchronized by executed step.
        n=max(map(len,all_rows))
        raw=HERE/(direction+'_comparison_raw.mp4')
        writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,1040))
        assert writer.isOpened()
        for step in range(1,n+1):
            panels=[]
            for method,rs in zip(METHODS,all_rows):
                current=min(step,len(rs));row=rs[current-1]
                im=cv2.imread(str(HERE/'runs'/(direction+'_'+method)/'frames'/f'{current:06d}.png'))
                im=cv2.resize(im,(640,480))
                panel=np.zeros((520,640,3),dtype=np.uint8);panel[40:]=im
                status=f'ENDED {len(rs)/30:.2f}s - FROZEN' if step>len(rs) else f't={step/30:.2f}s'
                a=row['twist_degrees']*(1 if direction=='left' else -1)
                cv2.putText(panel,f'{direction} {method} | {status} | requested {a:+.1f}deg',(8,27),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                panels.append(panel)
            writer.write(cv2.vconcat([cv2.hconcat(panels[:2]),cv2.hconcat(panels[2:])]))
        writer.release()
        subprocess.run([FFMPEG,'-nostdin','-v','error','-y','-i',str(raw),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(HERE/(direction+'_comparison.mp4'))],check=True)
        raw.unlink()
    fig.suptitle('Fixed Astra references + reference-initialized DDIM | physics 42 / noise 44\nNative failure terminates a trace; terminal rotation can include slipping/falling')
    fig.savefig(HERE/'curves.png',dpi=160)
    (HERE/'analysis.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
