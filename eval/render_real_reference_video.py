"""Combine actual camera recordings, with pose-log annotations (not rerenders)."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation as R
from analyze_real_reference_sim import rotation_metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    run=args.run;t=np.load(run/'trajectory.npz',allow_pickle=False)
    meta=json.loads((run/'run_metadata.json').read_text())
    ref_indices=np.flatnonzero(t['phase']=='reference');baseline=ref_indices[0]-1
    # The recorder also captured four screenshot frames into its initial video.
    mapping=[]
    for i in range(len(t['phase'])):
        mapping.append(i)
        if i==0 or i==baseline or i==ref_indices[-1] or i==len(t['phase'])-1:mapping.append(i)
    caps=[cv2.VideoCapture(str(run/('env%02d.mp4'%i))) for i in range(3)]
    for cap in caps:assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(mapping)
    spins=[]
    for i in range(3):
        spin,_,_=rotation_metrics(t['relative_quaternion'][baseline:,i]);spins.append(spin)
    initial=np.load(run/'initial_state.npz',allow_pickle=False)
    camera_targets=initial['object'][:,:3].astype(float)
    offset=np.array([.22,.36,.19]);forward=-offset/np.linalg.norm(offset)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
    focal=640/(2*np.tan(np.deg2rad(55/2)))
    def project(points,env):
        vec=points-camera_targets[env]-offset
        z=vec@forward
        return np.stack([320+focal*(vec@right)/z,240-focal*(vec@up)/z],axis=1).round().astype(int)
    movie=run/'comparison.mp4'
    with imageio.get_writer(str(movie),fps=30,codec='libx264',quality=8) as writer:
        for video_index,k in enumerate(mapping):
            phase=str(t['phase'][k]);j=int(t['index'][k])
            source_frame=325 if phase=='settle' else (min(400,326+j) if phase=='reference' else 400)
            real=cv2.imread('/home/carus/Data/bulb_tac_260909/episode_37/wrist/%06d.png'%source_frame)
            real=cv2.resize(real,(640,480))
            images=[real]
            headings=['REAL wrist camera | source frame %d'%source_frame]
            captions=['%s | socket contact in the real recording'%phase]
            for env,cap in enumerate(caps):
                ok,bgr=cap.read();assert ok
                pos=t['object_pose'][k,env,:3];rot=R.from_quat(t['object_pose'][k,env,3:])
                # Add reference axes projected from recorded object poses.
                pts=project(pos+rot.apply([[0,0,0],[.035,0,0],[0,.07,0],[0,0,.035]]),env)
                for q,color in zip(pts[1:],[(40,40,255),(40,230,40),(255,120,40)]):
                    if np.max(np.abs(pts[0]))<5000 and np.max(np.abs(q))<5000:
                        cv2.line(bgr,tuple(pts[0]),tuple(q),color,2,cv2.LINE_AA)
                angle=spins[env][k-baseline] if k>=baseline else None
                contacts=int((t['contact_force_norm'][k,env]>.05).sum())
                failed=bool(t['failure'][:k+1,env].any())
                noise='noise seed %d'%(42+env) if meta['mode']=='guided' else 'direct targets'
                headings.append('SIM env %d | %s | %s'%(env,noise,phase))
                spin_label='post-failure spin' if failed else 'right axial'
                captions.append(('settling' if angle is None else spin_label+': %+.1f deg'%angle)+' | tips: %d | failure: %s'%(contacts,failed))
                images.append(bgr)
            canvas=np.zeros((1088,1280,3),dtype=np.uint8)
            for i,im in enumerate(images):
                x,y=(i%2)*640,(i//2)*544
                canvas[y+64:y+544,x:x+640]=im
                cv2.putText(canvas,headings[i],(x+10,y+23),cv2.FONT_HERSHEY_SIMPLEX,.56,(245,245,245),1,cv2.LINE_AA)
                cv2.putText(canvas,captions[i],(x+10,y+49),cv2.FONT_HERSHEY_SIMPLEX,.49,(180,225,255),1,cv2.LINE_AA)
            writer.append_data(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
            if k==ref_indices[-1]:cv2.imwrite(str(run/'comparison_reference_end.jpg'),canvas)
    for cap in caps:cap.release()
    (run/'video_mapping.json').write_text(json.dumps(dict(source='actual rendered MP4 frames and recorded real PNGs',trace_index_per_video_frame=mapping,
        overlay='object axes projected from saved pose logs; axial spin from wrist-relative object orientation',video_time='30 FPS simulation clock, excluding model inference wall time'),indent=2)+'\n')
    print(movie)

if __name__=='__main__':main()
