"""Audit actual paired references and encode 600 real frames at the control rate."""
import hashlib,json,subprocess
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
ROOT=P.parents[2]
METHODS=['direct','edit010','edit020','edit035']
FFMPEG=ROOT/'eval/videos/20260914_10k_guides_1b_seed8/bin/ffmpeg'
LABELS={'direct':'Direct','edit010':'Noise 0.10 / DDIM6','edit020':'Noise 0.20 / DDIM6','edit035':'Noise 0.35 / DDIM6','edit015_ddim4':'Noise 0.15 / DDIM4'}
LABELS['guidance50_g4_ddim4']='Guidance50 / guide4 / DDIM4'
LABELS.update(guidance25_ddim4='Guidance scale 25 / DDIM4',guidance100_ddim4='Guidance scale 100 / DDIM4')

def load(direction,method,reviews,baseline):
    is_guidance=method.startswith('guidance')
    ddim_steps=4 if method=='edit015_ddim4' or is_guidance else 6
    guide_steps=4 if method=='guidance50_g4_ddim4' else 9
    folder=P/'runs'/(direction+'_'+method)
    rows=[json.loads(x) for x in (folder/'trajectory.jsonl').read_text().splitlines()]
    assert len(rows)==600 and [r['step'] for r in rows]==list(range(1,601))
    manifest=json.loads((folder/'manifest.json').read_text());dt=manifest['control_dt']
    assert abs(dt-1/30)<1e-6 and manifest['execution_steps']==2
    expected=dict(failure_obj_pos_thres_m=.05,failure_tip_pos_thres_m=.1,failure_obj_rot_thres_deg=180.,invalid_obj_pos_thres_m=.15,failure_tolerance_scale=10000.,fixed_tolerance_steps=20000,traj_steps_limit=12000,reset_on_reach_goal=False,cross_trajectory_goal_prob=.3)
    assert manifest['protocol']==expected,(manifest['protocol'],expected)
    assert manifest['data_indices']=='000-149'
    with np.load(folder/'initial_state.npz') as actual:
        for k,v in baseline.items(): np.testing.assert_array_equal(actual[k],v,err_msg=folder.name+':'+k)
    ref=np.load(P/(direction+'_reference.npz'))['actions']
    np.testing.assert_array_equal(np.array([r['astra_reference'] for r in rows],dtype=np.float32),ref[:600])
    model=json.loads((folder/'model.json').read_text());assert model['prior_ddim_steps']==ddim_steps
    if is_guidance:
        scale=float(method.split('_')[0].replace('guidance',''))
        assert model['guidance_scale']==scale and model['reference_editor'] is None
        assert model['guidance_steps']==guide_steps and model['noise_seed']==44 and model['fixed_noise']
        assert model['prior_residual_bound_rad'] is None and model['guidance_schedule_file'] is None
    errors=[]
    for step in range(0,600,2):
        with np.load(folder/f'prediction_{step:06d}.npz') as pred:
            np.testing.assert_array_equal(pred['reference'][0],ref[step:step+guide_steps])
            if method=='direct': np.testing.assert_array_equal(pred['prediction'][0],ref[step:step+2])
        meta=json.loads((folder/f'prediction_{step:06d}.json').read_text())
        assert meta['reference_hold_padding_steps']==0
        if is_guidance:
            assert 'reference_editor' not in meta
            assert np.isfinite(meta['mse_before']) and np.isfinite(meta['mse_after'])
        elif method!='direct':
            assert meta['reference_editor']['inference_steps']==ddim_steps
            errors.append(meta['edit_stats']['history_mask_max_error'])
    assert not errors or max(errors)==0
    review=reviews[folder.name]
    cutoff=review['last_confirmed_held_step']
    dropped=review['action']=='confirmed_drop'
    assert 1<=cutoff<=600 and (not dropped or cutoff<review['first_confirmed_separated_step']<=600)
    if dropped: (folder/'physical_drop_review.json').write_text(json.dumps(review,indent=2)+'\n')
    else: assert cutoff==600
    sign=1 if direction=='left' else -1
    angles=[sign*r['twist_degrees'] for r in rows[:cutoff]]
    first=np.asarray(rows[0]['state']['object_position'])
    translation=float(np.linalg.norm(first-baseline['object'][0,:3]))
    assert translation<.01
    summary=dict(direction=direction,method=method,steps=600,control_dt=dt,duration_s=600*dt,
        ddim_steps=ddim_steps,execution_steps=2,mass_kg=float(baseline['object_mass'][0]),object_friction=float(baseline['object_friction'][0,0]),
        physical_drop_confirmed=dropped,physical_review_pending=False,physical_review=review,
        observation_censored_without_confirmed_drop=not dropped,first_native_failure_step=next((r['step'] for r in rows if r['failure']),None),
        held_net_requested_deg=angles[-1],held_max_requested_deg=max([0.]+angles),
        first_control_step_object_translation_m=translation,initial_fields_identical=len(baseline),
        audited_prediction_windows=300,reference_padding_steps=0,history_mask_max_error=max(errors) if errors else None,
        reference_editor=model['reference_editor'],reference_file=str(P/(direction+'_reference.npz')))
    summary.update(guidance_scale=model['guidance_scale'],guidance_steps=model['guidance_steps'],
                   noise_seed=model.get('noise_seed'),fixed_noise=model['fixed_noise'])
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return folder,summary,rows

def panel(folder,summary,rows,step,direction,method):
    im=cv2.imread(str(folder/'frames'/f'{step:06d}.png'));assert im is not None
    # Remove the legacy source-camera text, preserving the physical scene below it.
    im=cv2.resize(im[80:],(640,426))
    out=np.zeros((510,640,3),np.uint8);out[55:481]=im
    dt=summary['control_dt'];review=summary['physical_review'];cutoff=review['last_confirmed_held_step']
    sign=1 if direction=='left' else -1
    cv2.putText(out,f'{direction.upper()} | {LABELS[method]}',(10,23),cv2.FONT_HERSHEY_SIMPLEX,.63,(255,255,255),1,cv2.LINE_AA)
    cycle=max(0,step-24)//200+1
    cv2.putText(out,f't = {step*dt:05.2f} / 20.00 s | 1x | step {step}/600 | cycle {cycle}',(10,46),cv2.FONT_HERSHEY_SIMPLEX,.5,(210,210,210),1,cv2.LINE_AA)
    separated=review.get('first_confirmed_separated_step',601)
    status='DROPPED; cyclic commands continue' if step>=separated else ('LOSS WINDOW' if step>cutoff else 'IN OBSERVATION')
    angle=sign*rows[min(step,cutoff)-1]['twist_degrees']
    cv2.putText(out,f'{status} | held turn {angle:+.1f} deg',(10,500),cv2.FONT_HERSHEY_SIMPLEX,.49,(60,170,255) if step>=separated else (170,230,170),1,cv2.LINE_AA)
    return out

def encode(direction,data):
    dt=data[0][1]['control_dt'];path=P/(direction+'_comparison_1x.mp4')
    command=[str(FFMPEG),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x1020','-r',str(1/dt),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    for step in range(1,601):
        panels=[panel(*item,step,direction,method) for item,method in zip(data,METHODS)]
        frame=cv2.vconcat([cv2.hconcat(panels[:2]),cv2.hconcat(panels[2:])])
        process.stdin.write(frame.tobytes())
        if step==600: cv2.imwrite(str(P/(direction+'_final.jpg')),frame)
    process.stdin.close();assert process.wait()==0
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=0
    while cap.read()[0]: count+=1
    cap.release();assert count==600 and abs(count/fps-600*dt)<.002
    return dict(file=str(path),frames=count,fps=fps,video_duration_s=count/fps,simulation_duration_s=600*dt,speed_ratio=(600*dt)/(count/fps),codec='H264/yuv420p',frozen_frames_added=0)

def main():
    reviews=json.loads((P/'physical_reviews.json').read_text())
    with np.load(P/'runs/left_direct/initial_state.npz') as z:baseline={k:z[k].copy() for k in z.files}
    assert abs(float(baseline['object_mass'][0])-.17)<1e-6 and np.allclose(baseline['object_friction'],2.2)
    summaries=[];videos={};fig,axes=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
    for i,d in enumerate(['left','right']):
        data=[load(d,m,reviews,baseline) for m in METHODS]
        for folder,s,rows in data:
            summaries.append(s);n=s['physical_review']['last_confirmed_held_step'];sign=1 if d=='left' else -1
            axes[i].plot(np.arange(1,n+1)*s['control_dt'],[sign*r['twist_degrees'] for r in rows[:n]],label=s['method'])
        axes[i].set(title=d,xlim=(0,20),xlabel='Actual simulation time (s)',ylabel='Requested-direction turn (deg)')
        axes[i].grid(alpha=.25);axes[i].legend()
        videos[d]=encode(d,data)
    fig.savefig(P/'held_turn_curves.png',dpi=150)
    result=dict(summaries=summaries,videos=videos)
    (P/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    hashes={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'build_reference.py',P/'run.py',P/'reference_config.json',P/'left_reference.npz',P/'right_reference.npz',P/'source/eval/astra_sim.py',P/'source/eval/longrun_review.py',P/'source/eval/reference_action_editor.py']}
    (P/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
