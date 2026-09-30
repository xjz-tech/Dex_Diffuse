"""Render saved states with one 1us physics tick to refresh GPU render transforms.

No policy is rerun. Each frame independently starts from the recorded state.
"""
import sys,json
from pathlib import Path
import numpy as np
sys.path.append('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/.worktrees/sim-real-holding-comparison/eval')
from real_sim_world import World
from isaacgym import gymapi
import cv2

out=Path(sys.argv[1]).resolve();dest=out/'shortest_seed65';dest.mkdir(exist_ok=True)
ini=np.load(out/'initial_state.npz');z=np.load(out/'rollouts.npz');idx=47;last=int(z['drop_steps'][idx])
w=World(n=1,dt=1e-6,demos=1,camera=True)
# Rendering reconstruction must not resolve the saved contact configuration.
# Disable contacts/gravity and velocities before the tiny transform-refresh tick.
params=w.g.get_sim_params(w.s);params.gravity=gymapi.Vec3(0,0,0);w.g.set_sim_params(w.s,params)
for ai in range(w.g.get_actor_count(w.e.envs[0])):
    actor=w.g.get_actor_handle(w.e.envs[0],ai)
    shapes=w.g.get_actor_rigid_shape_properties(w.e.envs[0],actor)
    for shape in shapes:shape.filter=1
    w.g.set_actor_rigid_shape_properties(w.e.envs[0],actor,shapes)
props=gymapi.CameraProperties();props.width=640;props.height=480;props.horizontal_fov=55
cams=[w.g.create_camera_sensor(w.e.envs[0],props) for _ in range(2)]
wrist=ini['wrist'][idx].copy();origin=wrist[:3]
for cam,offset in zip(cams,[[.12,.55,.15],[.45,.1,.08]]):
    w.g.set_camera_location(cam,w.e.envs[0],gymapi.Vec3(*(origin+offset)),gymapi.Vec3(*(origin+[-.08,0,-.12])))
video=cv2.VideoWriter(str(dest/'saved_state_replay_025x.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,560))
assert video.isOpened()
selected=[];metrics=[]
try:
    for step in range(-1,last+1):
        q=ini['q'][idx] if step<0 else z['qpos'][step,idx]
        obj=ini['object'][idx] if step<0 else z['object'][step,idx]
        t=(step+1)/30;dis=float(np.linalg.norm(obj[:3]-ini['object'][idx,:3]))
        render_obj=obj.copy();render_obj[7:]=0
        w.reset(q,wrist,render_obj)
        assert np.max(abs(w.state()[0]-q))<1e-6
        assert np.max(abs(w.objects()[0,:7]-obj[:7]))<1e-6
        w.step(1)
        render_q_error=float(np.max(abs(w.state()[0]-q)))
        render_obj_error=float(np.max(abs(w.objects()[0,:3]-obj[:3])))
        assert render_q_error<1e-4 and render_obj_error<1e-5,(render_q_error,render_obj_error)
        w.g.step_graphics(w.s);w.g.render_all_camera_sensors(w.s)
        imgs=[]
        for cam in cams:
            rgba=w.g.get_camera_image(w.s,w.e.envs[0],cam,gymapi.IMAGE_COLOR)
            imgs.append(cv2.cvtColor(np.asarray(rgba).reshape(480,640,4)[...,:3],cv2.COLOR_RGB2BGR))
        frame=np.zeros((560,1280,3),np.uint8);frame[80:]=np.concatenate(imgs,axis=1)
        title='SAVED STATE REPLAY | seed 65 | pose/placement 4 | 0.25x'
        detail=f't = {t:.3f} s | center displacement = {100*dis:.2f} cm | failure threshold = 5 cm'
        cv2.putText(frame,title,(15,29),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        cv2.putText(frame,detail,(15,62),cv2.FONT_HERSHEY_SIMPLEX,.65,(80,100,255) if dis>.05 else (190,240,190),2)
        for _ in range(30 if step in (-1,last) else 4):video.write(frame)
        if step in (-1,5,11,17,21):
            cv2.imwrite(str(dest/f'frame_{step+1:03d}.png'),frame);selected.append(frame)
        metrics.append({'step':step,'seconds':t,'center_displacement_m':dis,'object_xyz':obj[:3].tolist(),'render_q_error_rad':render_q_error,'render_obj_error_m':render_obj_error})
    cv2.imwrite(str(dest/'contact_sheet.png'),np.concatenate([cv2.resize(x,(768,336)) for x in selected],axis=0))
    (dest/'metrics.json').write_text(json.dumps(metrics,indent=2))
finally:
    video.release();w.close()
print(dest,flush=True)
