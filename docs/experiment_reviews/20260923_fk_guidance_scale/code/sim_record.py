"""Native sim_eval entry point with an isolated post-step CPU camera recorder.

This wraps rendering only. Native setup, step, reset, goal, observation and
failure logic are delegated unchanged to sim_eval.main.
"""
import os,sys,json
from pathlib import Path
ROOT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
sys.path.insert(0,str(ROOT/'eval'))
import sim_eval
from isaacgym import gymapi
import cv2
import numpy as np

make_config=sim_eval._make_task_config
validate_env=sim_eval._validate_environment
destroy_env=sim_eval._destroy_environment
recorders={}

def config(*args,**kwargs):
    cfg=make_config(*args,**kwargs)
    cfg.env.enableCameraSensors=True
    return cfg

def validate(env,manifest):
    result=validate_env(env,manifest)
    out=Path(os.environ['RECORD_DIR']);out.mkdir(parents=True,exist_ok=True)
    cp=gymapi.CameraProperties();cp.width=640;cp.height=480;cp.horizontal_fov=60
    camera=env.gym.create_camera_sensor(env.envs[0],cp)
    assert camera>=0
    origin=env._base_state[0,:3].cpu().numpy()
    env.gym.set_camera_location(camera,env.envs[0],gymapi.Vec3(*(origin+[-.10,.55,.10])),gymapi.Vec3(*(origin+[-.10,0.,-.14])))
    writer=cv2.VideoWriter(str(out/'live_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,528))
    assert writer.isOpened()
    state={'writer':writer,'frames':0,'closed':False,'camera':camera,'out':out}
    recorders[id(env)]=state
    native_step=env.step
    def step(actions):
        output=native_step(actions)
        if not state['closed']:
            g=env.gym;s=env.sim
            g.fetch_results(s,True);g.step_graphics(s);g.render_all_camera_sensors(s)
            rgba=np.asarray(g.get_camera_image(s,env.envs[0],camera,gymapi.IMAGE_COLOR)).reshape(480,640,4)
            panel=np.zeros((528,640,3),dtype=np.uint8);panel[48:]=cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR)
            state['frames']+=1
            cv2.putText(panel,os.environ['RUN_NAME'],(8,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1)
            cv2.putText(panel,f"env0 | step {state['frames']} | {state['frames']/30:.2f} sim s",(8,40),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1)
            writer.write(panel)
            if state['frames']==1:cv2.imwrite(str(out/'first_frame.png'),panel)
            if bool(output[2][0]):
                cv2.imwrite(str(out/'terminal_frame.png'),panel)
                writer.release();state['closed']=True
                # Keep camera alive until native cleanup; no physics API mutations.
        return output
    env.step=step
    return result

def destroy(env):
    state=recorders.pop(id(env),None)
    if state:
        state['writer'].release()
        (state['out']/'recording.json').write_text(json.dumps({'frames':state['frames'],'fps':30,'sim_seconds':state['frames']/30,'env':0,'timing':'one frame after each actual native step; first episode only'},indent=2))
        env.gym.destroy_camera_sensor(env.sim,env.envs[0],state['camera'])
    return destroy_env(env)

sim_eval._make_task_config=config
sim_eval._validate_environment=validate
sim_eval._destroy_environment=destroy
if __name__=='__main__':sim_eval.main()
