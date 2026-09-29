from pathlib import Path
import json,ast,hashlib
import numpy as np
ROOT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
OUT=Path(__file__).resolve().parent
S=Path('/home/carus/Data/bulb_tac_260909/episode_37')
r=OUT/'reference';r.mkdir(exist_ok=True)
s=np.load(S/'state.npy');a=np.load(S/'action.npy');start,end=160,250
names=json.loads(Path('/home/carus/Data/bulb_right_turn_reference_260909/id_37_f325_400/initial_state.json').read_text())['hand_joint_names']
np.savez_compressed(r/'reference.npz',hand_qpos_rad=s[start:end+1,9:],hand_target_rad=a[start:end,9:],hand_target_before_rad=a[start-1:end-1,9:],time_s=np.arange(end-start+1)/30,recorded_state31=s[start:end+1],source_state_frame_indices=np.arange(start,end+1))
(r/'initial_state.json').write_text(json.dumps(dict(hand_joint_names=names,source_episode=37,start_frame=start,end_frame=end,object_pose_recorded=False,selection='visually confirmed lifted horizontal bulb through vertical screw-down grasp; ends before socket contact'),indent=2))
sim=(ROOT/'eval/real_reference_sim.py').read_text().replace("ROOT = Path(__file__).resolve().parents[1]",f"ROOT = Path({str(ROOT)!r})").replace("REFERENCE = Path('/home/carus/Data/bulb_right_turn_reference_260909/id_37_f325_400/reference.npz')",f"REFERENCE = Path({str(r/'reference.npz')!r})")
sim=sim.replace("env.compute_observations();env.reset()", "env.compute_observations();env.reset()\n        env.gym.simulate(env.sim);env.gym.fetch_results(env.sim,True)\n        env.compute_observations()")
sim=sim.replace("base=arr(env._base_state[:,:7]);brot=R.from_quat(base[:,3:])", "env._base_state[:,:3]=torch.tensor([0.,0.,.6],device=env.device)\n        env._base_state[:,3:7]=torch.tensor([1.,0.,0.,0.],device=env.device)\n        base=arr(env._base_state[:,:7]);brot=R.from_quat(base[:,3:])")
sim=sim.replace("eye=center+np.array([.22,.36,.19])", "eye=center+np.array([.25,-.34,.12])")
sim=sim.replace("score=fit_drift+.05*q_rmse+100*ever_failure+.05*(contacts<2)", "axes=R.from_quat(settled_rot).apply([0,1,0])\n            horizontal_error=np.abs(axes[:,2])\n            score=fit_drift+.05*q_rmse+100*ever_failure+.05*(contacts<2)+.1*horizontal_error")
sim=sim.replace("native_failure=bool(ever_failure[i]),static_score=float(score[i])", "native_failure=bool(ever_failure[i]),horizontal_tilt_deg=float(np.degrees(np.arcsin(horizontal_error[i]))),static_score=float(score[i])")
sim=sim.replace("if args.phase=='evaluate':capture(phase,index)", "if args.phase=='evaluate':capture(phase,index)\n            if index==0 and phase=='settle':\n                assert np.max(np.abs(arr(env._base_state[:,:7])-base))<1e-5, 'fixed base snapped back'")
sim=sim.replace("'finite 75-action motion test'", "'finite 90-action horizontal-to-vertical test','wrist pose fixed fingers downward at xyz 0,0,0.6 and xyzw 1,0,0,0; engineering approximation of source wrist'")
sim=sim.replace("count=21 if args.phase=='fit' else 3","count=144 if args.phase=='fit' else 3").replace("offsets=[[0,0,0],[.005,0,0],[-.005,0,0],[0,.005,0],[0,-.005,0],[0,0,.005],[0,0,-.005]]","offsets=[[x,y,z] for x in [0.,.02,.04,.06] for y in [0.,.02,.04,.06] for z in [-.02,0.,.02]]")
(OUT/'sim.py').write_text(sim)
server=(ROOT/'eval/real_reference_server.py').read_text().replace("ROOT=Path(__file__).resolve().parents[1]",f"ROOT=Path({str(ROOT)!r})")
(OUT/'server.py').write_text(server)
print(OUT)
