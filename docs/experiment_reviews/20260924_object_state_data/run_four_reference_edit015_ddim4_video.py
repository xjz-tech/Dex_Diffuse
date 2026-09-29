"""Run three reference-edit noise levels at four/six local DDIM updates."""
import concurrent.futures,json,os,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
R=P/'reference_turn_baseline_20260926'
O=R/'four_reference_edit_noise_ddim_sweep_20260928'
EPISODES=(76,34,54,2)
RATIOS=(.15,.25,.30)
STEPS=(4,6)
SEED=44

def folder(ep,ratio,steps):
 return O/f'episode_{ep:02d}'/f'edit_{int(round(ratio*100)):03d}_ddim{steps}_seed{SEED}'

def run(job):
 ep,ratio,steps=job
 case=R/f'qualified_comparison/episode_{ep:02d}'
 out=folder(ep,ratio,steps).with_name(folder(ep,ratio,steps).name+'_video')
 out.mkdir(parents=True,exist_ok=True)
 if (out/'summary.json').exists() and (out/'predictions.json').exists():
  return (ep,ratio,steps,'existing')
 simenv=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PATH='/home/carus/miniforge3/envs/decv2/bin:'+os.environ['PATH'],PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
 serverenv=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')
 sock=Path(f'/tmp/refedit_sweep_{ep}_{int(ratio*100)}_{steps}_{os.getpid()}.sock')
 server=None
 try:
  command=['/home/carus/miniforge3/envs/dp/bin/python',str(P/'server_reference_action_editor.py'),'--socket',str(sock),'--log',str(out/'predictions.json'),'--checkpoint','/home/carus/data_usb/10B_obs_4-66.ckpt','--reference',str(case/'reference_full.npz'),'--reference-interpolation-threshold','.1','--noise-ratio',str(ratio),'--ddim-steps',str(steps)]
  with (out/'server.log').open('w') as log:
   server=subprocess.Popen(command,env=serverenv,stdout=log,stderr=subprocess.STDOUT)
  for _ in range(600):
   if sock.exists():break
   if server.poll() is not None:raise RuntimeError(f'server failed: {out}')
   time.sleep(.2)
  else:raise TimeoutError(out)
  command=['/home/carus/miniforge3/envs/decv2/bin/python',str(P/'compare_reference_edit_rollouts.py'),'--mode','guided','--out',str(out),'--reference',str(case/'reference_full.npz'),'--reference-id','0','--source-episode',str(ep),'--front-only','--audit-recording','--grasp-evidence','--reference-interpolation-threshold','.1','--object-mass-kg','.044','--friction','1.1','--guidance-steps','9','--guidance-scale','0','--execution-steps','2','--prior-noise-seed',str(SEED),'--socket',str(sock)]
  with (out/'sim.log').open('w') as log:
   subprocess.run(command,env=simenv,stdout=log,stderr=subprocess.STDOUT,check=True)
  assert server.wait(timeout=30)==0
  print('DONE',ep,ratio,steps,flush=True)
  return (ep,ratio,steps,'complete')
 finally:
  if server and server.poll() is None:
   server.terminate();server.wait()
  sock.unlink(missing_ok=True)

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 jobs=[(ep,.15,4) for ep in EPISODES]
 manifest=dict(jobs=jobs,episodes=EPISODES,source_frames={76:114,34:79,54:110,2:103},checkpoint='/home/carus/data_usb/10B_obs_4-66.ckpt',mass_kg=.044,friction=1.1,object_size_multiplier=1,reference_interpolation_threshold_rad=.1,execution_steps=2,history=3,future=9,denoising_schedule='local linear descent from closest trained noise ratio to timestep zero, rounded; exact schedule saved in each summary',prior_noise_seed=SEED,sim_seed=42,native_protocol='unchanged from compare_reference_edit_rollouts.py',selection='same four episodes previously qualified by original direct stable turn',video_recorded=True)
 (O/'video_015_ddim4_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  results=list(pool.map(run,jobs))
 (O/'video_015_ddim4_run_status.json').write_text(json.dumps(results,indent=2)+'\n')
 print('ALL VIDEO RUNS COMPLETE',len(results),flush=True)
