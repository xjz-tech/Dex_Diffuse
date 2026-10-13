import concurrent.futures,json,os,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926';O=R/'episode2_prior_palm_direction_20260928';C=R/'qualified_comparison/episode_02'
JOBS=[(name,offset,seed) for name,offset in [('original',[0.,0.,0.]),('toward_palm_20',[-.01,0.,-.02])] for seed in [44,45,46]]
(O/'direction_manifest.json').write_text(json.dumps(dict(jobs=JOBS,action_steps=180,ddim=4,execution_steps=2,guidance_scale=0,mass=.044,friction=1.1,note='toward_palm_20 is the furthest tested translation passing static inspection; not a full palm grasp. Selected before any prior rollouts. Same initial hand angles and bulb orientation. Post-settle states differ physically.'),indent=2)+'\n')
def run(job):
 name,off,seed=job;f=O/name/f'prior_seed{seed}';f.mkdir(parents=True,exist_ok=True)
 simenv=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PATH='/home/carus/miniforge3/envs/decv2/bin:'+os.environ['PATH'],PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
 serverenv=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')
 sock=Path(f'/tmp/palm_prior_{name}_{seed}_{os.getpid()}.sock');server=None
 try:
  with (f/'server.log').open('w') as log:
   server=subprocess.Popen(['/home/carus/miniforge3/envs/dp/bin/python',str(P/'server_autonomous_prior.py'),'--socket',str(sock),'--log',str(f/'predictions.json'),'--checkpoint','/home/carus/data_usb/10B_obs_4-66.ckpt','--reference',str(C/'reference_full.npz'),'--ddim-steps','4','--guidance-steps','4','--guidance-scale','0','--execution-steps','2','--reference-interpolation-threshold','.1'],env=serverenv,stdout=log,stderr=subprocess.STDOUT)
  for _ in range(600):
   if sock.exists():break
   if server.poll() is not None:raise RuntimeError('server failed')
   time.sleep(.2)
  else:raise TimeoutError()
  cmd=['/home/carus/miniforge3/envs/decv2/bin/python',str(P/'compare_prior_palm_rollouts.py'),'--mode','guided','--out',str(f),'--reference',str(C/'reference_full.npz'),'--reference-id','0','--source-episode','2','--front-only','--audit-recording','--grasp-evidence','--object-mass-kg','.044','--friction','1.1','--guidance-steps','4','--guidance-scale','0','--execution-steps','2','--reference-interpolation-threshold','.1','--action-limit','180','--prior-noise-seed',str(seed),'--object-wrist-offset',*map(str,off),'--socket',str(sock)]
  with (f/'sim.log').open('w') as log:subprocess.run(cmd,env=simenv,stdout=log,stderr=subprocess.STDOUT,check=True)
  assert server.wait(timeout=30)==0
  print('DONE',name,seed,flush=True)
 finally:
  if server and server.poll() is None:server.terminate();server.wait()
  sock.unlink(missing_ok=True)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,JOBS))
