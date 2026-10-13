import argparse,concurrent.futures,json,os,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926';O=R/'episode54_reference_edit_20260928';C=R/'qualified_comparison/episode_54'
def folder(method,seed):return O/(f'{method}_seed{seed}')
def run(job):
 method,ratio,seed=job;f=folder(method,seed);f.mkdir(parents=True,exist_ok=True)
 if (f/'summary.json').exists() and (f/'predictions.json').exists():return method,seed,'existing'
 simenv=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PATH='/home/carus/miniforge3/envs/decv2/bin:'+os.environ['PATH'],PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
 serverenv=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')
 sock=Path(f'/tmp/refedit54_{method}_{seed}_{os.getpid()}.sock');server=None
 try:
  standard=method=='standard_g4_s50';scale=50 if standard else 0;guidesteps=4 if standard else 9
  cmd=['/home/carus/miniforge3/envs/dp/bin/python',str(P/('server.py' if standard else 'server_reference_action_editor.py')),'--socket',str(sock),'--log',str(f/'predictions.json'),'--checkpoint','/home/carus/data_usb/10B_obs_4-66.ckpt','--reference',str(C/'reference_full.npz')]
  if standard:cmd+=['--ddim-steps','4','--guidance-steps','4','--guidance-scale','50','--execution-steps','2']
  else:cmd+=['--noise-ratio',str(ratio)]
  with (f/'server.log').open('w') as log:server=subprocess.Popen(cmd,env=serverenv,stdout=log,stderr=subprocess.STDOUT)
  for _ in range(600):
   if sock.exists():break
   if server.poll() is not None:raise RuntimeError(f'server failed: {f}')
   time.sleep(.2)
  else:raise TimeoutError(f)
  cmd=['/home/carus/miniforge3/envs/decv2/bin/python',str(P/'compare_reference_edit_rollouts.py'),'--mode','guided','--out',str(f),'--reference',str(C/'reference_full.npz'),'--reference-id','0','--source-episode','54','--front-only','--audit-recording','--grasp-evidence','--no-video','--object-mass-kg','.044','--friction','1.1','--guidance-steps',str(guidesteps),'--guidance-scale',str(scale),'--execution-steps','2','--prior-noise-seed',str(seed),'--socket',str(sock)]
  with (f/'sim.log').open('w') as log:subprocess.run(cmd,env=simenv,stdout=log,stderr=subprocess.STDOUT,check=True)
  assert server.wait(timeout=30)==0;print('DONE',method,seed,flush=True);return method,seed,'complete'
 finally:
  if server and server.poll() is None:server.terminate();server.wait()
  sock.unlink(missing_ok=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['zero','main'],required=True);args=parser.parse_args()
 if args.phase=='zero':jobs=[('zero',0.,44)]
 else:
  assert json.loads((O/'zero_verification.json').read_text())['full_physics_trace_exact']
  jobs=[(m,r,s) for s in [44,45,46] for m,r in [('standard_g4_s50',None),('edit010',.1),('edit020',.2),('edit035',.35)]]
 (O/f'{args.phase}_manifest.json').write_text(json.dumps(dict(jobs=jobs,episode=54,source_frame=110,mass=.044,friction=1.1,interpolation=False,exec_steps=2,ddim_updates=4,source='reference_initialized_editor versus standard noise_initialized_guidance',note='Noise ratio is normalized sqrt((1-alpha)/alpha), not a radian noise standard deviation. Actual discrete schedules are saved per run.'),indent=2)+'\n')
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,jobs))
