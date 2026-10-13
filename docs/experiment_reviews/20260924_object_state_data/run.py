from pathlib import Path
import subprocess,os,time,sys
P=Path(__file__).resolve().parent;ROOT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse');out=P/(sys.argv[1] if len(sys.argv)>1 else 'sweep');out.mkdir(exist_ok=True)
sock='/tmp/dex_objectstate_'+out.name+'.sock';m=os.environ.copy();m.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')
s=m.copy();s.update(PATH='/home/carus/miniforge3/envs/decv2/bin:'+s['PATH'],PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:'+str(ROOT/'eval'),LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
p=None
try:
 p=subprocess.Popen(['/home/carus/miniforge3/envs/dp/bin/python',str(P/'server.py'),'--socket',sock,'--checkpoint','/home/carus/data_usb/10B_obs_4-66.ckpt','--reference',str(P/'reference/reference.npz'),'--log',str(out/'predictions.json')],env=m,stdout=(out/'server.log').open('w'),stderr=subprocess.STDOUT)
 for i in range(600):
  if Path(sock).exists():break
  if p.poll() is not None:raise RuntimeError('model server failed')
  time.sleep(.2)
 else:raise TimeoutError()
 cmd=['/home/carus/miniforge3/envs/decv2/bin/python',str(P/'sim.py'),'--socket',sock,'--out',str(out)]
 cmd+=['--record-ids',sys.argv[2] if len(sys.argv)>2 else ','.join(map(str,list(range(9))+list(range(63,72))))]
 subprocess.run(cmd,env=s,stdout=(out/'sim.log').open('w'),stderr=subprocess.STDOUT,check=True)
finally:
 if p is not None:
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=10)
print('DONE',out)
