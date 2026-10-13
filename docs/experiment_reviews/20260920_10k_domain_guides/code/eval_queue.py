from pathlib import Path
import concurrent.futures as cf,subprocess,time,json,os
r=Path(__file__).resolve().parents[1]
names=['ordinary_1b','old10k','light_low','light_high','heavy_low','heavy_high','balanced']
def run(name):
 p=r/'evaluation'/name/'results.json'
 if name=='ordinary_1b':
  while not p.exists():
   try:os.kill(444125,0)
   except ProcessLookupError:raise RuntimeError('Baseline process exited without results')
   time.sleep(5)
 elif not p.exists():
  print('START',name,time.strftime('%FT%T'),flush=True)
  subprocess.run(['bash',str(r/'code/eval_one.sh'),name],check=True)
 assert json.loads(p.read_text())['complete']
 print('COMPLETE',name,time.strftime('%FT%T'),flush=True)
 return name
with cf.ThreadPoolExecutor(max_workers=2) as ex:
 futs={ex.submit(run,n):n for n in names}
 for f in cf.as_completed(futs):
  f.result()
  if (r/'evaluation/ordinary_1b/results.json').exists():
   with open(r/'report_progress.log','w') as log:subprocess.run(['/home/carus/miniforge3/envs/dp/bin/python',str(r/'code/report_results.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
with open(r/'videos.log','w') as log:subprocess.run(['/home/carus/miniforge3/envs/dp/bin/python',str(r/'code/videos.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
print('ALL COMPLETE',time.strftime('%FT%T'),flush=True)
