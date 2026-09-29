from pathlib import Path
import subprocess,json,time,signal
OUT=Path(__file__).resolve().parent
children={};results={};started=time.time()
def status():
 (OUT/'supervisor_status.json').write_text(json.dumps(dict(started_unix=started,elapsed_s=time.time()-started,children={s:dict(pid=p.pid,returncode=p.poll()) for s,p in children.items()},results=results),indent=2))
def stop(sig,frame):
 for p in children.values():
  if p.poll() is None:
   try:__import__('os').killpg(p.pid,signal.SIGTERM)
   except ProcessLookupError:pass
 status();raise SystemExit(128+sig)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
for seed in [42,123,2026]:
 children[seed]=subprocess.Popen(['bash',str(OUT/'run.sh'),str(seed)],cwd=str(OUT.parents[2]),start_new_session=True)
 status()
while any(p.poll() is None for p in children.values()):
 status();time.sleep(5)
for seed,p in children.items():results[seed]=p.returncode
status()
if any(results.values()):raise SystemExit('A seed failed; inspect logs, no successful-completion claim')
python='/home/carus/miniforge3/envs/dp/bin/python'
subprocess.run([python,str(OUT/'analyze.py')],check=True)
subprocess.run([python,str(OUT/'report.py')],check=True)
(OUT/'ALL_COMPLETE').write_text('All three planned runs and analysis completed.\n')
