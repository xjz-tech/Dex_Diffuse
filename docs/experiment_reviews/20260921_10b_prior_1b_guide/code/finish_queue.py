from pathlib import Path
import os,json,time,subprocess,traceback
R=Path(__file__).resolve().parents[1];ROOT=R.parents[2];PY='/home/carus/miniforge3/envs/dp/bin/python';os.environ['PYTHONDONTWRITEBYTECODE']='1'
status={'phase':'waiting_for_evaluation','pid':os.getpid()}
def save():(R/'postprocess_status.json').write_text(json.dumps(status,indent=2))
save()
try:
 while True:
  state=json.loads((R/'status.json').read_text())
  if state.get('error'):raise RuntimeError(state['error'])
  if state.get('complete'):break
  time.sleep(10)
 for script in ['report.py','videos.py']:
  status['phase']=script;save()
  with (R/(script+'.log')).open('w') as f:subprocess.run([PY,'-u',str(R/'code'/script)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
 status['phase']='complete_pending_visual_review'
except BaseException:
 status['error']=traceback.format_exc();raise
finally:save()
