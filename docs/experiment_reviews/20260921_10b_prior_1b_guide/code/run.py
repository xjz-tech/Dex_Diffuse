import os,json,time,subprocess,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
status={'started':time.strftime('%Y-%m-%d %H:%M:%S'),'pid':os.getpid(),'completed':[]}
def save():(R/'status.json').write_text(json.dumps(status,indent=2))
try:
 for name in ['ordinary_10b','guide_1b','guide_old10k']:
  status['active']=name;save()
  with (R/f'{name}.log').open('w') as f:
   subprocess.run(['bash',str(R/'code/eval_one.sh'),name],stdout=f,stderr=subprocess.STDOUT,check=True)
  status['completed'].append(name)
 status['active']=None;status['complete']=True
except BaseException:
 status['complete']=False;status['error']=traceback.format_exc();raise
finally:
 status['updated']=time.strftime('%Y-%m-%d %H:%M:%S');save()
