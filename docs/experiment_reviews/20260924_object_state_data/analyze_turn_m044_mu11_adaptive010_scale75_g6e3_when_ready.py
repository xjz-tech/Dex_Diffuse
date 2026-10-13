"""Analyze each episode after its complete simulation has been written."""
import concurrent.futures,json,time
import analyze_turn_m044_mu11_adaptive010_scale75_g6e3 as A
from pathlib import Path

def ready_analyze(job):
 ep,mode,ddim,scale=job
 folder=A.CASES/f'episode_{ep:02d}'/A.folder_name(mode,ddim,scale)
 for _ in range(1800):
  if (folder/'summary.json').exists() and (folder/'trace.json').exists() and (folder/'predictions.json').exists():
   return A.analyze(job)
  time.sleep(2)
 raise TimeoutError(folder)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(ready_analyze,A.JOBS))
 (A.OUT/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
 print('ALL ANALYZED',flush=True)
