"""Add DDIM6 for the three noise levels already run at DDIM4."""
import concurrent.futures,json,time
from run_four_reference_edit_noise_ddim_sweep import O,EPISODES,SEED,run
RATIOS=(.10,.20,.35)
if __name__=='__main__':
 while not (O/'run_status.json').exists():time.sleep(5)
 jobs=[(ep,ratio,6) for ep in EPISODES for ratio in RATIOS]
 (O/'extra_manifest.json').write_text(json.dumps(dict(jobs=jobs,reason='DDIM6 pairs for previously completed DDIM4 ratios',previous_ddim4_source='four_episode_reference_edit_adaptive010_video_20260928',all_other_settings='same as manifest.json'),indent=2)+'\n')
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,jobs))
 (O/'extra_run_status.json').write_text(json.dumps(results,indent=2)+'\n')
 print('ALL EXTRA RUNS COMPLETE',len(results),flush=True)
