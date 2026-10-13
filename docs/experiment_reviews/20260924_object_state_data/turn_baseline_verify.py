from pathlib import Path
import json,shutil,time,sys
import numpy as np
from turn_baseline_screen import check
from random4_geometry import metrics
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926/qualified_comparison'
def verify(ep):
 f=R/f'episode_{ep:02d}';run=f/'direct'
 while not (run/'summary.json').exists():time.sleep(2)
 screen=check(run);assert screen['preliminary_pass'],screen
 s=json.load(open(run/'summary.json'));t=json.load(open(run/'trace.json'));ev=f/'turn_evidence';ev.mkdir(exist_ok=True)
 first=screen['block'][0]-1
 for name in ['summary.json','initial_state.npz']:shutil.copy2(run/name,ev/name)
 (ev/'trace.json').write_text(json.dumps(t[:60]+t[60+first:60+first+30]))
 g=metrics(ev,all_frames=True);a=g['frames'][60:]
 assert all(r['near_contact_link_count']>=2 and r['mesh_vertex_gap_m']<.008 and r['mesh_table_clearance_m']>.08 for r in a),[(r['index'],r['near_contact_link_count']) for r in a]
 static=f/'static';static.mkdir(exist_ok=True)
 for name in ['summary.json','initial_state.npz']:shutil.copy2(run/name,static/name)
 (static/'trace.json').write_text(json.dumps(t[:60]));st=metrics(static)['static'];assert st['numeric_pass'],st
 screen.update(mesh_confirmed_30_steps=[first+1,first+30],minimum_near_object_contact_links=min(r['near_contact_link_count'] for r in a),minimum_mesh_table_clearance_m=min(r['mesh_table_clearance_m'] for r in a),static=st,baseline_verified=True)
 (f/'baseline_verified.json').write_text(json.dumps(screen,indent=2));print('BASELINE VERIFIED',ep,screen,flush=True)
 return screen
if __name__=='__main__':
 import concurrent.futures
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(verify,[76,34,54,2]))
