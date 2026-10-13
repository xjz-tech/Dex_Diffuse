"""Exact joint-space nearest neighbors; no physics or full grasp-coverage claim."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import pickle,json,hashlib,time
import numpy as np
from scipy.spatial import cKDTree
OUT=Path(__file__).resolve().parent;ROOT=Path('/home/carus/Program/dex-controller/data/retargeting/NOKOV-v3/mano2sharpa_rh/bulb2')
real=np.load(OUT/'real_states.npz');ref=[];rid=[];rf=[];sources=[]
for i in range(150):
 p=ROOT/f'{i:03d}.pkl'
 with p.open('rb') as f:d=pickle.load(f)
 q=np.array(d['opt_dof_pos'],dtype=np.float64)[::2] # Match raw 30Hz and avoid double-counting interpolation.
 assert q.shape[1]==22 and np.isfinite(q).all()
 ref.append(q);rid.append(np.full(len(q),i));rf.append(np.arange(len(q)))
 sources.append(dict(index=i,raw_sample_frames=len(q),pkl_frames=len(d['opt_dof_pos']),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
ref=np.concatenate(ref);rid=np.concatenate(rid);rf=np.concatenate(rf);q=real['qpos'].astype(np.float64);tree=cKDTree(ref)
print('reference',ref.shape,'real',q.shape,flush=True)
dists=[];indices=[]
for start in range(0,len(q),3000):
 dist,idx=tree.query(q[start:start+3000],workers=1);dists.append(dist);indices.append(idx)
 print('queried',min(start+3000,len(q)),flush=True)
idx=np.concatenate(indices);dist=np.concatenate(dists);rms=np.rad2deg(dist/np.sqrt(22));maxerr=np.rad2deg(np.abs(q-ref[idx]).max(1))
# Real commands are a different quantity, reported separately.
actdist,actidx=tree.query(real['action'],workers=1);actrms=np.rad2deg(actdist/np.sqrt(22))
# Reference novelty calibration: query every tenth ORIGINAL frame, exclude entire held-out demo fold.
fold=np.random.default_rng(20260912).permutation(150)%5;cal=[];calids=[]
for k in range(5):
 train=fold[rid]!=k;query=(fold[rid]==k)&(rf%10==0)
 ds,_=cKDTree(ref[train]).query(ref[query],workers=1);cal.extend(np.rad2deg(ds/np.sqrt(22)));calids.extend(rid[query]);print('calibration fold',k,flush=True)
cal=np.array(cal);cutoffs=np.percentile(cal,[50,90,95,99]);contact=real['contact_count'].sum(1)>0;contact2=(real['contact_count']>0).sum(1)>=2
np.savez_compressed(OUT/'nearest.npz',rms_deg=rms,max_error_deg=maxerr,nearest_ref_index=rid[idx],nearest_ref_frame=rf[idx],nearest_ref_qpos=ref[idx],action_rms_deg=actrms,calibration_rms_deg=cal,calibration_demo=np.array(calids),reference_qpos=ref,reference_demo=rid,reference_frame=rf)
rows=[]
for group,mask in [('all',np.ones(len(q),bool)),('tactile_any',contact),('tactile_two_or_more',contact2),('no_tactile_points',~contact)]:
 rows.append(dict(group=group,frames=int(mask.sum()),rms_percentiles=dict(zip(['p10','p25','p50','p75','p90','p95'],map(float,np.percentile(rms[mask],[10,25,50,75,90,95])))),threshold_coverage={str(t):float((rms[mask]<=t).mean()) for t in [5,10,15,20]},coverage_rms10_max25=float(((rms[mask]<=10)&(maxerr[mask]<=25)).mean()),reference_calibrated_coverage={str(p):float((rms[mask]<=t).mean()) for p,t in zip([50,90,95,99],cutoffs)},action_rms_percentiles=np.percentile(actrms[mask],[50,90,95]).tolist()))
episodes=json.loads((OUT/'episodes.json').read_text())
for e in episodes:
 mask=real['episode']==e['episode'];co=mask&contact
 e.update(rms_median_deg=float(np.median(rms[mask])),coverage10=float((rms[mask]<=10).mean()),contact_coverage10=float((rms[co]<=10).mean()) if co.any() else None,contact_rms_median=float(np.median(rms[co])) if co.any() else None,calibrated95_coverage=float((rms[mask]<=cutoffs[2]).mean()))
(OUT/'coverage.json').write_text(json.dumps(dict(metric='Euclidean joint angle distance / sqrt(22), converted to degrees; exact nearest reference among all 150, no object pose comparison',reference_frames=len(ref),real_frames=len(q),calibration='5-fold by demo; query every tenth original frame; 120 train reference demos vs 30 heldout; calibration is reference novelty, not control success',calibration_n=len(cal),calibration_cutoffs_deg=dict(zip(['p50','p90','p95','p99'],map(float,cutoffs))),groups=rows,episodes=episodes),indent=2))
(OUT/'reference_sources.json').write_text(json.dumps(sources,indent=2))
print(json.dumps(dict(calibration_cutoffs=cutoffs.tolist(),groups=rows),indent=2),flush=True)
