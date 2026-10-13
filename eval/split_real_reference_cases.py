"""Split a multi-placement batch into three-domain groups for existing reports."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('batch',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    meta=json.loads((args.batch/'run_metadata.json').read_text());cases=meta['placements'];n=len(cases)
    groups={}
    for i,case in enumerate(cases):groups.setdefault(case['group'],[]).append(i)
    tr=np.load(args.batch/'trajectory.npz',allow_pickle=False);init=np.load(args.batch/'initial_state.npz',allow_pickle=False)
    checks={}
    for label,ids in groups.items():
        assert len(ids)==3
        out=args.output/label;out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'trajectory.npz',**{k:(tr[k][:,ids] if tr[k].ndim>=2 and tr[k].shape[1]==n else tr[k]) for k in tr.files})
        np.savez_compressed(out/'initial_state.npz',**{k:(init[k][ids] if init[k].ndim>=1 and init[k].shape[0]==n else init[k]) for k in init.files})
        m=dict(meta)
        for key in ['placements','prior_noise_seeds','native_failure_during_settle','native_failure_anytime','settle_drift_m','settled_contacting_tip_count']:
            if m.get(key) is not None:m[key]=[m[key][i] for i in ids]
        m.update(placement=m['placements'][0],source_batch=str(args.batch),source_environment_indices=ids,pose_group=label)
        (out/'run_metadata.json').write_text(json.dumps(m,indent=2)+'\n')
        for local,source in enumerate(ids):
            for stem,suffix in [('env','.mp4')]:
                target=out/('env%02d'%local+suffix)
                if not target.exists():target.symlink_to((args.batch/('env%02d'%source+suffix)).resolve())
            for phase in ['settle_start','settled','reference_end','final']:
                src=args.batch/('%s_env%02d.jpg'%(phase,source));dest=out/('%s_env%02d.jpg'%(phase,local))
                if src.exists() and not dest.exists():dest.symlink_to(src.resolve())
        checks[label]=dict(original_environment_indices=ids,actual_mass_kg=init['object_mass'][ids].tolist(),
            actual_object_friction=init['object_friction'][ids].tolist())
        assert np.allclose(init['object_mass'][ids],.15,atol=1e-7)
        assert np.allclose(init['object_friction'][ids],2.2,atol=1e-6)
        print(out)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'split_checks.json').write_text(json.dumps(checks,indent=2)+'\n')

if __name__=='__main__':main()
