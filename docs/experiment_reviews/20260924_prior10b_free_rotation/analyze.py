from pathlib import Path
import argparse,json
import numpy as np

OUT=Path(__file__).resolve().parent

def aggregate(speed,delta):
    out={'frames':int(speed.size),'right_angle_deg':float(np.maximum(delta,0).sum()),'left_angle_deg':float(np.maximum(-delta,0).sum()),'speed_thresholds':{}}
    for t in [1,5,10,30]:
        r=int((speed>t).sum());l=int((speed < -t).sum());z=int((abs(speed)<=t).sum())
        out['speed_thresholds'][str(t)]=dict(right=r,left=l,slow=z,left_fraction_of_moving=None if r+l==0 else l/(r+l))
    interval_speed=np.asarray(delta,dtype=np.float64)*30.0
    out['interval_average_speed_thresholds']={}
    for t in [1,5,10,30]:
        right=int((interval_speed>t).sum());left=int((interval_speed < -t).sum());slow=int((abs(interval_speed)<=t).sum())
        out['interval_average_speed_thresholds'][str(t)]=dict(right=right,left=left,slow=slow,left_fraction_of_moving=None if right+left==0 else left/(right+left))
    return out

def classify(rows):
    out={'episodes':len(rows)}
    for t in [0,10,30,90,180]:
        r=sum(e['net_right_deg']>t for e in rows);l=sum(e['net_right_deg'] < -t for e in rows);z=len(rows)-r-l
        out[str(t)]=dict(right=r,left=l,small=z,left_fraction_of_directional=None if r+l==0 else l/(r+l))
    return out

def main():
    seeds=[];episodes=[];first30=[];vectors={k:[[],[]] for k in ['all','first_episode','first_episode_first30s']}
    for seed in [42,123,2026]:
        run=OUT/f'seed{seed}';paths=sorted(run.glob('trajectory_*.npz'))
        if not paths:continue
        fields=['global_step','episode','episode_step','right_increment_deg','right_speed_deg_s','done','failure','success','contact_force_norm']
        chunks={k:[] for k in fields}
        for path in paths:
            with np.load(path) as z:
                for k in fields:chunks[k].append(z[k])
        a={k:np.concatenate(v) for k,v in chunks.items()}
        assert np.array_equal(a['global_step'],np.arange(1,len(a['global_step'])+1))
        rows=[];rows30=[];N=a['episode'].shape[1]
        native={}
        if (run/'episodes.jsonl').exists():
            for line in (run/'episodes.jsonl').read_text().splitlines():
                e=json.loads(line);native[e['env'],e['episode']]=e
        for env in range(N):
            for ep in np.unique(a['episode'][:,env]):
                mask=a['episode'][:,env]==ep;d=a['right_increment_deg'][mask,env].astype(float);sp=a['right_speed_deg_s'][mask,env]
                steps=a['episode_step'][mask,env];assert np.array_equal(steps,np.arange(1,len(steps)+1))
                ended=bool(a['done'][mask,env][-1]);reason=native[(env,int(ep))]['reason'] if ended else ('observation_cap' if (run/'RUN_COMPLETE').exists() else 'ongoing')
                if ended:
                    assert (env,int(ep)) in native
                    assert native[env,int(ep)]['length']==len(d)
                row=dict(seed=seed,env=env,episode=int(ep),steps=len(d),duration_s=len(d)/30,reason=reason,ended=ended,net_right_deg=float(d.sum()),right_angle_deg=float(np.maximum(d,0).sum()),left_angle_deg=float(np.maximum(-d,0).sum()),right_frames=int((sp>5).sum()),left_frames=int((sp < -5).sum()),slow_frames=int((abs(sp)<=5).sum()))
                rows.append(row)
                if ep==0:
                    d30=d[:900];rows30.append(dict(seed=seed,env=env,steps=len(d30),net_right_deg=float(d30.sum()),ended_before_30s=ended and len(d)<900))
        seedmetrics=dict(seed=seed,steps=int(a['global_step'][-1]),complete=(run/'RUN_COMPLETE').exists(),episodes=classify(rows),first_episodes=classify([r for r in rows if r['episode']==0]),first_30s=classify(rows30))
        masks={'all':np.ones(a['episode'].shape,bool),'first_episode':a['episode']==0,'first_episode_first30s':(a['episode']==0)&(a['episode_step']<=900)}
        for name,mask in masks.items():
            sp=a['right_speed_deg_s'][mask];d=a['right_increment_deg'][mask];vectors[name][0].append(sp);vectors[name][1].append(d)
            seedmetrics[name+'_frames']=aggregate(sp,d)
        seedmetrics['time_bins']=[]
        for start in range(0,len(a['global_step']),1200):
            end=min(start+1200,len(a['global_step']))
            seedmetrics['time_bins'].append(dict(start_s=start/30,end_s=end/30,**aggregate(a['right_speed_deg_s'][start:end].ravel(),a['right_increment_deg'][start:end].ravel())))
        seeds.append(seedmetrics);episodes.extend(rows);first30.extend(rows30)
    result=dict(complete=len(seeds)==3 and all(r['complete'] for r in seeds),seeds=seeds,
        all_episodes=classify(episodes),first_episodes=classify([r for r in episodes if r['episode']==0]),first_30s=classify(first30),
        first_episode_duration_strata={str(t):classify([r for r in episodes if r['episode']==0 and r['steps']>=int(t*30)]) for t in [5,30,100,400]},
        total_native_failures=sum(r['reason']=='failure' for r in episodes),censored_episodes=sum(not r['ended'] for r in episodes),
        frame_metrics={name:aggregate(np.concatenate(v[0]),np.concatenate(v[1])) for name,v in vectors.items() if v[0]},
        notes=['All episodes retained; no initialization screening.','Primary frame metric uses instantaneous axial speed >5 deg/s; net angle integrates relative-quaternion increments.','Native failure is a proxy, not independently confirmed drop. Recorded fingertip-link contact forces are zero and unusable for contact filtering; no contact-based success claim.','Partial/censored net angles are observation endpoints, not completed rotations.','Repeated steps/episodes in one environment are not independent trials. First-episode summaries avoid extra weighting from frequent resets.'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'rotation_episodes.json').write_text(json.dumps(episodes,indent=2)+'\n');(OUT/'first30_episodes.json').write_text(json.dumps(first30,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='seeds'},indent=2))
if __name__=='__main__':main()
