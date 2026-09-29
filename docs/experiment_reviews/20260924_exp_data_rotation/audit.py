"""Read-only full-source direction audit. Right = clockwise viewed from dome."""
from pathlib import Path
import json, time, pickle
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

OUT = Path(__file__).resolve().parent
SOURCE = Path('/home/carus/Data/exp_data')
manifest = json.loads((SOURCE/'manifest.json').read_text())
sub = Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/data/sim_hand_10k_seed42/subset_manifest.json')
subset_ids = np.array([e['episode_id'] for e in json.loads(sub.read_text())['episodes']]) if sub.exists() else np.array([], dtype=int)
cache = SOURCE/'exp_data_mmap'
retained_ids = np.load(cache/'episode_ids.npy')
train_ids = retained_ids[~np.load(cache/'val_mask.npy')]
EP = {}
counts = {name: dict(rows=0, forward=0, reverse=0, negative_skip=0,
                    speed_counts={str(t):[0,0,0] for t in [1,5,10,30]},
                    right_speed_sum=0., left_speed_sum=0., adjacent_pairs=0,
                    right_angle_sum_deg=0.,left_angle_sum_deg=0.)
          for name in ['all','mmap_train','guide_10k']}
last_by_episode = {}
start = time.time()
for si, shard in enumerate(manifest['shards']):
    with h5py.File(SOURCE/shard['path'],'r') as f:
        ep=f['index/episode_id'][:]; step=f['index/step'][:]
        direction=f['index/traj_direction'][:]; skip=f['index/skip_steps'][:]
        q=f['object/quaternion_world'][:]; w=f['object/angular_velocity_world'][:]
        wrist=f['robot/wrist_pose_world'][:,3:7]
    # Shards interleave blocks from many episodes. Reassemble temporal order
    # per episode and preserve each episode's last sample across shards.
    order=np.lexsort((step,ep))
    ep,step,direction,skip,q,w,wrist=[x[order] for x in [ep,step,direction,skip,q,w,wrist]]
    # Recorded wrist is fixed within a rollout. Relative quaternions also
    # remove any wrist motion from the finite-difference rotation metric.
    objrot=R.from_quat(q)
    right_speed=-np.degrees(objrot.inv().apply(w)[:,1])
    rel=R.from_quat(wrist).inv()*objrot
    relq=rel.as_quat()
    prevq=np.concatenate([relq[:1],relq[:-1]])
    prevep=np.r_[ep[0],ep[:-1]]
    prevstep=np.r_[-999,step[:-1]]
    firsts=np.r_[0,np.flatnonzero(ep[1:]!=ep[:-1])+1]
    lasts=np.r_[firsts[1:]-1,len(ep)-1]
    for j,k in zip(firsts,lasts):
        previous=last_by_episode.get(int(ep[j]))
        if previous is not None:
            prevep[j]=ep[j];prevstep[j]=previous[0];prevq[j]=previous[1]
        last_by_episode[int(ep[j])]=(step[k],relq[k].copy())
    valid=(ep==prevep)&(step==prevstep+1)
    angle=np.zeros(len(ep))
    angle[valid]=-np.degrees((R.from_quat(prevq[valid]).inv()*R.from_quat(relq[valid])).as_rotvec()[:,1])
    masks={'all':np.ones(len(ep),bool),'mmap_train':np.isin(ep,train_ids),'guide_10k':np.isin(ep,subset_ids)}
    for name,mask in masks.items():
        c=counts[name]; s=right_speed[mask]; a=angle[mask&valid]
        c['rows']+=int(mask.sum());c['forward']+=int(((direction==1)&mask).sum());c['reverse']+=int(((direction==-1)&mask).sum());c['negative_skip']+=int(((skip<0)&mask).sum())
        for threshold in [1,5,10,30]:
            ns=[int((s>threshold).sum()),int((s < -threshold).sum()),int((np.abs(s)<=threshold).sum())]
            c['speed_counts'][str(threshold)]=[x+y for x,y in zip(c['speed_counts'][str(threshold)],ns)]
        c['right_speed_sum']+=float(np.maximum(s,0).sum());c['left_speed_sum']+=float(np.maximum(-s,0).sum())
        c['adjacent_pairs']+=len(a);c['right_angle_sum_deg']+=float(np.maximum(a,0).sum());c['left_angle_sum_deg']+=float(np.maximum(-a,0).sum())
    unique, inv=np.unique(ep,return_inverse=True)
    aggregate=[np.bincount(inv,weights=x,minlength=len(unique)) for x in [np.ones(len(ep)),direction==1,direction==-1,angle,np.abs(angle),right_speed,np.maximum(right_speed,0),np.maximum(-right_speed,0)]]
    for j,e in enumerate(unique):
        old=EP.setdefault(int(e),np.zeros(len(aggregate)))
        old+=np.array([x[j] for x in aggregate])
    if si%10==0 or si+1==len(manifest['shards']):
        print(json.dumps(dict(shards=si+1,total=len(manifest['shards']),seconds=round(time.time()-start,1),counts=counts['all'])),flush=True)

refs=[]
for i in range(150):
    path=Path('/home/carus/Program/dex-controller/data/NOKOV-v3/data/bulb2')/f'{i:03d}.h5'
    with h5py.File(path,'r') as f:
        obj=R.from_matrix(f['object/6dpose'][:,:3,:3])
    # Native loader uses optimized wrist, not raw motion-capture wrist.
    # Sample original 30Hz frames; the common mujoco2gym transform cancels.
    retarget=Path('/home/carus/Program/dex-controller/data/retargeting/NOKOV-v3/mano2sharpa_rh/bulb2')/f'{i:03d}.pkl'
    with retarget.open('rb') as handle:
        wrist=R.from_rotvec(pickle.load(handle)['opt_wrist_rot'][::2])
    assert len(obj)==len(wrist)
    rel=wrist.inv()*obj
    a=-np.degrees((rel[:-1].inv()*rel[1:]).as_rotvec()[:,1])
    refs.append(dict(demo=i,frames=len(rel),net_right_deg=float(a.sum()),right_deg=float(np.maximum(a,0).sum()),left_deg=float(np.maximum(-a,0).sum())))

ep_rows=[]
for e,v in sorted(EP.items()):
    ep_rows.append(dict(episode_id=e,rows=int(v[0]),forward=int(v[1]),reverse=int(v[2]),net_right_deg=float(v[3]),total_absolute_axial_deg=float(v[4]),right_speed_sum=float(v[6]),left_speed_sum=float(v[7])))
for name,ids in [('all',set(EP)),('mmap_train',set(map(int,train_ids))),('guide_10k',set(map(int,subset_ids)))]:
    rows=[r for r in ep_rows if r['episode_id'] in ids]
    counts[name]['episodes']=len(rows)
    counts[name]['episodes_with_both_playback_directions']=sum(r['forward']>0 and r['reverse']>0 for r in rows)
    counts[name]['episode_net_counts']={str(t):{'right':sum(r['net_right_deg']>t for r in rows),'left':sum(r['net_right_deg'] < -t for r in rows),'small':sum(abs(r['net_right_deg'])<=t for r in rows)} for t in [0,10,30,90]}
    counts[name]['episode_forward_only']=sum(r['reverse']==0 for r in rows)
    counts[name]['episode_reverse_only']=sum(r['forward']==0 for r in rows)
result=dict(source=str(SOURCE),manifest_transitions=manifest['total_transitions'],shards=len(manifest['shards']),
    convention='Right/clockwise viewed from dome (+local Y) toward thread (-local Y). Speed is negative object-local Y component; angle is summed negative body-Y rotation-vector increments of object relative to wrist. All rows, no success/contact filtering.',
    speed_count_order=['right','left','slow'],speed_threshold_unit='deg/s',counts=counts,
    references_150=dict(episodes=150,net_counts={str(t):{'right':sum(r['net_right_deg']>t for r in refs),'left':sum(r['net_right_deg'] < -t for r in refs),'small':sum(abs(r['net_right_deg'])<=t for r in refs)} for t in [0,10,30,90]},right_deg=sum(r['right_deg'] for r in refs),left_deg=sum(r['left_deg'] for r in refs)),
    pair_coverage=dict(expected=counts['all']['rows']-len(EP),actual=counts['all']['adjacent_pairs']),elapsed_s=time.time()-start)
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'episodes.json').write_text(json.dumps(ep_rows)+'\n')
(OUT/'references.json').write_text(json.dumps(refs,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
