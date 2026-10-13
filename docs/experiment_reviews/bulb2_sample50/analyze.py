"""Offline, single-threaded audit of 50 systematically sampled bulb2 demos."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import json
import pickle
from pathlib import Path
import numpy as np
import h5py
import trimesh
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/carus/Program/dex-controller')
OUT = Path(__file__).resolve().parent
SAMPLE = list(range(0, 150, 3))
mesh = trimesh.load(ROOT / 'data/NOKOV-v3/object/mesh/bulb2.obj', process=False)
verts = np.asarray(mesh.vertices)
center = (verts.min(0) + verts.max(0)) / 2
surface = verts[np.linspace(0, len(verts)-1, min(700,len(verts))).astype(int)]
tree = cKDTree(verts)
colors = ['#d64b43', '#319751', '#4273ca', '#d19b26', '#9749ae']
clips = {}
rows = []
for idx in SAMPLE:
    with h5py.File(ROOT / f'data/NOKOV-v3/data/bulb2/{idx:03d}.h5') as f:
        obj = f['object/6dpose'][:]
        joints = f['joint_position'][:]
        stamp = f['timestamp'][:]
    with open(ROOT / f'data/retargeting/NOKOV-v3/mano2sharpa_rh/bulb2/{idx:03d}.pkl', 'rb') as f:
        robot = pickle.load(f)
    assert obj.shape[0] == joints.shape[0] and np.isfinite(obj).all() and np.isfinite(joints).all()
    rotation = Rotation.from_matrix(obj[:,:3,:3])
    angle = np.rad2deg((rotation[0].inv()*rotation).magnitude())
    objcenter = np.einsum('tij,j->ti',obj[:,:3,:3],center) + obj[:,:3,3]
    translation = np.linalg.norm(objcenter-objcenter[0],axis=1)*100
    tips = joints[:,[7,11,15,19,23]]
    localtips = np.einsum('tji,tkj->tki',obj[:,:3,:3], tips-obj[:,None,:3,3])
    distances = tree.query(localtips.reshape(-1,3))[0].reshape(len(obj),5)*100
    aperture = np.linalg.norm(tips[:,0]-tips[:,1],axis=1)*100
    stride = np.r_[np.arange(0,len(obj),15),len(obj)-1]
    angular_path = np.rad2deg((rotation[stride[:-1]].inv()*rotation[stride[1:]]).magnitude()).sum()
    # Heuristic visual-review cue, not a measured contact/regrasp label.
    near = distances[stride] < 2
    pattern_changes = (near[1:]!=near[:-1]).sum()
    row = dict(index=idx,frames=len(obj),duration_nominal_s=(len(obj)-1)/30,
        max_rotation_from_start_deg=float(angle.max()),angular_path_0_5s_deg=float(angular_path),
        object_center_excursion_cm=float(translation.max()),
        object_center_bbox_diagonal_cm=float(np.linalg.norm(np.ptp(objcenter,axis=0))*100),
        min_fingertip_surface_distance_start_cm=float(distances[0].min()),
        max_nearest_fingertip_surface_distance_cm=float(distances.min(axis=1).max()),
        thumb_index_aperture_range_cm=float(np.ptp(aperture)),
        geometric_near_pattern_changes=int(pattern_changes),
        pkl_wrist_translation_max_abs=float(np.abs(robot['opt_wrist_pos']).max()),
        pkl_wrist_rotvec_max_abs=float(np.abs(robot['opt_wrist_rot']).max()))
    rows.append(row)
    clips[idx]=(obj,joints,angle,translation,distances,aperture,objcenter)

(OUT/'metrics.json').write_text(json.dumps({'sample':SAMPLE,'method':'indices 000,003,...147; 30Hz original H5; all frames analyzed; five keyframes visualized per demo; distances approximate surface proximity, not contact labels','rows':rows},indent=2))
for page in range(5):
    ids=SAMPLE[page*10:(page+1)*10]
    fig=plt.figure(figsize=(16,24), facecolor='white')
    fig.suptitle(f'Bulb2 reference audit | sample {page*10+1}-{page*10+10}/50\nRaw hand landmarks + object mesh; fixed view, no physics. Red=thumb, green=index, blue=middle.',fontsize=15,y=.995)
    for rownum,idx in enumerate(ids):
        obj,joints,angle,translation,distances,aperture,objcenter=clips[idx]
        times=np.linspace(0,len(obj)-1,5).astype(int)
        allpos=np.concatenate([joints.reshape(-1,3),objcenter])
        mid=(allpos.min(0)+allpos.max(0))/2
        span=max(np.ptp(allpos,axis=0).max(),.20)*.57
        for col,t in enumerate(times):
            ax=fig.add_subplot(10,6,rownum*6+col+1,projection='3d')
            posed=surface@obj[t,:3,:3].T+obj[t,:3,3]
            ax.scatter(*posed.T,c='#8a9096',s=.7,alpha=.28,depthshade=False)
            for fi,c in enumerate(colors):
                js=np.r_[0,np.arange(4+fi*4,8+fi*4)]
                pts=joints[t,js];ax.plot(*pts.T,c=c,lw=1.4)
                ax.scatter(*pts[-1],c=c,s=8)
            ax.set(xlim=(mid[0]-span,mid[0]+span),ylim=(mid[1]-span,mid[1]+span),zlim=(mid[2]-span,mid[2]+span))
            ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-60);ax.set_axis_off()
            ax.set_title(f'{idx:03d} | {t/30:.1f}s',fontsize=9,pad=-2)
        ax=fig.add_subplot(10,6,rownum*6+6)
        seconds=np.arange(len(obj))/30
        ax.plot(seconds,angle,c='#326ca8',lw=.9,label='rotation deg')
        ax.set_ylim(0,190);ax.set_yticks([0,90,180]);ax.tick_params(labelsize=7)
        ax2=ax.twinx();ax2.plot(seconds,translation,c='#d47d29',lw=.9,label='translation cm')
        ax2.set_ylim(0,max(5,translation.max()*1.1));ax2.tick_params(labelsize=7,colors='#b46520')
        ax.set_title(f'{idx:03d}: {angle.max():.0f} deg / {translation.max():.1f} cm',fontsize=8)
        ax.set_xlabel('seconds',fontsize=7);ax.grid(alpha=.2)
    fig.subplots_adjust(left=.005,right=.975,top=.963,bottom=.02,wspace=.03,hspace=.26)
    fig.savefig(OUT/f'sheet_{page+1}.png',dpi=120);plt.close(fig)
    print('Rendered page',page+1,flush=True)

print(json.dumps({'n':len(rows),'rotation_deg_range':[min(r['max_rotation_from_start_deg'] for r in rows),max(r['max_rotation_from_start_deg'] for r in rows)],'translation_cm_range':[min(r['object_center_excursion_cm'] for r in rows),max(r['object_center_excursion_cm'] for r in rows)],'largest_translation':sorted(rows,key=lambda r:r['object_center_excursion_cm'],reverse=True)[:8],'largest_aperture':sorted(rows,key=lambda r:r['thumb_index_aperture_range_cm'],reverse=True)[:8]},indent=2),flush=True)
