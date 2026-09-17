"""Read-only audit of the source dataset; outputs are written beside this script.

Run with numpy, pyarrow, pillow, matplotlib installed, e.g.:
PYTHONPATH=/tmp/bulb_dataset_analysis_deps /home/bighand/miniconda3/envs/tianji/bin/python analyze.py
"""
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path('/home/bighand/wangtianyu/projects/TacMP/data/realworld_bulb_sft_260909')
OUT = Path(__file__).resolve().parent


def stats(a):
    a = np.asarray(a).astype(float).ravel()
    a = a[np.isfinite(a)]
    return dict(zip(['min', 'p01', 'p05', 'median', 'p95', 'p99', 'max'],
                    np.percentile(a, [0, 1, 5, 50, 95, 99, 100]).tolist())) | {'mean': float(a.mean()), 'std': float(a.std()), 'n': len(a)} if len(a) else {}


def array(table, key):
    a = table[key].combine_chunks()
    shape = [len(a)]
    while hasattr(a, 'flatten') and not str(a.type).startswith('struct'):
        lengths = np.asarray(a.value_lengths())
        if not len(lengths) or not (lengths == lengths[0]).all():
            raise ValueError(f'{key}: ragged shape {np.unique(lengths)}')
        shape.append(int(lengths[0]))
        a = a.flatten()
    return a.to_numpy(zero_copy_only=False).reshape(shape)


def write_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    episodes, incomplete, anomalies = [], [], []
    arrays = defaultdict(list)
    compressed = Counter()
    contacts = Counter()
    contact_shapes = Counter()
    segments, tasks, schemas = Counter(), Counter(), Counter()
    before = {str(p.relative_to(ROOT)): (p.stat().st_size, p.stat().st_mtime_ns) for p in ROOT.rglob('*') if p.is_file()}
    for d in sorted((ROOT / 'rank_0').glob('id_*'), key=lambda p: int(p.name[3:])):
        info = json.loads((d / 'meta/info.json').read_text())
        files = sorted(d.glob('data/**/*.parquet'))
        if not files:
            pngs = defaultdict(list)
            for p in d.glob('images/**/*.png'):
                pngs[p.relative_to(d).parts[1]].append(int(p.stem.split('_')[-1]))
            pending = []
            for p in d.glob('timing/**/*.parquet'):
                q = pq.ParquetFile(p)
                pending.append({'path': str(p.relative_to(d)), 'rows': q.metadata.num_rows})
            incomplete.append({'id': d.name, 'meta_frames': info['total_frames'], 'bytes': sum(p.stat().st_size for p in d.rglob('*') if p.is_file()),
                               'png': {k: {'count': len(v), 'first': min(v), 'last': max(v), 'missing': sorted(set(range(max(v)+1))-set(v))} for k,v in pngs.items()}, 'pending': pending})
            continue
        assert len(files) == 1, files
        pf = pq.ParquetFile(files[0])
        schemas[str(pf.schema_arrow.remove_metadata())] += 1
        for r in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(r)
            for c in range(rg.num_columns):
                col = rg.column(c)
                compressed[col.path_in_schema.split('.')[0]] += col.total_compressed_size
        keys = [k for k in pf.schema_arrow.names if k not in ('image','extra_view_image','tactile_deform')]
        table = pf.read(columns=keys)
        a = {k: array(table,k) for k in keys if k != 'tactile_contact_points'}
        n = table.num_rows
        timing = pq.read_table(d / 'timing/episode_000000.parquet')
        t = {k: array(timing,k) for k in timing.column_names}
        epmeta = [json.loads(line) for line in (d/'meta/episodes.jsonl').read_text().splitlines()]
        tasks.update(json.loads(line)['task'] for line in (d/'meta/tasks.jsonl').read_text().splitlines())
        errors = []
        if info['total_frames'] != n or epmeta[0]['length'] != n: errors.append('metadata length mismatch')
        if timing.num_rows != n: errors.append('timing length mismatch')
        for k in ('frame_index','index'):
            if not np.array_equal(a[k], np.arange(n)): errors.append(f'{k} is not consecutive')
        for k in ('frame_index','episode_index','timestamp'):
            if not np.array_equal(a[k],t[k]): errors.append(f'timing {k} mismatch')
        if not (a['episode_index'] == 0).all(): errors.append('episode_index != 0')
        if not (a['task_index'] == 0).all(): errors.append('task_index != 0')
        if not np.array_equal(np.flatnonzero(a['done']), [n-1]): errors.append('done not only at last frame')
        nonfinite = {k: int((~np.isfinite(v)).sum()) for k,v in a.items()}
        if any(nonfinite.values()): errors.append(f'nonfinite {nonfinite}')
        nulls = {k: table[k].null_count for k in keys}
        if any(nulls.values()): errors.append(f'nulls {nulls}')
        for v in table['tactile_contact_points'].to_pylist():
            contacts[v] += 1
            try:
                obj = json.loads(v)
                contact_shapes[str([len(c) for c in obj])] += 1
            except (TypeError, ValueError): errors.append('malformed tactile_contact_points')
        segments.update(a['segment_id'].tolist())
        dt = np.diff(t['sample_monotonic_ns']) / 1e6
        age = (t['sample_monotonic_ns'][:,None] - t['tactile_receive_monotonic_ns']) / 1e6
        intersensor = np.ptp(t['tactile_receive_monotonic_ns'], axis=1) / 1e6
        fid_delta = np.diff(t['tactile_frame_ids'], axis=0)
        force = np.linalg.norm(a['tactile_f6'][:,:,:3],axis=2)
        xyz_step = np.linalg.norm(np.diff(a['state'][:,:3],axis=0),axis=1)
        action_xyz_step = np.linalg.norm(np.diff(a['actions'][:,:3],axis=0),axis=1)
        hand_step = np.max(np.abs(np.diff(a['actions'][:,9:],axis=0)),axis=1)
        xyz_error = np.linalg.norm(a['actions'][:,:3]-a['state'][:,:3],axis=1)
        hand_error = np.mean(np.abs(a['actions'][:,9:]-a['state'][:,9:]),axis=1)
        if (dt <= 0).any(): errors.append('nonmonotonic acquisition time')
        row = {'id': d.name,'frames': n,'nominal_seconds':n/info['fps'], 'elapsed_seconds': (t['sample_monotonic_ns'][-1]-t['sample_monotonic_ns'][0])/1e9,
               'effective_hz': 1000/dt.mean(), 'dt_p99_ms':float(np.percentile(dt,99)), 'dt_max_ms':float(dt.max()),'dt_gt50ms':int((dt>50).sum()),
               'state_step_max': float(xyz_step.max()), 'action_step_max':float(action_xyz_step.max()), 'hand_action_step_max':float(hand_step.max()),
               'xyz_tracking_error_mean':float(xyz_error.mean()),'hand_tracking_error_mean':float(hand_error.mean()), 'first_hand_tracking_error': float(hand_error[0]),
               'force_max':float(force.max()), 'tactile_age_max_ms':float(age.max()),'tactile_repeat_sensor_samples':int((fid_delta==0).sum()),
               'tactile_rewind_sensor_samples':int((fid_delta<0).sum()),'tactile_frame_id_gap_samples':int((fid_delta>1).sum()),
               'tactile_any_zero_frames':int(np.any(np.all(a['tactile_f6']==0,axis=2),axis=1).sum()),
               'state_unchanged_pairs':int(np.all(np.diff(a['state'],axis=0)==0,axis=1).sum()),
               'action_unchanged_pairs':int(np.all(np.diff(a['actions'],axis=0)==0,axis=1).sum()),
               'start_time':datetime.fromtimestamp(t['sample_time_ns'][0]/1e9,timezone(timedelta(hours=8))).isoformat(),
               'end_time':datetime.fromtimestamp(t['sample_time_ns'][-1]/1e9,timezone(timedelta(hours=8))).isoformat(),
               'data_bytes':files[0].stat().st_size,'errors':'; '.join(errors)}
        episodes.append(row)
        if errors: anomalies.append({'id':d.name,'errors':errors})
        for k in ('state','actions','tactile_f6','timestamp','done','segment_id'): arrays[k].append(a[k])
        for k,v in {'episode_id':np.full(n,int(d.name[3:])), 'frame':np.arange(n),'dt_ms':dt,'tactile_age_ms':age,
                    'tactile_skew_ms':intersensor,'tactile_frame_id_delta':fid_delta,'sample_monotonic_ns':t['sample_monotonic_ns'],
                    'obs_duration_ms':(t['observation_end_ns']-t['observation_start_ns'])/1e6,
                    'deadline_lateness_ms':(t['observation_start_ns']-t['observation_deadline_ns'])/1e6,
                    'state_xyz_step':xyz_step,'action_xyz_step':action_xyz_step,'hand_action_step':hand_step,
                    'xyz_tracking_error':xyz_error,'hand_tracking_error':hand_error}.items(): arrays[k].append(v)
        for k in ('state','actions'):
            v=a[k][:,3:9].reshape(-1,2,3)
            arrays[k+'_rot_norm_error'].append(np.abs(np.linalg.norm(v,axis=-1)-1))
            arrays[k+'_rot_dot'].append(np.abs((v[:,0]*v[:,1]).sum(-1)))
        if len(episodes)%15 == 0: print(f'low-dimensional audit {len(episodes)}/75',flush=True)
    all_a = {k:np.concatenate(v) for k,v in arrays.items()}
    np.savez_compressed(OUT/'audit_arrays.npz',**all_a)
    write_csv(OUT/'episodes.csv',episodes)
    dimensions=[]
    for k in ('state','actions','tactile_f6'):
        v=all_a[k].reshape(len(all_a[k]),-1)
        for j in range(v.shape[1]): dimensions.append({'field':k,'dim':j,**stats(v[:,j]),'nonfinite':int((~np.isfinite(v[:,j])).sum())})
    write_csv(OUT/'dimensions.csv',dimensions)
    force=np.linalg.norm(all_a['tactile_f6'][:,:,:3],axis=2)
    timing_summary={k:stats(all_a[k]) for k in ['dt_ms','tactile_age_ms','tactile_skew_ms','obs_duration_ms','deadline_lateness_ms']}
    timing_summary['tactile_age_by_sensor']=[stats(all_a['tactile_age_ms'][:,i]) for i in range(5)]
    timing_summary['frame_id_delta_counts']=dict(zip(*[x.tolist() for x in np.unique(all_a['tactile_frame_id_delta'],return_counts=True)]))
    summary={'source':str(ROOT),'audit_time':datetime.now(timezone.utc).isoformat(),'file_count':len(before),'total_bytes':sum(v[0] for v in before.values()),
             'complete_episodes':len(episodes),'frames':len(all_a['state']),'nominal_seconds':sum(e['nominal_seconds'] for e in episodes),'fps':30,
             'length_frames':stats([e['frames'] for e in episodes]),'effective_hz':stats([e['effective_hz'] for e in episodes]),
             'recording_start':min(e['start_time'] for e in episodes),'recording_end':max(e['end_time'] for e in episodes),
             'incomplete':incomplete,'tasks':dict(tasks),'segment_counts':dict(segments),
             'contacts':{'unique_strings':len(contacts),'all_empty_frames':contacts['[[], [], [], [], []]'],
                         'nonempty_frames':len(all_a['state'])-contacts['[[], [], [], [], []]']},'contact_shapes':dict(contact_shapes),
             'anomalies':anomalies,'schema_variants':len(schemas),'parquet_compressed_bytes':dict(compressed),'timing':timing_summary,
             'numeric':{k:stats(all_a[k]) for k in ('state_xyz_step','action_xyz_step','hand_action_step','xyz_tracking_error','hand_tracking_error','state_rot_norm_error','state_rot_dot','actions_rot_norm_error','actions_rot_dot')},
             'force_norm_by_sensor':[stats(force[:,i]) for i in range(5)],'force_norm_gt1_fraction':(force>1).mean(axis=0).tolist(),
             'force_norm_gt5_fraction':(force>5).mean(axis=0).tolist(),'force_norm_gt10_fraction':(force>10).mean(axis=0).tolist()}
    after = {str(p.relative_to(ROOT)): (p.stat().st_size,p.stat().st_mtime_ns) for p in ROOT.rglob('*') if p.is_file()}
    summary['source_file_stats_unchanged'] = before == after
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
