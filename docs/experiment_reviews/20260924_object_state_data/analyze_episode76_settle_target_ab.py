"""Validate isolation and summarize the fixed-target preload diagnostic."""

import json
import numpy as np
from scipy.spatial.transform import Rotation

from compare_corrected_rollouts import PROTOCOL
from random4_geometry import metrics
from run_episode76_settle_target_ab import OUT, ROOT


def main():
    baseline = ROOT / 'qualified_comparison/episode_76/direct_m044_mu11'
    original = np.load(baseline / 'initial_state.npz')
    source = np.load('/home/carus/Data/Object_state_data/episode_76/state.npy')[114, 9:]
    reference = np.load('/home/carus/Data/Object_state_data/episode_76/action.npy')[114, 9:]
    names = json.loads((ROOT.parent / 'reference/initial_state.json').read_text())['hand_joint_names']
    results = {}
    for mode in ('qpos', 'reference_action'):
        folder = OUT / mode
        ini = np.load(folder / 'initial_state.npz')
        changed = [k for k in original.files if not np.array_equal(ini[k], original[k])]
        assert set(changed) <= {'target', 'observation'}, changed
        assert np.array_equal(ini['q'][0], source)
        summary = json.loads((folder / 'summary.json').read_text())
        trace = json.loads((folder / 'trace.json').read_text())
        assert summary['native_protocol'] == PROTOCOL
        assert summary['steps'] == summary['intended_steps'] == dict(settle=60, action=0, hold=0)
        command = np.array([r['command'] for r in trace])
        executed = np.array([r['executed_target'] for r in trace])
        assert np.max(abs(command-command[0])) == 0
        assert np.max(abs(executed-executed[0])) == 0
        assert np.max(abs(executed-command)) < 1e-6
        assert np.max(abs(command[0]-(source if mode=='qpos' else reference))) < .004
        if mode == 'qpos':
            old = json.loads((baseline / 'trace.json').read_text())[:60]
            assert all(a[k] == b[k] for a,b in zip(trace,old)
                       for k in ('q','command','object_pose','native_failure'))
        geometry = metrics(folder, all_frames=False)
        pos = np.array([r['object_pose'][:3] for r in trace])
        rot = Rotation.from_quat(np.array([r['object_pose'][3:] for r in trace]))
        vel = np.array([r['object_velocity'] for r in trace])
        qd = np.array([r['qd'] for r in trace])
        q = np.array([r['q'] for r in trace])
        entries = {}
        for key,index in [('first_step',0),('last_step',59)]:
            delta = q[index]-source
            entries[key] = dict(object_shift_mm=1000*trace[index]['displacement_from_import_m'],
                object_rotation_deg=trace[index]['rotation_from_import_deg'],
                q_rmse_rad=float(np.sqrt(np.mean(delta**2))),
                q_max_abs_rad=float(abs(delta).max()),
                index_PIP_delta_rad=float(delta[2]), thumb_CMC_AA_delta_rad=float(delta[18]),
                q_delta_rad=dict(zip(names,delta.tolist())),
                linear_speed_m_s=float(np.linalg.norm(vel[index,:3])),
                angular_speed_rad_s=float(np.linalg.norm(vel[index,3:])))
        entries['last30'] = dict(
            linear_speed_median_m_s=float(np.median(np.linalg.norm(vel[-30:,:3],axis=1))),
            angular_speed_median_rad_s=float(np.median(np.linalg.norm(vel[-30:,3:],axis=1))),
            finite_difference_linear_speed_median_m_s=float(np.median(np.linalg.norm(np.diff(pos[-30:],axis=0),axis=1)*30)),
            finite_difference_angular_speed_median_rad_s=float(np.median((rot[-30:-1].inv()*rot[-29:]).magnitude()*30)),
            qd_rms_rad_s=float(np.sqrt(np.mean(qd[-30:]**2))))
        entries['geometry'] = geometry['static']
        entries['validation'] = dict(changed_initial_fields=changed, fixed_command=True,
                                     native_protocol_exact=True, first_native_failure=summary['first_native_failure'])
        results[mode] = entries
    (OUT/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
    for mode,r in results.items():
        print(mode,json.dumps({k:v for k,v in r.items() if k not in ('first_step','last_step')}),flush=True)
        for key in ('first_step','last_step'):
            print(key,{k:v for k,v in r[key].items() if k!='q_delta_rad'},flush=True)


if __name__ == '__main__':
    main()
