"""Read-only analysis of archived paired traces; writes only a new diagnosis folder."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from reference_resampling import interpolate_large_jumps

P = Path(__file__).resolve().parent
ROOT = P / 'reference_turn_baseline_20260926'
OUT = ROOT / 'mass_scale_diagnostic_20260927'
CASES = ROOT / 'qualified_comparison'
NAMES = json.loads((P / 'reference/initial_state.json').read_text())['hand_joint_names']

def load(folder):
    summary = json.loads((folder/'summary.json').read_text())
    trace = json.loads((folder/'trace.json').read_text())
    retention = json.loads((folder/'retention.json').read_text())
    return summary, trace, retention

def spans(flags):
    edges = np.diff(np.r_[False, flags, False].astype(int))
    return [[int(a+1), int(b), int(b-a)] for a,b in zip(np.where(edges==1)[0], np.where(edges==-1)[0])]

def analyze(folder, reference):
    s,t,r = load(folder)
    a = [x for x in t if x['phase']=='action']
    geo = [x for x in r['frames'] if x['phase']=='action']
    end = r['first_separation']['control_step']-1
    cmd,q,actual = (np.array([x[k] for x in a]) for k in ('command','q','executed_target'))
    angle = np.array([x['vertical_error_deg'] for x in a])
    valid = np.array([x['mesh_table_clearance_m']>.08 and x['near_contact_link_count']>=2 and x['mesh_vertex_gap_m']<.008 and x['object_contact_force_norm_N']>.1 for x in geo])
    valid[end:] = False
    intervals = spans(valid & (angle<=30))
    def rmse(x): return float(np.sqrt(np.mean(x*x)))
    def window(n):
        return dict(n=n,cmd_ref_rmse=rmse(cmd[:n]-reference[:n]),q_cmd_rmse=rmse(q[:n]-actual[:n]),q_ref_rmse=rmse(q[:n]-reference[:n]),cmd_actual_max=float(abs(cmd[:n]-actual[:n]).max()),cmd_jump_rms=rmse(np.diff(cmd[:n],axis=0)),finger_cmd_ref_rmse={finger:rmse((cmd[:n]-reference[:n])[:,[i for i,name in enumerate(NAMES) if f'_{finger}_' in name]]) for finger in ('thumb','index','middle','ring','pinky')})
    result = dict(folder=str(folder),episode=s['source_episode'],mass=s['mass_kg'],mu=s['friction'],ddim=s['prior']['ddim'],scale=s['guidance_scale'],first_separation_control_step=end+1,first_separation_progress=r['first_separation']['reference_action_number'],stable_vertical_intervals=intervals,longest_vertical_steps=max([x[2] for x in intervals],default=0),first_30deg_step=next((i+1 for i in range(end) if angle[i]<=30),None),minimum_angle_before_separation=float(angle[:end].min()),windows={str(n):window(n) for n in (30,60,120)},settle_angle=t[59]['vertical_error_deg'],settle_position=t[59]['object_pose'][:3],settle_q=t[59]['q'],native_failure=s['first_native_failure'])
    if s['source_episode']==54:
        result['checkpoints']=[dict(control_step=i+1,vertical_deg=float(angle[i]),contacts=geo[i]['near_contact_links'],object_force_N=geo[i]['object_contact_force_norm_N'],cmd_ref_rmse=rmse(cmd[i]-reference[i]),q_cmd_rmse=rmse(q[i]-actual[i])) for i in (0,14,29,44,59,74,89,104,119,134,149)]
        lag_errors={lag:rmse(cmd[10:120]-reference[10+lag:120+lag]) for lag in range(-10,11)}
        result['command_reference_shift_diagnostic']=dict(control_steps=[11,120],rmse_by_reference_shift=lag_errors,best_shift=min(lag_errors,key=lag_errors.get))
        result['joint_errors_first60']={name:dict(cmd_ref_rmse=rmse((cmd[:60]-reference[:60])[:,j]),cmd_ref_mean=float((cmd[:60]-reference[:60])[:,j].mean()),q_cmd_rmse=rmse((q[:60]-actual[:60])[:,j])) for j,name in enumerate(NAMES)}
    return result,(s,t,r,cmd,q,actual,angle,geo)

def main():
    OUT.mkdir(exist_ok=True)
    results=[]; raw={}; audits=[];cross=[]
    for ep in (76,34,54,2):
        case=CASES/f'episode_{ep:02d}'
        reference,progress=interpolate_large_jumps(np.load(case/'reference_full.npz')['hand_target_rad'],.12)
        for mass,mu,suffix in ((.17,2.2,''),(.044,1.1,'_m044_mu11')):
            for ddim in (4,16):
                for scale in (25,50):
                    name='guide1_adaptive012' if not suffix and ddim==4 and scale==50 else f'guide1_adaptive012_ddim{ddim}_scale{scale}{suffix}'
                    folder=case/name
                    assert np.array_equal(np.load(folder/'reference_progress.npy'),progress)
                    result,data=analyze(folder,reference[0]);results.append(result);raw[ep,mass,ddim,scale]=data
                    s=data[0]
                    assert s['mass_kg']==mass and s['friction']==mu and s['execution_steps']==1 and s['prior']['guidance_steps']==2
        with np.load(case/'guide1_adaptive012_ddim4_scale25/initial_state.npz') as a,np.load(case/'guide1_adaptive012_ddim4_scale25_m044_mu11/initial_state.npz') as b:
            unequal=[k for k in a.files if not np.array_equal(a[k],b[k])]
            audits.append(dict(episode=ep,initial_fields_differ=unequal,inertia_mass_ratio=(a['object_inertia']/np.where(abs(b['object_inertia'])>1e-12,b['object_inertia'],np.nan)).tolist()))
        a=raw[ep,.17,4,25][1][59];b=raw[ep,.044,4,25][1][59]
        cross.append(dict(episode=ep,settle_position_difference_mm=float(np.linalg.norm(np.array(a['object_pose'][:3])-b['object_pose'][:3])*1000),settle_rotation_difference_deg=float(np.degrees((Rotation.from_quat(a['object_pose'][3:]).inv()*Rotation.from_quat(b['object_pose'][3:])).magnitude())),settle_q_difference_rmse=float(np.sqrt(np.mean((np.array(a['q'])-b['q'])**2))),cross_physics_cmd_rmse_first60={str(scale):float(np.sqrt(np.mean((raw[ep,.17,4,scale][3][:60]-raw[ep,.044,4,scale][3][:60])**2))) for scale in (25,50)}))
    # Avoid JSON NaN for structurally zero inertia off-diagonal elements.
    for audit in audits:
        audit['inertia_mass_ratio']=[[v if np.isfinite(v) else None for v in row] for row in np.array(audit['inertia_mass_ratio']).reshape(3,3)]
    (OUT/'archive_metrics.json').write_text(json.dumps(dict(results=results,audits=audits,cross_physics=cross),indent=2,allow_nan=False)+'\n')
    lines=['# Archived trace comparison','', 'All errors in radians; prefixes use identical physical step counts and precede separation in episode54.','', '|ep|mass|DDIM|scale|stable steps|sep step|cmd-ref first60|q-cmd first60|max command clipping first120|','|---|---|---|---|---|---|---|---|---|']
    for x in results:
        w=x['windows']['60'];lines.append(f"|{x['episode']}|{x['mass']}|{x['ddim']}|{x['scale']}|{x['longest_vertical_steps']}|{x['first_separation_control_step']}|{w['cmd_ref_rmse']:.5f}|{w['q_cmd_rmse']:.5f}|{x['windows']['120']['cmd_actual_max']:.6f}|")
    (OUT/'archive_metrics.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(14,10),sharex=True)
    for col,mass in enumerate((.17,.044)):
        case=CASES/'episode_54';ref,_=interpolate_large_jumps(np.load(case/'reference_full.npz')['hand_target_rad'],.12)
        for scale,color in ((25,'#067a85'),(50,'#c65a18')):
            s,t,r,cmd,q,actual,angle,geo=raw[54,mass,4,scale]
            end=r['first_separation']['control_step']-1;x=np.arange(1,end+1)
            axes[0,col].plot(x,angle[:end],label=f'scale {scale}',color=color)
            axes[1,col].plot(x,np.sqrt(np.mean((cmd[:end]-ref[0,:end])**2,axis=1)),color=color)
            axes[2,col].plot(x,np.sqrt(np.mean((q[:end]-actual[:end])**2,axis=1)),color=color)
        axes[0,col].axhline(30,color='gray',ls='--');axes[0,col].legend();axes[0,col].set_title('170 g / friction 2.2' if col==0 else '44 g / friction 1.1')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.set_xlim(1,215)
        axes[2,col].set_xlabel('Actual control step (30 Hz)')
    for ax,label in zip(axes[:,0],('Bulb angle from vertical (deg)','Command - reference RMSE (rad)','Measured q - executed target RMSE (rad)')):ax.set_ylabel(label)
    fig.suptitle('Episode 54: better joint tracking does not imply better object rotation\nDDIM4, guide2/exec1, identical >0.12 rad interpolation schedule')
    fig.tight_layout();fig.savefig(OUT/'episode54_tracking.png',dpi=160)
    print('\n'.join(lines))

if __name__=='__main__':main()
