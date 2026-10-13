"""Episode54 factorial physics diagnosis, with original direct eligibility gate."""
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
from random4_geometry import metrics
from compare_corrected_rollouts import PROTOCOL

P=Path(__file__).resolve().parent
ROOT=P/'reference_turn_baseline_20260926'
OUT=ROOT/'mass_scale_diagnostic_20260927'
CASE=ROOT/'qualified_comparison/episode_54'
REF=CASE/'reference_full.npz'
SIMPY='/home/carus/miniforge3/envs/decv2/bin/python'
DPPY='/home/carus/miniforge3/envs/dp/bin/python'
SIMENV=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PATH='/home/carus/miniforge3/envs/decv2/bin:'+os.environ['PATH'],PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
DPENV=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')

def run(name,mass,mu,scale=None):
    folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
    sock=Path(f'/tmp/mass_scale_diagnosis_{os.getpid()}.sock')
    simcmd=[SIMPY,str(P/'compare_corrected_rollouts.py'),'--mode','direct' if scale is None else 'guided','--out',str(folder),'--reference',str(REF),'--reference-id','0','--source-episode','54','--front-only','--audit-recording','--grasp-evidence','--object-mass-kg',str(mass),'--friction',str(mu)]
    servercmd=None
    if scale is not None:
        simcmd+=['--no-video']
        simcmd+=['--socket',str(sock),'--guidance-steps','2','--execution-steps','1','--guidance-scale',str(scale),'--reference-interpolation-threshold','0.12']
        servercmd=[DPPY,str(P/'server.py'),'--socket',str(sock),'--log',str(folder/'predictions.json'),'--checkpoint','/home/carus/data_usb/10B_obs_4-66.ckpt','--reference',str(REF),'--ddim-steps','4','--guidance-steps','2','--execution-steps','1','--guidance-scale',str(scale),'--reference-interpolation-threshold','0.12']
    (folder/'commands.json').write_text(json.dumps(dict(sim=simcmd,server=servercmd),indent=2)+'\n')
    if not (folder/'summary.json').exists():
        server=None
        try:
            if servercmd:
                with (folder/'server.log').open('w') as log:server=subprocess.Popen(servercmd,env=DPENV,stdout=log,stderr=subprocess.STDOUT)
                start=time.monotonic()
                while not sock.exists():
                    if server.poll() is not None:raise RuntimeError('Server failed: '+name)
                    if time.monotonic()-start>120:raise TimeoutError(name)
                    time.sleep(.2)
            with (folder/'sim.log').open('w') as log:subprocess.run(simcmd,env=SIMENV,stdout=log,stderr=subprocess.STDOUT,check=True)
            if server is not None:assert server.wait(timeout=30)==0
        finally:
            if server is not None and server.poll() is None:server.terminate();server.wait()
            sock.unlink(missing_ok=True)
    summary=json.loads((folder/'summary.json').read_text());trace=json.loads((folder/'trace.json').read_text())
    assert summary['native_protocol']==PROTOCOL and summary['steps']==summary['intended_steps']
    assert summary['init_order']=='properties; simulate/fetch/observe; root import'
    assert not summary['stop_on_native_failure']
    gpath=folder/'grasp_metrics.json'
    geo=json.loads(gpath.read_text())['frames'] if gpath.exists() else metrics(folder,all_frames=True)['frames']
    flags=[]
    for row,g in zip(trace,geo):
        force=float(np.linalg.norm(row['object_contact_force']));g['object_contact_force_norm_N']=force
        g['separated']=bool((g['mesh_vertex_gap_m']>.005 and force<.05) or g['mesh_vertex_gap_m']>.02);flags.append(g['separated'])
    sep=next((i for i in range(len(flags)-2) if all(flags[i:i+3])),len(trace))
    good=[];best=[];spans=[]
    for i,(row,g) in enumerate(zip(trace,geo)):
        ok=i<sep and row['phase']=='action' and row['vertical_error_deg']<=30 and g['mesh_table_clearance_m']>.08 and g['near_contact_link_count']>=2 and g['mesh_vertex_gap_m']<.008 and g['object_contact_force_norm_N']>.1
        if ok:
            good.append(row['index']+1)
            if len(good)>len(best):best=good.copy()
        else:
            if good:spans.append([good[0],good[-1],len(good)])
            good=[]
    result=dict(name=name,mass=mass,mu=mu,scale=scale,baseline_qualified=len(best)>=30,longest_vertical_steps=len(best),vertical_intervals=spans,separation_phase=trace[sep]['phase'] if sep<len(trace) else None,separation_control_step=trace[sep]['index']+1 if sep<len(trace) else None,native_failure=summary['first_native_failure'],first_actual_step=dict(displacement_m=trace[0]['displacement_from_import_m'],rotation_deg=trace[0]['rotation_from_import_deg']),settle_end_angle=trace[59]['vertical_error_deg'])
    if scale is not None:
        a=[x for x in trace if x['phase']=='action'];result['angle_at_steps_30_60_90_120']=[a[i-1]['vertical_error_deg'] for i in (30,60,90,120)]
    (folder/'diagnostic_result.json').write_text(json.dumps(result,indent=2)+'\n')
    (folder/'retention.json').write_text(json.dumps(dict(frames=geo,first_separation=dict(control_step=result['separation_control_step'],reference_action_number=float(np.load(folder/'reference_progress.npy')[trace[sep]['index']]) if scale is not None and sep<len(trace) else result['separation_control_step'])),indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result

def main():
    OUT.mkdir(exist_ok=True)
    results=[]
    # Reproduction validates current runtime before interpreting interventions.
    results.append(run('reproduce_m044_mu11_s25',.044,1.1,25))
    new=json.loads((OUT/'reproduce_m044_mu11_s25/trace.json').read_text())
    old=json.loads((CASE/'guide1_adaptive012_ddim4_scale25_m044_mu11/trace.json').read_text())
    keys=('q','command','executed_target','object_pose','vertical_error_deg','native_failure','hand_body_pose')
    parity={k:all(a[k]==b[k] for a,b in zip(new,old)) for k in keys}
    parity['same_length']=len(new)==len(old)
    (OUT/'runtime_reproduction.json').write_text(json.dumps(parity,indent=2)+'\n')
    assert all(parity.values()),parity
    for mass,mu,label in ((.17,1.1,'m170_mu11'),(.044,2.2,'m044_mu22')):
        baseline=run('direct_'+label,mass,mu);results.append(baseline)
        if baseline['baseline_qualified']:
            for scale in (25,50):results.append(run('ddim4_'+label+f'_s{scale}',mass,mu,scale))
        else:print('BASELINE INELIGIBLE, no guided comparisons:',label,flush=True)
        (OUT/'factorial_results.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
